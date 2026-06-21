"""harness/fighter_adapter.py — the real GameAdapter for the Stage-1 fighter.

Conforms to the FROZEN ``GameAdapter`` interface (harness/interfaces.py). The
adapter OWNS the game: it builds arenas from a ``CurriculumSpec``, owns the two
reference policies (``scripted_expert`` == the scripted fighter, ``random_policy``),
and runs them in ``evaluate`` to produce a ScoreBundle.

Two arena-construction paths, both returning the SAME opaque ``_FighterArenaHandle``:
  * ``build(spec)``         — the contract path. Translates each gridworld-shaped
                              ``ArenaSpec`` (map_size / difficulty / ...) into
                              fighter geometry, deterministically. The
                              orchestrator only ever uses this.
  * ``arenas_from_configs`` — a convenience for the milestone / direct tests:
                              wrap explicit ``FighterArena`` objects. Lets
                              ``prove_ppo_learns.py`` dial platform_width /
                              gravity / knockback / spawn_gap directly to compare
                              two environments.

Scoring convention: a policy's "score" on an arena is its **win-rate as fighter 0
against the fixed scripted opponent (fighter 1)**, averaged over a fixed set of
seeds. This puts ``scripted_expert`` (≈ mirror of the opponent) high and
``random_policy`` low — exactly the strong-vs-weak gap the ONE scorer expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from contracts import ArenaSpec, CurriculumSpec
from games.fighter import (
    FighterArena,
    Policy,
    play_match,
    random_policy,
    scripted_fighter,
)
from harness.interfaces import Arena, GameAdapter

# How many seeds to average a win-rate over per arena. Fixed so scoring is
# deterministic and reproducible.
DEFAULT_EVAL_SEEDS = 25


# ---------------------------------------------------------------------------
# Opaque arena handle (the orchestrator never inspects this)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FighterArenaHandle:
    """What ``build`` returns: a concrete fighter arena plus the spec it came
    from. Opaque to the orchestrator — it only carries arenas between adapter
    methods. ``arena_spec`` is None for arenas built directly from a
    ``FighterArena`` (the milestone path)."""

    arena: FighterArena
    arena_spec: Optional[ArenaSpec] = None
    curriculum_id: str = ""


def _arena_from_spec(spec: ArenaSpec) -> FighterArena:
    """Deterministically translate a gridworld-shaped ArenaSpec into fighter
    geometry. The mapping is arbitrary but stable: bigger ``map_size`` => wider
    platform; higher ``difficulty`` => stronger gravity + knockback + tighter
    spawn (a harder fight). This lets a Teacher that emits ArenaSpecs still drive
    the fighter, without changing the frozen contract.
    """
    width = 6.0 + 0.5 * spec.map_size            # map_size 3..64 -> width 7.5..38
    gravity = 0.4 + 0.4 * spec.difficulty        # 0.4 .. 0.8
    knockback = 1.5 + 2.5 * spec.difficulty      # 1.5 .. 4.0
    spawn_gap = max(2.0, 0.4 * width)            # scale spawn with platform
    return FighterArena(
        platform_width=round(width, 3),
        gravity=round(gravity, 3),
        knockback=round(knockback, 3),
        spawn_gap=round(spawn_gap, 3),
    )


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class FighterGameAdapter(GameAdapter):
    """Real GameAdapter for the Stage-1 fighter (conforms to the frozen ABC)."""

    def __init__(self, *, eval_seeds: int = DEFAULT_EVAL_SEEDS):
        self._eval_seeds = eval_seeds

    # -- contract path ------------------------------------------------------

    def build(self, spec: CurriculumSpec) -> list[Arena]:
        """Construct one fighter arena per ArenaSpec in the curriculum."""
        return [
            _FighterArenaHandle(
                arena=_arena_from_spec(a),
                arena_spec=a,
                curriculum_id=spec.curriculum_id,
            )
            for a in spec.arenas
        ]

    # -- milestone / direct path -------------------------------------------

    def arenas_from_configs(
        self, configs: list[FighterArena], *, curriculum_id: str = "direct"
    ) -> list[Arena]:
        """Wrap explicit FighterArena objects as opaque arena handles.

        Used by ``prove_ppo_learns.py`` to vary platform_width / gravity /
        knockback / spawn_gap directly between config A and config B.
        """
        return [
            _FighterArenaHandle(arena=c, arena_spec=None, curriculum_id=curriculum_id)
            for c in configs
        ]

    # -- the two reference policies (game-owned) ----------------------------

    def scripted_expert(self) -> Policy:
        """STRONG reference: the scripted fighter (as fighter 0).

        Note: ``scripted_fighter`` needs the arena, but the GameAdapter contract
        exposes a parameterless ``scripted_expert()``. We return a *factory-bound*
        policy: a thin wrapper that, at evaluate time, is rebuilt per-arena. To
        keep the contract's parameterless shape while still being arena-aware,
        ``evaluate`` rebuilds the scripted policy for each arena and ``evaluate``
        is the only caller — so this method returns a sentinel the adapter
        recognises. See ``_resolve_policy``.
        """
        return _SCRIPTED_EXPERT

    def random_policy(self) -> Policy:
        """WEAK reference: uniform-random actions."""
        return _RANDOM_POLICY

    # -- evaluation / scoring ----------------------------------------------

    def evaluate(self, policy: Policy, arenas: list[Arena]) -> dict:
        """Run ``policy`` (as fighter 0) vs the fixed scripted opponent (fighter
        1) over each arena, averaging win-rate across a fixed seed set.

        Returns the ScoreBundle the ONE scorer expects:
            {"<label>": {"mean_score": float, "per_arena": [float, ...]}}
        The label is "scripted_expert" / "random_policy" for the two reference
        sentinels (so ``score_curriculum`` can pull them by name), else the
        policy's own ``label`` attr, else "player".
        """
        label = _label_for(policy)
        per_arena: list[float] = []
        for handle in arenas:
            arena = handle.arena
            agent = _resolve_policy(policy, arena, ego=0)
            opponent = scripted_fighter(arena, ego=1)
            wins = 0
            for s in range(self._eval_seeds):
                winner = play_match(arena, agent, opponent, seed=s)
                if winner == 0:
                    wins += 1
            per_arena.append(wins / self._eval_seeds)

        mean_score = sum(per_arena) / len(per_arena) if per_arena else 0.0
        return {label: {"mean_score": mean_score, "per_arena": per_arena}}


# ---------------------------------------------------------------------------
# Reference-policy sentinels + resolution
# ---------------------------------------------------------------------------

# The frozen GameAdapter exposes parameterless scripted_expert()/random_policy(),
# but the scripted fighter is arena-dependent. We hand back lightweight sentinels
# that evaluate() resolves into a concrete, arena-bound policy per arena. This
# keeps the contract's call shape intact while staying arena-aware.


class _ReferenceSentinel:
    """A named placeholder for a reference policy, resolved per-arena in evaluate."""

    def __init__(self, label: str):
        self.label = label

    def __call__(self, obs):  # pragma: no cover - sentinels are resolved first.
        raise RuntimeError(
            f"{self.label} sentinel must be resolved per-arena via evaluate(); "
            "it is not directly callable."
        )


_SCRIPTED_EXPERT = _ReferenceSentinel("scripted_expert")
_RANDOM_POLICY = _ReferenceSentinel("random_policy")


def _label_for(policy: Policy) -> str:
    if policy is _SCRIPTED_EXPERT:
        return "scripted_expert"
    if policy is _RANDOM_POLICY:
        return "random_policy"
    return getattr(policy, "label", "player")


def _resolve_policy(policy: Policy, arena: FighterArena, *, ego: int) -> Policy:
    """Turn a policy (sentinel or concrete callable) into an arena-bound callable."""
    if policy is _SCRIPTED_EXPERT:
        return scripted_fighter(arena, ego=ego)
    if policy is _RANDOM_POLICY:
        # Fixed seed offset per arena so random scoring is reproducible.
        return random_policy(seed=hash(arena) & 0xFFFF)
    # Already a concrete callable (e.g. a trained Player policy).
    return policy
