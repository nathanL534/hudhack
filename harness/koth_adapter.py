"""harness/koth_adapter.py — the GameAdapter for King of the Hill.

This is the DROP-IN proof of the transfer claim: ``KothGameAdapter`` conforms to
the SAME frozen ``GameAdapter`` interface (harness/interfaces.py) as
``FighterGameAdapter``, so KotH plugs into the harness / scorer / Modal / PPO
exactly like the fighter. A Teacher that emits ``ArenaSpec`` curricula drives
KotH through ``build`` without changing the frozen contract; PPO Players train on
the generated KotH arenas; the ONE scorer scores them — all unchanged.

It is a structural twin of ``harness/fighter_adapter.py`` (same opaque-handle
pattern, same reference-sentinel pattern, same scoring convention) so the only
thing that differs between the two games at the adapter layer is the
ArenaSpec -> geometry mapping and the underlying game module.

Two arena-construction paths, both returning the SAME opaque ``_KothArenaHandle``:
  * ``build(spec)``         — the contract path. Translates each gridworld-shaped
                              ``ArenaSpec`` into KotH geometry (platform + zone),
                              deterministically. The orchestrator only uses this.
  * ``arenas_from_configs`` — wrap explicit ``KothArena`` objects, for the
                              ``prove_koth_learns.py`` check and direct tests.

Scoring convention (identical to the fighter): a policy's "score" on an arena is
its **win-rate as player 0 against the parametric opponent (player 1)**, averaged
over a fixed seed set. This puts ``scripted_expert`` high and ``random_policy``
low — the strong-vs-weak gap the ONE scorer expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from contracts import ArenaSpec, CurriculumSpec
from games.koth import (
    KothArena,
    Policy,
    parametric_koth,
    play_match,
    random_policy,
    scripted_koth,
)
from harness.interfaces import Arena, GameAdapter

# How many seeds to average a win-rate over per arena (matches the fighter's
# default so the two adapters score on the same footing).
DEFAULT_EVAL_SEEDS = 25


# ---------------------------------------------------------------------------
# Opaque arena handle (the orchestrator never inspects this)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _KothArenaHandle:
    """What ``build`` returns: a concrete KotH arena plus the spec it came from.

    Opaque to the orchestrator — it only carries arenas between adapter methods.
    Mirrors ``_FighterArenaHandle`` so a Player trainer can pull the concrete
    arena off ``.arena`` exactly as it does for the fighter. ``arena_spec`` is
    None for arenas built directly from a ``KothArena`` (the direct path).
    """

    arena: KothArena
    arena_spec: Optional[ArenaSpec] = None
    curriculum_id: str = ""


def _arena_from_spec(spec: ArenaSpec) -> KothArena:
    """Deterministically translate a gridworld-shaped ArenaSpec into KotH geometry.

    The mapping is arbitrary but stable, and chosen so ``difficulty`` makes the
    arena genuinely HARDER (same direction as the fighter's mapping):
      * bigger ``map_size``  => wider platform (more ground to cover to the hill);
      * higher ``difficulty``=> SMALLER zone (harder to hold) and a zone pushed
        OFF-centre (a contested, asymmetric hill), plus a stronger parametric
        opponent via the difficulty field.
    This lets a Teacher that emits ArenaSpecs drive KotH without touching the
    frozen contract — the whole point of the drop-in.
    """
    width = 6.0 + 0.5 * spec.map_size                 # map_size 3..64 -> width 7.5..38
    # Zone half shrinks with difficulty: easy arenas have a big, easy-to-hold
    # hill; hard arenas a small one. Floor keeps it occupiable.
    zone_half = max(0.6, 0.22 * width * (1.0 - 0.6 * spec.difficulty))
    # Push the zone off-centre as difficulty rises (toward 0.5 +/- up to ~0.18),
    # so a hard hill is asymmetric. Deterministic, no rng.
    zone_center_frac = 0.5 + 0.18 * spec.difficulty
    return KothArena(
        platform_width=round(width, 3),
        zone_half=round(zone_half, 3),
        zone_center_frac=round(zone_center_frac, 3),
        # The difficulty dial also scales the PARAMETRIC opponent's strength
        # (high difficulty -> low epsilon -> reliable contester). Same direction
        # as the geometry knobs above, so the dial is monotone in hardness.
        difficulty=spec.difficulty,
    )


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class KothGameAdapter(GameAdapter):
    """Real GameAdapter for King of the Hill (conforms to the frozen ABC).

    Structural twin of ``FighterGameAdapter`` — same call shapes, same scoring
    convention — so KotH is a drop-in alternative game everywhere the fighter is
    used.
    """

    def __init__(self, *, eval_seeds: int = DEFAULT_EVAL_SEEDS):
        self._eval_seeds = eval_seeds

    # -- contract path ------------------------------------------------------

    def build(self, spec: CurriculumSpec) -> list[Arena]:
        """Construct one KotH arena per ArenaSpec in the curriculum."""
        return [
            _KothArenaHandle(
                arena=_arena_from_spec(a),
                arena_spec=a,
                curriculum_id=spec.curriculum_id,
            )
            for a in spec.arenas
        ]

    # -- direct path (for prove_koth_learns.py and tests) -------------------

    def arenas_from_configs(
        self, configs: list[KothArena], *, curriculum_id: str = "direct"
    ) -> list[Arena]:
        """Wrap explicit KothArena objects as opaque arena handles."""
        return [
            _KothArenaHandle(arena=c, arena_spec=None, curriculum_id=curriculum_id)
            for c in configs
        ]

    # -- the two reference policies (game-owned) ----------------------------

    def scripted_expert(self) -> Policy:
        """STRONG reference: the scripted KotH heuristic (resolved per-arena).

        Like the fighter adapter, the scripted policy is arena-dependent but the
        contract exposes a parameterless ``scripted_expert()``, so we hand back a
        named sentinel that ``evaluate`` resolves per-arena. See ``_resolve_policy``.
        """
        return _SCRIPTED_EXPERT

    def random_policy(self) -> Policy:
        """WEAK reference: uniform-random actions."""
        return _RANDOM_POLICY

    # -- evaluation / scoring ----------------------------------------------

    def evaluate(self, policy: Policy, arenas: list[Arena]) -> dict:
        """Run ``policy`` (as player 0) vs the PARAMETRIC opponent (player 1) over
        each arena, averaging win-rate across a fixed seed set.

        Identical scoring path to the fighter adapter: the opponent's strength
        scales with ``arena.difficulty`` so ``random_policy``'s win-rate (the
        solvability proxy ``p``) slides smoothly with difficulty instead of
        pinning. The opponent is seeded per eval-seed so its random mix is
        reproducible.

        Returns the ScoreBundle the ONE scorer expects:
            {"<label>": {"mean_score": float, "per_arena": [float, ...]}}
        """
        label = _label_for(policy)
        per_arena: list[float] = []
        for handle in arenas:
            arena = handle.arena
            agent = _resolve_policy(policy, arena, ego=0)
            wins = 0
            for s in range(self._eval_seeds):
                opponent = parametric_koth(arena, ego=1, seed=10_000 + s)
                winner = play_match(arena, agent, opponent, seed=s)
                if winner == 0:
                    wins += 1
            per_arena.append(wins / self._eval_seeds)

        mean_score = sum(per_arena) / len(per_arena) if per_arena else 0.0
        return {label: {"mean_score": mean_score, "per_arena": per_arena}}


# ---------------------------------------------------------------------------
# Reference-policy sentinels + resolution (mirror the fighter adapter)
# ---------------------------------------------------------------------------


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


def _resolve_policy(policy: Policy, arena: KothArena, *, ego: int) -> Policy:
    """Turn a policy (sentinel or concrete callable) into an arena-bound callable."""
    if policy is _SCRIPTED_EXPERT:
        return scripted_koth(arena, ego=ego)
    if policy is _RANDOM_POLICY:
        return random_policy(seed=hash(arena) & 0xFFFF)
    # Already a concrete callable (e.g. a trained Player policy).
    return policy
