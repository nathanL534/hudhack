"""output/stage6/config.py — the FAIRNESS INVARIANT, made structural.

The base-vs-trained comparison is only meaningful if base and trained Teachers are
scored on the IDENTICAL yardstick: same prompt + sampling for generation, same PPO
budget for the fresh Players, same held-out evaluation set. The legacy evaluator
relied on convention ("we pass the same args to both loops"). Stage 6 makes it
STRUCTURAL: one frozen ``EvalConfig`` object is built once and used to construct
EVERY payload for BOTH models. There is no per-model parameter anywhere — the only
thing that differs between base and trained is the Teacher HANDLE.

If you want to change a knob, you change it in ONE place and it applies to both
models by construction. ``EvalConfig.fingerprint()`` hashes the invariant so the
result JSON can prove the same config drove both halves.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvalConfig:
    """The single source of truth used IDENTICALLY for base and trained.

    Every field here is part of the fairness invariant. Frozen so it cannot drift
    mid-run; shared by reference so base and trained payloads are built the same.
    """

    # -- generation (prompt + sampling) — identical for both Teachers ---------
    temperature: float = 0.8
    # -- PPO budget for each fresh Player ------------------------------------
    # Larger budget is scoped to Stage-6 evaluation students only.
    ppo_episodes: int = 1500
    eval_seeds: int = 50
    architecture: str = "mlp"
    # -- experiment shape -----------------------------------------------------
    arenas_per_model: int = 5
    player_seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    held_out_grid: str = "full"  # "full" (15 arenas) | "diagonal" (5)
    # -- PRIMARY metric (Student-vs-Student head-to-head) shape ---------------
    # INDEPENDENT curriculum replicates: each generates fresh curricula from BOTH
    # Teachers, trains one Student per Teacher, freezes, fights. The CI is over
    # THESE replicates (not frames / duplicate matches), so >1 is required for a
    # real verdict.
    n_replicates: int = 4
    # Arenas the Teacher generates per curriculum (the Student trains on the SET).
    curriculum_arenas: int = 4
    # Base Teacher-GENERATION seed. Each replicate r uses ``curriculum_seed_base + r``
    # as its generation seed (applied IDENTICALLY to base and trained), so replicates
    # sample DIFFERENT curricula from the same Teacher — making the curriculum-level CI
    # capture curriculum-generation randomness, not just Student-training noise. Part of
    # the fairness invariant (in the fingerprint) since it shapes both halves equally.
    curriculum_seed_base: int = 0
    # Match seeds played per held-out arena per side (paired across the side swap).
    match_seeds_per_arena: int = 12
    # Held-out grid for the head-to-head (its own disjoint population).
    head_to_head_grid: str = "full"
    # -- OPPONENT LEAGUE (Stage-6 Student training only) — DEFAULT OFF --------
    # When ``opponent_league_enabled`` is False (the default), every Stage-6 Student
    # trains against the single difficulty-scaled parametric opponent — the EXACT
    # current behavior, byte-identical. When True, each Student trains against a
    # SPREAD of opponent styles drawn from ``opponent_league`` (the fix for the
    # 48.5%-draw head-to-head). ``opponent_league=None`` => the league module's
    # ``DEFAULT_LEAGUE`` is used. These flags live in the fairness invariant and are
    # applied IDENTICALLY to the base-Student and trained-Student training calls, so
    # both Students always face the SAME opponent distribution.
    opponent_league_enabled: bool = False
    opponent_league: list | None = None
    # Optional prior-Student artifact path: when set AND the league includes a
    # "prior_student" style, that frozen earlier Student is loaded as an opponent
    # (self-play-lite). Applied identically to both sides; ``None`` => no prior
    # student, the prior-student league entry is dropped by ``resolve_league``.
    prior_student_path: str | None = None

    def fingerprint(self) -> str:
        """A short stable hash of the invariant (recorded to prove fairness)."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["player_seeds"] = list(self.player_seeds)
        d["fingerprint"] = self.fingerprint()
        return d

    # -- the ONE place a worker payload is shaped (used for BOTH models) ------

    def train_payload(self, params: dict, held_out: list[dict], *, param_keys,
                      curriculum_id: str) -> dict:
        """Build the Modal worker payload for one (arena) under THIS invariant.

        Identical for base and trained: only ``params`` (the arena the Teacher
        emitted) and ``curriculum_id`` differ between rows; the PPO budget, eval
        seed count, architecture, and held-out set come from the shared config.
        """
        payload = {
            "difficulty": float(params["difficulty"]),
            "ppo_episodes": int(self.ppo_episodes),
            "eval_seeds": int(self.eval_seeds),
            "curriculum_id": curriculum_id,
            "architecture": self.architecture,
            "held_out_arenas": held_out,
        }
        for k in param_keys:
            if k == "difficulty":
                continue
            if k in params:
                payload[k] = float(params[k])
        return payload

    # -- league plumbing — applied IDENTICALLY to both Students (invariant) ----

    def apply_opponent_league(self, payload: dict) -> dict:
        """Inject the Stage-6 opponent-league flags into a Student-train payload.

        Called once per Student payload from the SINGLE payload builder, so the
        base-Student and trained-Student payloads receive the SAME league config by
        construction (the fairness invariant). When the league is disabled (the
        default) this adds NOTHING — the payload is returned unchanged, so the
        Student worker takes its original single-parametric-opponent path.
        """
        if not self.opponent_league_enabled:
            return payload
        payload["opponent_league_enabled"] = True
        if self.opponent_league is not None:
            payload["opponent_league"] = self.opponent_league
        if self.prior_student_path is not None:
            payload["prior_student_path"] = self.prior_student_path
        return payload


def smoke_config() -> EvalConfig:
    """A tiny deterministic config for the concurrency smoke (2x2 matrix)."""
    return EvalConfig(
        temperature=0.8,
        ppo_episodes=300,
        eval_seeds=10,
        arenas_per_model=2,
        player_seeds=(1, 2),
        held_out_grid="full",
        n_replicates=2,
        curriculum_arenas=2,
        match_seeds_per_arena=4,
        head_to_head_grid="diagonal",
    )


def head_to_head_smoke_config() -> EvalConfig:
    """The CHEAPEST head-to-head config — minimal PPO, 2 replicates, small grid.

    Just enough to exercise the full primary path (generate -> train 2 Students ->
    freeze -> side-swapped head-to-head -> curriculum-level CI) without a full
    Gate-4 budget. NOT a verdict (2 replicates / tiny budget is a smoke).
    """
    return EvalConfig(
        temperature=0.8,
        ppo_episodes=150,
        eval_seeds=8,
        arenas_per_model=2,
        player_seeds=(1,),
        held_out_grid="diagonal",
        n_replicates=2,
        curriculum_arenas=2,
        match_seeds_per_arena=6,
        head_to_head_grid="diagonal",
    )
