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
    ppo_episodes: int = 1000
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
    # Match seeds played per held-out arena per side (paired across the side swap).
    match_seeds_per_arena: int = 12
    # Held-out grid for the head-to-head (its own disjoint population).
    head_to_head_grid: str = "full"

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
