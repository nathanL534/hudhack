"""E2E driver: verify OUR harness.modal_fanout against REAL Modal + local fallback.

Run from repo root:
    .venv/bin/python output/modal_fanout_e2e.py

Exercises BOTH backends through the SAME frozen interface
(FanoutPlayerTrainer -> aggregate_experiment_result -> ExperimentResult):

  * modal_parallel=True  -> dispatches train_player to the deployed
                            'crucible-player' Modal app (real remote run).
  * modal_parallel=False -> runs local_smoke_worker in-process (no creds).

Asserts the aggregated ExperimentResult has the required shape and that both
backends agree (the smoke worker is deterministic in seed), then prints both.
"""

from __future__ import annotations

import json

from contracts import ExperimentResult, MatchResult, PlayerConfig, TrainingBudget
from harness.modal_fanout import FanoutPlayerTrainer, aggregate_experiment_result
from modal_player import local_smoke_worker

SEEDS = [1, 2, 3, 4, 5]
ARCH = "mlp"
CURRICULUM = "smoke-curriculum"


class _Arena:
    def __init__(self, curriculum_id: str):
        self.spec = {"curriculum_id": curriculum_id}


def _run(modal_parallel: bool) -> ExperimentResult:
    arenas = [_Arena(CURRICULUM)]
    trainer = FanoutPlayerTrainer(local_smoke_worker, max_workers=8)
    matches: list[MatchResult] = []
    for seed in SEEDS:
        cfg = PlayerConfig(
            architecture=ARCH,
            num_seeds=len(SEEDS),
            budget=TrainingBudget(episodes=1),
            seed=seed,
            modal_parallel=modal_parallel,
        )
        job = trainer.submit(cfg, arenas)
        matches.append(job.result())
    return aggregate_experiment_result(
        matches,
        reward=0.123,
        curricula_ids=[CURRICULUM],
        p_value=0.04,
        baseline=0.5,
    )


REQUIRED = ["per_seed_scores", "delta_per_arch", "n_seeds_completed", "p_value"]


def _check_shape(label: str, result: ExperimentResult) -> None:
    assert isinstance(result, ExperimentResult), f"{label}: not an ExperimentResult"
    for field in REQUIRED:
        assert hasattr(result, field), f"{label}: missing {field}"
    assert result.n_seeds_completed == len(SEEDS), (
        f"{label}: n_seeds_completed={result.n_seeds_completed} != {len(SEEDS)}"
    )
    assert ARCH in result.per_seed_scores, f"{label}: arch missing from per_seed_scores"
    assert len(result.per_seed_scores[ARCH]) == len(SEEDS), f"{label}: wrong seed count"
    assert ARCH in result.delta_per_arch, f"{label}: arch missing from delta_per_arch"
    assert isinstance(result.p_value, float), f"{label}: p_value not float"


def main() -> int:
    print("=== local fallback (modal_parallel=False) ===")
    local = _run(modal_parallel=False)
    _check_shape("local", local)
    print(json.dumps(local.model_dump(), indent=2, sort_keys=True))

    print("\n=== REAL Modal (modal_parallel=True) ===")
    remote = _run(modal_parallel=True)
    _check_shape("modal", remote)
    print(json.dumps(remote.model_dump(), indent=2, sort_keys=True))

    same = (
        local.per_seed_scores == remote.per_seed_scores
        and local.delta_per_arch == remote.delta_per_arch
    )
    print("\n=== verdict ===")
    print(f"shapes valid: True")
    print(f"local == modal (deterministic smoke worker agree): {same}")
    print("OK" if same else "MISMATCH")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
