#!/usr/bin/env python3
"""run_experiment.py — Phase 1 ACCEPTANCE TEST.

Wires FakeTeacher -> ExperimentRunner(FakeGameAdapter + FakePlayerTrainer,
num_seeds=4), runs the REAL reward, prints exactly five checkmarks, and saves
the ExperimentResult JSON to results/experiment_<curriculum_id>.json.

Exit code 0 on success; non-zero if any step fails. The reward is COMPUTED by
the real scorer, never hardcoded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from contracts import PlayerConfig, TrainingBudget
from harness.fakes import FakeGameAdapter, FakePlayerTrainer, FakeTeacher
from harness.runner import DefaultExperimentRunner
from harness.scoring import score_curriculum

NUM_SEEDS = 4
EXPECTED_ARENAS = 2


def main() -> int:
    teacher = FakeTeacher()
    game = FakeGameAdapter()
    trainer = FakePlayerTrainer()
    runner = DefaultExperimentRunner(trainer)

    player_config = PlayerConfig(
        architecture="mlp",
        num_seeds=NUM_SEEDS,
        budget=TrainingBudget(episodes=10),
        seed=0,
        modal_parallel=False,
    )

    # --- 1. curriculum validated ------------------------------------------
    spec = teacher.generate(game_spec={"game_id": "gridworld"}).validate()
    print("✓ curriculum validated")

    # --- 2. arenas instantiated -------------------------------------------
    arenas = game.build(spec)
    if len(arenas) != EXPECTED_ARENAS:
        print(f"FAIL: expected {EXPECTED_ARENAS} arenas, got {len(arenas)}", file=sys.stderr)
        return 1
    print(f"✓ {len(arenas)} arenas instantiated")

    # --- 3. seeded Player jobs + 4. real reward, via the runner -----------
    result = runner.run(teacher, game, player_config)
    if result.n_seeds_completed != NUM_SEEDS:
        print(
            f"FAIL: expected {NUM_SEEDS} seeds, got {result.n_seeds_completed}",
            file=sys.stderr,
        )
        return 1
    print(f"✓ {result.n_seeds_completed} seeded Player jobs completed")

    # The reward must be COMPUTED by the real scorer. Recompute independently to
    # prove it is not hardcoded, and confirm the runner agrees.
    scoring = score_curriculum(spec, game, mode="gap_proxy")
    if abs(scoring.reward - result.reward) > 1e-9:
        print(
            f"FAIL: runner reward {result.reward} != scorer reward {scoring.reward}",
            file=sys.stderr,
        )
        return 1
    print(f"✓ Teacher reward: {result.reward}")

    # --- 5. save the result as JSON ---------------------------------------
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"experiment_{spec.curriculum_id}.json"
    out_path.write_text(result.model_dump_json(indent=2))
    print("✓ experiment result saved as JSON")

    return 0


if __name__ == "__main__":
    sys.exit(main())
