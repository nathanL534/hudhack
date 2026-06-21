"""training/rft_run.py — the training driver.

SKELETON + runnable DUMMY loop. Today it runs a local "fake RFT" loop so the
whole project executes end-to-end:

    for step in range(N):
        params = teacher.generate(GAME_1)        # the rollout
        score  = score_env(params, GAME_1)       # the env evaluation
        r      = learnability(...)               # the reward
        log a StepRecord

It produces a list[StepRecord] (the pinned schema) and a dummy learning-curve
PNG. Because the stub Teacher is random, the curve will NOT trend up — that's
expected; it's a wiring test, not a result.

>>> WHERE THE REAL FIREWORKS RFT GOES (TODO, rag26) <<<
    Replace the local loop body with Fireworks Reinforcement Fine-Tuning:
      * Teacher rollouts run on Fireworks (Qwen3-4B) instead of RandomTeacher.
      * score_env becomes the reward fn behind a RemoteRolloutProcessor — the
        HUD harness ships each generated `params` out, scores it, and returns the
        learnability reward to the RFT optimizer.
      * StepRecord stays the contract: log teacher_step, game1_reward, and the
        held-out game2_winrate each optimizer step (see _heldout_game2_winrate).
    Keep the StepRecord logging + the PNG; just swap the loop body.

Run:  python training/rft_run.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from engine.games import GAME_1, GAME_2  # noqa: E402
from engine.score_env import score_env  # noqa: E402
from schemas import StepRecord  # noqa: E402
from training.reward import learnability  # noqa: E402
from training.teacher import default_teacher  # noqa: E402

N_STEPS = 30
OUT_PNG = os.path.join(os.path.dirname(__file__), "learning_curve.png")
OUT_LOG = os.path.join(os.path.dirname(__file__), "rft_log.json")


def _heldout_game2_winrate(params: dict) -> float:
    """Anti-gaming probe: how learnable are these params on the HELD-OUT GAME_2?

    Reuses the same env params on a different game mapping. If the Teacher is
    really learning learnability (not memorizing GAME_1), this co-rises with the
    GAME_1 reward. Here it's just the strong-weak gap on GAME_2.
    """
    s = score_env(params, GAME_2)
    return max(0.0, s["strong_score"] - s["weak_score"])


def run(n_steps: int = N_STEPS) -> list[StepRecord]:
    teacher = default_teacher(seed=0)
    log: list[StepRecord] = []

    for step in range(n_steps):
        # --- rollout ---  (TODO rag26: Fireworks Qwen3-4B generates here)
        params = teacher.generate(GAME_1)

        # --- reward ---   (TODO rag26: RemoteRolloutProcessor scores here)
        score = score_env(params, GAME_1)
        r = learnability(score["weak_score"], score["strong_score"], score["p"])

        # --- anti-gaming held-out probe ---
        g2 = _heldout_game2_winrate(params)

        log.append(StepRecord(teacher_step=step, game1_reward=float(r), game2_winrate=float(g2)))

    return log


def main() -> int:
    log = run()

    with open(OUT_LOG, "w") as f:
        json.dump(log, f, indent=2)
    print(f"wrote {len(log)} StepRecords -> {OUT_LOG}")

    steps = [rec["teacher_step"] for rec in log]
    rewards = [rec["game1_reward"] for rec in log]
    winrates = [rec["game2_winrate"] for rec in log]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, rewards, marker="o", label="GAME_1 learnability reward")
    plt.plot(steps, winrates, marker="x", label="GAME_2 held-out gap")
    plt.title("Dummy learning curve (random Teacher — wiring test, not a result)")
    plt.xlabel("teacher step")
    plt.ylabel("reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=110)
    print(f"saved learning curve -> {OUT_PNG}")

    print(
        f"mean GAME_1 reward={np.mean(rewards):.3f}  "
        f"mean GAME_2 gap={np.mean(winrates):.3f}  "
        "(flat is expected with the random stub Teacher)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
