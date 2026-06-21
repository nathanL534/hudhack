"""eval/plots.py — the two money plots, REAL against dummy data.

Two figures the demo needs:

  1. p*(1-p) histogram — "our learnability metric has signal". Drawn from real
     score_env calls over random params (same idea as prove_signal, kept here so
     eval/ is self-contained for the deck).

  2. anti-gaming co-rise plot — teacher_step on x; GAME_1 reward AND held-out
     GAME_2 win-rate on y. The CLAIM: if both rise together, the Teacher learned
     genuine learnability, not GAME_1 memorization. Today it reads the dummy
     StepRecord log from rft_run.py (or synthesizes one if absent).

Run:  python eval/plots.py   (run training/rft_run.py first for real-ish curve 2)
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

from engine.games import GAME_1  # noqa: E402
from engine.score_env import score_env  # noqa: E402
from schemas import StepRecord  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HIST_PNG = os.path.join(HERE, "plot_learnability_hist.png")
CORISE_PNG = os.path.join(HERE, "plot_anti_gaming_corise.png")
RFT_LOG = os.path.join(os.path.dirname(HERE), "training", "rft_log.json")


def plot_learnability_hist(n_samples: int = 50) -> None:
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_samples):
        params = {
            "difficulty": float(rng.uniform(0, 1)),
            "obstacle_density": float(rng.uniform(0, 0.45)),
            "goal_distance": float(rng.uniform(0.1, 1)),
        }
        p = score_env(params, GAME_1)["p"]
        vals.append(p * (1 - p))

    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=20, range=(0, 0.25), color="#55A868", edgecolor="white")
    plt.axvline(0.2, color="crimson", linestyle="--", label="learnable gate = 0.2")
    plt.title("Learnability metric p*(1-p) over random envs")
    plt.xlabel("p * (1 - p)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HIST_PNG, dpi=110)
    print(f"saved -> {HIST_PNG}")


def _load_or_synthesize_log() -> list[StepRecord]:
    if os.path.exists(RFT_LOG):
        with open(RFT_LOG) as f:
            return json.load(f)
    # Synthesize a co-rising dummy curve so the plot is meaningful pre-training.
    rng = np.random.default_rng(1)
    log: list[StepRecord] = []
    for step in range(30):
        trend = step / 29.0
        log.append(
            StepRecord(
                teacher_step=step,
                game1_reward=float(np.clip(trend + rng.normal(0, 0.08), 0, 1)),
                game2_winrate=float(np.clip(trend * 0.9 + rng.normal(0, 0.08), 0, 1)),
            )
        )
    return log


def plot_anti_gaming_corise() -> None:
    log = _load_or_synthesize_log()
    steps = [r["teacher_step"] for r in log]
    g1 = [r["game1_reward"] for r in log]
    g2 = [r["game2_winrate"] for r in log]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, g1, marker="o", label="GAME_1 reward (trained)")
    plt.plot(steps, g2, marker="x", label="GAME_2 win-rate (held-out)")
    plt.title("Anti-gaming: held-out GAME_2 should co-rise with GAME_1 reward")
    plt.xlabel("teacher step")
    plt.ylabel("reward / win-rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CORISE_PNG, dpi=110)
    print(f"saved -> {CORISE_PNG}")


def main() -> int:
    plot_learnability_hist()
    plot_anti_gaming_corise()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
