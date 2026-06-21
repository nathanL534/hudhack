"""engine/prove_signal.py — the MIDNIGHT GATE.

Question we must answer before building anything on top of the engine:
    Does our learnability metric p*(1-p) actually have signal across random
    environments? If the median is near zero, every env is trivially solvable or
    trivially impossible, and there is nothing for the Teacher to learn.

This script:
  1. draws 50 random Game param sets
  2. scores each with score_env -> EnvScore
  3. computes the learnability quantity p*(1-p) for each
  4. prints the median and reports PASS/FAIL against the 0.2 gate
  5. saves a histogram PNG of p*(1-p)

Run:  python engine/prove_signal.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Allow `python engine/prove_signal.py` from repo root without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # headless: write PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from engine.games import GAME_1  # noqa: E402
from engine.score_env import score_env  # noqa: E402

N_SAMPLES = 50
GATE = 0.2
OUT_PNG = os.path.join(os.path.dirname(__file__), "prove_signal_hist.png")


def random_params(rng: np.random.Generator) -> dict:
    return {
        "difficulty": float(rng.uniform(0.0, 1.0)),
        "obstacle_density": float(rng.uniform(0.0, 0.45)),
        "goal_distance": float(rng.uniform(0.1, 1.0)),
    }


def main() -> int:
    rng = np.random.default_rng(0)
    learnability = []
    for _ in range(N_SAMPLES):
        params = random_params(rng)
        score = score_env(params, GAME_1)
        p = score["p"]
        learnability.append(p * (1.0 - p))

    arr = np.array(learnability)
    median = float(np.median(arr))
    passed = median > GATE

    print(f"samples={N_SAMPLES}  median p*(1-p)={median:.3f}  gate>{GATE}")
    print("MIDNIGHT GATE:", "PASS ✅" if passed else "FAIL ❌ (engine needs a wider learnable band)")

    plt.figure(figsize=(7, 4))
    plt.hist(arr, bins=20, range=(0, 0.25), color="#4C72B0", edgecolor="white")
    plt.axvline(GATE, color="crimson", linestyle="--", label=f"gate = {GATE}")
    plt.axvline(median, color="black", linestyle="-", label=f"median = {median:.3f}")
    plt.title("Learnability signal: distribution of p*(1-p)")
    plt.xlabel("p * (1 - p)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=110)
    print(f"saved histogram -> {OUT_PNG}")

    # Report-only: don't hard-exit nonzero, the scaffold should always run green.
    # TODO(nathan): once the engine is tuned, make this `return 0 if passed else 1`
    # so CI / the midnight check can gate on it.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
