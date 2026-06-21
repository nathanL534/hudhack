"""eval/headtohead.py — held-out head-to-head evaluation.

The FINAL claim of the project is a head-to-head: take two players (e.g. a
policy trained on Teacher-generated envs vs. a baseline) and compete them over
many held-out seeds. Report player A's win-rate and a one-sided binomial
p-value for "A beats B more than half the time".

This is runnable-ish today: it competes two DUMMY players (A is given a small
fixed edge) so the stats path is exercised. The real players plug into
`compete_on_seed`.

Run:  python eval/headtohead.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import HeadToHead  # noqa: E402

N_SEEDS = 30  # spec: N >= 30

# A Competitor decides, for a given seed, the "skill draw" it brings. Real
# players will instead run a policy on a GridWorld built from that seed and
# return their score. Signature is intentionally simple so it's easy to swap.
Competitor = Callable[[int], float]


def dummy_player(edge: float, seed_offset: int) -> Competitor:
    """A stand-in competitor: returns a noisy score with a fixed `edge`."""
    def play(seed: int) -> float:
        rng = np.random.default_rng(seed + seed_offset)
        return float(rng.normal(loc=edge, scale=1.0))

    return play


def compete_on_seed(seed: int, player_a: Competitor, player_b: Competitor) -> HeadToHead:
    """Run ONE head-to-head on a seed. TODO: build a GridWorld(seed) and roll
    each player's real policy; here we just compare their dummy scores."""
    a, b = player_a(seed), player_b(seed)
    if a > b:
        winner = "A"
    elif b > a:
        winner = "B"
    else:
        winner = "draw"
    return HeadToHead(seed=seed, winner=winner)


def _binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """One-sided binomial p-value: P(X >= k) under Binomial(n, p). No scipy."""
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
    return total


def run_headtohead(
    n_seeds: int = N_SEEDS,
    player_a: Competitor | None = None,
    player_b: Competitor | None = None,
) -> dict:
    """Run n_seeds matches; return win-rate for A and a one-sided binomial p."""
    player_a = player_a or dummy_player(edge=0.3, seed_offset=0)
    player_b = player_b or dummy_player(edge=0.0, seed_offset=9999)

    results = [compete_on_seed(seed, player_a, player_b) for seed in range(n_seeds)]
    wins_a = sum(1 for r in results if r["winner"] == "A")
    draws = sum(1 for r in results if r["winner"] == "draw")
    decisive = n_seeds - draws

    win_rate = wins_a / n_seeds
    # p-value tests "A wins at least wins_a of the decisive games by chance".
    p_value = _binom_sf(wins_a, decisive) if decisive else 1.0

    return {
        "n_seeds": n_seeds,
        "wins_a": wins_a,
        "draws": draws,
        "win_rate": win_rate,
        "p_value": p_value,
        "results": results,
    }


def main() -> int:
    out = run_headtohead()
    print(
        f"A wins {out['wins_a']}/{out['n_seeds']} "
        f"(draws={out['draws']})  win_rate={out['win_rate']:.2f}  "
        f"one-sided binomial p={out['p_value']:.4f}"
    )
    print("significant at 0.05:", out["p_value"] < 0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
