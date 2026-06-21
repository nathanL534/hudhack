"""output/stage6/aggregate.py — statistics for the decider (no scipy dependency).

Pure, dependency-free aggregation so it runs anywhere (the worktree venv has numpy
+ matplotlib but NOT scipy). Covers:

  * ``mean`` / ``sample_std`` — basic moments.
  * ``mean_confidence_interval`` — a Student-t CI on the mean, using a small
    embedded t-table (two-sided 95% by default) so we need no scipy.
  * ``parameter_diversity`` — how varied the Teacher's generated arenas are
    (collapse-to-one-config is an anti-gaming red flag).

These are the only numbers the verdict and anti-gaming checks depend on, so they
are unit-tested directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Two-sided 95% Student-t critical values by degrees of freedom (df = n-1).
# Embedded so we avoid a scipy dependency. df>=31 uses the z=1.96 normal limit.
_T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}
_Z_95 = 1.96


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def sample_std(xs: list[float]) -> float:
    """Sample standard deviation (n-1 denominator). 0 for <2 points."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T_TABLE_95:
        return _T_TABLE_95[df]
    return _Z_95  # df > 30 -> normal approximation


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    low: float
    high: float
    half_width: float
    std: float
    n: int
    confidence: float = 0.95

    def as_dict(self) -> dict:
        return {
            "mean": round(self.mean, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
            "half_width": round(self.half_width, 6),
            "std": round(self.std, 6),
            "n": self.n,
            "confidence": self.confidence,
        }


def mean_confidence_interval(xs: list[float], confidence: float = 0.95) -> ConfidenceInterval:
    """Student-t CI on the mean (95% by default).

    For n < 2 the interval collapses to the point estimate with zero width (we
    cannot estimate variance from a single sample). The standard error is
    ``std / sqrt(n)`` and the half-width is ``t_crit * se``.
    """
    n = len(xs)
    m = mean(xs)
    if n < 2:
        return ConfidenceInterval(mean=m, low=m, high=m, half_width=0.0, std=0.0,
                                  n=n, confidence=confidence)
    s = sample_std(xs)
    se = s / math.sqrt(n)
    t = _t_critical_95(n - 1)  # only 95% is tabulated; confidence kept for record
    half = t * se
    return ConfidenceInterval(mean=m, low=m - half, high=m + half, half_width=half,
                              std=s, n=n, confidence=confidence)


def bootstrap_ci(
    xs: list[float], *, confidence: float = 0.95, n_boot: int = 2000, seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap CI on the mean — robust for small replicate counts.

    Resamples ``xs`` with replacement ``n_boot`` times, takes the mean of each
    resample, and reads the percentile interval. For n<2 it collapses to the point
    estimate (you cannot bootstrap a single sample). Used at the curriculum-
    replicate level where the t-CI's normality assumption is shakiest.
    """
    import random

    n = len(xs)
    m = mean(xs)
    if n < 2:
        return ConfidenceInterval(mean=m, low=m, high=m, half_width=0.0, std=0.0,
                                  n=n, confidence=confidence)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = 1.0 - confidence
    lo = means[int((alpha / 2.0) * n_boot)]
    hi = means[min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot))]
    half = (hi - lo) / 2.0
    return ConfidenceInterval(mean=m, low=lo, high=hi, half_width=half,
                              std=sample_std(xs), n=n, confidence=confidence)


def paired_advantage(trained_wins: int, base_wins: int, draws: int) -> dict:
    """Win/loss/draw rates + the paired win-rate advantage for ONE match population.

    ``advantage`` = trained_win_rate - base_win_rate. Draws count toward neither
    win rate (so the three rates sum to 1.0); the advantage is the head-to-head
    edge after draws are removed from both sides equally.
    """
    total = trained_wins + base_wins + draws
    if total == 0:
        return {"trained_win_rate": 0.0, "base_win_rate": 0.0, "draw_rate": 0.0,
                "advantage": 0.0, "n": 0}
    tw = trained_wins / total
    bw = base_wins / total
    dr = draws / total
    return {
        "trained_win_rate": round(tw, 6),
        "base_win_rate": round(bw, 6),
        "draw_rate": round(dr, 6),
        "advantage": round(tw - bw, 6),
        "n": total,
    }


def parameter_diversity(arenas: list[dict], param_keys) -> dict:
    """Quantify how varied a Teacher's generated arenas are.

    Collapse to a single configuration is a gaming red flag (the Teacher found one
    arena that scores well and emits only that). We report:
      * ``n_arenas`` / ``n_unique`` — distinct rounded param tuples.
      * ``unique_fraction`` — n_unique / n_arenas (1.0 = all distinct).
      * ``per_param_std`` — spread of each knob across arenas (0 = constant).
      * ``mean_param_std`` — average per-knob std (a single collapse scalar).
      * ``collapsed`` — True iff every arena is the SAME config (n_unique == 1).
    """
    keys = list(param_keys)
    n = len(arenas)
    if n == 0:
        return {
            "n_arenas": 0, "n_unique": 0, "unique_fraction": 0.0,
            "per_param_std": {k: 0.0 for k in keys}, "mean_param_std": 0.0,
            "collapsed": True,
        }
    # Distinct configurations (rounded to 6 dp to ignore float noise).
    tuples = {tuple(round(float(a.get(k, 0.0)), 6) for k in keys) for a in arenas}
    n_unique = len(tuples)
    per_param_std = {
        k: round(sample_std([float(a.get(k, 0.0)) for a in arenas]), 6) for k in keys
    }
    mean_param_std = round(mean(list(per_param_std.values())), 6) if keys else 0.0
    return {
        "n_arenas": n,
        "n_unique": n_unique,
        "unique_fraction": round(n_unique / n, 6),
        "per_param_std": per_param_std,
        "mean_param_std": mean_param_std,
        "collapsed": n_unique == 1,
    }
