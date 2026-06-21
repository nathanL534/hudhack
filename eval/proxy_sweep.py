"""Stage 2: compare a cheap Teacher proxy with actual PPO learning gain.

This module is intentionally backend-agnostic.  Callers provide two functions:

* ``proxy_fn(params)``: cheap score suitable for the hot Teacher loop.
* ``learning_fn(params, seed)``: expensive before/after Player experiment.

The sweep persists every observation and reports Pearson/Spearman correlation.
The real fighter can be wired in after its randomized-start work stabilizes;
tests use deterministic fakes so this scaffold is usable immediately.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


Params = dict[str, float]
ProxyFn = Callable[[Params], float]
LearningFn = Callable[[Params, int], float]


@dataclass(frozen=True)
class SweepRow:
    candidate_id: str
    params: Params
    proxy_score: float
    learning_gains: list[float]

    @property
    def mean_learning_gain(self) -> float:
        return sum(self.learning_gains) / len(self.learning_gains)


@dataclass(frozen=True)
class SweepReport:
    rows: list[SweepRow]
    pearson_r: float
    spearman_r: float

    def model_dump(self) -> dict:
        return {
            "rows": [
                {**asdict(row), "mean_learning_gain": row.mean_learning_gain}
                for row in self.rows
            ],
            "pearson_r": self.pearson_r,
            "spearman_r": self.spearman_r,
        }


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denom if denom else 0.0


def _ranks(values: list[float]) -> list[float]:
    """Average ranks for ties, zero-based."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def run_proxy_sweep(
    candidates: Iterable[Params],
    *,
    proxy_fn: ProxyFn,
    learning_fn: LearningFn,
    seeds: Iterable[int],
    output_path: Path | None = None,
) -> SweepReport:
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("at least one learning seed is required")

    rows: list[SweepRow] = []
    for index, params in enumerate(candidates):
        normalized = {key: float(value) for key, value in params.items()}
        rows.append(
            SweepRow(
                candidate_id=f"candidate-{index:03d}",
                params=normalized,
                proxy_score=float(proxy_fn(normalized)),
                learning_gains=[
                    float(learning_fn(normalized, seed)) for seed in seed_list
                ],
            )
        )

    proxies = [row.proxy_score for row in rows]
    gains = [row.mean_learning_gain for row in rows]
    report = SweepReport(
        rows=rows,
        pearson_r=_pearson(proxies, gains),
        spearman_r=_pearson(_ranks(proxies), _ranks(gains)),
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.model_dump(), indent=2))
    return report

