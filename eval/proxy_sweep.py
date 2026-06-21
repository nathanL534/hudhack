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


# ---------------------------------------------------------------------------
# REAL fighter wiring (the correlation gate)
# ---------------------------------------------------------------------------
#
# Everything above is backend-agnostic so the deterministic test can drive it
# with fakes. The factories below bind the GENERIC sweep to the REAL fighter:
#
#   * ``make_gap_proxy_fn``     -> the cheap GAP PROXY that the hot Teacher RFT
#                                  loop would optimise (scripted-strong vs
#                                  random-weak win-rate gap), computed through
#                                  the SAME ``FighterGameAdapter.evaluate`` path
#                                  ``harness.scoring.score_curriculum`` uses.
#   * ``make_ppo_learning_fn``  -> the EXPENSIVE real signal: train a small PPO
#                                  Player on the arena and measure held-out
#                                  after-minus-before win-rate (the real PPO body
#                                  in ``modal_player``), locally or on Modal.
#
# The correlation gate runner (output/run_correlation_gate.py) builds these two
# and feeds them straight into ``run_proxy_sweep`` — no bespoke correlation math,
# the same Pearson/Spearman this module already computes.


def make_gap_proxy_fn(*, eval_seeds: int = 40) -> ProxyFn:
    """Build the cheap GAP PROXY ``proxy_fn`` over a real ``FighterArena``.

    The proxy is ``strong_score - weak_score`` — the scripted_expert vs
    random_policy win-rate gap on the arena — computed through
    ``FighterGameAdapter.evaluate``, the identical scoring path the ONE scorer
    (``harness.scoring.score_curriculum``) and the difficulty sweep use. No PPO
    training: this is what makes it cheap enough for the hot RFT loop.

    The returned fn takes a params dict carrying ``difficulty`` (and optionally
    the four geometry knobs) and returns the gap as a float. The adapter is built
    once and closed over, so repeated calls don't re-pay construction.
    """
    from games.fighter import FighterArena
    from harness.fighter_adapter import FighterGameAdapter

    adapter = FighterGameAdapter(eval_seeds=eval_seeds)
    defaults = FighterArena()

    def proxy_fn(params: Params) -> float:
        arena = FighterArena(
            platform_width=float(params.get("platform_width", defaults.platform_width)),
            gravity=float(params.get("gravity", defaults.gravity)),
            knockback=float(params.get("knockback", defaults.knockback)),
            spawn_gap=float(params.get("spawn_gap", defaults.spawn_gap)),
            difficulty=float(params.get("difficulty", defaults.difficulty)),
        )
        cid = f"gate-d{arena.difficulty}"
        arenas = adapter.arenas_from_configs([arena], curriculum_id=cid)

        def _score(policy) -> float:
            (entry,) = adapter.evaluate(policy, arenas).values()
            return float(entry["mean_score"])

        strong = _score(adapter.scripted_expert())
        weak = _score(adapter.random_policy())
        return strong - weak

    return proxy_fn


def make_ppo_learning_fn(
    *,
    backend: str = "local",
    episodes: int = 600,
    eval_seeds: int = 40,
    modal_app: str = "crucible-player",
    modal_function: str = "train_player",
) -> LearningFn:
    """Build the real PPO LEARNING ``learning_fn`` (after-minus-before win-rate).

    ``backend="local"`` runs the REAL PPO body (``modal_player.local_worker`` —
    SB3 PPO on the arena vs the difficulty-scaled parametric opponent) in this
    process. ``backend="modal"`` invokes the deployed ``train_player`` remotely
    (real parallelism when many arenas/seeds are mapped). Both return the SAME
    ``improvement`` field (held-out AFTER win-rate minus an untrained net's
    BEFORE), so the learning signal is identical whichever backend runs it.

    Note: this evaluates ONE (params, seed) per call (the generic sweep loops
    seeds itself). On Modal that's one ``fn.remote`` per call — fine for the
    gate's grid, and a thread-pooled caller still gets concurrency. The local
    backend pays the PPO cost in-process; keep ``episodes`` short.
    """
    def _payload(params: Params, seed: int) -> dict:
        p = {key: float(value) for key, value in params.items()}
        p.update(
            {
                "ppo_episodes": int(episodes),
                "eval_seeds": int(eval_seeds),
                "curriculum_id": f"gate-d{p.get('difficulty', 1.0)}",
                "architecture": "mlp",
            }
        )
        return p

    if backend == "local":
        from modal_player import local_worker

        def learning_fn(params: Params, seed: int) -> float:
            row = local_worker(_payload(params, seed), int(seed))
            return float(row["improvement"])

        return learning_fn

    if backend == "modal":
        import modal

        fn = modal.Function.from_name(modal_app, modal_function)

        def learning_fn(params: Params, seed: int) -> float:
            row = fn.remote(_payload(params, seed), int(seed))
            return float(row["improvement"])

        return learning_fn

    raise ValueError(f"unknown learning backend: {backend!r} (use 'local' or 'modal')")

