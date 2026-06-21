"""output/modal_player_sweep.py — REAL PPO sweep on Modal, in PARALLEL.

Two jobs in one file:

  * ``sweep``        — dispatch ~6 difficulties x 3 seeds as CONCURRENT Modal jobs
                       through OUR harness.modal_fanout (FanoutPlayerTrainer ->
                       PlayerFanout._run_modal -> fn.map), aggregate the per-seed
                       real PPO learning into a frozen ExperimentResult, and
                       report wall-clock + job count. Compares against a
                       sequential-time estimate (one remote call per arena).

  * ``gate``         — THE CORRELATION GATE (the science check before Teacher RFT):
                       on ~10-20 arenas compute BOTH the cheap gap proxy
                       (scripted-strong vs random-weak score gap, the same number
                       score_curriculum/the difficulty sweep use) AND the REAL PPO
                       learning improvement (train a Player on Modal, measure
                       after-before). Report Pearson + Spearman correlation and a
                       scatter summary. If the proxy does NOT predict real PPO
                       learning, RFT would optimize the wrong thing — and the gate
                       SAYS SO.

Run from repo root (.venv has modal + sb3):
    .venv/bin/python output/modal_player_sweep.py sweep
    .venv/bin/python output/modal_player_sweep.py gate
    .venv/bin/python output/modal_player_sweep.py all      # sweep then gate

The real PPO worker (modal_player.train_player) must already be deployed:
    modal deploy modal_player.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

# Repo root on path (script lives in output/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import MatchResult  # noqa: E402
from harness.modal_fanout import (  # noqa: E402
    FanoutConfig,
    PlayerFanout,
    aggregate_experiment_result,
)

OUTPUT_DIR = Path(__file__).resolve().parent

# Sweep grid: 6 difficulties spanning the curve (easy -> the learnable band).
SWEEP_DIFFICULTIES = [0.2, 0.4, 0.6, 0.75, 0.85, 0.9]
SWEEP_SEEDS = [1, 2, 3]
# Short-but-real PPO budget per job (this is a trend check, not a tuned agent).
SWEEP_EPISODES = 800
SWEEP_EVAL_SEEDS = 40

MODAL_APP = "crucible-player"
MODAL_FN = "train_player"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Arena:
    """Minimal arena handle carrying a spec dict (read by FanoutPlayerTrainer)."""

    def __init__(self, payload: dict):
        self.spec = dict(payload)


def _modal_fn():
    import modal

    return modal.Function.from_name(MODAL_APP, MODAL_FN)


def _payload(difficulty: float, *, episodes: int, eval_seeds: int) -> dict:
    return {
        "difficulty": float(difficulty),
        "ppo_episodes": int(episodes),
        "eval_seeds": int(eval_seeds),
        "curriculum_id": f"d{difficulty}",
        "architecture": "mlp",
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _spearman(xs: list[float], ys: list[float]) -> float:
    def _rank(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    return _pearson(_rank(xs), _rank(ys))


# ---------------------------------------------------------------------------
# Task 3: parallel sweep on Modal
# ---------------------------------------------------------------------------


def run_sweep() -> dict:
    """Dispatch (difficulty x seed) REAL PPO jobs concurrently via fan-out.

    Each (difficulty, seed) is ONE Modal job. We submit one job per seed for each
    difficulty through FanoutPlayerTrainer (modal_parallel=True), which routes to
    PlayerFanout._run_modal -> fn.map — concurrent remote execution. We also time
    a single remote call to estimate the sequential cost and prove the parallel
    win.
    """
    n_jobs = len(SWEEP_DIFFICULTIES) * len(SWEEP_SEEDS)
    print(f"=== PARALLEL Modal PPO sweep: {len(SWEEP_DIFFICULTIES)} difficulties "
          f"x {len(SWEEP_SEEDS)} seeds = {n_jobs} concurrent jobs ===")
    print(f"    per-job budget: {SWEEP_EPISODES} episodes, {SWEEP_EVAL_SEEDS} eval seeds")

    # Measure a single remote call first (a sequential-per-job baseline). This is
    # one warm call; the parallel wall-clock below covers ALL n_jobs at once.
    fn = _modal_fn()
    t0 = time.time()
    one = fn.remote(_payload(0.4, episodes=SWEEP_EPISODES, eval_seeds=SWEEP_EVAL_SEEDS), 999)
    single_call_s = time.time() - t0
    assert one.get("status") == "ppo", f"expected real PPO result, got {one!r}"
    print(f"    single warm remote call: {single_call_s:.1f}s "
          f"(sequential estimate for {n_jobs} jobs ~= {single_call_s * n_jobs:.0f}s)")

    matches: list[MatchResult] = []
    rows: list[dict] = []

    # Dispatch THROUGH OUR harness: PlayerFanout(backend="modal") -> _run_modal ->
    # fn.map. Each difficulty's seeds go out as ONE concurrent map; we run the 6
    # difficulties back-to-back, so the whole sweep is 6 concurrent maps of 3.
    fanout = PlayerFanout(
        FanoutConfig(backend="modal", max_workers=n_jobs, modal_app=MODAL_APP,
                     modal_function=MODAL_FN),
        local_worker=None,
    )
    t0 = time.time()
    for d in SWEEP_DIFFICULTIES:
        payload = _payload(d, episodes=SWEEP_EPISODES, eval_seeds=SWEEP_EVAL_SEEDS)
        d_results = fanout.run(payload, list(SWEEP_SEEDS))  # concurrent over seeds
        for res in d_results:
            assert res.get("status") == "ppo", f"non-PPO result: {res!r}"
            rows.append(res)
            matches.append(
                MatchResult(
                    seed=int(res["seed"]),
                    architecture="mlp",
                    curriculum_id=res["curriculum_id"],
                    arena_scores=[float(res["after_winrate"])],
                    mean_score=float(res["after_winrate"]),
                )
            )
    wall_s = time.time() - t0

    # Aggregate into the frozen ExperimentResult (real reward carried through —
    # here we pass the mean improvement as a stand-in reward for the sweep view).
    mean_improvement = sum(r["improvement"] for r in rows) / len(rows)
    experiment = aggregate_experiment_result(
        matches,
        reward=mean_improvement,
        curricula_ids=sorted({r["curriculum_id"] for r in rows}),
        p_value=1.0,
        baseline=0.0,
    )

    speedup = (single_call_s * n_jobs) / wall_s if wall_s > 0 else float("inf")
    summary = {
        "n_jobs": n_jobs,
        "difficulties": SWEEP_DIFFICULTIES,
        "seeds": SWEEP_SEEDS,
        "episodes": SWEEP_EPISODES,
        "eval_seeds": SWEEP_EVAL_SEEDS,
        "wall_clock_s": round(wall_s, 2),
        "single_call_s": round(single_call_s, 2),
        "sequential_estimate_s": round(single_call_s * n_jobs, 1),
        "speedup_vs_sequential": round(speedup, 2),
        "parallel_beat_sequential": wall_s < single_call_s * n_jobs,
        "mean_improvement": round(mean_improvement, 4),
        "rows": rows,
        "experiment_result": experiment.model_dump(),
    }

    print(f"    PARALLEL wall-clock for all {n_jobs} jobs: {wall_s:.1f}s")
    print(f"    speedup vs sequential estimate: {speedup:.2f}x "
          f"(beat sequential: {summary['parallel_beat_sequential']})")
    print(f"    mean real PPO improvement across sweep: {mean_improvement:+.4f}")

    out = OUTPUT_DIR / "modal_player_sweep_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"    wrote {out}")
    return summary


# ---------------------------------------------------------------------------
# Task 4: THE CORRELATION GATE
# ---------------------------------------------------------------------------

# Arenas for the gate: difficulties spanning the curve, repeated so we get
# 10-20 (arena, gap_proxy, real_improvement) points. We vary difficulty (the
# Teacher's primary dial) at fixed default geometry so the proxy and the PPO
# signal see the SAME arena.
GATE_DIFFICULTIES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
GATE_SEEDS = [11, 22]          # 2 PPO seeds per difficulty -> 20 PPO points
GATE_EPISODES = 1000
GATE_EVAL_SEEDS = 50


def _gap_proxy(difficulty: float) -> float:
    """The CHEAP proxy: scripted-strong vs random-weak win-rate gap on the arena.

    This is exactly what harness.scoring.score_curriculum computes (strong_score
    - weak_score), via the same FighterGameAdapter.evaluate path. No PPO training.
    """
    from games.fighter import FighterArena
    from harness.fighter_adapter import FighterGameAdapter

    adapter = FighterGameAdapter(eval_seeds=GATE_EVAL_SEEDS)
    arenas = adapter.arenas_from_configs(
        [FighterArena(difficulty=difficulty)], curriculum_id=f"gate-d{difficulty}"
    )

    def _score(policy) -> float:
        (entry,) = adapter.evaluate(policy, arenas).values()
        return float(entry["mean_score"])

    strong = _score(adapter.scripted_expert())
    weak = _score(adapter.random_policy())
    return strong - weak


def run_gate() -> dict:
    """THE CORRELATION GATE: does the cheap gap proxy predict real PPO learning?

    (a) gap proxy   — computed locally (cheap, no training), per difficulty.
    (b) real PPO    — train a Player on Modal per (difficulty, seed), measure
                      held-out after-before improvement.
    Correlate (a) vs (b). Positive correlation => the proxy is a valid RFT
    objective. Non-positive => RFT would optimize the wrong thing.
    """
    print("=== THE CORRELATION GATE: gap proxy vs REAL PPO learning ===")
    print(f"    {len(GATE_DIFFICULTIES)} difficulties x {len(GATE_SEEDS)} seeds "
          f"= {len(GATE_DIFFICULTIES) * len(GATE_SEEDS)} PPO arenas")

    # (a) cheap proxy per difficulty (local).
    print("    computing gap proxy (local, cheap)...")
    proxy_by_d = {d: _gap_proxy(d) for d in GATE_DIFFICULTIES}
    for d in GATE_DIFFICULTIES:
        print(f"      d={d:<4} gap_proxy={proxy_by_d[d]:+.3f}")

    # (b) real PPO improvement per (difficulty, seed), dispatched in PARALLEL.
    print("    training real PPO Players on Modal (parallel)...")
    import modal

    fn = modal.Function.from_name(MODAL_APP, MODAL_FN)
    flat = [
        (d, s, _payload(d, episodes=GATE_EPISODES, eval_seeds=GATE_EVAL_SEEDS))
        for d in GATE_DIFFICULTIES
        for s in GATE_SEEDS
    ]
    payloads = [p for _, _, p in flat]
    seeds = [s for _, s, _ in flat]

    t0 = time.time()
    results = list(fn.map(payloads, seeds))
    wall_s = time.time() - t0
    print(f"    PPO sweep wall-clock: {wall_s:.1f}s for {len(flat)} jobs")

    points = []  # (difficulty, seed, gap_proxy, real_improvement)
    for (d, s, _), res in zip(flat, results):
        assert res.get("status") == "ppo", f"non-PPO result: {res!r}"
        points.append(
            {
                "difficulty": d,
                "seed": s,
                "gap_proxy": proxy_by_d[d],
                "improvement": float(res["improvement"]),
                "before": float(res["before_winrate"]),
                "after": float(res["after_winrate"]),
            }
        )

    xs = [p["gap_proxy"] for p in points]
    ys = [p["improvement"] for p in points]
    pearson = _pearson(xs, ys)
    spearman = _spearman(xs, ys)

    # Diagnostic: correlation of (-difficulty) vs improvement. If this is much
    # stronger than the gap-proxy correlation, the proxy's FAILURE is its own
    # non-monotonicity (it humps in the middle) — real PPO learning is NOT
    # unpredictable, the gap proxy is just the wrong predictor.
    neg_difficulty = [-p["difficulty"] for p in points]
    pearson_difficulty = _pearson(neg_difficulty, ys)

    # Per-difficulty mean improvement (the scatter summary at arena granularity).
    by_d = {}
    for d in GATE_DIFFICULTIES:
        imps = [p["improvement"] for p in points if p["difficulty"] == d]
        by_d[d] = {
            "gap_proxy": round(proxy_by_d[d], 4),
            "mean_improvement": round(sum(imps) / len(imps), 4),
        }

    # Gate verdict — called HONESTLY against precommitted thresholds. A near-zero
    # correlation is NOT a pass: r in [-0.2, 0.2] is statistically
    # indistinguishable from no relationship, so the proxy does not predict real
    # PPO learning and RFT against it would optimize the WRONG thing.
    #   PASS  : r >= 0.5  (the proxy meaningfully predicts PPO learning)
    #   WEAK  : 0.2 <= r < 0.5  (some signal, but a risky RFT objective)
    #   FAIL  : r < 0.2  (no usable relationship — do NOT RFT against this proxy)
    GATE_PASS = 0.5
    GATE_WEAK = 0.2
    r = pearson if not math.isnan(pearson) else 0.0
    if r >= GATE_PASS:
        verdict = "PASS"
    elif r >= GATE_WEAK:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    # "proxy predicts PPO" is true ONLY for a meaningful (>= WEAK) positive r.
    positive = r >= GATE_WEAK

    print()
    print("    --- scatter (per difficulty) ---")
    print(f"    {'d':>5} {'gap_proxy':>10} {'mean_improve':>13}")
    for d in GATE_DIFFICULTIES:
        print(f"    {d:>5} {by_d[d]['gap_proxy']:>10.3f} {by_d[d]['mean_improvement']:>13.3f}")
    print()
    print(f"    Pearson r  (gap_proxy vs real PPO improvement) = {pearson:+.3f}")
    print(f"    Spearman rho                                   = {spearman:+.3f}")
    print(f"    GATE VERDICT: {verdict}")
    if verdict == "PASS":
        print("      => the gap proxy POSITIVELY predicts real PPO learning.")
        print("         RFT against the proxy optimizes the right thing.")
    elif verdict == "WEAK":
        print("      => the gap proxy weakly predicts real PPO learning (risky objective).")
        print("         RFT against the proxy is NOT safe without a stronger proxy.")
    else:
        print("      => the gap proxy does NOT predict real PPO learning (r ~ 0).")
        print("         RFT against the proxy would optimize the WRONG thing. DO NOT RFT.")

    summary = {
        "n_points": len(points),
        "difficulties": GATE_DIFFICULTIES,
        "seeds": GATE_SEEDS,
        "episodes": GATE_EPISODES,
        "eval_seeds": GATE_EVAL_SEEDS,
        "wall_clock_s": round(wall_s, 2),
        "pearson_r": round(pearson, 4) if not math.isnan(pearson) else None,
        "spearman_rho": round(spearman, 4) if not math.isnan(spearman) else None,
        "pearson_neg_difficulty_r": round(pearson_difficulty, 4) if not math.isnan(pearson_difficulty) else None,
        "gate_verdict": verdict,
        "proxy_predicts_ppo": bool(positive),
        "per_difficulty": by_d,
        "points": points,
    }
    out = OUTPUT_DIR / "correlation_gate_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"    wrote {out}")
    return summary


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("sweep", "all"):
        run_sweep()
        print()
    if cmd in ("gate", "all"):
        run_gate()
    if cmd not in ("sweep", "gate", "all"):
        print(f"unknown command {cmd!r}; use sweep | gate | all", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
