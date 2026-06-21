#!/usr/bin/env python3
"""output/run_correlation_gate.py — RUN the correlation gate (the science check
that gates Teacher RFT).

THE QUESTION (one, answered empirically):

    Does the CHEAP gap proxy that the hot Teacher RFT loop would optimise
    actually PREDICT real PPO learning? If yes, RFT against the proxy optimises
    the right thing. If no, RFT would train Qwen on a lie — and this gate is the
    *save* that stops it.

HOW (through the REAL functions, via the generic eval.proxy_sweep scaffold):

  (a) cheap GAP PROXY per arena  — scripted-strong vs random-weak win-rate gap,
      ``eval.proxy_sweep.make_gap_proxy_fn`` (the SAME FighterGameAdapter.evaluate
      path harness.scoring.score_curriculum uses). No PPO. ~0.5s/arena.

  (b) real PPO LEARNING per arena x seed — train a small PPO Player on the arena
      (modal_player's real PPO body) and measure held-out AFTER-minus-BEFORE
      win-rate. Run on Modal in PARALLEL when the deployed worker is reachable,
      else locally with a short budget.

  correlate (a) vs (b) with Pearson + Spearman — computed by the generic
  ``run_proxy_sweep`` (no bespoke math) — and write the scatter + coefficients to
  output/correlation_gate_results.json.

VERDICT: Pearson/Spearman >= ~0.3-0.5 and positive => PASS (RFT unlocked).
Near-zero or negative => FAIL (the proxy is broken; DO NOT start RFT).

Run from repo root (.venv has modal + sb3):

    .venv/bin/python output/run_correlation_gate.py            # auto backend
    .venv/bin/python output/run_correlation_gate.py --local    # force local PPO
    .venv/bin/python output/run_correlation_gate.py --modal    # force Modal PPO

Modal mode needs the worker deployed:  modal deploy modal_player.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Repo root on path (this script lives in output/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from eval.proxy_sweep import make_gap_proxy_fn, run_proxy_sweep  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUTPUT_DIR / "correlation_gate_results.json"

MODAL_APP = "crucible-player"
MODAL_FN = "train_player"

# --- the gate grid ---------------------------------------------------------
# ~15 candidate arenas spanning the difficulty curve (the Teacher's primary
# dial) at fixed default geometry, so the cheap proxy and the real PPO signal
# see the IDENTICAL arena. The span deliberately crosses the learnable band
# (~0.85-0.9) AND the easy/hard tails so BOTH signals vary — a flat predictor or
# a flat target makes the correlation undefined, which would be a non-answer.
GATE_DIFFICULTIES = [
    0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95,
]
GATE_SEEDS = [11, 22, 33]  # 3 PPO seeds per arena

# Budgets: short but REAL. A budget too small reports 0 learning everywhere
# (zero variance -> undefined correlation), so this is calibrated to a budget
# where the learnable arenas show measurable lift while the hard tail stays low.
GATE_PROXY_EVAL_SEEDS = 40   # win-rate averaging seeds for the cheap proxy
GATE_PPO_EPISODES = 600      # ~36k timesteps/job
GATE_PPO_EVAL_SEEDS = 40     # held-out eval seeds for before/after win-rate


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _modal_reachable() -> bool:
    """True iff a real remote PPO job comes back from the deployed worker.

    ``Function.from_name`` is lazy (no network), so we actually invoke one tiny
    job and check the status. Any failure (no creds, not deployed, import error)
    falls back to local.
    """
    try:
        import modal

        fn = modal.Function.from_name(MODAL_APP, MODAL_FN)
        row = fn.remote(
            {
                "difficulty": 0.6,
                "ppo_episodes": 50,
                "eval_seeds": 5,
                "curriculum_id": "gate-probe",
            },
            0,
        )
        return isinstance(row, dict) and row.get("status") == "ppo"
    except Exception as exc:  # noqa: BLE001 - any failure => use local backend
        print(f"    [modal] unreachable ({type(exc).__name__}: {exc}); using local.")
        return False


# ---------------------------------------------------------------------------
# learning_fn builders (each returns a learning_fn(params, seed) -> improvement)
# ---------------------------------------------------------------------------


def _candidates() -> list[dict]:
    return [{"difficulty": float(d)} for d in GATE_DIFFICULTIES]


def _payload(difficulty: float, seed: int) -> dict:
    return {
        "difficulty": float(difficulty),
        "ppo_episodes": int(GATE_PPO_EPISODES),
        "eval_seeds": int(GATE_PPO_EVAL_SEEDS),
        "curriculum_id": f"gate-d{difficulty}",
        "architecture": "mlp",
    }


def _build_modal_learning_fn():
    """Dispatch ALL (arena, seed) PPO jobs to Modal in ONE parallel map, then
    return a learning_fn that just reads the precomputed improvement.

    The generic ``run_proxy_sweep`` loops seeds sequentially; pre-dispatching via
    ``fn.map`` gives us TRUE Modal parallelism (all 45 jobs in flight at once)
    while still routing the correlation through the generic scaffold. Returns
    (learning_fn, rows, wall_clock_s).
    """
    import modal

    fn = modal.Function.from_name(MODAL_APP, MODAL_FN)
    flat = [(d, s) for d in GATE_DIFFICULTIES for s in GATE_SEEDS]
    payloads = [_payload(d, s) for d, s in flat]
    seeds = [s for _, s in flat]

    print(f"    dispatching {len(flat)} REAL PPO jobs to Modal in parallel "
          f"({GATE_PPO_EPISODES} eps, {GATE_PPO_EVAL_SEEDS} eval seeds each)...")
    t0 = time.time()
    results = list(fn.map(payloads, seeds))
    wall_s = time.time() - t0
    print(f"    Modal PPO map wall-clock: {wall_s:.1f}s")

    lookup: dict[tuple[float, int], dict] = {}
    for (d, s), row in zip(flat, results):
        if not (isinstance(row, dict) and row.get("status") == "ppo"):
            raise RuntimeError(f"non-PPO result for d={d} seed={s}: {row!r}")
        lookup[(round(float(d), 6), int(s))] = row

    def learning_fn(params: dict, seed: int) -> float:
        d = round(float(params["difficulty"]), 6)
        return float(lookup[(d, int(seed))]["improvement"])

    return learning_fn, lookup, wall_s


def _build_local_learning_fn():
    """Run the REAL PPO body in-process (no Modal). Returns (learning_fn, rows
    dict, wall_clock_s). Slower (sequential) but credential-free."""
    from modal_player import local_worker

    flat = [(d, s) for d in GATE_DIFFICULTIES for s in GATE_SEEDS]
    print(f"    running {len(flat)} REAL PPO jobs LOCALLY (sequential, "
          f"{GATE_PPO_EPISODES} eps each)...")
    lookup: dict[tuple[float, int], dict] = {}
    t0 = time.time()
    for i, (d, s) in enumerate(flat, 1):
        row = local_worker(_payload(d, s), int(s))
        lookup[(round(float(d), 6), int(s))] = row
        print(f"      [{i:>2}/{len(flat)}] d={d:<4} seed={s:<3} "
              f"imp={row['improvement']:+.3f} "
              f"(before={row['before_winrate']:.2f} after={row['after_winrate']:.2f})")
    wall_s = time.time() - t0
    print(f"    local PPO wall-clock: {wall_s:.1f}s")

    def learning_fn(params: dict, seed: int) -> float:
        d = round(float(params["difficulty"]), 6)
        return float(lookup[(d, int(seed))]["improvement"])

    return learning_fn, lookup, wall_s


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _verdict(pearson: float, spearman: float) -> tuple[str, bool]:
    """GO/NO-GO. Positive AND either coefficient clears ~0.3 => RFT unlocked.

    >= 0.5 on the primary (Pearson) => clean PASS. 0.3-0.5 => WEAK-PASS (proxy
    predicts, but noisily — RFT is defensible). <= ~0 or negative => FAIL: the
    proxy does NOT track real PPO learning; RFT would optimise a lie. NaN (no
    variance in one axis) => FAIL (a non-answer is not a GO).
    """
    if math.isnan(pearson) or math.isnan(spearman):
        return "FAIL", False
    best = max(pearson, spearman)
    positive = pearson > 0.0 and spearman > 0.0
    if positive and pearson >= 0.5:
        return "PASS", True
    if positive and best >= 0.3:
        return "WEAK-PASS", True
    return "FAIL", False


def run_gate(backend: str) -> dict:
    print("=" * 68)
    print("  THE CORRELATION GATE — cheap gap proxy vs REAL PPO learning")
    print("=" * 68)
    print(f"  arenas: {len(GATE_DIFFICULTIES)} difficulties x {len(GATE_SEEDS)} "
          f"seeds = {len(GATE_DIFFICULTIES) * len(GATE_SEEDS)} PPO jobs")
    print(f"  backend: {backend}")
    print("-" * 68)

    # (a) cheap GAP PROXY per arena (local, fast) — the real wired proxy_fn.
    proxy_fn = make_gap_proxy_fn(eval_seeds=GATE_PROXY_EVAL_SEEDS)
    print("  computing cheap gap proxy (local)...")
    proxy_t0 = time.time()
    proxy_by_d = {d: proxy_fn({"difficulty": d}) for d in GATE_DIFFICULTIES}
    proxy_s = time.time() - proxy_t0
    for d in GATE_DIFFICULTIES:
        print(f"    d={d:<5} gap_proxy={proxy_by_d[d]:+.3f}")
    print(f"  gap proxy wall-clock: {proxy_s:.1f}s")

    # (b) real PPO LEARNING per arena x seed.
    print()
    if backend == "modal":
        learning_fn, rows_lookup, learn_s = _build_modal_learning_fn()
    else:
        learning_fn, rows_lookup, learn_s = _build_local_learning_fn()

    # Correlate via the GENERIC scaffold (single source of Pearson/Spearman).
    print()
    print("  correlating gap proxy vs real PPO learning (run_proxy_sweep)...")
    report = run_proxy_sweep(
        _candidates(),
        proxy_fn=proxy_fn,
        learning_fn=learning_fn,
        seeds=GATE_SEEDS,
    )

    pearson = report.pearson_r
    spearman = report.spearman_r
    verdict, go = _verdict(pearson, spearman)

    # --- scatter (per arena) ----------------------------------------------
    scatter = []
    for row in report.rows:
        d = float(row.params["difficulty"])
        scatter.append(
            {
                "difficulty": d,
                "gap_proxy": round(row.proxy_score, 4),
                "learning_gains": [round(g, 4) for g in row.learning_gains],
                "mean_learning_gain": round(row.mean_learning_gain, 4),
            }
        )

    print()
    print("  --- scatter (per arena) ---")
    print(f"  {'difficulty':>10} {'gap_proxy':>10} {'mean_learn':>11} "
          f"{'per-seed learning':>22}")
    for s in scatter:
        seeds_str = "[" + ", ".join(f"{g:+.2f}" for g in s["learning_gains"]) + "]"
        print(f"  {s['difficulty']:>10.2f} {s['gap_proxy']:>+10.3f} "
              f"{s['mean_learning_gain']:>+11.3f} {seeds_str:>22}")

    # Raw per-(arena, seed) points for the record.
    points = []
    for d in GATE_DIFFICULTIES:
        for s in GATE_SEEDS:
            row = rows_lookup[(round(float(d), 6), int(s))]
            points.append(
                {
                    "difficulty": float(d),
                    "seed": int(s),
                    "gap_proxy": round(proxy_by_d[d], 4),
                    "before_winrate": round(float(row["before_winrate"]), 4),
                    "after_winrate": round(float(row["after_winrate"]), 4),
                    "improvement": round(float(row["improvement"]), 4),
                }
            )

    print()
    print("=" * 68)
    print(f"  Pearson  r   (gap_proxy vs real PPO learning) = {pearson:+.4f}")
    print(f"  Spearman rho                                  = {spearman:+.4f}")
    print(f"  GATE VERDICT: {verdict}   ->   {'GO (RFT unlocked)' if go else 'NO-GO (DO NOT RFT)'}")
    print("=" * 68)
    if go:
        print("  => the cheap proxy POSITIVELY predicts real PPO learning.")
        print("     RFT against the proxy optimises the right thing. UNLOCKED.")
    else:
        print("  => the cheap proxy does NOT positively predict real PPO learning.")
        print("     RFT against it would train Qwen on a lie. DO NOT START RFT.")
        print("     (This is a SAVE, not a failure of the run.)")

    summary = {
        "question": "Does the cheap gap proxy positively predict real PPO learning?",
        "backend": backend,
        "n_arenas": len(GATE_DIFFICULTIES),
        "n_seeds_per_arena": len(GATE_SEEDS),
        "n_points": len(points),
        "difficulties": GATE_DIFFICULTIES,
        "seeds": GATE_SEEDS,
        "ppo_episodes": GATE_PPO_EPISODES,
        "ppo_eval_seeds": GATE_PPO_EVAL_SEEDS,
        "proxy_eval_seeds": GATE_PROXY_EVAL_SEEDS,
        "proxy_wall_clock_s": round(proxy_s, 2),
        "learning_wall_clock_s": round(learn_s, 2),
        "pearson_r": None if math.isnan(pearson) else round(pearson, 4),
        "spearman_rho": None if math.isnan(spearman) else round(spearman, 4),
        "gate_verdict": verdict,
        "go_no_go": "GO" if go else "NO-GO",
        "rft_unlocked": bool(go),
        "scatter_per_arena": scatter,
        "points": points,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {RESULTS_PATH}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the correlation gate.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--modal", action="store_true", help="force Modal PPO backend")
    group.add_argument("--local", action="store_true", help="force local PPO backend")
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore")

    if args.modal:
        backend = "modal"
    elif args.local:
        backend = "local"
    else:
        print("  selecting backend (probing Modal)...")
        backend = "modal" if _modal_reachable() else "local"

    run_gate(backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
