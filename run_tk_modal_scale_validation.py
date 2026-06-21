"""run_tk_modal_scale_validation.py — LARGE concurrent validation of Target Knockback.

The earlier 3-seed run (d=0.5) gave a noisy +0.053 improvement, dragged by one
high-baseline seed. This driver fans out a MUCH bigger grid through the already-deployed
``crucible-player-tk`` / ``train_tk_player`` worker to average out the noisy untrained-net
baseline and get a ROBUST learning signal:

  * 12 seeds x 4 difficulties {0.45, 0.5, 0.55, 0.6} = 48 jobs
  * ALL 48 jobs fanned out in ONE concurrent ``fn.map`` (like the broad-eval), NOT a
    sequential loop.
  * Real worker budget: ppo_episodes=2000, eval_seeds=50.

For each difficulty it reports per-seed before/after/improvement, the MEAN improvement
and std, AND the AFTER-winrate mean/std (the earlier run's claim was that after-winrate is
tight even when improvement is noisy — i.e. training is the stabilizer; we confirm/refute
that at scale). Captures 1-2 trained-Player replays and writes a results JSON + verdict.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

# .env ships BLANK MODAL_TOKEN_ID/SECRET that break Modal env-var auth — pop them so the
# njlee007 profile credentials are used instead (same fix as run_tk_modal_validation.py).
for _k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
    if not os.environ.get(_k):
        os.environ.pop(_k, None)
os.environ.setdefault("MODAL_PROFILE", "njlee007")

import modal

HERE = Path(__file__).parent
REPLAYS_DIR = HERE / "replays"
RESULTS_PATH = HERE / "tk_modal_scale_validation_results.json"

SEEDS = list(range(1, 13))  # 12 seeds: 1..12
DIFFICULTIES = [0.45, 0.5, 0.55, 0.6]
PPO_EPISODES = 2000
EVAL_SEEDS = 50

# Capture a trained-Player replay for ONE seed at these two candidate learnable bands.
REPLAY_DIFFICULTIES = {0.5: 1, 0.55: 1}  # difficulty -> seed that carries the capture flag


def _ms(xs):
    """mean, sample-std (0 if <2 points)."""
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> int:
    fn = modal.Function.from_name("crucible-player-tk", "train_tk_player")

    # Build ONE flat list of (payload, seed) across the whole 12x4 grid so a single
    # fn.map fans them ALL out concurrently. Tag each job so we can regroup the rows.
    payloads: list[dict] = []
    seeds: list[int] = []
    job_keys: list[tuple[float, int]] = []  # (difficulty, seed) per job, in map order
    for d in DIFFICULTIES:
        for s in SEEDS:
            p = {
                "difficulty": d,
                "ppo_episodes": PPO_EPISODES,
                "eval_seeds": EVAL_SEEDS,
                "curriculum_id": f"tk-scale-d{d}",
            }
            if REPLAY_DIFFICULTIES.get(d) == s:
                p["capture_replay_id"] = f"tk_modal_trained_scale_d{d}_seed{s}"
            payloads.append(p)
            seeds.append(s)
            job_keys.append((d, s))

    n_jobs = len(payloads)
    print(
        f"Fanning out {n_jobs} jobs ({len(SEEDS)} seeds x {len(DIFFICULTIES)} difficulties) "
        f"in ONE concurrent fn.map via crucible-player-tk.train_tk_player ..."
    )
    print(f"  budget: ppo_episodes={PPO_EPISODES}, eval_seeds={EVAL_SEEDS}")
    print(f"  difficulties: {DIFFICULTIES}")
    print(f"  seeds: {SEEDS}")
    t0 = time.time()

    # ONE big concurrent map over (payload, seed) — order_outputs=True (default) so rows
    # line up with job_keys positionally.
    rows = list(fn.map(payloads, seeds))
    dt = time.time() - t0
    print(f"  all {len(rows)} rows back in {dt:.1f}s\n")

    # Regroup rows by difficulty using positional job_keys.
    by_diff: dict[float, list[dict]] = {d: [] for d in DIFFICULTIES}
    captured_replays: list[dict] = []
    for (d, s), r in zip(job_keys, rows):
        by_diff[d].append(r)
        if "replay" in r:
            captured_replays.append(r)

    # ---- Per-difficulty aggregation + console report ----
    per_difficulty = {}
    print("=" * 72)
    print("PER-DIFFICULTY RESULTS")
    print("=" * 72)
    for d in DIFFICULTIES:
        rs = sorted(by_diff[d], key=lambda r: int(r["seed"]))
        befores = [r["before_winrate"] for r in rs]
        afters = [r["after_winrate"] for r in rs]
        improves = [r["improvement"] for r in rs]

        bm, bs = _ms(befores)
        am, as_ = _ms(afters)
        im, is_ = _ms(improves)
        n_pos = sum(1 for x in improves if x > 0)

        print(f"\n--- difficulty d={d}  (n={len(rs)} seeds) ---")
        print(f"  {'seed':>4}  {'before':>7}  {'after':>7}  {'improve':>8}")
        for r in rs:
            print(
                f"  {int(r['seed']):>4}  {r['before_winrate']:>7.3f}  "
                f"{r['after_winrate']:>7.3f}  {r['improvement']:>+8.3f}"
            )
        print(f"  before  mean={bm:.3f}  std={bs:.3f}")
        print(f"  after   mean={am:.3f}  std={as_:.3f}   <-- after-winrate spread")
        print(f"  improve mean={im:+.3f}  std={is_:.3f}   ({n_pos}/{len(rs)} seeds improved)")

        per_difficulty[str(d)] = {
            "difficulty": d,
            "n_seeds": len(rs),
            "per_seed": [
                {
                    "seed": int(r["seed"]),
                    "before_winrate": r["before_winrate"],
                    "after_winrate": r["after_winrate"],
                    "improvement": r["improvement"],
                }
                for r in rs
            ],
            "before_mean": bm,
            "before_std": bs,
            "after_mean": am,
            "after_std": as_,
            "improvement_mean": im,
            "improvement_std": is_,
            "n_positive": n_pos,
        }

    # ---- Verdict logic ----
    # Best learnable band = highest mean improvement, tie-broken by tighter improvement std.
    ranked = sorted(
        per_difficulty.values(),
        key=lambda x: (x["improvement_mean"], -x["improvement_std"]),
        reverse=True,
    )
    best = ranked[0]
    best_d = best["difficulty"]

    # Robustness heuristic: a band is "robust" if mean improvement clears its own std
    # error of the mean (mean > SEM, i.e. mean*sqrt(n) > std) AND a majority of seeds
    # improved AND after-winrate is tighter than the before-winrate (training stabilizes).
    def _robust(x):
        n = x["n_seeds"]
        sem = (x["improvement_std"] / (n ** 0.5)) if n > 1 else float("inf")
        mean_clears_sem = x["improvement_mean"] > sem and x["improvement_mean"] > 0
        majority = x["n_positive"] >= (n / 2 + 0.5)
        training_stabilizes = x["after_std"] < x["before_std"]
        return {
            "mean_clears_sem": bool(mean_clears_sem),
            "sem": sem,
            "majority_improved": bool(majority),
            "training_stabilizes": bool(training_stabilizes),
            "robust": bool(mean_clears_sem and majority),
        }

    for d_str, x in per_difficulty.items():
        x["robustness"] = _robust(x)

    any_robust = any(x["robustness"]["robust"] for x in per_difficulty.values())
    best_robust = per_difficulty[str(best_d)]["robustness"]["robust"]

    # Overall after-winrate stabilizer check across the grid.
    stabilizer_hits = sum(
        1 for x in per_difficulty.values() if x["robustness"]["training_stabilizes"]
    )

    if best_robust:
        verdict = (
            f"ROBUST: TK reliably improves a Player at scale. Best learnable band is "
            f"d={best_d} (improvement mean={best['improvement_mean']:+.3f} "
            f"std={best['improvement_std']:.3f}, "
            f"{best['n_positive']}/{best['n_seeds']} seeds improved). "
            f"Clean enough to include TK in the 2-game RFT."
        )
    elif any_robust:
        robust_bands = [
            x["difficulty"] for x in per_difficulty.values() if x["robustness"]["robust"]
        ]
        verdict = (
            f"PARTIALLY ROBUST: best-mean band d={best_d} is not the cleanest, but "
            f"band(s) {robust_bands} pass the robustness bar. TK can plausibly be "
            f"included in the 2-game RFT at a robust band, but signal is band-sensitive."
        )
    else:
        verdict = (
            f"STILL NOISY: no difficulty band clears the robustness bar (mean improvement "
            f"> SEM and majority of seeds improving) at n={len(SEEDS)} seeds. Best band by "
            f"mean is d={best_d} ({best['improvement_mean']:+.3f}+/-{best['improvement_std']:.3f}). "
            f"TK's learning signal is NOT clean enough to confidently include in the 2-game "
            f"RFT without more seeds or a worker-budget change."
        )

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  best learnable band: d={best_d}")
    print(f"  after-winrate stabilizer holds in {stabilizer_hits}/{len(DIFFICULTIES)} bands")
    print(f"  {verdict}")

    # ---- Write captured replays ----
    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    replay_paths = []
    for r in captured_replays:
        rid = r["replay"]["replay_id"]
        path = REPLAYS_DIR / f"{rid}.json"
        path.write_text(json.dumps(r["replay"]["data"], indent=2))
        meta = r["replay"]["data"]["meta"]
        replay_paths.append(str(path))
        print(
            f"  replay written: {path}  "
            f"(frames={meta['frames']} winner={meta['winner']} "
            f"p1={meta['p1_policy']} p2={meta['p2_policy']})"
        )

    # ---- Save results JSON ----
    results = {
        "experiment": "tk_modal_scale_validation",
        "worker": {"app": "crucible-player-tk", "function": "train_tk_player"},
        "grid": {
            "seeds": SEEDS,
            "difficulties": DIFFICULTIES,
            "n_jobs": n_jobs,
            "ppo_episodes": PPO_EPISODES,
            "eval_seeds": EVAL_SEEDS,
        },
        "wall_clock_seconds": dt,
        "per_difficulty": per_difficulty,
        "best_learnable_difficulty": best_d,
        "after_winrate_stabilizer_bands": stabilizer_hits,
        "any_band_robust": any_robust,
        "best_band_robust": best_robust,
        "verdict": verdict,
        "replay_paths": replay_paths,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n  results written: {RESULTS_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
