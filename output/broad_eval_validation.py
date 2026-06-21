"""output/broad_eval_validation.py — THE decisive reward-validation experiment.

THE OPEN QUESTION. A prior multi-seed sweep (reward = mean held-out PPO transfer
on the FIXED 2-opponent turtle reference set, d=0.75/0.85) ranked a trivial-
looking training arena d=0.25 as the PEAK reward:

    d=0.15->0.277  d=0.25->0.563(peak)  d=0.55->0.243
    d=0.65->0.080  d=0.85->0.013        d=0.97->0.020

Is d=0.25 the genuine best CURRICULUM (its trained Players transfer to a BROAD
range of unseen fighter tasks), or did it just learn to beat those two specific
turtle references? The narrow held-out set (2 opponents, same archetype, same
physics) cannot tell those apart.

THIS EXPERIMENT re-ranks the candidates against a STRUCTURALLY DIVERSE held-out
population (output/broad_eval_set: 5 physics variants x 3 difficulties = 15
unseen arenas, none equal to any training arena). For each training difficulty
in {0.25, 0.55, 0.85} we train FRESH PPO Players (5 seeds) and score each on the
SAME 15-arena population, then aggregate per-difficulty mean transfer + per-seed
std. The reward FORMULA is unchanged (still mean held-out PPO improvement); only
the held-out POPULATION is broadened, via the worker's opt-in held_out_arenas key.

VERDICT logic:
  * d=0.25 still transfers best on the diverse population
        => it is the genuine reward winner (NOT degenerate); the held-out-
           improvement reward is valid to optimize.
  * d=0.25 collapses on the diverse population while d=0.55 holds up
        => it was gaming the narrow turtle benchmark; the reward needs redesign.

Run from repo root (.venv has modal + sb3); the worker must be deployed with the
held_out_arenas key (modal deploy modal_player.py):

    .venv/bin/python output/broad_eval_validation.py --seeds 1 2 3 4 5
    .venv/bin/python output/broad_eval_validation.py --backend local --seeds 1 2  # cheap, no Modal
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Repo root on path (script lives in output/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from output.broad_eval_set import (  # noqa: E402
    build_broad_eval_arenas,
    assert_disjoint_from_training,
    payload_arenas,
)

OUTPUT_DIR = Path(__file__).resolve().parent

MODAL_APP = "crucible-player"
MODAL_FN = "train_player_transfer"

# The candidates under re-ranking (the training difficulties). 0.25 was the prior
# peak; 0.55 is mid-band; 0.85 is the turtle-active upper end.
TRAINING_DIFFICULTIES = (0.25, 0.55, 0.85)
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
# Keep the existing ~1200-episode training budget (do NOT change it — the prior
# sweep used the same budget, so the re-rank is apples-to-apples).
DEFAULT_EPISODES = 1200
# Eval seeds PER held-out arena. The diverse population already averages over 15
# arenas, so a moderate per-arena eval-seed count gives a stable mean.
DEFAULT_EVAL_SEEDS = 30

# The training arena's geometry mirrors nested_reward._spec_for(d): map_size=10
# -> platform_width 11.0, default physics. Fixed so the ONLY training variable is
# difficulty (matching the prior sweep).
TRAIN_PLATFORM_WIDTH = 11.0


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))  # sample std


def _train_payload(difficulty: float, held_out: list[dict], *, episodes: int,
                   eval_seeds: int) -> dict:
    return {
        "difficulty": float(difficulty),
        "platform_width": TRAIN_PLATFORM_WIDTH,
        "ppo_episodes": int(episodes),
        "eval_seeds": int(eval_seeds),
        "curriculum_id": f"broadval-d{difficulty}",
        "architecture": "mlp",
        # THE diverse population: same 15 arenas for every training difficulty.
        "held_out_arenas": held_out,
    }


def _run_jobs(flat: list[tuple[float, int, dict]], *, backend: str) -> list[dict]:
    """Dispatch all (difficulty, seed) jobs and return rows in the same order.

    Modal: one ``fn.map`` over all jobs -> concurrent remote PPO (the 15 jobs run
    in parallel, so wall-clock ~= one job, not 15). Local: sequential in-process
    (correctness fallback, no credits)."""
    payloads = [p for _, _, p in flat]
    seeds = [s for _, s, _ in flat]
    if backend == "modal":
        import modal

        fn = modal.Function.from_name(MODAL_APP, MODAL_FN)
        return list(fn.map(payloads, seeds))
    if backend == "local":
        from modal_player import local_transfer_worker

        return [local_transfer_worker(p, s) for p, s in zip(payloads, seeds)]
    raise ValueError(f"unknown backend {backend!r} (use 'modal' or 'local')")


def run(args) -> dict:
    seeds = list(args.seeds)
    # Build the diverse FIXED population once; assert it is disjoint from training.
    eval_arenas = build_broad_eval_arenas(grid="full")
    assert_disjoint_from_training(eval_arenas, TRAINING_DIFFICULTIES)
    held_out = payload_arenas(eval_arenas)

    n_jobs = len(TRAINING_DIFFICULTIES) * len(seeds)
    print("=== BROAD-POPULATION REWARD VALIDATION ===")
    print(f"    backend={args.backend}  training_difficulties={list(TRAINING_DIFFICULTIES)}  "
          f"seeds={seeds}")
    print(f"    diverse held-out population: {len(eval_arenas)} arenas "
          f"(physics x difficulty), disjoint from training")
    print(f"    per training difficulty: {len(seeds)} fresh PPO seeds; "
          f"{n_jobs} jobs total; episodes={args.episodes} eval_seeds={args.eval_seeds}")
    print(f"    held-out arena names: {[a['name'] for a in eval_arenas]}")
    print()

    flat = [
        (d, s, _train_payload(d, held_out, episodes=args.episodes, eval_seeds=args.eval_seeds))
        for d in TRAINING_DIFFICULTIES
        for s in seeds
    ]
    t0 = time.time()
    rows = _run_jobs(flat, backend=args.backend)
    wall = time.time() - t0
    for (d, s, _), r in zip(flat, rows):
        assert r.get("status") == "ppo_transfer", f"non-transfer result: {r!r}"

    # --- aggregate per training difficulty ---
    per_difficulty = []
    by_d_rows: dict[float, list[tuple[int, dict]]] = {d: [] for d in TRAINING_DIFFICULTIES}
    for (d, s, _), r in zip(flat, rows):
        by_d_rows[d].append((s, r))

    for d in TRAINING_DIFFICULTIES:
        srows = by_d_rows[d]
        per_seed_improvement = [float(r["held_out_improvement"]) for _, r in srows]
        per_seed = [
            {
                "seed": s,
                "broad_improvement": round(float(r["held_out_improvement"]), 4),
                "before": round(float(r["before_winrate"]), 4),
                "after": round(float(r["after_winrate"]), 4),
                "train_arena_winrate": round(float(r["train_arena_winrate"]), 4),
                "per_arena_after": [round(x, 3) for x in r["held_out_after_per_arena"]],
            }
            for s, r in srows
        ]
        mean_impr = _mean(per_seed_improvement)
        std_impr = _std(per_seed_improvement)
        per_difficulty.append(
            {
                "training_difficulty": d,
                "n_seeds": len(srows),
                "mean_broad_improvement": round(mean_impr, 4),
                "std_broad_improvement": round(std_impr, 4),
                "per_seed_improvement": [round(x, 4) for x in per_seed_improvement],
                "mean_train_arena_winrate": round(
                    _mean([float(r["train_arena_winrate"]) for _, r in srows]), 4
                ),
                "per_seed": per_seed,
            }
        )

    # --- the verdict ---
    by_d = {row["training_difficulty"]: row for row in per_difficulty}
    ranked = sorted(per_difficulty, key=lambda r: r["mean_broad_improvement"], reverse=True)
    winner = ranked[0]
    d025 = by_d.get(0.25)
    d055 = by_d.get(0.55)

    winner_is_025 = abs(winner["training_difficulty"] - 0.25) < 1e-9
    # "collapses" = d=0.25 no longer the top, AND d=0.55 now beats it on the broad
    # population (the redesign-trigger case the task names explicitly).
    d025_collapsed = (
        d025 is not None
        and d055 is not None
        and not winner_is_025
        and d055["mean_broad_improvement"] > d025["mean_broad_improvement"]
    )

    if winner_is_025:
        decision = "REWARD_VALID"
        verdict_text = (
            "d=0.25 still transfers BEST on the diverse held-out population -> it is "
            "the GENUINE reward winner, NOT a narrow-benchmark exploit. The held-out-"
            "improvement reward is valid to optimize."
        )
    elif d025_collapsed:
        decision = "REWARD_NEEDS_REDESIGN"
        verdict_text = (
            "d=0.25 COLLAPSES on the diverse population while d=0.55 holds up -> the "
            "prior d=0.25 peak was gaming the narrow turtle benchmark. The held-out-"
            "improvement reward needs redesign before Teacher RFT."
        )
    else:
        decision = "INCONCLUSIVE"
        verdict_text = (
            f"d=0.25 is no longer the top on the diverse population (winner: "
            f"d={winner['training_difficulty']}), but d=0.55 does not clearly beat it "
            "-> neither clean case obtains; inspect the per-difficulty table."
        )

    # --- print the verdict table ---
    print(f"    parallel wall-clock for {n_jobs} jobs: {wall:.1f}s")
    print()
    print("    === BROAD-POPULATION VERDICT TABLE ===")
    print(f"    {'train_d':>8} {'mean_broad_transfer':>20} {'std(5 seeds)':>13} "
          f"{'per-seed':>0}")
    for row in per_difficulty:
        marker = "  <-- prior peak" if abs(row["training_difficulty"] - 0.25) < 1e-9 else ""
        print(f"    {row['training_difficulty']:>8} "
              f"{row['mean_broad_improvement']:>+20.4f} "
              f"{row['std_broad_improvement']:>13.4f}  "
              f"{row['per_seed_improvement']}{marker}")
    print()
    print(f"    broad-population winner: d={winner['training_difficulty']} "
          f"(mean transfer {winner['mean_broad_improvement']:+.4f})")
    print(f"    DECISION: {decision}")
    print(f"    {verdict_text}")

    summary = {
        "experiment": "broad_eval_validation",
        "question": (
            "Is training d=0.25 the genuine best curriculum (broad transfer) or "
            "did it game the narrow 2-turtle held-out benchmark?"
        ),
        "backend": args.backend,
        "training_difficulties": list(TRAINING_DIFFICULTIES),
        "seeds": seeds,
        "episodes": args.episodes,
        "eval_seeds_per_arena": args.eval_seeds,
        "n_jobs": n_jobs,
        "train_platform_width": TRAIN_PLATFORM_WIDTH,
        "wall_clock_s": round(wall, 2),
        "diverse_held_out_population": eval_arenas,
        "n_held_out_arenas": len(eval_arenas),
        "prior_narrow_sweep": {
            "held_out": "2 turtle opponents d=0.75/0.85, default geometry",
            "reward_by_difficulty": {
                "0.15": 0.277, "0.25": 0.563, "0.55": 0.243,
                "0.65": 0.080, "0.85": 0.013, "0.97": 0.020,
            },
            "winner": 0.25,
        },
        "per_difficulty": per_difficulty,
        "broad_population_winner": {
            "training_difficulty": winner["training_difficulty"],
            "mean_broad_improvement": winner["mean_broad_improvement"],
        },
        "decision": decision,
        "verdict": verdict_text,
        "rows": rows,
    }
    out = OUTPUT_DIR / "broad_eval_validation.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"    wrote {out}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Broad-population reward validation.")
    p.add_argument("--backend", choices=["modal", "local"], default="modal")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    p.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=DEFAULT_EVAL_SEEDS)
    args = p.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
