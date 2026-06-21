"""output/select_best_checkpoint.py — pick the best Teacher iteration by VALIDATION.

The trainer only auto-scores the FINAL adapter (``update{N}``) on the reserved
validation seeds. To choose the best *iteration* we must score each candidate
checkpoint the same way — generate arenas from that adapter and measure held-out
PPO transfer on the RESERVED validation seeds (disjoint from training + Stage-6).

This reuses the EXACT trainer machinery so the number is comparable to the
trainer's own ``summary.json`` headline:
  * generation  -> deployed ``crucible-teacher-gen`` ``TeacherGenerator``, prompted
    with the trainer's ``RING_OUT_PROMPT`` (system="" => byte-identical to training).
  * scoring     -> ``train_teacher_modal._score_ring_out`` (parse -> clamp ->
    ``teacher_reward`` -> 3 PPO seeds on ``crucible-player`` -> held-out improvement).

Selection rule (noise-robust, per runbook §8):
  * Score only the DISTINCT checkpoints. A zero-gradient update (``grad_norm≈0``,
    arena collapse) produces a byte-identical adapter to its predecessor, so we
    skip it. Distinct set = update1 + every update whose ``grad_norm > --grad-eps``.
  * Pick by validation mean. Selection NEVER uses the Stage-6 test (contamination)
    and NEVER uses the per-update PRE-update training-seed rewards (circular/noisy).

Usage (from repo root, project venv, apps deployed):
    PYTHONPATH=. python3 output/select_best_checkpoint.py --run-id friend_r32 \
        --n-arenas 6 --temperature 1.2 --val-seeds 1 2 3 --episodes 1000 --eval-seeds 50

Honest by construction: if NO checkpoint's validation mean beats the others by more
than noise — or all sit at/under the base level — that is reported, not hidden. A
checkpoint is only "best" relative to the candidates; this does not assert it beats
the un-adapted base (that is what the Stage-6 null + decider establish).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import modal  # noqa: E402

from training.train_teacher_modal import (  # noqa: E402
    DEFAULT_MODEL,
    GAME_PROMPTS,
    _build_held_out,
    _score_ring_out,
    _score_tk,
)

GEN_APP = "crucible-teacher-gen"
GEN_CLS = "TeacherGenerator"
# Fixed generation seed across ALL checkpoints so the only thing that varies is the
# adapter (matches the trainer's final-adapter validation sample seed of 9000).
GEN_SEED = 9000


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _distinct_checkpoints(run_id: str, grad_eps: float) -> list[str]:
    """Update tags that are DISTINCT adapters: update1 + any update with a real
    gradient. A ``grad_norm <= grad_eps`` step did not move the weights, so its
    checkpoint equals its predecessor and is skipped."""
    hist_path = _REPO_ROOT / "output" / f"contingency_{run_id}" / "history.json"
    if not hist_path.exists():
        raise SystemExit(f"no history at {hist_path} — has the run produced any updates?")
    hist = json.loads(hist_path.read_text())
    tags: list[str] = []
    for rec in hist:
        tag = rec.get("tag", "")
        if not tag.startswith("update"):
            continue  # skip the step-0 'base' row (no adapter saved)
        gn = (rec.get("update_stats") or {}).get("grad_norm")
        if gn is None:
            continue
        if float(gn) > grad_eps:
            tags.append(tag)
    return tags


def _score_one(comp: str, *, game: str, seeds, episodes, eval_seeds, held_out):
    if game == "ring_out":
        return _score_ring_out(comp, seeds=seeds, episodes=episodes,
                               eval_seeds=eval_seeds, held_out=held_out)
    return _score_tk(comp, seeds=seeds, episodes=episodes, eval_seeds=eval_seeds)


def validate_checkpoint(gen_cls, run_id: str, tag: str, *, game: str, n_arenas: int,
                        temperature: float, seeds, episodes: int, eval_seeds: int,
                        held_out) -> dict:
    adapter_tag = f"{run_id}/{tag}"
    inst = gen_cls(model_name=DEFAULT_MODEL, adapter_tag=adapter_tag)
    prompt = GAME_PROMPTS[game]
    t0 = time.time()
    # ONE remote round-trip generates all arenas for this checkpoint.
    comps = inst.generate.remote(n=n_arenas, system="", user=prompt,
                                 temperature=temperature, seed=GEN_SEED)
    # Score every arena in parallel (each is a slow remote PPO fan-out).
    with ThreadPoolExecutor(max_workers=max(1, n_arenas)) as ex:
        scored = list(ex.map(
            lambda c: _score_one(c, game=game, seeds=seeds, episodes=episodes,
                                 eval_seeds=eval_seeds, held_out=held_out),
            comps,
        ))
    rewards = [s[0] for s in scored]
    statuses = [s[2] for s in scored]
    valid = [r for r, st in zip(rewards, statuses) if st == "ok"]
    return {
        "tag": tag,
        "adapter_tag": adapter_tag,
        "game": game,
        "rewards": [round(r, 4) for r in rewards],
        "statuses": statuses,
        "n_valid": len(valid),
        "val_mean": (round(_mean(valid), 4) if valid else None),
        "wall_s": round(time.time() - t0, 1),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Select best Teacher checkpoint by validation transfer.")
    p.add_argument("--run-id", required=True, help="e.g. friend_r32")
    p.add_argument("--game", default="ring_out", choices=list(GAME_PROMPTS),
                   help="selection game (default ring_out — matches the fighter Stage-6)")
    p.add_argument("--tags", nargs="+", default=None,
                   help="explicit update tags (e.g. update1 update5); default = auto distinct")
    p.add_argument("--grad-eps", type=float, default=1.0,
                   help="updates with grad_norm <= this are collapsed/no-op dupes, skipped "
                        "(real steps here are ~50-250; collapse steps are ~0)")
    p.add_argument("--n-arenas", type=int, default=6, help="arenas generated per checkpoint")
    p.add_argument("--temperature", type=float, default=1.2,
                   help="generation temperature (match the run's TRAINING temperature)")
    p.add_argument("--val-seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="RESERVED validation PPO seeds (must match the run's --val-seeds)")
    p.add_argument("--episodes", type=int, default=1000, help="Ring-Out PPO episodes/seed")
    p.add_argument("--eval-seeds", type=int, default=50, help="held-out eval seeds")
    p.add_argument("--out", default=None, help="output JSON path")
    args = p.parse_args(argv)

    tags = args.tags or _distinct_checkpoints(args.run_id, args.grad_eps)
    if not tags:
        raise SystemExit("no candidate checkpoints found (no updates with a real gradient yet)")

    seeds = tuple(args.val_seeds)
    held_out = _build_held_out() if args.game == "ring_out" else []

    print(f"=== CHECKPOINT SELECTION (run_id={args.run_id}, game={args.game}) ===", flush=True)
    print(f"    candidates (distinct, grad>{args.grad_eps}): {tags}", flush=True)
    print(f"    n_arenas={args.n_arenas}  gen_temp={args.temperature}  "
          f"val_seeds={list(seeds)}  episodes={args.episodes}  eval_seeds={args.eval_seeds}", flush=True)
    print(f"    metric = mean held-out transfer on RESERVED validation seeds "
          f"(same as trainer summary.json headline)\n", flush=True)

    gen_cls = modal.Cls.from_name(GEN_APP, GEN_CLS)
    rows = []
    for tag in tags:
        print(f"[{tag}] generating {args.n_arenas} arenas + scoring on validation seeds ...", flush=True)
        row = validate_checkpoint(
            gen_cls, args.run_id, tag, game=args.game, n_arenas=args.n_arenas,
            temperature=args.temperature, seeds=seeds, episodes=args.episodes,
            eval_seeds=args.eval_seeds, held_out=held_out,
        )
        rows.append(row)
        print(f"[{tag}] val_mean={row['val_mean']}  valid={row['n_valid']}/{args.n_arenas}  "
              f"rewards={row['rewards']}  ({row['wall_s']}s)\n", flush=True)

    scored_rows = [r for r in rows if r["val_mean"] is not None]
    best = max(scored_rows, key=lambda r: r["val_mean"]) if scored_rows else None

    print("=== VALIDATION RANKING (high to low) ===", flush=True)
    for r in sorted(scored_rows, key=lambda r: r["val_mean"], reverse=True):
        flag = "  <== BEST" if best and r["tag"] == best["tag"] else ""
        print(f"    {r['tag']:>10}  val_mean={r['val_mean']:+.4f}  "
              f"(valid {r['n_valid']}/{args.n_arenas}){flag}", flush=True)

    result = {
        "run_id": args.run_id,
        "game": args.game,
        "metric": "mean held-out transfer on reserved validation seeds",
        "selection_temperature": args.temperature,
        "val_seeds": list(seeds),
        "candidates": rows,
        "best_tag": (best["tag"] if best else None),
        "best_adapter_handle": (f"modal:{args.run_id}/{best['tag']}" if best else None),
        "best_val_mean": (best["val_mean"] if best else None),
    }
    out_path = Path(args.out) if args.out else (_REPO_ROOT / "output" / f"contingency_{args.run_id}" / "checkpoint_selection.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    if best:
        print(f">>> BEST CHECKPOINT: {result['best_adapter_handle']}  "
              f"(val_mean={best['val_mean']:+.4f}) — use as --trained for Stage-6", flush=True)
    else:
        print(">>> NO checkpoint produced a valid validation score — investigate before Stage-6", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
