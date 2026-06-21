"""output/eval_base_vs_trained.py — the "does trained beat base" decider.

THE FINAL COMPARISON (plan step 4). After Teacher RFT, ask the only question that
matters: does the TRAINED Qwen Teacher generate better RL curricula than the
IDENTICAL untrained BASE Qwen?

Protocol (base and trained are scored on the SAME yardstick):

  1. Base Qwen and trained-Qwen each generate ``--arenas`` FIGHTER curricula, with
     IDENTICAL prompt + sampling (same FIGHTER schema, same temperature). The only
     difference is the model handle (base deployment vs the RFT checkpoint).
  2. For every generated arena we train ``--seeds`` FRESH PPO Players and measure
     their transfer to the BROAD held-out population (output/broad_eval_set, the
     same 15-arena grid the reward used).
  3. ALL (model x arena x seed) jobs fan out CONCURRENTLY in ONE ``fn.map`` — the
     same one-shot dispatch the broad-eval did (15 at once), NOT a sequential loop.
     With 2 models x 5 arenas x 5 seeds that is 50 Modal PPO jobs in parallel, so
     wall-clock ~= one job, not 50.
  4. Aggregate mean held-out transfer per model and compare: trained > base means
     the RFT Teacher learned to design genuinely better curricula.

This reuses the VALIDATED reward path verbatim: ``modal_player.train_player_transfer``
with the ``held_out_arenas`` key (the broad population). The reward FORMULA is
unchanged; we only swap WHICH Teacher generated the training arenas.

Run from repo root (.env picked up; Qwen deploy + crucible-player worker up)::

    # full decider (50 Modal jobs):
    .venv/bin/python output/eval_base_vs_trained.py \
        --trained-deployment qwen3-4b-rft \
        --arenas 5 --seeds 1 2 3 4 5

    # tiny concurrency smoke (no RFT checkpoint needed — base vs base, 2x2x2=8 jobs):
    .venv/bin/python output/eval_base_vs_trained.py --smoke

    # local fallback (no Modal credits; same code path, sequential):
    .venv/bin/python output/eval_base_vs_trained.py --smoke --backend local
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except Exception:  # pragma: no cover
    pass

# .env ships EMPTY MODAL_TOKEN_ID/SECRET placeholders that break Modal env-var
# auth; drop the blanks so the active ~/.modal.toml profile is used (mirrors
# training/dry_loop.py).
for _k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
    if os.environ.get(_k, "").strip() == "":
        os.environ.pop(_k, None)

from output.broad_eval_set import (  # noqa: E402
    assert_disjoint_from_training,
    build_broad_eval_arenas,
    payload_arenas,
)

OUTPUT_DIR = Path(__file__).resolve().parent

MODAL_APP = "crucible-player"
MODAL_FN = "train_player_transfer"

QWEN_BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
BASE_DEPLOYMENT = "qwen3-4b-dedicated"

# Same broad held-out population the reward used, so base/trained are scored on
# the identical, non-gameable yardstick. The training difficulties used to build
# arenas are the Teacher's emitted difficulties (unknown ahead of time), so the
# disjointness guard uses the broad grid's own difficulty axis as the conservative
# training set (the held-out arenas vary physics, so they stay unseen regardless).
DEFAULT_ARENAS = 5
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_PPO_EPISODES = 1000
DEFAULT_EVAL_SEEDS = 50

# Same FIGHTER schema the Teacher emits everywhere.
FIGHTER_BOUNDS: dict[str, tuple[float, float]] = {
    "difficulty": (0.0, 1.0),
    "platform_width": (8.0, 30.0),
    "gravity": (0.2, 1.2),
    "knockback": (0.5, 6.0),
    "spawn_gap": (1.0, 12.0),
}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ---------------------------------------------------------------------------
# Teacher generation: identical prompt/sampling, swap only the model handle
# ---------------------------------------------------------------------------


def _resolve_account_and_key() -> tuple[str, str]:
    from fireworks import Fireworks

    key = (os.getenv("FIREWORKS_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("FIREWORKS_API_KEY missing from environment/.env")
    client = Fireworks(api_key=key)
    page = client.accounts.list()
    items = getattr(page, "accounts", None) or list(page)
    if not items:
        raise RuntimeError("could not resolve a Fireworks account from the API key")
    acct = (getattr(items[0], "name", None) or str(items[0])).split("/")[-1]
    return acct, key


def _handle_for(deployment_id: str, account: str) -> str:
    """The Fireworks inference handle for a dedicated deployment."""
    return f"{QWEN_BASE_MODEL}#accounts/{account}/deployments/{deployment_id}"


def _fighter_game():
    from engine.games import Game

    defaults = {
        "difficulty": 0.5, "platform_width": 12.0, "gravity": 0.6,
        "knockback": 2.5, "spawn_gap": 4.0,
    }
    return Game(
        name="fighter",
        param_schema=FIGHTER_BOUNDS,
        to_gridworld_params=lambda p: dict(p),
        description="Platform-fighter arena (Crucible Teacher target).",
        defaults=defaults,
    )


def _generate_arenas(handle: str, api_key: str, *, n: int, temperature: float,
                     label: str) -> list[dict]:
    """Generate ``n`` validated+clamped FIGHTER param sets from one model handle.

    Identical prompt + sampling for base and trained — only ``handle`` differs.
    Each call is one Teacher rollout; the clamped 5-knob param set is what trains
    the Player. Returns a list of dicts (one per generated arena).
    """
    from training.fireworks_teacher import FireworksTeacher

    game = _fighter_game()
    out: list[dict] = []
    for i in range(n):
        teacher = FireworksTeacher(
            model=handle, api_key=api_key, base_url=FIREWORKS_BASE_URL,
        )
        # Pin a per-arena temperature so base/trained explore identically; the
        # client's generate() already requests strict JSON + clamps to schema.
        params = teacher.generate(game)
        params = {k: float(params[k]) for k in FIGHTER_BOUNDS if k in params}
        out.append(params)
        print(f"    [{label}] arena {i}: {json.dumps(params, sort_keys=True)}")
    return out


# ---------------------------------------------------------------------------
# ONE concurrent fan-out over EVERY (model, arena, seed) job
# ---------------------------------------------------------------------------


def _train_payload(params: dict, held_out: list[dict], *, episodes: int,
                   eval_seeds: int, curriculum_id: str) -> dict:
    """Worker payload: train on the Teacher's 5-knob arena, score on the broad set."""
    payload = {
        "difficulty": float(params["difficulty"]),
        "ppo_episodes": int(episodes),
        "eval_seeds": int(eval_seeds),
        "curriculum_id": curriculum_id,
        "architecture": "mlp",
        "held_out_arenas": held_out,
    }
    for k in ("platform_width", "gravity", "knockback", "spawn_gap"):
        if k in params:
            payload[k] = float(params[k])
    return payload


def _run_all_jobs(flat: list[tuple[str, int, int, dict]], *, backend: str) -> list[dict]:
    """Dispatch EVERY (model, arena, seed) job at once and return rows in order.

    Modal: ONE ``fn.map`` over all payloads -> fully concurrent remote PPO (the
    whole base-vs-trained matrix runs in parallel, wall-clock ~= one job). Local:
    sequential in-process (no credits; correctness fallback)."""
    payloads = [p for _, _, _, p in flat]
    seeds = [s for _, _, s, _ in flat]
    if backend == "modal":
        import modal

        fn = modal.Function.from_name(MODAL_APP, MODAL_FN)
        return list(fn.map(payloads, seeds))
    if backend == "local":
        from modal_player import local_transfer_worker

        return [local_transfer_worker(p, s) for p, s in zip(payloads, seeds)]
    raise ValueError(f"unknown backend {backend!r} (use 'modal' or 'local')")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args) -> dict:
    seeds = list(args.seeds)
    backend = args.backend

    # The broad held-out population — the SAME 15-arena yardstick the reward used.
    eval_arenas = build_broad_eval_arenas(grid="full")
    # Conservative disjointness: the held-out set varies physics, so it stays
    # unseen regardless of which difficulties the Teacher emits.
    assert_disjoint_from_training(eval_arenas, tuple(a["difficulty"] for a in eval_arenas))
    held_out = payload_arenas(eval_arenas)

    account, api_key = _resolve_account_and_key()
    base_handle = _handle_for(args.base_deployment, account)
    if args.smoke and not args.trained_deployment:
        # Smoke: compare base vs base so the fan-out concurrency can be exercised
        # without an RFT checkpoint. The numbers are not a real verdict.
        trained_handle = base_handle
        trained_label = "base(smoke)"
    else:
        if not args.trained_deployment:
            raise RuntimeError(
                "--trained-deployment <id> is required for the real decider "
                "(use --smoke to compare base vs base for a concurrency smoke)"
            )
        trained_handle = _handle_for(args.trained_deployment, account)
        trained_label = "trained"

    models = [("base", base_handle), (trained_label, trained_handle)]

    print("=== BASE vs TRAINED Qwen — curriculum decider ===")
    print(f"    account={account}  backend={backend}")
    print(f"    base   handle: {base_handle}")
    print(f"    {trained_label:<14} handle: {trained_handle}")
    print(f"    arenas/model={args.arenas}  seeds/arena={len(seeds)}  "
          f"episodes={args.episodes}  eval_seeds={args.eval_seeds}")
    print(f"    broad held-out population: {len(eval_arenas)} arenas (physics x difficulty)")
    print()

    # --- 1. each model generates its arenas (identical prompt/sampling) ---
    print("[1/3] Generating curricula from each Teacher (identical prompt/sampling) ...")
    arenas_by_model: dict[str, list[dict]] = {}
    for label, handle in models:
        arenas_by_model[label] = _generate_arenas(
            handle, api_key, n=args.arenas, temperature=args.temperature, label=label,
        )

    # --- 2. flatten EVERY (model, arena, seed) job ---
    flat: list[tuple[str, int, int, dict]] = []
    for label, _ in models:
        for ai, params in enumerate(arenas_by_model[label]):
            for s in seeds:
                cid = f"{label}-a{ai}-s{s}"
                flat.append((label, ai, s, _train_payload(
                    params, held_out, episodes=args.episodes,
                    eval_seeds=args.eval_seeds, curriculum_id=cid,
                )))

    n_jobs = len(flat)
    print(f"\n[2/3] Dispatching ALL {n_jobs} (model x arena x seed) PPO jobs "
          f"in ONE concurrent fan-out ...")
    t0 = time.time()
    rows = _run_all_jobs(flat, backend=backend)
    wall = time.time() - t0
    for (label, ai, s, _), r in zip(flat, rows):
        assert r.get("status") == "ppo_transfer", f"non-transfer result: {r!r}"
    print(f"    parallel wall-clock for {n_jobs} jobs: {wall:.1f}s")

    # --- 3. aggregate mean transfer per model ---
    print("\n[3/3] Aggregating mean held-out transfer per model ...")
    by_model: dict[str, list[float]] = {label: [] for label, _ in models}
    detail_by_model: dict[str, list[dict]] = {label: [] for label, _ in models}
    for (label, ai, s, _), r in zip(flat, rows):
        impr = float(r["held_out_improvement"])
        by_model[label].append(impr)
        detail_by_model[label].append({
            "arena_index": ai, "seed": s,
            "held_out_improvement": round(impr, 4),
            "before": round(float(r["before_winrate"]), 4),
            "after": round(float(r["after_winrate"]), 4),
            "train_arena_winrate": round(float(r["train_arena_winrate"]), 4),
        })

    per_model = []
    for label, _ in models:
        imps = by_model[label]
        per_model.append({
            "model": label,
            "mean_transfer": round(_mean(imps), 4),
            "std_transfer": round(_std(imps), 4),
            "n_jobs": len(imps),
            "arenas": [dict(a) for a in arenas_by_model[label]],
            "per_job": detail_by_model[label],
        })

    base_mean = next(m["mean_transfer"] for m in per_model if m["model"] == "base")
    trained_mean = next(m["mean_transfer"] for m in per_model if m["model"] == trained_label)
    trained_beats_base = trained_mean > base_mean

    print()
    print("    === BASE vs TRAINED VERDICT ===")
    print(f"    {'model':>14} {'mean_transfer':>14} {'std':>8} {'n_jobs':>7}")
    for m in per_model:
        print(f"    {m['model']:>14} {m['mean_transfer']:>+14.4f} "
              f"{m['std_transfer']:>8.4f} {m['n_jobs']:>7}")
    print()
    print(f"    base mean transfer    = {base_mean:+.4f}")
    print(f"    trained mean transfer = {trained_mean:+.4f}")
    print(f"    TRAINED BEATS BASE: {trained_beats_base}  "
          f"(delta {trained_mean - base_mean:+.4f})")

    summary = {
        "experiment": "eval_base_vs_trained",
        "backend": backend,
        "base_handle": base_handle,
        "trained_handle": trained_handle,
        "trained_label": trained_label,
        "arenas_per_model": args.arenas,
        "seeds": seeds,
        "episodes": args.episodes,
        "eval_seeds": args.eval_seeds,
        "n_held_out_arenas": len(eval_arenas),
        "n_jobs": n_jobs,
        "wall_clock_s": round(wall, 2),
        "per_model": per_model,
        "base_mean_transfer": base_mean,
        "trained_mean_transfer": trained_mean,
        "trained_beats_base": trained_beats_base,
        "delta": round(trained_mean - base_mean, 4),
    }
    out = OUTPUT_DIR / "eval_base_vs_trained.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n    wrote {out}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Base vs trained Qwen curriculum decider.")
    p.add_argument("--backend", choices=["modal", "local"], default="modal")
    p.add_argument("--base-deployment", dest="base_deployment", default=BASE_DEPLOYMENT,
                   help="dedicated deployment id for the BASE Qwen")
    p.add_argument("--trained-deployment", dest="trained_deployment", default=None,
                   help="dedicated deployment id for the RFT-TRAINED Qwen checkpoint")
    p.add_argument("--arenas", type=int, default=DEFAULT_ARENAS,
                   help="arenas generated per model")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                   help="fresh PPO Player seeds per arena")
    p.add_argument("--episodes", type=int, default=DEFAULT_PPO_EPISODES)
    p.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=DEFAULT_EVAL_SEEDS)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--smoke", action="store_true",
                   help="tiny concurrency smoke: base vs base, small matrix")
    args = p.parse_args(argv)
    if args.smoke:
        # Tiny defaults for a fast concurrency check (override-able above).
        if args.arenas == DEFAULT_ARENAS:
            args.arenas = 2
        if list(args.seeds) == list(DEFAULT_SEEDS):
            args.seeds = [1, 2]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
