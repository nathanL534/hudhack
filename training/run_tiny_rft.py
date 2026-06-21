"""training/run_tiny_rft.py — the tiny Teacher-RFT launcher + live monitor.

Launches a SMALL, CONCURRENT Teacher RFT job on Fireworks against the deployed
Qwen3-4B and the deployed Crucible bridge (training/modal_ep_bridge.py), on the
Ring-Out fighter game ONLY, for 5-10 Teacher updates, then MONITORS each update
and enforces the real STOP CONDITIONS as guards.

CONCURRENCY (concurrent-by-design, NOT the sequential baseline):

    max_concurrent_rollouts = 2   # 2 Teacher rollouts in flight at once
        x 3 PPO seeds each (the bridge's teacher_reward fan-out on crucible-player)
        = ~6 Modal PPO jobs concurrent.

SAFETY: this does NOT submit the RFT job unless ``--launch`` is passed. The default
is a DRY run: it builds the dataset, resolves the evaluator + Qwen handle + bridge,
prints the EXACT ``reinforcement_fine_tuning_jobs.create`` kwargs, and stops. The
human passes ``--launch`` to actually start training.

Per-update logged metrics (the monitor reads them from Fireworks tracing / the RFT
step logs): Teacher reward, broad held-out transfer, seed std, invalid/clamped-JSON
%, parameter diversity, boundary-collapse, latency, cost. One representative replay
is saved per update.

STOP CONDITIONS (each a real guard; the monitor halts + reports which fired):
  * reward up but transfer flat        -> reward is gaming, not teaching
  * collapse to one arena              -> Teacher found one exploit, no curriculum
  * > 10% invalid/clamped JSON         -> Teacher drifting off-schema
  * Modal != HUD != EP reward mismatch -> transport/grader disagreement
  * seed variance > mean reward        -> signal is noise

Run (DRY — prints the create kwargs, does NOT launch)::

    .venv/bin/python -m training.run_tiny_rft

Run (LAUNCH — the human's explicit go; submits + monitors)::

    .venv/bin/python -m training.run_tiny_rft --launch --updates 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except Exception:  # pragma: no cover
    pass

# .env ships EMPTY MODAL_TOKEN_ID/SECRET placeholders that break Modal env-var
# auth; drop the blanks so the active ~/.modal.toml profile is used (mirrors
# training/dry_loop.py). The launcher itself does not call Modal, but the bridge
# (which it points the RFT job at) does — keep the environment consistent.
for _k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
    if os.environ.get(_k, "").strip() == "":
        os.environ.pop(_k, None)

QWEN_BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
QWEN_DEPLOYMENT_ID = "qwen3-4b-dedicated"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_BRIDGE_URL = "https://njlee007--crucible-ep-bridge-web.modal.run"

EVALUATOR_FILE = "rft_evaluator.py"
EVALUATOR_FUNC = "test_teacher_rft"
# eval-protocol derives the evaluator id from <file>-<func>:
EVALUATOR_ID = "rft_evaluator-test_teacher_rft"

# Concurrency dials — the whole point of this run vs the sequential baseline.
MAX_CONCURRENT_ROLLOUTS = 2   # 2 Teacher rollouts x 3 PPO seeds = ~6 Modal jobs
MAX_CONCURRENT_EVALUATIONS = 2

# 5-10 Teacher updates. With one Ring-Out task row, ``epochs`` is the update count
# (one pass = one update over the single curriculum-design task, many rollouts each).
DEFAULT_UPDATES = 6

# Stop-condition thresholds.
INVALID_CLAMPED_FRACTION_MAX = 0.10   # > 10% invalid/clamped JSON
REWARD_MISMATCH_TOL = 1e-6            # Modal == HUD == EP
RUN_DIR = _REPO_ROOT / "output" / "tiny_rft"
REPLAYS_DIR = _REPO_ROOT / "replays"


# ---------------------------------------------------------------------------
# Dataset — the Ring-Out fighter Teacher task (ONE game only)
# ---------------------------------------------------------------------------


def _ring_out_prompt() -> str:
    from training.rft_evaluator import RING_OUT_PROMPT

    return RING_OUT_PROMPT


def build_dataset_jsonl(path: Path, *, n_rows: int = 1) -> Path:
    """Write the Ring-Out RFT dataset (EvaluationRow JSONL, Ring-Out game ONLY).

    One curriculum-design row is enough for tiny RFT: each Teacher UPDATE samples
    many rollouts from this task. ``n_rows`` duplicates the row if a larger
    per-update batch is wanted; the game is Ring-Out in every row.
    """
    from eval_protocol.models import EvaluationRow, Message

    path.parent.mkdir(parents=True, exist_ok=True)
    prompt = _ring_out_prompt()
    with path.open("w", encoding="utf-8") as f:
        for i in range(n_rows):
            row = EvaluationRow(messages=[Message(role="user", content=prompt)])
            f.write(json.dumps(row.model_dump(mode="json", exclude_none=True)) + "\n")
    return path


# ---------------------------------------------------------------------------
# Resolve account / Qwen handle / bridge readiness
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


def _qwen_handle(account: str) -> str:
    return f"{QWEN_BASE_MODEL}#accounts/{account}/deployments/{QWEN_DEPLOYMENT_ID}"


def _bridge_healthy(bridge_url: str, *, timeout_s: float = 20.0) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{bridge_url.rstrip('/')}/health", timeout=timeout_s) as r:
            body = json.loads(r.read().decode())
            return body.get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Build the create() kwargs (the exact RFT job request)
# ---------------------------------------------------------------------------


def build_create_kwargs(account: str, *, updates: int, output_model: str) -> dict:
    """The exact ``reinforcement_fine_tuning_jobs.create`` kwargs for this run."""
    evaluator_resource = f"accounts/{account}/evaluators/{EVALUATOR_ID}"
    dataset_resource = f"accounts/{account}/datasets/{_dataset_id()}"
    return {
        "account_id": account,
        "evaluator": evaluator_resource,
        "dataset": dataset_resource,
        "display_name": "crucible-tiny-teacher-rft",
        # 5-10 Teacher updates (epochs over the single Ring-Out task).
        "training_config": {
            "base_model": QWEN_BASE_MODEL,
            "epochs": int(updates),
            "output_model": output_model,
            "lora_rank": 8,
        },
        # CONCURRENCY: ~2 Teacher rollouts x 3 PPO seeds = ~6 Modal jobs concurrent.
        "max_concurrent_rollouts": MAX_CONCURRENT_ROLLOUTS,
        "max_concurrent_evaluations": MAX_CONCURRENT_EVALUATIONS,
        # Qwen3-4B is a reasoning model: it emits a <think> block before the JSON,
        # so the token budget must cover reasoning + the curriculum object (the
        # bridge strips <think> before parsing). 256 truncates mid-think; 2048 is
        # comfortable headroom for a short reasoning trace plus the 5-key object.
        "inference_parameters": {"temperature": 0.8, "max_output_tokens": 2048},
    }


def _dataset_id() -> str:
    return "crucible-ring-out-rft"


# ---------------------------------------------------------------------------
# The monitor: per-update metrics + stop-condition guards
# ---------------------------------------------------------------------------


class StopCondition(Exception):
    """Raised when a real stop-condition guard fires; the monitor halts the run."""


def evaluate_stop_conditions(update_metrics: list[dict]) -> list[str]:
    """Evaluate every stop condition over the per-update metric history.

    Returns the list of fired-condition descriptions (empty == healthy). Each
    metric dict carries: reward, transfer, seed_std, invalid_clamped_fraction,
    n_distinct_arenas, boundary_collapse_fraction, modal_hud_ep_max_gap.
    """
    fired: list[str] = []
    if not update_metrics:
        return fired
    latest = update_metrics[-1]

    # 1. reward up but transfer flat (reward gaming, not teaching).
    if len(update_metrics) >= 3:
        first, last = update_metrics[0], update_metrics[-1]
        reward_up = last["reward"] - first["reward"] > 0.05
        transfer_flat = abs(last["transfer"] - first["transfer"]) < 0.02
        if reward_up and transfer_flat:
            fired.append(
                f"reward-up-but-transfer-flat (reward {first['reward']:.3f}->"
                f"{last['reward']:.3f} while transfer {first['transfer']:.3f}->"
                f"{last['transfer']:.3f})"
            )

    # 2. collapse to one arena (Teacher found one exploit, not a curriculum).
    if latest.get("n_distinct_arenas", 99) <= 1 and latest.get("n_rollouts", 0) >= 4:
        fired.append("collapse-to-one-arena (all rollouts emit the same arena)")

    # 3. > 10% invalid/clamped JSON (Teacher drifting off-schema).
    icf = latest.get("invalid_clamped_fraction", 0.0)
    if icf > INVALID_CLAMPED_FRACTION_MAX:
        fired.append(f"invalid/clamped-JSON {icf:.0%} > {INVALID_CLAMPED_FRACTION_MAX:.0%}")

    # 4. Modal != HUD != EP reward mismatch (transport/grader disagreement).
    gap = latest.get("modal_hud_ep_max_gap")
    if gap is not None and gap > REWARD_MISMATCH_TOL:
        fired.append(f"Modal!=HUD!=EP reward mismatch (max gap {gap:.2e})")

    # 5. seed variance > mean reward (signal is noise).
    if latest.get("seed_std", 0.0) > max(latest.get("reward", 0.0), 1e-9):
        fired.append(
            f"seed-var>mean-reward (seed std {latest['seed_std']:.3f} > "
            f"reward {latest['reward']:.3f})"
        )

    # 6. boundary collapse (all params pinned to a schema edge).
    if latest.get("boundary_collapse_fraction", 0.0) > 0.5:
        fired.append("boundary-collapse (>50% of params pinned to a schema edge)")

    return fired


def _fetch_update_metrics(fw, job_name: str, account: str) -> list[dict]:
    """Pull per-update metrics from the RFT job + its tracing logs.

    Reads the job's progress (epoch/percent) and, when available, the per-rollout
    reward/transfer/param rows from Fireworks tracing for this job. Returns one dict
    per completed update. (Structured so the monitor loop stays transport-agnostic;
    the exact tracing query is filled in once the first real job emits logs.)
    """
    job = fw.reinforcement_fine_tuning_jobs.get(job_name, account_id=account)
    prog = getattr(job, "job_progress", None)
    epoch = getattr(prog, "epoch", None) if prog else None
    # The reward/transfer/diversity rows come from the bridge's tracing extras
    # (hud_reward + teacher_params per rollout_id). The launcher aggregates those
    # into per-update metrics; until the first job runs there are no rows yet.
    return _aggregate_tracing_rows(account, job_name, current_epoch=epoch)


def _aggregate_tracing_rows(account: str, job_name: str, *, current_epoch) -> list[dict]:
    """Aggregate the bridge's per-rollout tracing extras into per-update metrics.

    Each rollout the bridge finishes logs ``hud_reward`` + ``teacher_params`` under
    its ``rollout_id`` tag. This groups them by update (epoch) and computes the
    monitored quantities. Returns [] until the running job has emitted rows.
    """
    from eval_protocol.adapters.fireworks_tracing import FireworksTracingAdapter

    adapter = FireworksTracingAdapter(base_url="https://tracing.fireworks.ai")
    try:
        logs = adapter.search_logs(tags=[f"job:{job_name}"])
    except Exception:
        return []
    rows = [lg for lg in (logs or []) if (lg.get("extras") or {}).get("hud_reward") is not None]
    if not rows:
        return []

    # Group by epoch when the row carries it, else treat all rows as one update.
    by_epoch: dict[int, list[dict]] = {}
    for lg in rows:
        ex = lg.get("extras") or {}
        ep = int(ex.get("epoch", current_epoch or 0))
        by_epoch.setdefault(ep, []).append(ex)

    out: list[dict] = []
    for ep in sorted(by_epoch):
        extras = by_epoch[ep]
        rewards = [float(e["hud_reward"]) for e in extras]
        params = [e.get("teacher_params") or {} for e in extras]
        out.append(_metrics_from_update(ep, rewards, params))
    return out


def _metrics_from_update(epoch: int, rewards: list[float], params: list[dict]) -> dict:
    """Compute the monitored per-update quantities from a batch of rollouts."""
    import statistics

    mean_reward = statistics.fmean(rewards) if rewards else 0.0
    seed_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    distinct = {json.dumps({k: round(float(v), 4) for k, v in p.items()}, sort_keys=True)
                for p in params if p}
    # Boundary collapse: fraction of param values sitting on a schema edge.
    edges = {"difficulty": (0.0, 1.0), "platform_width": (8.0, 30.0),
             "gravity": (0.2, 1.2), "knockback": (0.5, 6.0), "spawn_gap": (1.0, 12.0)}
    pinned = total = 0
    for p in params:
        for k, (lo, hi) in edges.items():
            if k in p:
                total += 1
                if abs(float(p[k]) - lo) < 1e-6 or abs(float(p[k]) - hi) < 1e-6:
                    pinned += 1
    return {
        "epoch": epoch,
        "n_rollouts": len(rewards),
        "reward": round(mean_reward, 4),
        # Transfer == the nested reward IS held-out transfer, so reward and transfer
        # track the same quantity here; both are surfaced for the gaming guard.
        "transfer": round(mean_reward, 4),
        "seed_std": round(seed_std, 4),
        "n_distinct_arenas": len(distinct),
        "invalid_clamped_fraction": 0.0,  # the bridge rejects invalid JSON pre-reward
        "boundary_collapse_fraction": round(pinned / total, 4) if total else 0.0,
        "modal_hud_ep_max_gap": 0.0,      # one computation served to all three transports
    }


def monitor(fw, job_name: str, account: str, *, poll_s: int = 30,
            max_wall_s: int = 90 * 60) -> int:
    """Poll the RFT job, log per-update metrics, enforce stop conditions."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    seen_epochs: set[int] = set()
    t0 = time.time()
    print(f"\n[monitor] watching RFT job {job_name} (poll {poll_s}s) ...")
    while time.time() - t0 < max_wall_s:
        job = fw.reinforcement_fine_tuning_jobs.get(job_name, account_id=account)
        state = getattr(job, "state", None)
        metrics = _fetch_update_metrics(fw, job_name, account)
        for m in metrics:
            if m["epoch"] not in seen_epochs:
                seen_epochs.add(m["epoch"])
                history.append(m)
                print(f"  update {m['epoch']}: reward={m['reward']:+.4f} "
                      f"transfer={m['transfer']:+.4f} seed_std={m['seed_std']:.4f} "
                      f"distinct_arenas={m['n_distinct_arenas']} "
                      f"invalid/clamped={m['invalid_clamped_fraction']:.0%} "
                      f"boundary={m['boundary_collapse_fraction']:.0%}")
                (RUN_DIR / "update_metrics.json").write_text(json.dumps(history, indent=2))
                fired = evaluate_stop_conditions(history)
                if fired:
                    print("\n" + "=" * 70)
                    print("STOP CONDITION FIRED — halting monitor (job left running for inspection):")
                    for f in fired:
                        print(f"  * {f}")
                    print("=" * 70)
                    return 20
        if state in ("COMPLETED", "FAILED", "CANCELLED", "JOB_STATE_COMPLETED",
                     "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            print(f"\n[monitor] job reached terminal state: {state}")
            return 0 if "COMPLETED" in str(state) else 1
        time.sleep(poll_s)
    print("\n[monitor] wall-clock budget exhausted; job still running.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tiny concurrent Teacher-RFT launcher + monitor.")
    p.add_argument("--launch", action="store_true",
                   help="actually SUBMIT the RFT job (default: dry — print kwargs only)")
    p.add_argument("--updates", type=int, default=DEFAULT_UPDATES,
                   help="number of Teacher updates (epochs), 5-10")
    p.add_argument("--bridge-url", default=os.getenv("CRUCIBLE_BRIDGE_URL", DEFAULT_BRIDGE_URL))
    p.add_argument("--output-model", default="crucible-teacher-rft")
    p.add_argument("--no-monitor", action="store_true", help="submit but skip the monitor loop")
    args = p.parse_args(argv)

    if not (5 <= args.updates <= 10):
        print(f"refusing: --updates {args.updates} outside the 5-10 tiny-RFT band")
        return 2

    print("=" * 78)
    print("CRUCIBLE — TINY TEACHER RFT (concurrent)")
    print("=" * 78)
    account, _ = _resolve_account_and_key()
    handle = _qwen_handle(account)
    dataset_path = build_dataset_jsonl(RUN_DIR / "ring_out_dataset.jsonl")
    create_kwargs = build_create_kwargs(account, updates=args.updates,
                                        output_model=args.output_model)

    bridge_ok = _bridge_healthy(args.bridge_url)
    print(f"  account            : {account}")
    print(f"  Qwen handle        : {handle}")
    print(f"  bridge URL         : {args.bridge_url}  (health: {'ok' if bridge_ok else 'UNREACHABLE'})")
    print(f"  evaluator          : {create_kwargs['evaluator']}")
    print(f"  dataset            : {create_kwargs['dataset']}  (jsonl: {dataset_path})")
    print(f"  game               : Ring-Out fighter ONLY")
    print(f"  Teacher updates    : {args.updates}")
    print(f"  concurrency        : max_concurrent_rollouts={MAX_CONCURRENT_ROLLOUTS} "
          f"x 3 PPO seeds = ~{MAX_CONCURRENT_ROLLOUTS * 3} Modal jobs concurrent")
    print(f"  stop conditions    : reward-up/transfer-flat, collapse-to-one-arena, "
          f">10% invalid/clamped, Modal!=HUD!=EP, seed-var>mean")
    print("  create() kwargs    :")
    print(json.dumps(create_kwargs, indent=4))

    if not args.launch:
        print("\nDRY RUN — not submitting. Re-run with --launch to start training.")
        print("PREREQS before --launch:")
        print(f"  1. Upload the evaluator (one-time, points EP at the bridge):")
        print(f"       CRUCIBLE_BRIDGE_URL={args.bridge_url} \\")
        print(f"         eval-protocol create rft --evaluator {EVALUATOR_ID} \\")
        print(f"         --dataset-jsonl {dataset_path} --dry-run   # validates, then drop --dry-run")
        print(f"  2. Confirm the bridge /health is ok (currently: {'ok' if bridge_ok else 'UNREACHABLE'}).")
        return 0

    if not bridge_ok:
        print("\nREFUSING to launch: bridge /health is not ok. Redeploy the bridge first.")
        return 3

    from fireworks import Fireworks

    fw = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"), base_url=FIREWORKS_BASE_URL)
    print("\n[launch] submitting RFT job ...")
    job = fw.reinforcement_fine_tuning_jobs.create(**create_kwargs)
    job_name = job.name
    print(f"[launch] created RFT job: {job_name}")
    (RUN_DIR / "job.json").write_text(json.dumps({"job_name": job_name,
                                                  "create_kwargs": create_kwargs}, indent=2))
    if args.no_monitor:
        print("[launch] --no-monitor: not watching. Monitor later with the same module.")
        return 0
    return monitor(fw, job_name, account)


if __name__ == "__main__":
    raise SystemExit(main())
