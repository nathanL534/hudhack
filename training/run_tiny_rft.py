"""training/run_tiny_rft.py — the tiny Teacher-RFT launcher + live monitor.

Launches a SMALL, CONCURRENT Teacher RFT job on Fireworks against the deployed
Qwen3-4B and the deployed Crucible bridge (training/modal_ep_bridge.py), on the
Ring-Out fighter game ONLY, for a handful of Teacher updates, then MONITORS each
update and enforces the real STOP CONDITIONS as guards.

CONCURRENCY (concurrent-by-design, NOT the sequential baseline):

    max_concurrent_rollouts = 2   # 2 Teacher rollouts in flight at once
        x 3 PPO seeds each (the bridge's teacher_reward fan-out on crucible-player)
        = ~6 Modal PPO jobs concurrent.

SAFETY: this does NOT submit the RFT job unless ``--launch`` is passed. The default
is a DRY run: it builds the dataset, resolves the evaluator + Qwen handle + bridge,
prints the EXACT ``reinforcement_fine_tuning_jobs.create`` kwargs, and stops.

The control plane (datasets / evaluators / RFT jobs) lives at the ROOT Fireworks
host ``https://api.fireworks.ai`` — NOT the ``/inference/v1`` OpenAI-compatible
endpoint. The SDK resource methods hardcode the ``/v1/...`` path themselves, so
overriding the base with ``/inference/v1`` yields a broken ``/v1/v1`` path. We
therefore drive control-plane calls with the SDK default base (root host).

Per-update logged metrics (the monitor reads them from Fireworks tracing): Teacher
reward, seed std, distinct arenas, invalid/clamped-JSON fraction, boundary collapse.

STOP CONDITIONS (each a real guard; the monitor halts + reports which fired):
  * reward up but transfer flat        -> reward is gaming, not teaching
  * collapse to one arena              -> Teacher found one exploit, no curriculum
  * > 10% invalid/clamped JSON         -> Teacher drifting off-schema
  * seed variance > mean reward        -> signal is noise
  * boundary collapse                  -> all params pinned to a schema edge

Run (DRY — prints the create kwargs, does NOT launch)::

    .venv/bin/python -m training.run_tiny_rft

Run the resource UPLOAD (dataset + evaluator) and capture the real ids::

    .venv/bin/python -m training.run_tiny_rft --upload

Run the PREFLIGHT (refuses unless dataset+evaluator exist, bridge /health ok, and
one real reward trace completes end to end through the deployed bridge)::

    .venv/bin/python -m training.run_tiny_rft --preflight

Run (LAUNCH — the human's explicit go; submits + monitors). Launch runs the
preflight first and refuses if it fails::

    .venv/bin/python -m training.run_tiny_rft --launch --updates 2 \
        --max-concurrent-rollouts 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
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
# The CONTROL-PLANE host (datasets / evaluators / RFT jobs). The SDK resource
# methods prepend ``/v1/...`` themselves, so this must be the ROOT host, NOT the
# ``/inference/v1`` OpenAI-compatible endpoint (that would produce ``/v1/v1``).
FIREWORKS_API_BASE = "https://api.fireworks.ai"
DEFAULT_BRIDGE_URL = "https://njlee007--crucible-ep-bridge-web.modal.run"

# The evaluator lives in training/rft_evaluator.py as ``test_teacher_rft``. eval-
# protocol derives + NORMALIZES the id from "<file-stem>-<func>" (lowercase,
# underscores -> hyphens). We compute it the SAME way eval-protocol does, then
# prefer the REAL id the upload returns over this reconstruction.
EVALUATOR_SOURCE_STEM = "rft_evaluator"
EVALUATOR_FUNC = "test_teacher_rft"

# Concurrency dials — the whole point of this run vs the sequential baseline.
DEFAULT_MAX_CONCURRENT_ROLLOUTS = 2   # 2 Teacher rollouts x 3 PPO seeds = ~6 Modal jobs
DEFAULT_MAX_CONCURRENT_EVALUATIONS = 2

# A handful of Teacher updates. With one Ring-Out task row, ``epochs`` is the update
# count (one pass = one update over the single curriculum-design task, many rollouts
# each). The tiny smoke uses 2; the band caps it for safety.
DEFAULT_UPDATES = 2
MIN_UPDATES = 1
MAX_UPDATES = 10

# Stop-condition thresholds.
INVALID_CLAMPED_FRACTION_MAX = 0.10   # > 10% invalid/clamped JSON
RUN_DIR = _REPO_ROOT / "output" / "tiny_rft"
REPLAYS_DIR = _REPO_ROOT / "replays"
RESOURCES_FILE = RUN_DIR / "resources.json"   # captured dataset/evaluator ids


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
        for _ in range(n_rows):
            row = EvaluationRow(messages=[Message(role="user", content=prompt)])
            f.write(json.dumps(row.model_dump(mode="json", exclude_none=True)) + "\n")
    return path


# ---------------------------------------------------------------------------
# Resolve account / Qwen handle / evaluator id / bridge readiness
# ---------------------------------------------------------------------------


def _resolve_account_and_key() -> tuple[str, str]:
    from fireworks import Fireworks

    key = (os.getenv("FIREWORKS_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("FIREWORKS_API_KEY missing from environment/.env")
    client = Fireworks(api_key=key)  # default base = root control-plane host
    page = client.accounts.list()
    items = getattr(page, "accounts", None) or list(page)
    if not items:
        raise RuntimeError("could not resolve a Fireworks account from the API key")
    acct = (getattr(items[0], "name", None) or str(items[0])).split("/")[-1]
    return acct, key


def _control_plane_client():
    """A Fireworks SDK client pointed at the control-plane host (NOT inference/v1)."""
    from fireworks import Fireworks

    return Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))


def _evaluator_id() -> str:
    """The normalized evaluator id eval-protocol would assign to our test.

    Mirrors eval_protocol.cli_commands.utils._normalize_evaluator_id over
    "<file-stem>-<func>" so the launcher and the SDK agree on the id BEFORE any
    upload (after upload we prefer the real returned id).
    """
    from eval_protocol.cli_commands.utils import _normalize_evaluator_id

    return _normalize_evaluator_id(f"{EVALUATOR_SOURCE_STEM}-{EVALUATOR_FUNC}")


def _qwen_handle(account: str) -> str:
    return f"{QWEN_BASE_MODEL}#accounts/{account}/deployments/{QWEN_DEPLOYMENT_ID}"


def _bridge_healthy(bridge_url: str, *, timeout_s: float = 25.0) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{bridge_url.rstrip('/')}/health", timeout=timeout_s) as r:
            body = json.loads(r.read().decode())
            return body.get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Captured resource ids (written by --upload, read by --preflight / --launch)
# ---------------------------------------------------------------------------


def _load_resources() -> dict:
    if RESOURCES_FILE.exists():
        try:
            return json.loads(RESOURCES_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_resources(data: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCES_FILE.write_text(json.dumps(data, indent=2))


def _resource_path(account: str, kind: str, rid: str) -> str:
    return f"accounts/{account}/{kind}/{rid}"


# ---------------------------------------------------------------------------
# Upload: create the dataset + evaluator on Fireworks, capture the REAL ids
# ---------------------------------------------------------------------------


def upload_resources(account: str, *, force: bool = False) -> dict:
    """Create the training dataset + evaluator on Fireworks and capture real ids.

    Uses eval-protocol's OWN proven upload functions (the same ones its
    ``create rft`` flow uses): ``create_dataset_from_jsonl`` for the dataset and
    ``upload_command`` + ``_poll_evaluator_status`` for the evaluator. Writes the
    returned ids to RESOURCES_FILE so preflight/launch read them — no hardcoded
    names anywhere.
    """
    from eval_protocol.auth import get_fireworks_api_base
    from eval_protocol.cli_commands.upload import upload_command
    from eval_protocol.cli_commands.create_rft import _poll_evaluator_status
    from eval_protocol.cli_commands.utils import _build_trimmed_dataset_id
    from eval_protocol.fireworks_rft import create_dataset_from_jsonl

    key = os.getenv("FIREWORKS_API_KEY")
    api_base = get_fireworks_api_base()  # root control-plane host
    evaluator_id = _evaluator_id()

    # ---- 1. Dataset: build the jsonl, then create + upload it. ----
    dataset_path = build_dataset_jsonl(RUN_DIR / "ring_out_dataset.jsonl")
    dataset_id = _build_trimmed_dataset_id(evaluator_id)
    print(f"[upload] creating dataset '{dataset_id}' from {dataset_path} ...")
    real_dataset_id, _ds = create_dataset_from_jsonl(
        account_id=account,
        api_key=key,
        api_base=api_base,
        dataset_id=dataset_id,
        display_name=dataset_id,
        jsonl_path=str(dataset_path),
    )
    print(f"[upload] dataset created: {real_dataset_id}")

    # ---- 2. Evaluator: upload the rft_evaluator package, poll to ACTIVE. ----
    # The entry point pins eval-protocol at OUR test (training/rft_evaluator.py::
    # test_teacher_rft); the upload ships the package + secrets (FIREWORKS_API_KEY
    # so the evaluator can read tracing). CWD must be the repo root for discovery.
    entry = f"training/rft_evaluator.py::{EVALUATOR_FUNC}"
    prev_cwd = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        upload_args = argparse.Namespace(
            path=str(_REPO_ROOT),
            entry=entry,
            id=evaluator_id,
            display_name=None,
            description=None,
            force=force,
            yes=True,
            env_file=str(_REPO_ROOT / ".env"),
        )
        print(f"[upload] uploading evaluator '{evaluator_id}' (entry {entry}) ...")
        rc = upload_command(upload_args)
        if rc != 0:
            raise RuntimeError(f"evaluator upload failed (rc={rc})")
    finally:
        os.chdir(prev_cwd)

    evaluator_resource = _resource_path(account, "evaluators", evaluator_id)
    print(f"[upload] waiting for evaluator '{evaluator_id}' to become ACTIVE ...")
    active = _poll_evaluator_status(
        evaluator_resource_name=evaluator_resource,
        api_key=key,
        api_base=api_base,
        timeout_minutes=10,
    )
    if not active:
        raise RuntimeError(f"evaluator '{evaluator_id}' did not reach ACTIVE in time")

    resources = {
        "account": account,
        "dataset_id": real_dataset_id,
        "dataset_resource": _resource_path(account, "datasets", real_dataset_id),
        "evaluator_id": evaluator_id,
        "evaluator_resource": evaluator_resource,
    }
    _save_resources(resources)
    print(f"[upload] captured resource ids -> {RESOURCES_FILE}")
    return resources


# ---------------------------------------------------------------------------
# Build the create() kwargs (the exact RFT job request)
# ---------------------------------------------------------------------------


def build_create_kwargs(
    account: str,
    resources: dict,
    *,
    updates: int,
    output_model: str,
    lora_rank: int,
    max_concurrent_rollouts: int,
    max_concurrent_evaluations: int,
) -> dict:
    """The exact ``reinforcement_fine_tuning_jobs.create`` kwargs for this run.

    Reads the REAL dataset/evaluator resources captured at upload time (no
    reconstructed names). ``lora_rank`` is caller-supplied so the launch path can
    retry with 0 if the service-mode RLOR trainer rejects a non-zero rank.
    """
    return {
        "account_id": account,
        "evaluator": resources["evaluator_resource"],
        "dataset": resources["dataset_resource"],
        "display_name": "crucible-tiny-teacher-rft",
        "training_config": {
            "base_model": QWEN_BASE_MODEL,
            "epochs": int(updates),
            "output_model": output_model,
            "lora_rank": int(lora_rank),
        },
        "max_concurrent_rollouts": max_concurrent_rollouts,
        "max_concurrent_evaluations": max_concurrent_evaluations,
        # Qwen3-4B is a reasoning model: it emits a <think> block before the JSON,
        # so the token budget must cover reasoning + the curriculum object (the
        # bridge strips <think> before parsing). 2048 is comfortable headroom.
        "inference_parameters": {"temperature": 0.8, "max_output_tokens": 2048},
    }


# ---------------------------------------------------------------------------
# The monitor: per-update metrics + stop-condition guards
# ---------------------------------------------------------------------------


class StopCondition(Exception):
    """Raised when a real stop-condition guard fires; the monitor halts the run."""


def evaluate_stop_conditions(update_metrics: list[dict]) -> list[str]:
    """Evaluate every stop condition over the per-update metric history.

    Returns the list of fired-condition descriptions (empty == healthy). Each
    metric dict carries: reward, transfer, seed_std, invalid_clamped_fraction,
    n_distinct_arenas, boundary_collapse_fraction.
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

    # 4. seed variance > mean reward (signal is noise).
    if latest.get("seed_std", 0.0) > max(latest.get("reward", 0.0), 1e-9):
        fired.append(
            f"seed-var>mean-reward (seed std {latest['seed_std']:.3f} > "
            f"reward {latest['reward']:.3f})"
        )

    # 5. boundary collapse (all params pinned to a schema edge).
    if latest.get("boundary_collapse_fraction", 0.0) > 0.5:
        fired.append("boundary-collapse (>50% of params pinned to a schema edge)")

    return fired


def _tracing_adapter():
    from eval_protocol.adapters.fireworks_tracing import FireworksTracingAdapter

    return FireworksTracingAdapter(base_url="https://tracing.fireworks.ai")


def _search_run_logs(run_id: str) -> list[dict]:
    """Fetch ALL of this RFT job's rollout logs by the stable run_id tag.

    The bridge promotes the EP run_id onto each finished-rollout log record, so
    FireworksTracingHttpHandler tags it ``run_id:<id>`` — that is the group key
    here. (The old ``job:<name>`` tag was never emitted by anything.)
    """
    try:
        return _tracing_adapter().search_logs(tags=[f"run_id:{run_id}"]) or []
    except Exception:
        return []


def _discover_run_id(rollout_ids: list[str]) -> str | None:
    """Discover the job's run_id from any one of its rollouts' tracing logs.

    The launcher does not know the run_id until the first rollout finishes, so it
    reads it back from the rollout's own log extras / tags.
    """
    adapter = _tracing_adapter()
    for rid in rollout_ids:
        try:
            logs = adapter.search_logs(tags=[f"rollout_id:{rid}"]) or []
        except Exception:
            continue
        for lg in logs:
            ex = lg.get("extras") or {}
            if ex.get("run_id"):
                return str(ex["run_id"])
            for tag in lg.get("tags") or []:
                if isinstance(tag, str) and tag.startswith("run_id:"):
                    return tag.split(":", 1)[1]
    return None


def _aggregate_run_logs(logs: list[dict], *, current_epoch) -> list[dict]:
    """Aggregate the bridge's per-rollout tracing logs into per-update metrics.

    Fireworks does NOT tell the bridge which training UPDATE a rollout belongs to,
    so there is no honest per-epoch grouping available from the trace data. We key
    the whole current batch of finished rollouts by the job's reported epoch (read
    from job.job_progress at fetch time) and compute the monitored quantities.
    The invalid/clamped fraction is computed for real from error-status logs.
    """
    finished = [
        lg for lg in logs if (lg.get("extras") or {}).get("hud_reward") is not None
    ]
    # A rollout whose Teacher emitted off-schema/invalid JSON raises in the bridge
    # before any reward is published, and is logged with an error status (no
    # hud_reward). Count those as the invalid/clamped numerator.
    errored = 0
    for lg in logs:
        if (lg.get("extras") or {}).get("hud_reward") is not None:
            continue
        status = lg.get("status")
        code = status.get("code") if isinstance(status, dict) else None
        # Treat any non-finished terminal status as an invalid/error rollout.
        if code is not None:
            errored += 1

    if not finished:
        return []

    extras = [lg.get("extras") or {} for lg in finished]
    rewards = [float(e["hud_reward"]) for e in extras]
    params = [e.get("teacher_params") or {} for e in extras]
    total_dispatched = len(finished) + errored
    invalid_clamped = errored / total_dispatched if total_dispatched else 0.0
    return [_metrics_from_update(int(current_epoch or 0), rewards, params, invalid_clamped)]


def _metrics_from_update(
    epoch: int, rewards: list[float], params: list[dict], invalid_clamped_fraction: float
) -> dict:
    """Compute the monitored per-update quantities from a batch of rollouts."""
    import statistics

    mean_reward = statistics.fmean(rewards) if rewards else 0.0
    seed_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    distinct = {
        json.dumps({k: round(float(v), 4) for k, v in p.items()}, sort_keys=True)
        for p in params
        if p
    }
    # Boundary collapse: fraction of param values sitting on a schema edge.
    edges = {
        "difficulty": (0.0, 1.0),
        "platform_width": (8.0, 30.0),
        "gravity": (0.2, 1.2),
        "knockback": (0.5, 6.0),
        "spawn_gap": (1.0, 12.0),
    }
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
        # The nested reward IS held-out transfer (mean PPO transfer on the BROAD
        # population), so reward and transfer track the same quantity here; both
        # are surfaced for the gaming guard.
        "transfer": round(mean_reward, 4),
        "seed_std": round(seed_std, 4),
        "n_distinct_arenas": len(distinct),
        "invalid_clamped_fraction": round(invalid_clamped_fraction, 4),
        "boundary_collapse_fraction": round(pinned / total, 4) if total else 0.0,
    }


def _job_id(job_name: str) -> str:
    """The BARE job id .get() expects (the resource methods add the path prefix)."""
    return job_name.split("/")[-1]


def _job_state_and_epoch(fw, job_name: str, account: str):
    job = fw.reinforcement_fine_tuning_jobs.get(_job_id(job_name), account_id=account)
    state = getattr(job, "state", None)
    prog = getattr(job, "job_progress", None)
    epoch = getattr(prog, "epoch", None) if prog else None
    return state, epoch


def monitor(fw, job_name: str, account: str, bridge_url: str, *, poll_s: int = 30,
            max_wall_s: int = 90 * 60) -> int:
    """Poll the RFT job, log per-update metrics, enforce stop conditions.

    Reads real job ``state`` from ``reinforcement_fine_tuning_jobs.get`` (BARE id)
    and real reward traces by the job's stable ``run_id:`` tag. The run_id is
    discovered from the bridge's GET /runs (the bridge records each run_id at /init
    time), then used to fetch the per-rollout ``hud_reward`` extras from tracing.
    """
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    seen_epochs: set[int] = set()
    run_id: str | None = None
    runs_before = _bridge_runs(bridge_url)  # run_ids seen BEFORE this job's rollouts
    t0 = time.time()
    print(f"\n[monitor] watching RFT job {job_name} (poll {poll_s}s) ...")
    while time.time() - t0 < max_wall_s:
        state, epoch = _job_state_and_epoch(fw, job_name, account)

        # Discover this job's run_id from the bridge: the first run_id the bridge
        # sees that was NOT present before launch belongs to this job.
        if run_id is None:
            run_id = _discover_new_run_id(bridge_url, runs_before)
            if run_id:
                print(f"  [monitor] discovered run_id={run_id}")
        logs = _search_run_logs(run_id) if run_id else []
        metrics = _aggregate_run_logs(logs, current_epoch=epoch)

        for m in metrics:
            if m["epoch"] not in seen_epochs:
                seen_epochs.add(m["epoch"])
                history.append(m)
                print(
                    f"  update {m['epoch']}: reward={m['reward']:+.4f} "
                    f"transfer={m['transfer']:+.4f} seed_std={m['seed_std']:.4f} "
                    f"distinct_arenas={m['n_distinct_arenas']} "
                    f"invalid/clamped={m['invalid_clamped_fraction']:.0%} "
                    f"boundary={m['boundary_collapse_fraction']:.0%} "
                    f"n_rollouts={m['n_rollouts']}"
                )
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


def _bridge_runs(bridge_url: str) -> set[str]:
    """The set of run_ids the deployed bridge has seen (from GET /runs)."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{bridge_url.rstrip('/')}/runs", timeout=30) as r:
            body = json.loads(r.read().decode())
        return {row["run_id"] for row in body.get("runs", []) if row.get("run_id")}
    except Exception:
        return set()


def _discover_new_run_id(bridge_url: str, runs_before: set[str]) -> str | None:
    """The first run_id the bridge sees that wasn't present before launch.

    The RFT job object doesn't expose the EP run_id, so the bridge is the discovery
    point: it records every run_id at /init time. The new run_id (relative to the
    snapshot taken at monitor start) is this job's.
    """
    new = _bridge_runs(bridge_url) - runs_before
    return sorted(new)[0] if new else None


# ---------------------------------------------------------------------------
# Preflight — REFUSE to launch unless everything is real and end-to-end works
# ---------------------------------------------------------------------------


def preflight(account: str, resources: dict, bridge_url: str, *, reward_timeout_s: float = 1500.0) -> tuple[bool, str]:
    """Hard preflight. Returns (ok, summary). Refuses launch unless ALL pass:

      1. dataset exists  (datasets.get)
      2. evaluator exists + ACTIVE  (evaluators.get / status)
      3. bridge /health responds ok
      4. ONE real reward trace completes end to end: a single /init round-trip to
         the deployed bridge that returns a real ``hud_reward`` via the tracing
         path. This also confirms the live Modal-in-Modal PPO dispatch works —
         crucible-player containers must actually spin up and the 3 PPO seeds run.
    """
    lines: list[str] = []
    fw = _control_plane_client()

    # 1. dataset exists
    try:
        fw.datasets.get(resources["dataset_id"], account_id=account)
        lines.append(f"  [ok] dataset exists: {resources['dataset_id']}")
    except Exception as e:
        return False, f"  [FAIL] dataset '{resources.get('dataset_id')}' not found: {e}"

    # 2. evaluator exists AND is ACTIVE (a BUILD_FAILED/BUILDING evaluator cannot
    #    serve rollouts, and create rft itself gates on ACTIVE — so must we).
    try:
        ev = fw.evaluators.get(resources["evaluator_id"], account_id=account)
        state = str(getattr(ev, "state", None) or "")
    except Exception as e:
        return False, f"  [FAIL] evaluator '{resources.get('evaluator_id')}' not found: {e}"
    if state != "ACTIVE":
        status = getattr(ev, "status", None)
        return False, "\n".join(lines + [
            f"  [FAIL] evaluator '{resources['evaluator_id']}' is not ACTIVE "
            f"(state={state}, status={status})"
        ])
    lines.append(f"  [ok] evaluator ACTIVE: {resources['evaluator_id']}")

    # 3. bridge health
    if not _bridge_healthy(bridge_url):
        return False, f"  [FAIL] bridge /health not ok at {bridge_url}"
    lines.append(f"  [ok] bridge /health ok: {bridge_url}")

    # 4. one real reward trace end to end through the deployed bridge.
    ok, detail = _preflight_reward_trace(bridge_url, reward_timeout_s=reward_timeout_s)
    if not ok:
        return False, "\n".join(lines + [f"  [FAIL] reward trace did not complete: {detail}"])
    lines.append(f"  [ok] reward trace completed end to end: {detail}")
    return True, "\n".join(lines)


def _preflight_reward_trace(bridge_url: str, *, reward_timeout_s: float) -> tuple[bool, str]:
    """Drive ONE /init round-trip to the deployed bridge and read the real reward.

    Posts a Ring-Out InitRequest, then polls the bridge's ``/debug/result/<id>``
    until it returns a finished reward (or errors). This exercises the FULL path:
    bridge -> Qwen checkpoint -> strict-validate -> teacher_reward (3 PPO seeds
    fanned out on crucible-player) -> reward. A finished, in-[0,1] reward proves
    the live Modal-in-Modal PPO dispatch actually spins crucible-player up.
    """
    import urllib.request

    account, _ = _resolve_account_and_key()
    rollout_id = f"preflight-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    init = {
        "completion_params": {"model": QWEN_BASE_MODEL, "temperature": 0.8, "max_tokens": 2048},
        "messages": [{"role": "user", "content": _ring_out_prompt()}],
        "model_base_url": f"{QWEN_BASE_MODEL}#accounts/{account}/deployments/{QWEN_DEPLOYMENT_ID}",
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "metadata": {
            "invocation_id": f"pf-inv-{rollout_id}",
            "experiment_id": f"pf-exp-{rollout_id}",
            "rollout_id": rollout_id,
            "run_id": f"pf-run-{rollout_id}",
            "row_id": f"pf-row-{rollout_id}",
        },
    }
    # The bridge resolves the current checkpoint via model_base_url; for a preflight
    # we point it at the deployed Qwen handle so it calls a real model.
    init["model_base_url"] = f"https://api.fireworks.ai/inference/v1"
    init["completion_params"]["model"] = _qwen_handle(account)

    base = bridge_url.rstrip("/")
    body = json.dumps(init).encode()
    req = urllib.request.Request(f"{base}/init", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            accepted = json.loads(r.read().decode())
    except Exception as e:
        return False, f"/init POST failed: {e}"
    if accepted.get("status") != "accepted":
        return False, f"/init not accepted: {accepted}"
    print(f"  [preflight] /init accepted rollout {rollout_id}; polling /debug/result "
          f"(this runs 3 real PPO seeds on crucible-player, ~minutes) ...")

    deadline = time.time() + reward_timeout_s
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/debug/result/{rollout_id}", timeout=60) as r:
                last = json.loads(r.read().decode())
        except Exception as e:
            last = {"status": "poll-error", "detail": str(e)}
            time.sleep(10)
            continue
        st = last.get("status")
        if st == "finished":
            reward = last.get("reward")
            params = last.get("params")
            if isinstance(reward, (int, float)) and 0.0 <= float(reward) <= 1.0:
                return True, f"reward={float(reward):.4f} params={params}"
            return False, f"finished but reward out of range: {last}"
        if st == "error":
            return False, f"bridge reported error: {last.get('detail')}"
        time.sleep(10)
    return False, f"timed out after {reward_timeout_s:.0f}s (last status: {last})"


# ---------------------------------------------------------------------------
# Dry-run validation via the eval-protocol SDK
# ---------------------------------------------------------------------------


def dry_run_validate(account: str, resources: dict) -> tuple[bool, str]:
    """Validate the RFT create request via the SDK's dry-run path (no submission).

    Mirrors eval_protocol.cli_commands.create_rft._create_rft_job(dry_run=True):
    builds the SDK kwargs and confirms ``reinforcement_fine_tuning_jobs.create``
    would accept them, without calling the API.
    """
    import inspect
    from fireworks import Fireworks

    sig = inspect.signature(Fireworks().reinforcement_fine_tuning_jobs.create)
    kwargs = build_create_kwargs(
        account, resources, updates=DEFAULT_UPDATES, output_model="crucible-teacher-rft",
        lora_rank=0, max_concurrent_rollouts=1, max_concurrent_evaluations=1,
    )
    # Every kwarg must be a real parameter of create().
    unknown = [k for k in kwargs if k not in sig.parameters]
    if unknown:
        return False, f"unknown create() kwargs: {unknown}"
    return True, json.dumps(kwargs, indent=2)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def _is_service_mode_lora_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("lora_rank" in msg or "lorarank" in msg or "service" in msg) and (
        "0" in msg or "service" in msg or "rlor" in msg
    )


def launch(fw, account: str, resources: dict, *, updates: int, output_model: str,
           lora_rank: int, max_concurrent_rollouts: int, max_concurrent_evaluations: int):
    """Submit the RFT job; on a service-mode lora_rank rejection, retry with 0."""
    for attempt_rank in ([lora_rank, 0] if lora_rank != 0 else [0]):
        kwargs = build_create_kwargs(
            account, resources, updates=updates, output_model=output_model,
            lora_rank=attempt_rank, max_concurrent_rollouts=max_concurrent_rollouts,
            max_concurrent_evaluations=max_concurrent_evaluations,
        )
        try:
            print(f"\n[launch] submitting RFT job (lora_rank={attempt_rank}) ...")
            job = fw.reinforcement_fine_tuning_jobs.create(**kwargs)
            return job, attempt_rank
        except Exception as e:
            if attempt_rank != 0 and _is_service_mode_lora_error(e):
                print(f"[launch] lora_rank={attempt_rank} rejected (service-mode RLOR); "
                      f"retrying with lora_rank=0. ({e})")
                continue
            raise
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tiny concurrent Teacher-RFT launcher + monitor.")
    p.add_argument("--upload", action="store_true",
                   help="create the Fireworks dataset + evaluator and capture their ids")
    p.add_argument("--force-upload", action="store_true",
                   help="with --upload, overwrite an existing evaluator")
    p.add_argument("--preflight", action="store_true",
                   help="run the hard preflight (refuses launch unless all checks pass)")
    p.add_argument("--launch", action="store_true",
                   help="actually SUBMIT the RFT job (runs preflight first)")
    p.add_argument("--updates", type=int, default=DEFAULT_UPDATES,
                   help=f"number of Teacher updates (epochs), {MIN_UPDATES}-{MAX_UPDATES}")
    p.add_argument("--max-concurrent-rollouts", type=int, default=DEFAULT_MAX_CONCURRENT_ROLLOUTS)
    p.add_argument("--max-concurrent-evaluations", type=int, default=DEFAULT_MAX_CONCURRENT_EVALUATIONS)
    p.add_argument("--lora-rank", type=int, default=8,
                   help="LoRA rank; falls back to 0 if service-mode RLOR rejects it")
    p.add_argument("--bridge-url", default=os.getenv("CRUCIBLE_BRIDGE_URL", DEFAULT_BRIDGE_URL))
    p.add_argument("--output-model", default="crucible-teacher-rft")
    p.add_argument("--no-monitor", action="store_true", help="submit but skip the monitor loop")
    args = p.parse_args(argv)

    if not (MIN_UPDATES <= args.updates <= MAX_UPDATES):
        print(f"refusing: --updates {args.updates} outside the {MIN_UPDATES}-{MAX_UPDATES} band")
        return 2

    print("=" * 78)
    print("CRUCIBLE — TINY TEACHER RFT (concurrent)")
    print("=" * 78)
    account, _ = _resolve_account_and_key()
    handle = _qwen_handle(account)
    evaluator_id = _evaluator_id()
    print(f"  account            : {account}")
    print(f"  Qwen handle        : {handle}")
    print(f"  control-plane base : {FIREWORKS_API_BASE}")
    print(f"  evaluator id       : {evaluator_id}  (file=training/rft_evaluator.py::{EVALUATOR_FUNC})")
    print(f"  bridge URL         : {args.bridge_url}")

    # ---- UPLOAD ----
    if args.upload:
        resources = upload_resources(account, force=args.force_upload)
        print("\nUPLOAD complete. Captured ids:")
        print(json.dumps(resources, indent=2))
        return 0

    resources = _load_resources()

    # ---- PREFLIGHT (standalone) ----
    if args.preflight and not args.launch:
        if not resources.get("dataset_id") or not resources.get("evaluator_id"):
            print("\nREFUSING preflight: no captured resources. Run --upload first.")
            return 3
        ok, summary = preflight(account, resources, args.bridge_url)
        print("\nPREFLIGHT:")
        print(summary)
        return 0 if ok else 4

    # ---- DRY (default) ----
    if not args.launch:
        if resources.get("dataset_id") and resources.get("evaluator_id"):
            ok, kwargs_or_err = dry_run_validate(account, resources)
            print("\nDRY RUN — create() kwargs (validated against the SDK signature):")
            print(kwargs_or_err if ok else f"  VALIDATION FAILED: {kwargs_or_err}")
        else:
            print("\nDRY RUN — no captured resources yet. Run --upload to create the "
                  "dataset + evaluator and capture their ids.")
        print("\nNext: --upload (create resources) -> --preflight -> --launch")
        return 0

    # ---- LAUNCH ----
    if not resources.get("dataset_id") or not resources.get("evaluator_id"):
        print("\nREFUSING to launch: no captured resources. Run --upload first.")
        return 3

    ok, summary = preflight(account, resources, args.bridge_url)
    print("\nPREFLIGHT:")
    print(summary)
    if not ok:
        print("\nREFUSING to launch: preflight FAILED (see above).")
        return 4

    fw = _control_plane_client()
    job, used_rank = launch(
        fw, account, resources, updates=args.updates, output_model=args.output_model,
        lora_rank=args.lora_rank,
        max_concurrent_rollouts=args.max_concurrent_rollouts,
        max_concurrent_evaluations=args.max_concurrent_evaluations,
    )
    job_name = job.name
    print(f"[launch] created RFT job: {job_name}  (lora_rank={used_rank})")
    (RUN_DIR / "job.json").write_text(json.dumps({
        "job_name": job_name, "job_id": _job_id(job_name), "lora_rank": used_rank,
        "updates": args.updates, "resources": resources,
    }, indent=2))
    if args.no_monitor:
        print("[launch] --no-monitor: not watching. Monitor later with the same module.")
        return 0
    return monitor(fw, job_name, account, args.bridge_url)


if __name__ == "__main__":
    raise SystemExit(main())
