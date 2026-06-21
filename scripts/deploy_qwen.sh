#!/usr/bin/env bash
#
# deploy_qwen.sh — Deploy a Qwen3-4B base model to a DEDICATED Fireworks endpoint.
#
# This account has no serverless access to Qwen3-4B, so serving inference (and the
# RFT round-trip smoke test) requires a dedicated on-demand deployment. This script
# creates one, waits for it to become READY, prints the deployment id + the
# OpenAI-compatible model handle you point inference at, then runs a tiny
# completion to prove it serves real tokens.
#
# COST: a dedicated Qwen3-4B deployment bills while UP (~$7/hr on 1x H100-class
# accelerator). It does NOT stop when this script exits. Run scripts/teardown_qwen.sh
# to delete it and STOP billing.
#
# Tooling note: firectl is not installed in this environment, so this script drives
# the Fireworks control-plane REST API via the installed `fireworks` Python SDK
# (.venv/bin/python). If you later install firectl, the same deployment is visible
# via `firectl list deployments`.
#
# Usage:
#   scripts/deploy_qwen.sh                       # deploys accounts/fireworks/models/qwen3-4b
#   scripts/deploy_qwen.sh qwen3-4b-instruct-2507  # deploys the instruct variant
#   MODEL_SLUG=qwen3-4b scripts/deploy_qwen.sh   # same, via env var
#
# Env overrides (all optional):
#   MODEL_SLUG          base model slug under accounts/fireworks/models (default: qwen3-4b)
#   DEPLOYMENT_ID       fixed deployment id to make this idempotent (default: qwen3-4b-dedicated)
#   ACCELERATOR_TYPE    e.g. NVIDIA_H100_80GB (default: let Fireworks pick the default shape)
#   MIN_REPLICAS        min replica count (default: 1 — keeps it warm; set 0 to scale-to-zero)
#   WAIT_TIMEOUT_SECS   how long to wait for READY (default: 900)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"

# ---- positional arg or env for the model slug -------------------------------
MODEL_SLUG="${1:-${MODEL_SLUG:-qwen3-4b}}"
DEPLOYMENT_ID="${DEPLOYMENT_ID:-qwen3-4b-dedicated}"
MIN_REPLICAS="${MIN_REPLICAS:-1}"
WAIT_TIMEOUT_SECS="${WAIT_TIMEOUT_SECS:-900}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-}"   # empty => omit => Fireworks default shape

# ---- preflight --------------------------------------------------------------
if [[ ! -x "$PY" ]]; then
  echo "ERROR: $PY not found. Create the venv first (it ships with the repo)." >&2
  exit 2
fi
if [[ ! -f "$ROOT/.env" ]]; then
  echo "ERROR: missing .env at repo root. Copy .env.example and set FIREWORKS_API_KEY." >&2
  exit 2
fi
"$PY" -c 'import fireworks' 2>/dev/null || {
  echo "ERROR: the 'fireworks' Python SDK is not installed in .venv." >&2
  echo "       Install it with: $PY -m pip install fireworks" >&2
  exit 2
}

echo "==> Deploying base model: accounts/fireworks/models/${MODEL_SLUG}"
echo "==> Target deployment id : ${DEPLOYMENT_ID}"
echo "==> Min replicas         : ${MIN_REPLICAS}"
[[ -n "$ACCELERATOR_TYPE" ]] && echo "==> Accelerator          : ${ACCELERATOR_TYPE}"
echo

# The key is read inside Python from .env via python-dotenv and is NEVER printed.
MODEL_SLUG="$MODEL_SLUG" \
DEPLOYMENT_ID="$DEPLOYMENT_ID" \
MIN_REPLICAS="$MIN_REPLICAS" \
WAIT_TIMEOUT_SECS="$WAIT_TIMEOUT_SECS" \
ACCELERATOR_TYPE="$ACCELERATOR_TYPE" \
"$PY" - <<'PY'
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
load_dotenv(Path.cwd() / ".env")

from fireworks import Fireworks  # noqa: E402

API_KEY = os.environ.get("FIREWORKS_API_KEY", "").strip()
if not API_KEY:
    print("ERROR: FIREWORKS_API_KEY missing/empty in .env", file=sys.stderr)
    sys.exit(2)

MODEL_SLUG = os.environ["MODEL_SLUG"]
DEPLOYMENT_ID = os.environ["DEPLOYMENT_ID"]
MIN_REPLICAS = int(os.environ["MIN_REPLICAS"])
WAIT_TIMEOUT = int(os.environ["WAIT_TIMEOUT_SECS"])
ACCELERATOR_TYPE = os.environ.get("ACCELERATOR_TYPE", "").strip() or None

BASE_MODEL = f"accounts/fireworks/models/{MODEL_SLUG}"

# Resolve the account from the API key (no hardcoding).
client = Fireworks(api_key=API_KEY)
accts = list(client.accounts.list())
acct_objs = getattr(accts[0], "accounts", None) if accts and hasattr(accts[0], "accounts") else accts
# accounts.list() returns a page; iterate to find the account name.
acct_name = None
try:
    page = client.accounts.list()
    items = getattr(page, "accounts", None) or list(page)
    first = items[0]
    acct_name = getattr(first, "name", None) or str(first)
except Exception:
    pass
if not acct_name:
    print("ERROR: could not resolve account from API key", file=sys.stderr)
    sys.exit(2)
ACCOUNT_ID = acct_name.split("/")[-1]  # accounts/<id> -> <id>

fw = Fireworks(api_key=API_KEY, account_id=ACCOUNT_ID)

# 0) Confirm the base model exists + is READY before spending money.
try:
    m = fw.models.get(MODEL_SLUG, account_id="fireworks")
    if getattr(m, "state", None) not in (None, "READY"):
        print(f"WARNING: base model {BASE_MODEL} state={m.state} (not READY)", file=sys.stderr)
except Exception as e:
    print(f"ERROR: base model {BASE_MODEL} not resolvable: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(2)

def deployment_handle(dep_id: str) -> str:
    """OpenAI-compatible model string that routes inference to this dedicated deployment."""
    return f"{BASE_MODEL}#accounts/{ACCOUNT_ID}/deployments/{dep_id}"

def find_existing():
    """Idempotency: reuse a live deployment for this base model if one exists."""
    for d in fw.deployments.list():
        if getattr(d, "base_model", None) != BASE_MODEL:
            continue
        short = (d.name or "").split("/")[-1]
        if short == DEPLOYMENT_ID or getattr(d, "state", None) in ("READY", "CREATING", "UPDATING"):
            return d
    return None

existing = find_existing()
if existing is not None:
    dep_id = (existing.name or "").split("/")[-1]
    print(f"==> Reusing existing deployment '{dep_id}' (state={existing.state}); not creating a new one.")
    dep = existing
else:
    print(f"==> Creating dedicated deployment '{DEPLOYMENT_ID}' for {BASE_MODEL} ...")
    kwargs = dict(
        base_model=BASE_MODEL,
        deployment_id=DEPLOYMENT_ID,
        min_replica_count=MIN_REPLICAS,
        max_replica_count=max(MIN_REPLICAS, 1),
        display_name=f"qwen3-4b dedicated ({MODEL_SLUG})",
        description="Dedicated serving endpoint for Qwen3-4B (HUD hackathon). Bills while up.",
    )
    if ACCELERATOR_TYPE:
        kwargs["accelerator_type"] = ACCELERATOR_TYPE
    dep = fw.deployments.create(**kwargs)
    print(f"==> Create accepted: {dep.name} (state={dep.state})")

dep_id = (dep.name or "").split("/")[-1]

# Wait for READY.
print(f"==> Waiting up to {WAIT_TIMEOUT}s for deployment to become READY ...")
deadline = time.time() + WAIT_TIMEOUT
last_state = None
while time.time() < deadline:
    cur = fw.deployments.get(dep_id)
    st = getattr(cur, "state", None)
    if st != last_state:
        print(f"    state={st} replicas={getattr(cur, 'replica_count', '?')}")
        last_state = st
    if st == "READY":
        dep = cur
        break
    if st in ("FAILED", "DELETING", "DELETED"):
        print(f"ERROR: deployment entered terminal state {st}", file=sys.stderr)
        sys.exit(1)
    time.sleep(10)
else:
    print(f"ERROR: timed out after {WAIT_TIMEOUT}s waiting for READY (last state={last_state}).", file=sys.stderr)
    print("       The deployment may still be coming up; check teardown to avoid charges.", file=sys.stderr)
    sys.exit(1)

handle = deployment_handle(dep_id)

print()
print("============================================================")
print("DEPLOYMENT IS LIVE (and BILLING ~\\$7/hr until torn down)")
print("============================================================")
print(f"  deployment_name : {dep.name}")
print(f"  deployment_id   : {dep_id}")
print(f"  base_model      : {BASE_MODEL}")
print(f"  state           : {dep.state}")
print(f"  inference handle: {handle}")
print(f"  endpoint base   : https://api.fireworks.ai/inference/v1")
print("============================================================")

# ---- tiny inference smoke test against the DEDICATED endpoint ----------------
print()
print("==> Running one tiny inference call against the deployed endpoint ...")
try:
    resp = fw.chat.completions.create(
        model=handle,
        messages=[{"role": "user", "content": "Reply with exactly: deployment online"}],
        max_tokens=16,
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    print("==> COMPLETION OK. Model returned:")
    print(f"    {text!r}")
except Exception as e:
    print(f"WARNING: inference smoke test failed: {type(e).__name__}: {e}", file=sys.stderr)
    print("         The deployment is READY but the round-trip call errored.", file=sys.stderr)
    print("         The endpoint is still BILLING — tear it down if you are not using it.", file=sys.stderr)
    sys.exit(1)

print()
print("Reminder: this endpoint keeps billing until you run:")
print("  scripts/teardown_qwen.sh")
PY
