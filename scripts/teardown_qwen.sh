#!/usr/bin/env bash
#
# teardown_qwen.sh — Delete the dedicated Qwen3-4B Fireworks deployment to STOP billing.
#
# A dedicated deployment bills (~$7/hr) for as long as it is up. This script finds
# the deployment created by deploy_qwen.sh and deletes it. Run this the moment you
# are done testing.
#
# Tooling note: firectl is not installed here, so this drives the Fireworks
# control-plane REST API via the installed `fireworks` Python SDK (.venv/bin/python).
#
# Usage:
#   scripts/teardown_qwen.sh                 # deletes deployment id 'qwen3-4b-dedicated'
#   scripts/teardown_qwen.sh <deployment_id> # delete a specific deployment id
#   DEPLOYMENT_ID=foo scripts/teardown_qwen.sh
#   ALL=1 scripts/teardown_qwen.sh           # delete ALL deployments on the account (careful)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"

DEPLOYMENT_ID="${1:-${DEPLOYMENT_ID:-qwen3-4b-dedicated}}"
DELETE_ALL="${ALL:-0}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: $PY not found." >&2
  exit 2
fi
if [[ ! -f "$ROOT/.env" ]]; then
  echo "ERROR: missing .env at repo root." >&2
  exit 2
fi

echo "==> Tearing down dedicated Qwen3-4B deployment(s) to stop billing ..."
[[ "$DELETE_ALL" == "1" ]] && echo "==> ALL=1 set: will delete EVERY deployment on the account." \
                           || echo "==> Target deployment id: ${DEPLOYMENT_ID}"
echo

DEPLOYMENT_ID="$DEPLOYMENT_ID" \
DELETE_ALL="$DELETE_ALL" \
"$PY" - <<'PY'
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")

from fireworks import Fireworks  # noqa: E402

API_KEY = os.environ.get("FIREWORKS_API_KEY", "").strip()
if not API_KEY:
    print("ERROR: FIREWORKS_API_KEY missing/empty in .env", file=sys.stderr)
    sys.exit(2)

DEPLOYMENT_ID = os.environ["DEPLOYMENT_ID"]
DELETE_ALL = os.environ.get("DELETE_ALL", "0") == "1"

# Resolve account from the key.
client = Fireworks(api_key=API_KEY)
page = client.accounts.list()
items = getattr(page, "accounts", None) or list(page)
acct_name = getattr(items[0], "name", None) or str(items[0])
ACCOUNT_ID = acct_name.split("/")[-1]

fw = Fireworks(api_key=API_KEY, account_id=ACCOUNT_ID)

deps = list(fw.deployments.list())
if not deps:
    print("==> No deployments found on the account. Nothing to bill, nothing to delete.")
    sys.exit(0)

if DELETE_ALL:
    targets = deps
else:
    targets = [d for d in deps if (d.name or "").split("/")[-1] == DEPLOYMENT_ID]
    if not targets:
        print(f"==> No deployment matched id '{DEPLOYMENT_ID}'.")
        print("    Current deployments on the account:")
        for d in deps:
            print(f"      - {(d.name or '').split('/')[-1]} | base={d.base_model} | state={d.state}")
        print("    Re-run with the right id, or ALL=1 to delete everything.")
        sys.exit(1)

deleted = 0
for d in targets:
    dep_id = (d.name or "").split("/")[-1]
    print(f"==> Deleting deployment '{dep_id}' (base={d.base_model}, state={d.state}) ...")
    try:
        fw.deployments.delete(dep_id)
        print(f"    deleted: {dep_id}")
        deleted += 1
    except Exception as e:
        print(f"    ERROR deleting {dep_id}: {type(e).__name__}: {e}", file=sys.stderr)

# Verify nothing live remains for what we targeted.
remaining = [
    (d.name or "").split("/")[-1]
    for d in fw.deployments.list()
    if getattr(d, "state", None) not in ("DELETED", "DELETING")
]
print()
print(f"==> Deleted {deleted} deployment(s).")
if remaining:
    print(f"==> Still present (not DELETED): {remaining}")
    print("    If any of these are unexpected, re-run teardown with their id.")
else:
    print("==> No active deployments remain. Billing for dedicated serving has stopped.")
PY

echo
echo "==> Teardown complete. Verify in the Fireworks dashboard if you want a second confirmation."
