#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Missing .env. Copy .env.example and add FIREWORKS_API_KEY." >&2
  exit 2
fi

echo "1/3 Testing the Fireworks-hosted base Teacher..."
.venv/bin/python scripts/fireworks_base_smoke.py

echo "2/3 Creating/updating the Modal secret..."
FIREWORKS_API_KEY="$(
  .venv/bin/python -c \
    'from dotenv import dotenv_values; print(dotenv_values(".env").get("FIREWORKS_API_KEY", ""))'
)"
if [[ -z "$FIREWORKS_API_KEY" ]]; then
  echo "FIREWORKS_API_KEY is empty in .env" >&2
  exit 2
fi
.venv/bin/modal secret create crucible-fireworks \
  "FIREWORKS_API_KEY=$FIREWORKS_API_KEY" --force
unset FIREWORKS_API_KEY

echo "3/3 Deploying the Fireworks/HUD sidecar on Modal..."
.venv/bin/modal deploy training/modal_ep_bridge.py

echo
echo "Copy the deployed HTTPS URL, then run:"
echo "  .venv/bin/python scripts/fireworks_sidecar_smoke.py <MODAL_URL>"
