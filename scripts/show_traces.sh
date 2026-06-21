#!/usr/bin/env bash
# show_traces.sh — grab replay traces into the viewer after ANY run (e.g. Stage-6).
#
# What it does:
#   1. Regenerates replays/manifest.json from whatever is in replays/*.json
#      (so new h2h_fighter_*/checkpoint_*/etc. traces are listed immediately).
#   2. Serves the REPO ROOT on a static http.server so BOTH /viewer/ and
#      /replays/ resolve, and the viewer can fetch the live files.
#
# It serves the directory THIS script lives under (repo root = scripts/..),
# so you always get the main checkout you ran it from — not some other worktree.
#
# Usage (from anywhere):
#   scripts/show_traces.sh              # manifest + serve on :8001
#   PORT=8002 scripts/show_traces.sh    # pick a different port
#   scripts/show_traces.sh --no-serve   # only regenerate the manifest, don't serve
#
# Then open:  http://localhost:<PORT>/viewer/viewer.html
#
# Notes:
#   - If the chosen port is already in use, this script leaves that server
#     alone and tells you (it does NOT kill anything). Pass PORT=... to pick
#     a free port, or just open the existing server's URL if it's the root.
#   - Re-run any time after a new run; the manifest regenerates from disk and
#     the viewer's dropdown picks up new traces on reload.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${PORT:-8001}"
HOST="127.0.0.1"

# Prefer the repo venv python by absolute path; fall back to python3 on PATH.
PY="$REPO_ROOT/.venv/bin/python3"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || true)"
fi
if [ -z "$PY" ]; then
  echo "error: no python3 found (looked for $REPO_ROOT/.venv/bin/python3 and PATH)" >&2
  exit 1
fi

echo "[show_traces] repo root : $REPO_ROOT"
echo "[show_traces] python     : $PY"

# 1) Regenerate the manifest from current replays/*.json
"$PY" "$REPO_ROOT/viewer/make_manifest.py"

if [ "${1:-}" = "--no-serve" ]; then
  echo "[show_traces] --no-serve: manifest updated, not serving."
  exit 0
fi

# 2) Serve the repo root (so /viewer/ and /replays/ both resolve).
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[show_traces] port $PORT already in use — NOT touching it." >&2
  echo "[show_traces] If that server already serves THIS repo root, just open:" >&2
  echo "              http://localhost:$PORT/viewer/viewer.html" >&2
  echo "[show_traces] Otherwise re-run with a free port, e.g. PORT=8002 $0" >&2
  exit 1
fi

echo ""
echo "[show_traces] serving $REPO_ROOT on http://$HOST:$PORT"
echo "[show_traces] OPEN ->  http://localhost:$PORT/viewer/viewer.html"
echo "[show_traces] (Ctrl-C to stop)"
echo ""
cd "$REPO_ROOT"
exec "$PY" -m http.server "$PORT" --bind "$HOST"
