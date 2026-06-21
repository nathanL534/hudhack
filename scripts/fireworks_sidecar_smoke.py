#!/usr/bin/env python3
"""Run one real Qwen rollout through the deployed Modal/HUD sidecar.

Usage:
    python scripts/fireworks_sidecar_smoke.py https://<modal-host>

The script POSTs a real Eval Protocol InitRequest, then polls the sidecar's
debug result endpoint. The production RemoteRolloutProcessor still consumes the
same reward from Fireworks tracing.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request_json(url: str, *, payload: dict | None = None) -> tuple[int, dict]:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode())
    except HTTPError as exc:
        text = exc.read().decode(errors="replace")
        try:
            detail = json.loads(text)
        except json.JSONDecodeError:
            detail = {"detail": text}
        return exc.code, detail


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: fireworks_sidecar_smoke.py https://<modal-host>", file=sys.stderr)
        return 2

    base_url = sys.argv[1].rstrip("/")
    rollout_id = f"crucible-smoke-{uuid.uuid4().hex[:12]}"
    payload = {
        "completion_params": {
            "model": "accounts/fireworks/models/qwen3-4b",
            "temperature": 0.8,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You design fighter RL curricula. Return strict JSON only, "
                    "with every requested parameter."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return difficulty, platform_width, gravity, knockback, and "
                    "spawn_gap. Choose a moderately learnable arena."
                ),
            },
        ],
        "model_base_url": "https://api.fireworks.ai/inference/v1",
        "metadata": {
            "invocation_id": rollout_id,
            "experiment_id": "crucible-live-smoke",
            "rollout_id": rollout_id,
            "run_id": rollout_id,
            "row_id": "fighter-smoke-1",
        },
    }

    status, response = request_json(f"{base_url}/init", payload=payload)
    if status not in (200, 202):
        print(json.dumps({"init_status": status, "response": response}, indent=2))
        return 1

    deadline = time.time() + 120
    while time.time() < deadline:
        status, result = request_json(f"{base_url}/debug/result/{rollout_id}")
        if result.get("status") == "finished":
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if result.get("status") == "error" or status >= 500:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
        time.sleep(2)

    print(json.dumps({"status": "timeout", "rollout_id": rollout_id}, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
