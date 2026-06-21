"""Stage 3: Fireworks-hosted Teacher client with an injectable offline transport.

Fireworks exposes an OpenAI-compatible chat-completions API.  Keeping the HTTP
transport injectable lets tests validate prompts, JSON parsing, clamping, and
retry behavior without credentials or network access.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Optional

from engine.games import Game
from training.teacher import Teacher, _clamp_to_schema


Transport = Callable[[dict], dict]


class FireworksTeacher(Teacher):
    def __init__(
        self,
        *,
        model: str = "accounts/fireworks/models/qwen3-4b",
        api_key: Optional[str] = None,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        transport: Optional[Transport] = None,
        max_retries: int = 2,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        self.base_url = base_url.rstrip("/")
        self._transport = transport or self._http_transport
        self.max_retries = max_retries

    def generate(self, game: Game) -> dict:
        prompt = {
            "game": game.name,
            "parameter_schema": {
                key: {"minimum": low, "maximum": high}
                for key, (low, high) in game.param_schema.items()
            },
            "instruction": (
                "Return one JSON object containing only parameter keys. "
                "Choose a potentially learnable environment."
            ),
        }
        payload = {
            "model": self.model,
            "temperature": 0.8,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You design RL training environments. Output strict JSON "
                        "only; never output code or alter the true objective."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
        }

        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                response = self._transport(payload)
                content = response["choices"][0]["message"]["content"]
                parsed = json.loads(content) if isinstance(content, str) else content
                if not isinstance(parsed, dict):
                    raise ValueError("Teacher response must be a JSON object")
                return _clamp_to_schema(parsed, game)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise ValueError(f"Fireworks Teacher returned invalid JSON: {last_error}")

    def _http_transport(self, payload: dict) -> dict:
        if not self.api_key:
            raise RuntimeError("FIREWORKS_API_KEY is required for live Teacher calls")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"Fireworks request failed ({exc.code}): {body}") from exc

