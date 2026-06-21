"""Stage 3: Fireworks-hosted Teacher client with an injectable offline transport.

Fireworks exposes an OpenAI-compatible chat-completions API.  Keeping the HTTP
transport injectable lets tests validate prompts, JSON parsing, clamping, and
retry behavior without credentials or network access.

Two contracts live here, at two levels:

  * ``generate(game) -> dict`` — the Teacher's I/O contract (see teacher.py):
    a params dict for ONE arena, every key in the game's ``param_schema`` and
    every value strictly clamped into range.  This is the RFT rollout unit.
  * ``to_curriculum(params, game, ...) -> CurriculumSpec`` — the bridge to the
    frozen runner-level contract.  The Teacher's raw params are turned into a
    validated ``ArenaSpec``/``CurriculumSpec`` here, at the seam, so the
    orchestrator only ever sees contract-valid curricula.

Offline mode: ``FireworksTeacher.offline(params=...)`` returns a Teacher backed
by a canned transport, so the whole pipeline is runnable and testable without a
``FIREWORKS_API_KEY``.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Callable, Optional

from contracts import ArenaSpec, CurriculumSpec
from engine.games import Game
from training.teacher import Teacher, _clamp_to_schema


Transport = Callable[[dict], dict]


def _strict_validate_params(parsed: object, game: Game) -> dict:
    """Strict-validate a raw Teacher response, then clamp it to the schema.

    Strict means: it must be a JSON object, it must carry at least one key the
    game actually declares, and every numeric value must be finite.  A response
    that is a list, a bare scalar, or carries only junk keys is rejected (so we
    do not silently fall back to all-defaults and mask a broken Teacher).
    """
    if not isinstance(parsed, dict):
        raise ValueError("Teacher response must be a JSON object")
    known = set(game.param_schema)
    overlap = known.intersection(parsed)
    if not overlap:
        raise ValueError(
            f"Teacher response has no known parameter keys; expected any of {sorted(known)}"
        )
    for key in overlap:
        value = parsed[key]
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parameter {key!r} is not numeric: {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"parameter {key!r} is not finite: {value!r}")
    return _clamp_to_schema(parsed, game)


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

    @classmethod
    def offline(cls, params: Optional[dict] = None, **kwargs) -> "FireworksTeacher":
        """A key-free Teacher backed by a canned response.

        Returns a valid, schema-clamped params dict for any game without a
        network call, so the pipeline and tests run without ``FIREWORKS_API_KEY``.
        Defaults to a mid-difficulty arena when no params are supplied.
        """
        canned = params or {"difficulty": 0.5, "obstacle_density": 0.2, "goal_distance": 0.6}

        def _mock_transport(_payload: dict) -> dict:
            return {"choices": [{"message": {"content": json.dumps(canned)}}]}

        return cls(transport=_mock_transport, **kwargs)

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
                return _strict_validate_params(parsed, game)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise ValueError(f"Fireworks Teacher returned invalid JSON: {last_error}")

    def to_curriculum(
        self,
        game: Game,
        *,
        curriculum_id: str,
        n_arenas: int = 1,
    ) -> CurriculumSpec:
        """Generate ``n_arenas`` and wrap them in a validated CurriculumSpec.

        This is the boundary that turns the Teacher's params dicts into the
        frozen runner-level contract, so the orchestrator only ever handles
        contract-valid curricula.  Validation is the frozen ``CurriculumSpec``.
        """
        arenas = [_params_to_arena_spec(self.generate(game), game) for _ in range(n_arenas)]
        return CurriculumSpec(
            game_id=game.name,
            curriculum_id=curriculum_id,
            arenas=arenas,
        ).validate()

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


def _params_to_arena_spec(params: dict, game: Game) -> ArenaSpec:
    """Map a Teacher params dict onto the frozen ArenaSpec.

    ``difficulty`` carries directly; ``map_size`` scales with difficulty so a
    harder arena is a bigger grid; ``hazard_density`` reads from the game's
    obstacle knob.  All bounds are enforced by ArenaSpec itself on construction.
    """
    difficulty = float(params.get("difficulty", 0.5))
    hazard = float(params.get("obstacle_density", 0.15))
    map_size = int(round(5 + difficulty * 27))  # 5..32, inside ArenaSpec's [3, 64]
    return ArenaSpec(
        map_size=map_size,
        doors=0,
        keys=0,
        hazard_density=min(1.0, max(0.0, hazard)),
        difficulty=min(1.0, max(0.0, difficulty)),
    )
