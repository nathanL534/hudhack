"""HUD environment for grading Teacher-generated fighter curricula.

HUD owns the task contract and the reward.  The scorer is injected so the same
task can use a fast local proxy during development and Modal PPO validation in
the real run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from hud import Environment

TeacherScorer = Callable[[dict[str, float]], float]

env = Environment("crucible-teacher")


def clamp_reward(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def parse_teacher_params(answer: str, bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Parse strict JSON and clamp every declared parameter to its safe range."""
    raw = json.loads(answer)
    if not isinstance(raw, dict):
        raise ValueError("Teacher output must be a JSON object")

    params: dict[str, float] = {}
    for key, (low, high) in bounds.items():
        if key not in raw:
            raise ValueError(f"Teacher output is missing parameter {key!r}")
        value = float(raw[key])
        params[key] = min(float(high), max(float(low), value))
    return params


def score_teacher_answer(
    answer: str,
    *,
    bounds: dict[str, tuple[float, float]],
    scorer: TeacherScorer,
) -> tuple[float, dict[str, float]]:
    """The shared HUD grading core used by local, Modal, and RFT execution."""
    params = parse_teacher_params(answer, bounds)
    return clamp_reward(scorer(params)), params


@env.template(id="design_curriculum")
async def design_curriculum(
    game_description: str,
    parameter_bounds: dict[str, list[float]],
):
    """Generate one safe parameter set whose measured learning value is high."""
    prompt = {
        "role": "curriculum_designer",
        "game": game_description,
        "parameter_bounds": parameter_bounds,
        "instruction": "Return only one JSON object with every parameter key.",
    }
    answer = yield json.dumps(prompt, sort_keys=True)

    # The deployed task uses the configured scorer. The remote RFT bridge drives
    # the same grading core with its injected local/Modal scorer.
    bounds = {key: (float(value[0]), float(value[1])) for key, value in parameter_bounds.items()}
    scorer = _configured_scorer
    reward, _ = score_teacher_answer(answer or "{}", bounds=bounds, scorer=scorer)
    yield reward


def _unconfigured_scorer(_: dict[str, float]) -> float:
    raise RuntimeError("configure_scorer() must be called before serving the HUD environment")


_configured_scorer: TeacherScorer = _unconfigured_scorer


def configure_scorer(scorer: TeacherScorer) -> None:
    """Configure the scorer used when this HUD environment is served directly."""
    global _configured_scorer
    _configured_scorer = scorer


async def run_template_smoke(
    answer: str,
    *,
    bounds: dict[str, tuple[float, float]],
    scorer: TeacherScorer,
) -> float:
    """Drive HUD's prompt -> answer -> reward generator without credentials."""
    parameter_bounds = {key: [low, high] for key, (low, high) in bounds.items()}
    configure_scorer(scorer)
    generator = design_curriculum.func("fighter", parameter_bounds)
    await anext(generator)
    return float(await generator.asend(answer))

