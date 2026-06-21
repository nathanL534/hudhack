"""HUD environment for grading Teacher-generated fighter curricula.

HUD owns the task contract and the reward.  The scorer is injected so the same
task can use a fast local proxy during development and Modal PPO validation in
the real run.

Load-bearing wiring: the default configured scorer is :func:`fighter_scorer`,
which routes the Teacher's parameter set through the REAL Crucible fighter
reward (``harness.scoring.score_curriculum`` over ``FighterGameAdapter``). When
this env is served (locally via ``hud.eval.LocalRuntime`` or hosted), the reward
HUD records on each trace is the genuine band-gated gap-proxy — strong (scripted)
minus weak (random) — computed on local CPU, not a placeholder. ``run_hud_eval.py``
drives a minimal HUD run that captures real traces from exactly this path.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import Any

# This module is re-imported in a child process by ``hud.eval.LocalRuntime``
# (``python -m hud.environment.server``), and ``load_module`` only puts THIS
# file's directory (training/) on ``sys.path`` — not the repo root. The real
# fighter scorer lives at the repo root (``harness/``, ``contracts.py``,
# ``games/``), so the repo root must be importable when the env is served.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hud import Environment

TeacherScorer = Callable[[dict[str, float]], float]

env = Environment("crucible-teacher")

# The Teacher parameter set the fighter task grades. Bounds are the safe ranges
# the boundary validator clamps to; ``difficulty`` is the dial the band-gated
# reward is actually sensitive to (see fighter_adapter._arena_from_spec).
FIGHTER_BOUNDS: dict[str, tuple[float, float]] = {
    "difficulty": (0.0, 1.0),
    "map_size": (3.0, 64.0),
}


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


# ---------------------------------------------------------------------------
# The REAL Crucible reward (gap proxy), wired into the HUD grader path.
# ---------------------------------------------------------------------------

# How many seeds the fighter win-rate is averaged over per arena when scoring
# inside the HUD task. Smaller than the harness default (25) so a HUD trace
# grades in well under a second on CPU — load-bearing without burning compute.
HUD_EVAL_SEEDS = 16


def params_to_curriculum(params: dict[str, float], *, curriculum_id: str = "hud-teacher"):
    """Map a validated Teacher parameter set onto a frozen ``CurriculumSpec``.

    The Teacher emits dials (``difficulty``, ``map_size``); the fighter adapter
    (``_arena_from_spec``) turns them into platform geometry + opponent strength.
    Imported lazily so the decorative ``from hud_teacher_env import ...`` path
    (and credential-free template smoke) never pulls the fighter engine.
    """
    from contracts import ArenaSpec, CurriculumSpec

    difficulty = float(params["difficulty"])
    map_size = int(round(float(params["map_size"])))
    return CurriculumSpec(
        game_id="fighter",
        curriculum_id=curriculum_id,
        arenas=[
            ArenaSpec(
                map_size=map_size,
                doors=0,
                keys=0,
                hazard_density=0.0,
                difficulty=difficulty,
            )
        ],
    )


def fighter_scorer(params: dict[str, float]) -> float:
    """The REAL fighter reward: ``harness.scoring.score_curriculum`` (gap proxy).

    This is the scorer the served HUD env uses by default, so the reward HUD
    records per trace is the genuine band-gated strong-minus-weak gap — NOT the
    placeholder triangle function. Runs on local CPU (free); HUD orchestrates
    the task and captures the result + trace.
    """
    from harness.fighter_adapter import FighterGameAdapter
    from harness.scoring import score_curriculum

    spec = params_to_curriculum(params)
    adapter = FighterGameAdapter(eval_seeds=HUD_EVAL_SEEDS)
    result = score_curriculum(spec, adapter)
    return float(result.reward)


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


# Default to the REAL fighter reward so a served env (LocalRuntime / hosted) is
# load-bearing out of the box: every HUD trace records the genuine gap proxy.
# Callers that need a different scorer (Modal PPO, a test fake) still override
# via ``configure_scorer``.
_configured_scorer: TeacherScorer = fighter_scorer


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

