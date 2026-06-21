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

# The FIGHTER Teacher parameter set the task grades. These are the FIVE fighter
# knobs the Modal PPO worker (``modal_player._arena_from_payload``) reads verbatim
# to build the training arena: opponent strength (``difficulty``) plus the four
# physics/geometry dials (``platform_width`` / ``gravity`` / ``knockback`` /
# ``spawn_gap``). Bounds are the safe ranges the boundary validator clamps to;
# they match the schema the deploy/test path (scripts/deploy_qwen_test.py) and the
# Modal EP bridge advertise, so the Teacher emits ONE coherent schema everywhere.
FIGHTER_BOUNDS: dict[str, tuple[float, float]] = {
    "difficulty": (0.0, 1.0),
    "platform_width": (8.0, 30.0),
    "gravity": (0.2, 1.2),
    "knockback": (0.5, 6.0),
    "spawn_gap": (1.0, 12.0),
}

# Fighter geometry knobs (everything except the opponent-strength ``difficulty``
# dial). Used to split a validated FIGHTER param set into the difficulty dial vs
# the physics override the worker payload carries verbatim.
FIGHTER_GEOMETRY_KEYS: tuple[str, ...] = ("platform_width", "gravity", "knockback", "spawn_gap")


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

    The FIGHTER Teacher emits ``difficulty`` plus the four physics knobs
    (``platform_width`` / ``gravity`` / ``knockback`` / ``spawn_gap``). The frozen
    ``ArenaSpec`` only carries ``map_size`` + ``difficulty``, so this derives a
    ``map_size`` from ``platform_width`` (the inverse of the adapter's
    ``width = 6.0 + 0.5*map_size`` mapping) and keeps ``difficulty`` verbatim. The
    full geometry is threaded separately into the Modal worker payload (see
    ``geometry_override`` in the reward path), so nothing is lost — the
    ``CurriculumSpec`` is the contract object, the geometry override is the verbatim
    physics. ``map_size`` is still accepted directly for the legacy two-dial path.

    Imported lazily so the decorative ``from hud_teacher_env import ...`` path (and
    credential-free template smoke) never pulls the fighter engine.
    """
    from contracts import ArenaSpec, CurriculumSpec

    difficulty = float(params["difficulty"])
    if "map_size" in params:
        map_size = int(round(float(params["map_size"])))
    elif "platform_width" in params:
        # Invert the adapter geometry map (width = 6.0 + 0.5*map_size).
        map_size = int(round((float(params["platform_width"]) - 6.0) / 0.5))
    else:
        map_size = 10
    map_size = max(3, min(64, map_size))
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


def fighter_geometry_override(params: dict[str, float]) -> dict[str, float]:
    """Extract the verbatim FIGHTER physics knobs from a validated param set.

    Returns only the geometry keys present in ``params`` (``platform_width`` /
    ``gravity`` / ``knockback`` / ``spawn_gap``). The Modal worker reads these
    directly so the Teacher's emitted physics trains the Player exactly as emitted.
    """
    return {k: float(params[k]) for k in FIGHTER_GEOMETRY_KEYS if k in params}


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


def _default_served_scorer() -> TeacherScorer:
    """Pick the scorer a freshly-served env uses.

    ``hud.eval.LocalRuntime`` re-imports THIS module in a child subprocess, so a
    ``configure_scorer`` call in the parent never reaches the served grader. The
    env var ``CRUCIBLE_HUD_SCORER`` is the cross-process switch the child reads at
    import time:

      * ``nested`` -> the REAL nested-RL PPO held-out-transfer reward, served via
        the dry-loop sidecar (``training.dry_loop_sidecar.served_nested_scorer``).
        This is what makes the FIGHTER dry run's HUD-recorded reward BE the nested
        PPO improvement (not the gap proxy) while keeping HUD load-bearing.
      * unset / anything else -> the gap-proxy ``fighter_scorer`` (default), so the
        existing ``run_hud_eval`` smoke and tests are untouched.
    """
    if os.getenv("CRUCIBLE_HUD_SCORER", "").strip().lower() == "nested":
        from training.dry_loop_sidecar import served_nested_scorer

        return served_nested_scorer
    return fighter_scorer


# Default to the REAL fighter reward so a served env (LocalRuntime / hosted) is
# load-bearing out of the box: every HUD trace records a genuine fighter reward.
# When CRUCIBLE_HUD_SCORER=nested, the served grader uses the nested-RL PPO reward
# instead. Callers that need a different scorer (a test fake) still override via
# ``configure_scorer``.
_configured_scorer: TeacherScorer = _default_served_scorer()


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

