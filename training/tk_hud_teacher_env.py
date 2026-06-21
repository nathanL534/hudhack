"""training/tk_hud_teacher_env.py — HUD environment for Target Knockback (game #2).

This is the Target-Knockback twin of ``training/hud_teacher_env.py``. It serves a
SEPARATE HUD env (``crucible-teacher-tk``) so TK can be graded on HUD WITHOUT
touching the fighter's ``crucible-teacher`` env. It reuses the SHARED grading core
(``hud_teacher_env.score_teacher_answer`` / ``parse_teacher_params`` / ``clamp_reward``)
— that core is already game-agnostic (it takes injected ``bounds`` + ``scorer``), so
TK plugs in by SUPPLYING its own bounds and scorer, exactly the "extend, don't fork"
seam the games registry encodes.

Load-bearing wiring: the default served scorer is :func:`tk_gap_scorer`, which runs
the REAL Crucible TK reward (``harness.scoring.score_curriculum`` over
``TargetKnockbackGameAdapter``) on local CPU. So the reward HUD records per trace is
the genuine band-gated strong-minus-weak gap on the actual TK sim — not a placeholder.
When ``CRUCIBLE_HUD_SCORER=nested`` the served grader instead reads the nested-RL PPO
held-out-transfer reward via the TK sidecar (``training.tk_dry_loop_sidecar``), so the
integrated dry run's HUD-recorded reward IS the nested TK improvement.

This module is re-imported in a child process by ``hud.eval.LocalRuntime``; like the
fighter env, it puts the repo root on ``sys.path`` so ``harness`` / ``contracts`` /
``games`` resolve when served.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hud import Environment

# Reuse the SHARED grading core + the TeacherScorer type from the fighter env. These
# are game-agnostic (bounds + scorer are injected), so TK does not duplicate them.
from training.hud_teacher_env import (
    TeacherScorer,
    clamp_reward,
    parse_teacher_params,
    score_teacher_answer,
)

# The TK param schema, sourced from the SHARED registry (single source of truth). The
# boundary validator clamps every emitted key into its range.
from games_registry import TARGET_KNOCKBACK

env = Environment("crucible-teacher-tk")

# TK Teacher schema: difficulty + fighter physics knobs + the two TK zone dials.
TK_BOUNDS: dict[str, tuple[float, float]] = dict(TARGET_KNOCKBACK.bounds)

# Geometry knobs threaded VERBATIM into the worker payload (everything except the
# opponent-strength ``difficulty`` dial).
TK_GEOMETRY_KEYS: tuple[str, ...] = TARGET_KNOCKBACK.geometry_keys

# How many seeds the TK win-rate is averaged over per arena inside the HUD task.
# Small enough to grade in well under a second on CPU; load-bearing without burning
# compute (mirrors the fighter env's HUD_EVAL_SEEDS).
HUD_EVAL_SEEDS = 16


def params_to_tk_curriculum(params: dict[str, float], *, curriculum_id: str = "hud-tk"):
    """Map a validated TK Teacher param set onto a frozen ``CurriculumSpec``.

    The frozen ``ArenaSpec`` only carries ``map_size`` + ``difficulty``, so this
    derives ``map_size`` from ``platform_width`` (the inverse of the TK adapter's
    ``width = 6.0 + 0.5*map_size`` mapping) and keeps ``difficulty`` verbatim. The
    full TK geometry (physics + zone dials) is threaded separately into the Modal
    worker payload (see ``tk_geometry_override``), so nothing is lost: the spec is
    the contract object, the override carries the verbatim geometry. ``map_size`` is
    still accepted directly for a legacy two-dial path.

    Imported lazily so a credential-free template smoke never pulls the game engine.
    """
    from contracts import ArenaSpec, CurriculumSpec

    difficulty = float(params["difficulty"])
    if "map_size" in params:
        map_size = int(round(float(params["map_size"])))
    elif "platform_width" in params:
        map_size = int(round((float(params["platform_width"]) - 6.0) / 0.5))
    else:
        map_size = 10
    map_size = max(3, min(64, map_size))
    return CurriculumSpec(
        game_id="target_knockback",
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


def tk_geometry_override(params: dict[str, float]) -> dict[str, float]:
    """Extract the verbatim TK geometry knobs (physics + zone) from a param set."""
    return {k: float(params[k]) for k in TK_GEOMETRY_KEYS if k in params}


def tk_gap_scorer(params: dict[str, float]) -> float:
    """The REAL TK reward: ``harness.scoring.score_curriculum`` (gap proxy).

    The served HUD env uses this by default, so the reward HUD records per trace is
    the genuine band-gated strong-minus-weak gap computed on the actual TK sim via
    ``TargetKnockbackGameAdapter``. Runs on local CPU (free).
    """
    from harness.scoring import score_curriculum
    from harness.target_knockback_adapter import TargetKnockbackGameAdapter

    spec = params_to_tk_curriculum(params)
    adapter = TargetKnockbackGameAdapter(eval_seeds=HUD_EVAL_SEEDS)
    result = score_curriculum(spec, adapter)
    return float(result.reward)


@env.template(id="design_tk_curriculum")
async def design_tk_curriculum(
    game_description: str,
    parameter_bounds: dict[str, list[float]],
):
    """Generate one safe TK parameter set whose measured learning value is high."""
    prompt = {
        "role": "curriculum_designer",
        "game": game_description,
        "parameter_bounds": parameter_bounds,
        "instruction": "Return only one JSON object with every parameter key.",
    }
    answer = yield json.dumps(prompt, sort_keys=True)

    bounds = {key: (float(value[0]), float(value[1])) for key, value in parameter_bounds.items()}
    scorer = _configured_scorer
    reward, _ = score_teacher_answer(answer or "{}", bounds=bounds, scorer=scorer)
    yield reward


def _default_served_scorer() -> TeacherScorer:
    """Pick the scorer a freshly-served TK env uses.

    ``hud.eval.LocalRuntime`` re-imports THIS module in a child subprocess, so a
    parent ``configure_scorer`` never reaches the served grader. ``CRUCIBLE_HUD_SCORER``
    is the cross-process switch the child reads at import time:

      * ``nested`` -> the REAL nested-RL TK PPO held-out-transfer reward, served via
        the TK dry-loop sidecar. This makes the TK dry run's HUD-recorded reward BE
        the nested TK improvement while keeping HUD load-bearing.
      * unset / anything else -> the gap-proxy ``tk_gap_scorer`` (default), so the
        TK HUD smoke + tests run on local CPU with no Modal credits.
    """
    if os.getenv("CRUCIBLE_HUD_SCORER", "").strip().lower() == "nested":
        from training.tk_dry_loop_sidecar import served_tk_nested_scorer

        return served_tk_nested_scorer
    return tk_gap_scorer


_configured_scorer: TeacherScorer = _default_served_scorer()


def configure_scorer(scorer: TeacherScorer) -> None:
    """Configure the scorer used when this TK HUD environment is served directly."""
    global _configured_scorer
    _configured_scorer = scorer


async def run_template_smoke(
    answer: str,
    *,
    bounds: dict[str, tuple[float, float]] | None = None,
    scorer: TeacherScorer | None = None,
) -> float:
    """Drive HUD's prompt -> answer -> reward generator without credentials."""
    bounds = bounds or TK_BOUNDS
    scorer = scorer or tk_gap_scorer
    parameter_bounds = {key: [low, high] for key, (low, high) in bounds.items()}
    configure_scorer(scorer)
    generator = design_tk_curriculum.func("target_knockback", parameter_bounds)
    await anext(generator)
    return float(await generator.asend(answer))
