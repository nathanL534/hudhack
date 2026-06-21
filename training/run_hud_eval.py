"""Minimal HUD eval that makes Crucible load-bearing on HUD.

What this proves
----------------
HUD is not a decorative import here: this script runs the ``crucible-teacher``
HUD environment through HUD's *real* wire protocol and records the genuine
Crucible reward on a HUD trace.

The flow per task:

  1. ``LocalRuntime`` serves ``hud_teacher_env.py`` in a child process (the same
     ``python -m hud.environment.server`` entry a hosted/container env runs).
  2. ``Task.run(agent, runtime=...)`` opens a HUD ``Job`` (``POST /trace/job/.../enter``),
     drives the agent over the channel, and grades on exit.
  3. The env's ``design_curriculum`` grader calls ``fighter_scorer`` →
     ``harness.scoring.score_curriculum`` over ``FighterGameAdapter`` — the REAL
     band-gated gap proxy (strong scripted minus weak random), on local CPU.
  4. HUD reports the reward + trace to the platform (``POST /trace/.../exit``)
     under the job id.

So the reward HUD records IS the real fighter reward — we cross-check each one
against a direct local ``score_curriculum`` call and assert they match.

Credit usage: the agent is deterministic (it submits a fixed JSON curriculum,
no LLM inference), and we run only a few tasks. No model tokens are sampled
through HUD's gateway; the only platform usage is the job + trace reporting
calls, which are free metadata writes.

Run (from the repo root so ``.env`` with HUD_API_KEY is picked up)::

    .venv/bin/python -m training.run_hud_eval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# Repo root on path so ``harness`` / ``contracts`` resolve when run as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hud.agents.base import Agent
from hud.eval import LocalRuntime
from hud.settings import settings

from harness.fighter_adapter import FighterGameAdapter
from harness.scoring import score_curriculum
from training.hud_teacher_env import (
    FIGHTER_BOUNDS,
    HUD_EVAL_SEEDS,
    design_curriculum,
    params_to_curriculum,
)

# The Teacher curricula this smoke submits. Each is a full FIGHTER param set (the
# five knobs in FIGHTER_BOUNDS) that lands in the learnable band, so the recorded
# reward is a real, non-trivial gap proxy (not 0.0 from the band gate). difficulty
# is the dial the gap-proxy reward responds to; the geometry knobs are the arena
# physics the Player trains under.
SMOKE_CURRICULA: list[dict[str, float]] = [
    {"difficulty": 0.22, "platform_width": 12.0, "gravity": 0.6, "knockback": 2.5, "spawn_gap": 4.0},
    {"difficulty": 0.25, "platform_width": 12.0, "gravity": 0.6, "knockback": 2.5, "spawn_gap": 4.0},
    {"difficulty": 0.30, "platform_width": 12.0, "gravity": 0.6, "knockback": 2.5, "spawn_gap": 4.0},
]

ENV_SOURCE = os.path.join(_REPO_ROOT, "training", "hud_teacher_env.py")


class FixedCurriculumAgent(Agent):
    """A deterministic agent: submits one preset Teacher curriculum as its answer.

    No LLM, no token sampling — the answer is the JSON ``curriculum`` it was
    constructed with. It fills ``run.trace.content`` (the graded answer) and
    records an assistant step so the HUD trace shows a real trajectory.
    """

    def __init__(self, curriculum: dict[str, float]) -> None:
        self._answer = json.dumps(curriculum, sort_keys=True)

    async def __call__(self, run) -> None:  # noqa: ANN001 - hud.eval.Run
        # mcp.types + hud.types are how the SDK shapes a trajectory step.
        import mcp.types as mcp_types

        from hud.types import Step

        # Reading the prompt is part of a faithful trajectory even though the
        # answer is fixed; it lands the curriculum-design prompt on the trace.
        _ = run.prompt_text
        run.record(
            Step(
                source="agent",
                messages=[
                    mcp_types.PromptMessage(
                        role="assistant",
                        content=mcp_types.TextContent(type="text", text=self._answer),
                    )
                ],
            )
        )
        # The graded answer the env's grader receives as the second ``yield``.
        run.trace.content = self._answer


def _local_reward(curriculum: dict[str, float]) -> float:
    """Direct local fighter reward — the ground truth HUD's record must match."""
    spec = params_to_curriculum(curriculum)
    return float(score_curriculum(spec, FighterGameAdapter(eval_seeds=HUD_EVAL_SEEDS)).reward)


async def run_eval(curricula: list[dict[str, float]]) -> int:
    if not settings.api_key:
        print("HUD_API_KEY not found in settings — traces will NOT report to the platform.")
        print("Run from the repo root so .env is picked up, or `hud set HUD_API_KEY=...`.")
    print(f"telemetry_enabled={settings.telemetry_enabled}  api_key_present={bool(settings.api_key)}")
    print(f"platform: {settings.hud_api_url}  web: {settings.hud_web_url}")
    print(f"serving env source: {ENV_SOURCE}")
    print(f"fighter eval seeds per trace: {HUD_EVAL_SEEDS}")
    print("-" * 72)

    bounds = {k: [lo, hi] for k, (lo, hi) in FIGHTER_BOUNDS.items()}
    runtime = LocalRuntime(ENV_SOURCE, env="crucible-teacher")

    all_match = True
    for i, curriculum in enumerate(curricula, start=1):
        # The concrete HUD Task row minted from the template (args = task inputs).
        task = design_curriculum("Stage-1 2D fighter; pick a learnable curriculum", bounds)
        agent = FixedCurriculumAgent(curriculum)

        expected = _local_reward(curriculum)
        job = await task.run(agent, runtime=runtime)

        run = job.runs[0]
        recorded = run.reward
        match = abs(recorded - expected) < 1e-9
        all_match = all_match and match

        print(f"[task {i}] curriculum={curriculum}")
        print(f"         HUD job id : {job.id}")
        print(f"         trace id   : {run.trace_id}")
        print(f"         status     : {run.trace.status}")
        print(f"         HUD reward : {recorded:.4f}")
        print(f"         local score_curriculum : {expected:.4f}")
        print(f"         REAL-REWARD MATCH      : {match}")
        if settings.api_key and run.trace_id:
            print(f"         job url    : {settings.hud_web_url}/jobs/{job.id}")
        print("-" * 72)

    print(f"all rewards matched local score_curriculum: {all_match}")
    return 0 if all_match else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal load-bearing HUD eval for Crucible.")
    parser.add_argument(
        "-n",
        "--num-tasks",
        type=int,
        default=len(SMOKE_CURRICULA),
        help="How many curricula to run (1-3). Keep it minimal to conserve HUD usage.",
    )
    args = parser.parse_args()
    n = max(1, min(args.num_tasks, len(SMOKE_CURRICULA)))
    rc = asyncio.run(run_eval(SMOKE_CURRICULA[:n]))
    sys.exit(rc)


if __name__ == "__main__":
    main()
