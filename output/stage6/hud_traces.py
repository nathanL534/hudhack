"""output/stage6/hud_traces.py — OPTIONAL HUD trace capture for the decider.

Stage 6's structured output records HUD trace IDs/URLs "when available". HUD traces
are a side effect of running a generated curriculum through the served
``crucible-teacher`` env (see ``training/dry_loop._run_hud``); the trace URL is then
assembled as ``f"{settings.hud_web_url}/jobs/{job.id}"``. There is no URL-returning
helper, so we reproduce that minimal pattern here.

This is STRICTLY optional and best-effort: if HUD / its deps are not importable, or
no ``HUD_API_KEY`` is set, capture returns an empty list and the decider is
unaffected (it never blocks the verdict on tracing). Enable via the CLI
``--hud-traces`` flag; off by default so the core run stays fast and dependency-light.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


async def _trace_one(clamped: dict, bounds: dict) -> dict:
    """Run ONE generated arena through the served crucible-teacher env; read the trace."""
    import os

    import mcp.types as mcp_types
    from hud.agents.base import Agent
    from hud.eval import LocalRuntime
    from hud.settings import settings
    from hud.types import Step

    from training.hud_teacher_env import design_curriculum

    answer = json.dumps(clamped, sort_keys=True)

    class FixedCurriculumAgent(Agent):
        async def __call__(self, run) -> None:  # noqa: ANN001
            _ = run.prompt_text
            run.record(Step(
                source="agent",
                messages=[mcp_types.PromptMessage(
                    role="assistant",
                    content=mcp_types.TextContent(type="text", text=answer),
                )],
            ))
            run.trace.content = answer

    env_source = os.path.join(str(_REPO_ROOT), "training", "hud_teacher_env.py")
    runtime = LocalRuntime(env_source, env="crucible-teacher")
    bounds_list = {k: [lo, hi] for k, (lo, hi) in bounds.items()}
    task = design_curriculum("Stage-1 2D fighter; pick a learnable FIGHTER curriculum",
                             bounds_list)
    job = await task.run(FixedCurriculumAgent(), runtime=runtime)
    run = job.runs[0]
    url = None
    if settings.api_key and run.trace_id:
        url = f"{settings.hud_web_url}/jobs/{job.id}"
    return {
        "trace_id": str(run.trace_id) if run.trace_id else "",
        "url": url,
        "status": str(run.trace.status),
        "reward": float(run.reward),
    }


def capture_traces(arenas_by_model: dict[str, list[dict]], bounds: dict,
                   *, enabled: bool = False, limit_per_model: int = 1) -> list[dict]:
    """Capture up to ``limit_per_model`` HUD traces per model (best-effort).

    Returns ``[]`` (and never raises) when disabled or when HUD is unavailable, so
    the decider degrades gracefully on machines without a HUD key.
    """
    if not enabled:
        return []
    try:
        traces: list[dict] = []
        for model, arenas in arenas_by_model.items():
            for arena in arenas[:limit_per_model]:
                t = asyncio.run(_trace_one(arena, bounds))
                t["model"] = model
                traces.append(t)
        return traces
    except Exception as exc:  # pragma: no cover - HUD-env dependent
        return [{"trace_id": "", "url": None, "status": f"unavailable: {exc}",
                 "model": "all"}]
