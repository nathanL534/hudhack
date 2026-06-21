"""Eval Protocol remote rollout server for Fireworks Teacher RFT.

Fireworks supplies the current checkpoint through ``model_base_url``.  This
server calls it, sends the generated JSON through the HUD grader, and publishes
the resulting reward to Fireworks tracing under the exact rollout id.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from eval_protocol import (
    FireworksTracingHttpHandler,
    InitRequest,
    RolloutIdFilter,
    Status,
)
from openai import AsyncOpenAI

from training.hud_teacher_env import TeacherScorer, score_teacher_answer

logger = logging.getLogger("crucible.ep_remote")


class CompletionClient(Protocol):
    class Chat(Protocol):
        class Completions(Protocol):
            async def create(self, **kwargs: Any) -> Any: ...

        completions: Completions

    chat: Chat


ClientFactory = Callable[[str, str], CompletionClient]
CompletionReporter = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class RolloutResult:
    rollout_id: str
    reward: float
    params: dict[str, float]
    answer: str


def _default_client_factory(base_url: str, api_key: str) -> CompletionClient:
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def _message_dicts(request: InitRequest) -> list[dict[str, Any]]:
    if not request.messages:
        raise ValueError("messages is required")
    return [message.model_dump(exclude_none=True) for message in request.messages]


def _completion_kwargs(request: InitRequest) -> dict[str, Any]:
    params = dict(request.completion_params)
    params["messages"] = _message_dicts(request)
    if request.tools:
        params["tools"] = request.tools
    return params


def configure_fireworks_tracing() -> None:
    """Attach the official tracing sink once in live mode."""
    root = logging.getLogger()
    if not any(isinstance(handler, FireworksTracingHttpHandler) for handler in root.handlers):
        root.addHandler(FireworksTracingHttpHandler())
    root.setLevel(logging.INFO)


def tracing_reporter(rollout_id: str, extras: dict[str, Any]) -> None:
    rollout_logger = logging.getLogger(f"crucible.ep_remote.{rollout_id}")
    rollout_logger.addFilter(RolloutIdFilter(rollout_id))
    # Promote the EP run/experiment identifiers (already inside ``extras``) onto the
    # log RECORD so FireworksTracingHttpHandler emits ``run_id:<id>`` /
    # ``experiment_id:<id>`` TAGS, not just rollout_id. Fireworks never tells the
    # bridge which training UPDATE a rollout belongs to, so there is no real
    # per-epoch tag to emit — but run_id is stable for the whole RFT job, which is
    # exactly the group key the launcher's monitor needs to find + aggregate every
    # rollout of this job (it cannot enumerate rollout_ids ahead of time).
    record_extra: dict[str, Any] = {"status": Status.rollout_finished(), "extras": extras}
    for key in ("run_id", "experiment_id"):
        val = extras.get(key)
        if val:
            record_extra[key] = val
    rollout_logger.info("Teacher rollout completed", extra=record_extra)


async def execute_rollout(
    request: InitRequest,
    *,
    bounds: dict[str, tuple[float, float]],
    scorer: TeacherScorer,
    client_factory: ClientFactory = _default_client_factory,
    reporter: CompletionReporter = tracing_reporter,
) -> RolloutResult:
    """Call the current Teacher checkpoint, grade with HUD, and report reward."""
    if not request.model_base_url:
        raise ValueError("model_base_url is required")
    api_key = request.api_key or os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("Fireworks API key is required")

    client = client_factory(request.model_base_url, api_key)
    completion = await client.chat.completions.create(**_completion_kwargs(request))
    answer = completion.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Teacher returned an empty response")

    # The scorer fans out the PPO seeds with Modal's SYNCHRONOUS ``fn.map``, which
    # cannot be iterated from inside this async function ("You can't iter(
    # Function.map()) from an async function"). Run the sync scorer off the event
    # loop via run_in_executor — the intended design: the scorer stays synchronous,
    # the async boundary calls it in a thread.
    loop = asyncio.get_running_loop()
    reward, params = await loop.run_in_executor(
        None, lambda: score_teacher_answer(answer, bounds=bounds, scorer=scorer)
    )
    messages = _message_dicts(request) + [{"role": "assistant", "content": answer}]
    extras = {
        "messages": messages,
        "hud_reward": reward,
        "teacher_params": params,
        # Stamp the EP run/experiment ids so the launcher's monitor can (a) discover
        # this job's run_id from any one finished rollout and (b) group every rollout
        # of the job together. These come straight off the InitRequest metadata.
        "run_id": request.metadata.run_id,
        "experiment_id": request.metadata.experiment_id,
    }
    reporter(request.metadata.rollout_id, extras)
    return RolloutResult(request.metadata.rollout_id, reward, params, answer)


def create_ep_app(
    *,
    bounds: dict[str, tuple[float, float]],
    scorer: TeacherScorer,
    client_factory: ClientFactory = _default_client_factory,
    reporter: CompletionReporter = tracing_reporter,
    live_tracing: bool = True,
):
    """Create the real ``POST /init`` endpoint expected by Eval Protocol."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    if live_tracing:
        configure_fireworks_tracing()

    app = FastAPI(title="Crucible Fireworks-HUD Rollout Bridge")
    app.state.tasks = set()
    app.state.results = {}
    # run_id -> list of this run's rollout_ids (newest last). The launcher's monitor
    # cannot enumerate rollout ids ahead of time and the RFT job object does not
    # expose the EP run_id, so the bridge surfaces the run_ids it has actually seen
    # (recorded at /init time, before the rollout even finishes) via GET /runs. That
    # is the monitor's honest discovery handle for the run_id:<id> tracing tag.
    app.state.run_index: dict[str, list[str]] = {}
    app.state.run_order: list[str] = []

    async def _run(request: InitRequest) -> None:
        rollout_id = request.metadata.rollout_id
        try:
            result = await execute_rollout(
                request,
                bounds=bounds,
                scorer=scorer,
                client_factory=client_factory,
                reporter=reporter,
            )
            app.state.results[rollout_id] = result
        except Exception as exc:
            app.state.results[rollout_id] = exc
            rollout_logger = logging.getLogger(f"crucible.ep_remote.{rollout_id}")
            rollout_logger.addFilter(RolloutIdFilter(rollout_id))
            rollout_logger.exception(
                "Teacher rollout failed",
                extra={"status": Status.rollout_internal_error(str(exc))},
            )

    @app.get("/health")
    async def health():
        return {"status": "ok", "protocol": "eval-protocol"}

    @app.post("/init", status_code=202)
    async def init(request: InitRequest):
        if not request.messages:
            return JSONResponse(status_code=422, content={"detail": "messages is required"})
        # Record the run_id -> rollout_id mapping at accept time so the monitor can
        # discover the run before any rollout finishes.
        run_id = request.metadata.run_id
        rollout_id = request.metadata.rollout_id
        if run_id:
            ids = app.state.run_index.setdefault(run_id, [])
            if rollout_id not in ids:
                ids.append(rollout_id)
            if run_id not in app.state.run_order:
                app.state.run_order.append(run_id)
        task = asyncio.create_task(_run(request))
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return {"status": "accepted", "rollout_id": rollout_id}

    @app.get("/runs")
    async def runs(limit: int = 20):
        """Recently-seen EP run_ids (newest last) + their rollout ids.

        The launcher's monitor reads this to discover the RFT job's run_id, which it
        then uses as the ``run_id:<id>`` tracing tag to fetch + aggregate rewards.
        Returns only ids — never credentials, prompts, or rewards.
        """
        order = app.state.run_order[-limit:]
        return {"runs": [{"run_id": r, "rollout_ids": app.state.run_index.get(r, [])} for r in order]}

    @app.get("/debug/result/{rollout_id}")
    async def debug_result(rollout_id: str):
        """Small smoke-test endpoint; Eval Protocol itself polls tracing.

        This intentionally returns only status, reward, and validated parameters.
        It never returns credentials or the full request payload.
        """
        result = app.state.results.get(rollout_id)
        if result is None:
            return {"status": "pending", "rollout_id": rollout_id}
        if isinstance(result, Exception):
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "rollout_id": rollout_id,
                    "detail": str(result),
                },
            )
        return {
            "status": "finished",
            "rollout_id": rollout_id,
            "reward": result.reward,
            "params": result.params,
        }

    return app
