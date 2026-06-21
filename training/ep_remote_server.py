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
    rollout_logger.info(
        "Teacher rollout completed",
        extra={"status": Status.rollout_finished(), "extras": extras},
    )


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

    reward, params = score_teacher_answer(answer, bounds=bounds, scorer=scorer)
    messages = _message_dicts(request) + [{"role": "assistant", "content": answer}]
    extras = {
        "messages": messages,
        "hud_reward": reward,
        "teacher_params": params,
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
        task = asyncio.create_task(_run(request))
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return {"status": "accepted", "rollout_id": request.metadata.rollout_id}

    return app

