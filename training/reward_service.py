"""Stage 3: credential-free reward-service core plus optional FastAPI app.

This is the seam used by a future Eval Protocol RemoteRolloutProcessor.  The
service accepts Teacher-generated parameters, scores them through an injected
function, and always returns the caller's rollout id with the reward.

The HTTP ``/init`` handler is the silent-corruption fix from the architecture
review.  With n concurrent rollouts (n=4 by default), every reward MUST be
tagged with its own ``rollout_id`` or rewards cross-tag silently — A's reward
filed under B's id, a bug that only surfaces at real batch size.  So ``/init``:

  (a) is ``async def`` — it never blocks the event loop on CPU-bound scoring;
  (b) reads ``rollout_id`` from request metadata and threads it into EVERY log
      line of a per-rollout background task, so concurrent rollouts never
      cross-tag;
  (c) returns 200 fast and runs the real scoring in the background via
      ``run_in_executor`` (the executor keeps CPU-bound work off the loop);
  (d) validates the incoming curriculum against the frozen ``CurriculumSpec``
      at parse time, rejecting a malformed curriculum with 422 before any
      background work is scheduled.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from pydantic import ValidationError

from contracts import CurriculumSpec, ScoringContext

logger = logging.getLogger("crucible.reward_service")


ScoreParams = Callable[[dict[str, float]], float]


@dataclass(frozen=True)
class RewardRequest:
    params: dict[str, float]
    rollout_id: Optional[str] = None


@dataclass(frozen=True)
class RewardResponse:
    reward: float
    rollout_id: Optional[str]
    status: str = "finished"


class RewardService:
    def __init__(self, scorer: ScoreParams):
        self._scorer = scorer

    def score(self, request: RewardRequest) -> RewardResponse:
        reward = min(1.0, max(0.0, float(self._scorer(request.params))))
        return RewardResponse(reward=reward, rollout_id=request.rollout_id)


def _extract_rollout_id(body: dict[str, Any]) -> Optional[str]:
    """Pull the rollout id from the top level or the metadata envelope.

    Eval Protocol threads it through ``metadata``; we accept either so the
    service works regardless of which envelope the caller uses.  The id is
    modeled by the frozen ``ScoringContext`` contract.
    """
    metadata = body.get("metadata") or {}
    context = ScoringContext(rollout_id=body.get("rollout_id") or metadata.get("rollout_id"))
    return context.rollout_id


def _parse_curriculum(body: dict[str, Any]) -> Optional[CurriculumSpec]:
    """Validate the incoming curriculum against the frozen contract, if present.

    Returns the validated spec, or ``None`` when the body carries no curriculum
    (a bare params request).  Raises ``ValidationError`` for a malformed one so
    the handler can reject it with 422 BEFORE scheduling any background work.
    """
    raw = body.get("curriculum")
    if raw is None:
        return None
    return CurriculumSpec.model_validate(raw)


def create_fastapi_app(service: RewardService):
    """Create the optional HTTP wrapper; import FastAPI only when installed."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - optional integration
        raise RuntimeError("install fastapi and uvicorn to run the reward service") from exc

    app = FastAPI(title="Crucible Reward Service")

    # Where finished background rewards land, keyed by rollout_id, so a poller /
    # the Eval Protocol processor can collect them after the fast 200.
    app.state.results = {}

    async def _score_in_background(request: RewardRequest) -> None:
        """Run CPU-bound scoring off the event loop, tagged with rollout_id.

        EVERY log line carries the rollout_id so concurrent n=4 rollouts can be
        disentangled in the logs — this is the cross-tag guard.
        """
        rid = request.rollout_id
        logger.info("rollout_id=%s scoring started", rid)
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(None, service.score, request)
        except Exception:  # pragma: no cover - defensive; log under the right id
            logger.exception("rollout_id=%s scoring failed", rid)
            app.state.results[rid] = RewardResponse(reward=0.0, rollout_id=rid, status="error")
            return
        logger.info("rollout_id=%s scoring finished reward=%.4f", rid, response.reward)
        app.state.results[rid] = response

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/init", status_code=202)
    async def init(body: dict):
        # (d) Validate the curriculum at parse time; reject malformed ones now.
        try:
            _parse_curriculum(body)
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"detail": exc.errors()})

        rollout_id = _extract_rollout_id(body)
        try:
            params = {k: float(v) for k, v in body.get("params", {}).items()}
        except (TypeError, ValueError) as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": f"params must be a scalar map: {exc}"},
            )

        request = RewardRequest(params=params, rollout_id=rollout_id)

        # (b)+(c) Hand scoring to a per-rollout background task and return fast.
        # asyncio.create_task fires-and-forgets; run_in_executor inside it keeps
        # the CPU-bound scorer off the event loop.
        asyncio.create_task(_score_in_background(request))
        logger.info("rollout_id=%s accepted (scoring in background)", rollout_id)
        return {"status": "accepted", "rollout_id": rollout_id}

    @app.get("/result/{rollout_id}")
    async def result(rollout_id: str):
        """Collect a finished reward by rollout_id (None until the task lands)."""
        response = app.state.results.get(rollout_id)
        if response is None:
            return {"status": "pending", "rollout_id": rollout_id}
        return asdict(response)

    return app
