"""Stage 3: credential-free reward-service core plus optional FastAPI app.

This is the seam used by a future Eval Protocol RemoteRolloutProcessor.  The
service accepts Teacher-generated parameters, scores them through an injected
function, and always returns the caller's rollout id with the reward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional


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


def create_fastapi_app(service: RewardService):
    """Create the optional HTTP wrapper; import FastAPI only when installed."""
    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover - optional integration
        raise RuntimeError("install fastapi and uvicorn to run the reward service") from exc

    app = FastAPI(title="Crucible Reward Service")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/init")
    def init(body: dict):
        request = RewardRequest(
            params={k: float(v) for k, v in body.get("params", {}).items()},
            rollout_id=body.get("rollout_id")
            or body.get("metadata", {}).get("rollout_id"),
        )
        return asdict(service.score(request))

    return app

