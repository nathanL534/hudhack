"""Fireworks RFT evaluator for rewards produced by the HUD remote environment."""

from __future__ import annotations

from eval_protocol.models import EvaluateResult, EvaluationRow


def apply_hud_reward(row: EvaluationRow) -> EvaluationRow:
    """Copy the terminal HUD reward from remote-rollout metadata into EP."""
    extra = row.execution_metadata.extra or {}
    reward = extra.get("hud_reward")
    if reward is None:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            reason="Remote rollout did not publish hud_reward",
            is_score_valid=False,
        )
        return row

    score = min(1.0, max(0.0, float(reward)))
    row.evaluation_result = EvaluateResult(
        score=score,
        reason="Reward produced by the Crucible HUD environment",
        is_score_valid=True,
    )
    return row


def remote_processor(remote_base_url: str):
    """Build the processor used by the Fireworks RFT evaluator."""
    from eval_protocol import RemoteRolloutProcessor

    return RemoteRolloutProcessor(
        remote_base_url=remote_base_url,
        model_base_url="https://tracing.fireworks.ai",
        poll_interval=1.0,
        timeout_seconds=900.0,
    )

