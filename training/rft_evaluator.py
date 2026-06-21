"""training/rft_evaluator.py — the eval-protocol evaluator for Teacher RFT.

This is the evaluator Fireworks RFT runs on every Teacher rollout. It is a thin
``@evaluation_test`` that:

  1. Hands each rollout to the deployed Crucible bridge via ``RemoteRolloutProcessor``
     (the bridge calls the current Qwen checkpoint, gets a FIGHTER curriculum JSON,
     strict-validates it, and computes the VALIDATED nested-RL reward by fanning 3
     PPO seeds out on the ``crucible-player`` Modal app — see training/modal_ep_bridge.py).
  2. Reads the terminal ``hud_reward`` the bridge published into the rollout's tracing
     metadata and turns it into the EP ``EvaluateResult`` score (``apply_hud_reward``).

So the RFT optimization signal IS the nested PPO held-out-transfer reward, end to end.

The bridge URL is read from ``CRUCIBLE_BRIDGE_URL`` (the launcher sets it; default is
the deployed bridge). Point it at a different deployment by exporting that env var.

Used by the launcher (``training/run_tiny_rft.py``) and by the eval-protocol CLI::

    CRUCIBLE_BRIDGE_URL=https://njlee007--crucible-ep-bridge-web.modal.run \
        eval-protocol create rft --evaluator rft_evaluator-test_teacher_rft

NOTE: this module deliberately does NOT use ``from __future__ import annotations``
— eval-protocol's @evaluation_test validates the ``row``/return annotations by
identity (``is EvaluationRow``), which PEP-563 stringized annotations would break.
"""

import os

from eval_protocol import RemoteRolloutProcessor, evaluation_test
from eval_protocol.models import EvaluationRow

from training.fireworks_rft_eval import apply_hud_reward

# The deployed Crucible Fireworks<->HUD bridge (nested-reward server). Overridable
# so a re-deploy or a staging bridge can be pointed at without editing this file.
BRIDGE_URL = os.getenv(
    "CRUCIBLE_BRIDGE_URL",
    "https://njlee007--crucible-ep-bridge-web.modal.run",
)

# The Ring-Out fighter Teacher prompt + the FIGHTER schema. ONE row: the RFT loop
# generates many rollouts per update from this single curriculum-design task (the
# game is Ring-Out ONLY — no other game in the dataset).
RING_OUT_PROMPT = (
    "You design RL training curricula for a Stage-1 2D platform fighter "
    "(the Ring-Out duel: two fighters, knock the opponent off the platform). "
    "Return ONE JSON object with EXACTLY these keys, each a number in range: "
    "difficulty [0.0,1.0], platform_width [8.0,30.0], gravity [0.2,1.2], "
    "knockback [0.5,6.0], spawn_gap [1.0,12.0]. Choose a LEARNABLE arena whose "
    "trained Player transfers broadly. Output strict JSON only; no prose, no code."
)


@evaluation_test(
    input_messages=[[[{"role": "user", "content": RING_OUT_PROMPT}]]],
    # Qwen3-4B reasons before answering: leave room for the <think> block plus the
    # 5-key curriculum object (the bridge strips <think> before parsing). 256 would
    # truncate mid-reasoning and yield no JSON.
    completion_params=[{"temperature": 0.8, "max_tokens": 2048}],
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url=BRIDGE_URL,
        model_base_url="https://tracing.fireworks.ai",
        poll_interval=2.0,
        # One rollout = one full nested reward (3 PPO Players, ~minutes). Give the
        # bridge a generous window before EP declares the rollout timed out.
        timeout_seconds=1500.0,
    ),
    # Keep EP-side concurrency bounded so it never outpaces the RFT job's own
    # max_concurrent_rollouts (the real Modal-job dial is set in the launcher).
    max_concurrent_rollouts=2,
    max_concurrent_evaluations=2,
    passed_threshold=0.0,
)
def test_teacher_rft(row: EvaluationRow) -> EvaluationRow:
    """Score one Teacher rollout = the bridge's published nested-RL reward."""
    return apply_hud_reward(row)
