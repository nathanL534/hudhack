"""training/rft_evaluator_tk.py — the eval-protocol evaluator for Target-Knockback RFT.

The Target-Knockback twin of ``training/rft_evaluator.py`` (the fighter evaluator). It
is a thin ``@evaluation_test`` that:

  1. Hands each rollout to the deployed TK Crucible bridge via ``RemoteRolloutProcessor``
     (the bridge calls the current Qwen checkpoint, gets a TK curriculum JSON, strict-
     validates it, and computes the nested-RL TK reward by fanning PPO seeds out on the
     ISOLATED ``crucible-player-tk`` Modal app — see training/modal_ep_bridge_tk.py).
  2. Reads the terminal ``hud_reward`` the bridge published into the rollout's tracing
     metadata and turns it into the EP ``EvaluateResult`` score (``apply_hud_reward``,
     the SAME EP->score adapter the fighter evaluator uses — game-agnostic).

So the TK RFT optimization signal IS the nested PPO held-out-transfer reward, end to end.

The TK bridge is ISOLATED from the fighter bridge: its URL comes from
``CRUCIBLE_TK_BRIDGE_URL`` (default is the TK bridge deployment), so launching a TK RFT
NEVER points at or disturbs the live fighter bridge.

NOTE: like the fighter evaluator, this module deliberately does NOT use
``from __future__ import annotations`` — eval-protocol's @evaluation_test validates the
``row``/return annotations by identity, which PEP-563 stringized annotations break.
"""

import os

from eval_protocol import RemoteRolloutProcessor, evaluation_test
from eval_protocol.models import EvaluationRow

from training.fireworks_rft_eval import apply_hud_reward
from games_registry import TARGET_KNOCKBACK

# The deployed TK Crucible bridge (nested-reward server). Overridable so a re-deploy or
# a LOCAL/ephemeral bridge (the dry round-trip uses uvicorn) can be targeted without
# editing this file. NEVER the fighter's CRUCIBLE_BRIDGE_URL.
BRIDGE_URL = os.getenv(
    TARGET_KNOCKBACK.bridge_url_env,
    "https://njlee007--crucible-ep-bridge-tk-web.modal.run",
)

# The Target-Knockback Teacher prompt (from the shared registry — single source of
# truth). ONE row: the RFT loop generates many rollouts per update from this single
# curriculum-design task (the game is Target Knockback ONLY).
TK_PROMPT = TARGET_KNOCKBACK.prompt


@evaluation_test(
    input_messages=[[[{"role": "user", "content": TK_PROMPT}]]],
    # Qwen3 reasons before answering: leave room for the <think> block plus the 7-key
    # TK curriculum object (the bridge strips <think> before parsing).
    completion_params=[{"temperature": 0.8, "max_tokens": 2048}],
    rollout_processor=RemoteRolloutProcessor(
        remote_base_url=BRIDGE_URL,
        model_base_url="https://tracing.fireworks.ai",
        poll_interval=2.0,
        # One rollout = one full nested TK reward (PPO Players, ~minutes). Give the
        # bridge a generous window before EP declares the rollout timed out.
        timeout_seconds=1500.0,
    ),
    max_concurrent_rollouts=2,
    max_concurrent_evaluations=2,
    passed_threshold=0.0,
)
def test_teacher_rft_tk(row: EvaluationRow) -> EvaluationRow:
    """Score one TK Teacher rollout = the TK bridge's published nested-RL reward."""
    return apply_hud_reward(row)
