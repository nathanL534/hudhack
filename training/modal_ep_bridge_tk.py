"""Modal deployment entrypoint for the Target-Knockback Fireworks <-> HUD bridge.

The Target-Knockback twin of ``training/modal_ep_bridge.py``. This is a BRAND-NEW,
SEPARATE Modal app (``crucible-ep-bridge-tk``) — ISOLATED from the fighter bridge
(``crucible-ep-bridge``) so a live fighter RFT against the deployed fighter bridge is
NEVER disturbed by a TK deploy or a TK RFT.

The served reward IS the nested-RL TK Teacher reward (held-out PPO transfer), the same
number the TK scale validation measured (d=0.55 clean band). It is NOT a placeholder.

How the reward is served (one /init round-trip):

    Fireworks RFT POSTs /init with the current Qwen checkpoint + TK Teacher prompt
        -> create_ep_app calls the checkpoint, gets the TK curriculum JSON
        -> strict-validates + clamps to TK_BOUNDS
        -> ``tk_nested_reward_scorer`` runs ``tk_teacher_reward(...)`` which fans the
           PPO seeds out IN PARALLEL on the ISOLATED ``crucible-player-tk`` Modal app
        -> reward = mean held-out improvement across seeds, clamped to [0, 1].

This bridge container only DISPATCHES the PPO seeds (the heavy SB3/torch training runs
inside ``crucible-player-tk``), so its image needs the local source packages + ``modal``
to call ``crucible-player-tk`` — not torch.

Deploy (the SAME secret the fighter bridge uses; this only READS the key)::

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET   # use the ~/.modal.toml njlee007 profile
    modal deploy training/modal_ep_bridge_tk.py
"""

from __future__ import annotations

import modal

# DISTINCT app name — never crucible-ep-bridge.
app = modal.App("crucible-ep-bridge-tk")

# The TK schema, sourced from the shared registry so the bridge, the HUD env, and the
# RFT evaluator all advertise ONE coherent TK schema.
from games_registry import TARGET_KNOCKBACK  # noqa: E402

TK_BOUNDS: dict[str, tuple[float, float]] = dict(TARGET_KNOCKBACK.bounds)

# Nested-reward budget — matches the TK scale-validation budget so the served reward is
# comparable to the validated d=0.55 learnable band.
NESTED_SEEDS = (1, 2, 3)
NESTED_PPO_EPISODES = 2000
NESTED_EVAL_SEEDS = 50

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi==0.138.0",
        "eval-protocol==0.3.31",
        "openai==2.43.0",
        "hud-python==0.6.6",
        # numpy/gymnasium are imported transitively when the bridge builds the TK
        # curriculum spec. torch/SB3 are NOT needed here: the PPO seeds train inside
        # crucible-player-tk; this container only dispatches.
        "numpy",
        "gymnasium",
        # `modal` itself, so fn.map to the crucible-player-tk app works for the
        # in-rollout seed fan-out.
        "modal",
    )
    # Ship the local source the TK nested reward imports: training/ (ep_remote_server +
    # tk_hud_teacher_env + hud_teacher_env + tk_dry_loop_sidecar), output/
    # (nested_reward_tk), games/ (target_knockback + fighter), harness/, the top-level
    # contracts module, and the games_registry. Without these the container raises
    # ModuleNotFoundError when create_ep_app/web imports the reward path.
    .add_local_python_source(
        "training",
        "output",
        "games",
        "harness",
        "contracts",
        "games_registry",
    )
)


def tk_nested_reward_scorer(params: dict[str, float]) -> float:
    """The nested-RL Target-Knockback reward for a clamped TK param set.

    Runs ``tk_teacher_reward`` with the scale-validation budget: the Teacher's emitted
    TK geometry trains the Player verbatim, the PPO seeds fan out in parallel on the
    ISOLATED ``crucible-player-tk`` Modal app, and the returned value is the mean
    held-out improvement across seeds, clamped to [0, 1].
    """
    from output.nested_reward_tk import tk_teacher_reward

    return tk_teacher_reward(
        params,
        backend="modal",
        seeds=NESTED_SEEDS,
        episodes=NESTED_PPO_EPISODES,
        eval_seeds=NESTED_EVAL_SEEDS,
        curriculum_id="tk-rft",
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("crucible-fireworks")],
    timeout=1800,
    # The async reward results live in app.state.results (in-memory); pin to ONE
    # always-warm container so /init and the /debug/result poll share memory and the
    # background task survives (same reasoning as the fighter bridge).
    min_containers=1,
    max_containers=1,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    from training.ep_remote_server import create_ep_app

    return create_ep_app(
        bounds=TK_BOUNDS,
        scorer=tk_nested_reward_scorer,
    )
