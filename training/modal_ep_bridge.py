"""Modal deployment entrypoint for the Fireworks <-> HUD rollout bridge.

The served reward IS the VALIDATED nested-RL Teacher reward (held-out PPO transfer
on the BROAD population) — the same reward the dry loop proved end to end. It is
NOT the old placeholder triangle scorer and NOT the cheap strong-vs-weak gap proxy.

How the reward is served (one /init round-trip):

    Fireworks RFT POSTs /init with the current Qwen checkpoint + Teacher prompt
        -> create_ep_app calls the checkpoint, gets the Teacher's curriculum JSON
        -> strict-validates + clamps to FIGHTER_BOUNDS
        -> ``nested_reward_scorer`` runs ``teacher_reward(...)`` with the EXACT
           dry-loop config: 3 PPO seeds, 1000 episodes, 50 eval seeds, the BROAD
           ``full`` 15-arena held-out grid, the 5-knob geometry override
        -> the 3 PPO seeds fan out on the deployed ``crucible-player`` Modal app
           IN PARALLEL (one ``fn.map`` — already how ``teacher_reward`` works)
        -> reward = mean held-out transfer across seeds, clamped to [0, 1]
        -> HUD stays load-bearing: this nested number is what the grader records.

This bridge container only DISPATCHES the PPO seeds (the heavy SB3/torch training
runs inside ``crucible-player``), so its image needs the local source packages and
``modal`` to call ``crucible-player`` — not torch.

Create a Modal secret named ``crucible-fireworks`` containing ``FIREWORKS_API_KEY``
before deployment (the same key authorizes model calls and Fireworks tracing)::

    modal deploy training/modal_ep_bridge.py
"""

from __future__ import annotations

import modal

app = modal.App("crucible-ep-bridge")

# The FIGHTER schema the Teacher emits everywhere (matches FIGHTER_BOUNDS in
# training/hud_teacher_env.py and the deploy/test path). One coherent schema.
FIGHTER_BOUNDS: dict[str, tuple[float, float]] = {
    "difficulty": (0.0, 1.0),
    "platform_width": (8.0, 30.0),
    "gravity": (0.2, 1.2),
    "knockback": (0.5, 6.0),
    "spawn_gap": (1.0, 12.0),
}

# Nested-reward budget — IDENTICAL to the proven dry loop (training/dry_loop.py):
# 3 fresh PPO Players (seeds), a short-but-real PPO budget, enough held-out eval
# seeds for a stable win-rate, the BROAD (full) 15-arena held-out grid.
NESTED_SEEDS = (1, 2, 3)
NESTED_PPO_EPISODES = 1000
NESTED_EVAL_SEEDS = 50
NESTED_HELD_OUT_GRID = "full"  # 5 physics variants x 3 difficulties = 15 arenas
NESTED_BACKEND = "modal"       # PPO seeds fan out on the deployed crucible-player app

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi==0.138.0",
        "eval-protocol==0.3.31",
        "openai==2.43.0",
        "hud-python==0.6.6",
        # numpy/gymnasium are imported transitively when the bridge builds the
        # held-out population (output.broad_eval_set -> games.fighter -> numpy)
        # and the curriculum spec (contracts). torch/SB3 are NOT needed here: the
        # PPO seeds train inside crucible-player, this container only dispatches.
        "numpy",
        "gymnasium",
        # `modal` itself is present in the runtime; pin it so fn.map to the
        # crucible-player app works for the in-rollout seed fan-out.
        "modal",
    )
    # Ship the local source the nested reward imports: training/ (ep_remote_server
    # + hud_teacher_env + dry_loop_sidecar), output/ (nested_reward + broad_eval_set),
    # games/ (fighter), harness/ (adapters/scoring read by the spec path), and the
    # top-level contracts module. Without these the container raises
    # ModuleNotFoundError when create_ep_app/web imports the reward path.
    .add_local_python_source(
        "training",
        "output",
        "games",
        "harness",
        "contracts",
    )
)


def nested_reward_scorer(params: dict[str, float]) -> float:
    """The VALIDATED nested-RL Teacher reward for a clamped FIGHTER param set.

    Runs ``teacher_reward`` with the EXACT dry-loop config: the Teacher's emitted
    geometry trains the Player verbatim (``geometry_override``), the 3 PPO seeds
    fan out in parallel on the deployed ``crucible-player`` Modal app, and transfer
    is scored on the BROAD ``full`` held-out population. The returned value is the
    mean held-out improvement across seeds, already clamped to [0, 1]. This is the
    SAME number the dry loop asserted Modal == HUD == Eval-Protocol agree on.
    """
    from output.broad_eval_set import build_broad_eval_arenas, payload_arenas
    from output.nested_reward import teacher_reward
    from training.hud_teacher_env import fighter_geometry_override, params_to_curriculum

    spec = params_to_curriculum(params, curriculum_id="rft")
    held_out = payload_arenas(build_broad_eval_arenas(grid=NESTED_HELD_OUT_GRID))
    return teacher_reward(
        spec,
        backend=NESTED_BACKEND,
        seeds=NESTED_SEEDS,
        episodes=NESTED_PPO_EPISODES,
        eval_seeds=NESTED_EVAL_SEEDS,
        held_out_arenas=held_out,
        geometry_override=fighter_geometry_override(params),
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("crucible-fireworks")],
    timeout=1800,
    # The async reward results live in ``app.state.results`` (in-memory). On a
    # horizontally-scaled serverless deploy, ``/init`` and the ``/debug/result``
    # poll can hit DIFFERENT containers -> result not found -> poll-error; an idle
    # container can also be recycled mid-computation, killing the ~2-min reward
    # task. Pin to ONE always-warm container so every request shares the same
    # memory and the background task survives. (Fine for the tiny RFT: ~2 rollouts;
    # for scale, swap the in-memory store for a modal.Dict.)
    min_containers=1,
    max_containers=1,
)
# One Teacher rollout = one full nested reward (3 PPO seeds, ~minutes). Keep the
# per-container input concurrency low so each rollout's seed fan-out gets its own
# slice of Modal capacity rather than thrashing one container; the RFT launcher's
# max_concurrent_rollouts is the real concurrency dial (~2 rollouts).
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    from training.ep_remote_server import create_ep_app

    return create_ep_app(
        bounds=FIGHTER_BOUNDS,
        scorer=nested_reward_scorer,
    )
