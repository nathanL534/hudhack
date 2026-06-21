"""Modal deployment entrypoint for the Fireworks <-> HUD rollout bridge.

Create a Modal secret named ``crucible-fireworks`` containing
``FIREWORKS_API_KEY`` before deployment. The same key authorizes model calls
and Fireworks tracing.
"""

from __future__ import annotations

import modal

app = modal.App("crucible-ep-bridge")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi==0.138.0",
        "eval-protocol==0.3.31",
        "openai==2.43.0",
        "hud-python==0.6.6",
    )
    # Ship the local `training` package (ep_remote_server + hud_teacher_env) into
    # the image; without this the container raises ModuleNotFoundError: 'training'
    # when `web()` imports the bridge.
    .add_local_python_source("training")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("crucible-fireworks")],
    timeout=1800,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    # Temporary deterministic proxy. Replace this function with the real
    # fighter/Modal PPO validator after its correlation gate passes.
    def scorer(params: dict[str, float]) -> float:
        difficulty = params["difficulty"]
        return max(0.0, 1.0 - abs(difficulty - 0.5) * 2.0)

    from training.ep_remote_server import create_ep_app

    return create_ep_app(
        bounds={
            "difficulty": (0.0, 1.0),
            "platform_width": (8.0, 30.0),
            "gravity": (0.2, 1.2),
            "knockback": (0.5, 6.0),
            "spawn_gap": (1.0, 12.0),
        },
        scorer=scorer,
    )

