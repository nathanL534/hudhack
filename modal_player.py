"""Deployable Modal worker for Stage 4.

Deploy with:

    modal deploy modal_player.py

The worker currently returns a deterministic smoke result.  Replace the body of
``train_player`` with the serialized fighter/PPO adapter after Stage 1 contracts
are finalized; the caller interface will not change.
"""

from __future__ import annotations

try:
    import modal
except ImportError:  # pragma: no cover - optional integration
    modal = None


if modal is not None:  # pragma: no branch
    app = modal.App("crucible-player")
    image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "numpy",
        "gymnasium",
        "stable-baselines3",
        "torch",
        "pydantic",
    )

    @app.function(image=image, timeout=900)
    def train_player(payload: dict, seed: int) -> dict:
        # Smoke implementation verifies deployment, auth, fan-out, ordering,
        # timeout behavior, and serialization before real PPO is moved here.
        return {
            "seed": int(seed),
            "curriculum_id": payload.get("curriculum_id", "smoke"),
            "mean_score": float((seed % 10) / 10.0),
            "status": "smoke",
        }

