"""Deployable Modal worker for Stage 4.

Deploy with:

    modal deploy modal_player.py

The worker currently returns a deterministic smoke result.  Replace the body of
``_smoke_result`` (and the Modal ``train_player`` wrapper) with the serialized
fighter/PPO adapter after Stage 1 contracts are finalized; the caller interface
will not change.

``local_smoke_worker`` is the credential-free fallback: it is the SAME function
body without the Modal decorator, so ``FanoutPlayerTrainer(local_smoke_worker)``
runs the whole fan-out path locally with no Modal creds.  The fields it emits
(``seed``, ``curriculum_id``, ``arena_scores``, ``mean_score``) are exactly what
``harness.modal_fanout._row_to_match_result`` reads to build a frozen
``MatchResult``.
"""

from __future__ import annotations

try:
    import modal
except ImportError:  # pragma: no cover - optional integration
    modal = None


def _smoke_result(payload: dict, seed: int) -> dict:
    """Deterministic smoke result, shaped to the MatchResult bridge.

    Verifies deployment, auth, fan-out, ordering, timeout behavior, and
    serialization before real PPO is moved here.  Replace this body with the
    real PPO training call; keep the return shape.
    """
    mean_score = float((seed % 10) / 10.0)
    return {
        "seed": int(seed),
        "curriculum_id": payload.get("curriculum_id", "smoke"),
        "arena_scores": [mean_score],
        "mean_score": mean_score,
        "status": "smoke",
    }


def local_smoke_worker(payload: dict, seed: int) -> dict:
    """Credential-free local fallback: same body, no Modal required."""
    return _smoke_result(payload, seed)


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
        return _smoke_result(payload, seed)
