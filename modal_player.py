"""Deployable Modal worker for Stage 4 — the REAL fighter PPO Player trainer.

Deploy with:

    modal deploy modal_player.py

What changed (vs the old smoke worker): ``train_player`` now runs the SAME real
PPO Player training that ``harness/ppo_trainer.py`` (``PPOPlayerTrainer``) and
``prove_ppo_learns.py`` use locally — a small MLP trained by SB3 PPO on a single
``FighterArena`` (difficulty + geometry) against the difficulty-scaled parametric
opponent — and returns its **real learning improvement**: held-out win-rate AFTER
training minus BEFORE (an untrained net of the identical architecture), scored
through ``FighterGameAdapter.evaluate`` (the very same path the reference
policies use). Nothing is stubbed.

The Modal image ships the local ``games`` + ``harness`` Python packages via
``add_local_python_source("games", "harness")`` — exactly the bridge fix — so the
container can import ``games.fighter`` / ``harness.ppo_trainer`` /
``harness.fighter_adapter`` / ``contracts``.

Return shape (consumed by ``harness.modal_fanout._row_to_match_result`` to build
a frozen ``MatchResult``, and by the correlation gate to read the improvement):

    {
      "seed": int,
      "curriculum_id": str,
      "arena_scores": [after_winrate],   # held-out AFTER win-rate (the contract score)
      "mean_score": after_winrate,
      "before_winrate": float,           # untrained-net held-out win-rate
      "after_winrate": float,            # trained held-out win-rate
      "improvement": float,              # after - before  (the REAL PPO learning signal)
      "difficulty": float,
      "status": "ppo",
    }

``local_worker`` is the credential-free, no-Modal fallback: the SAME body without
the Modal decorator, so ``FanoutPlayerTrainer(local_worker)`` runs the whole
fan-out path locally. ``local_smoke_worker`` is kept as a deterministic,
torch-free fallback for shape/auth/fan-out smoke tests that must not pay PPO cost.
"""

from __future__ import annotations

try:
    import modal
except ImportError:  # pragma: no cover - optional integration
    modal = None


# ---------------------------------------------------------------------------
# Payload -> FighterArena (the arena under test)
# ---------------------------------------------------------------------------


def _arena_from_payload(payload: dict):
    """Build the concrete ``FighterArena`` this job trains/evaluates on.

    The payload carries the arena dials directly (difficulty + the four geometry
    knobs ``prove_ppo_learns`` varies). Missing knobs fall back to the
    ``FighterArena`` defaults, so a payload of just ``{"difficulty": 0.85}`` is a
    valid single-dial arena. Import is INSIDE the function so the module imports
    cleanly without ``games`` on the path (e.g. at Modal deploy registration on a
    machine where only the decorator matters).
    """
    from games.fighter import FighterArena

    defaults = FighterArena()
    return FighterArena(
        platform_width=float(payload.get("platform_width", defaults.platform_width)),
        gravity=float(payload.get("gravity", defaults.gravity)),
        knockback=float(payload.get("knockback", defaults.knockback)),
        spawn_gap=float(payload.get("spawn_gap", defaults.spawn_gap)),
        difficulty=float(payload.get("difficulty", defaults.difficulty)),
    )


def _real_ppo_result(payload: dict, seed: int) -> dict:
    """Train ONE real PPO Player on the payload's arena and measure its learning.

    This is the body shared by the Modal worker and the local fallback. It
    reproduces the ``prove_ppo_learns.py`` before/after protocol for a single
    arena and a single seed:

      * BEFORE  — an untrained PPO net (identical architecture) scored on a
                  HELD-OUT arena of the same config (unseen eval seeds).
      * AFTER   — train the Player (SB3 PPO, short budget) on the arena, then
                  score the trained policy on the same held-out arena.
      * improvement = AFTER - BEFORE  (the real PPO learning signal).

    All scoring goes through ``FighterGameAdapter.evaluate`` — the SAME path the
    scripted/random references use — so the number is directly comparable to the
    cheap gap proxy.
    """
    import warnings

    warnings.filterwarnings("ignore")

    from contracts import PlayerConfig, TrainingBudget
    from harness.fighter_adapter import FighterGameAdapter
    from harness.ppo_trainer import (
        PPOPlayerTrainer,
        _MultiArenaFighterEnv,
        _make_policy_from_model,
    )

    episodes = int(payload.get("ppo_episodes", 1000))   # ~60k timesteps at the default
    eval_seeds = int(payload.get("eval_seeds", 50))

    arena = _arena_from_payload(payload)
    # Held-out arena = same config family; the adapter scores over a fixed eval
    # seed set the trainer never saw (training draws its own episode seeds).
    held_out = type(arena)(
        platform_width=arena.platform_width,
        gravity=arena.gravity,
        knockback=arena.knockback,
        spawn_gap=arena.spawn_gap,
        difficulty=arena.difficulty,
    )

    adapter = FighterGameAdapter(eval_seeds=eval_seeds)
    held = adapter.arenas_from_configs([held_out], curriculum_id=payload.get("curriculum_id", "modal"))
    train_arenas = adapter.arenas_from_configs([arena], curriculum_id=payload.get("curriculum_id", "modal"))

    def _label_score(bundle: dict) -> float:
        (entry,) = bundle.values()
        return float(entry["mean_score"])

    # BEFORE: untrained net (no learning), same architecture, on held-out.
    from stable_baselines3 import PPO

    before_env = _MultiArenaFighterEnv([arena], seed=seed)
    before_model = PPO(
        "MlpPolicy",
        before_env,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": [64, 64]},
        device="cpu",
    )
    before_policy = _make_policy_from_model(before_model)
    before_winrate = _label_score(adapter.evaluate(before_policy, held))

    # AFTER: train, then score the trained policy on held-out.
    trainer = PPOPlayerTrainer(eval_seeds=eval_seeds)
    config = PlayerConfig(
        architecture=str(payload.get("architecture", "mlp")),
        num_seeds=1,
        budget=TrainingBudget(episodes=episodes),
        seed=int(seed),
        modal_parallel=False,  # we ARE the remote worker; train locally in-container.
    )
    job = trainer.submit(config, train_arenas)
    job.result()  # collect the in-distribution MatchResult (contract step)
    after_winrate = _label_score(adapter.evaluate(job.policy, held))

    improvement = after_winrate - before_winrate
    return {
        "seed": int(seed),
        "curriculum_id": payload.get("curriculum_id", "modal"),
        "arena_scores": [float(after_winrate)],
        "mean_score": float(after_winrate),
        "before_winrate": float(before_winrate),
        "after_winrate": float(after_winrate),
        "improvement": float(improvement),
        "difficulty": float(arena.difficulty),
        "status": "ppo",
    }


def local_worker(payload: dict, seed: int) -> dict:
    """Credential-free local fallback: the REAL PPO body, no Modal required."""
    return _real_ppo_result(payload, seed)


def _smoke_result(payload: dict, seed: int) -> dict:
    """Deterministic, torch-free smoke result (shape-only).

    Kept for fan-out / auth / serialization smoke tests that must not pay PPO
    cost. NOT used by the real worker.
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
    """Credential-free, torch-free smoke fallback (deterministic in seed)."""
    return _smoke_result(payload, seed)


if modal is not None:  # pragma: no branch
    app = modal.App("crucible-player")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "numpy",
            "gymnasium",
            "stable-baselines3",
            "torch",
            "pydantic",
        )
        # Ship the local game + harness packages into the image so the container
        # can import games.fighter / harness.ppo_trainer / harness.fighter_adapter
        # / contracts — exactly the bridge fix (without this: ModuleNotFoundError).
        .add_local_python_source("games", "harness", "contracts")
    )

    @app.function(image=image, timeout=1800)
    def train_player(payload: dict, seed: int) -> dict:
        """REAL remote PPO Player training (no stub). Returns learning improvement."""
        return _real_ppo_result(payload, seed)
