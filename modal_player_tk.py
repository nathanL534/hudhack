"""modal_player_tk.py — ISOLATED Modal worker for Target Knockback (game #2).

This is a BRAND-NEW, SEPARATE Modal app (``crucible-player-tk``) that trains a real
SB3 PPO Player on a Target-Knockback arena and returns the same before/after learning
schema the fighter worker returns. It exists so Target Knockback can be validated and
scaled on Modal CONCURRENTLY with an in-flight fighter RFT — WITHOUT touching the
fighter's worker (``modal_player.py`` / app ``crucible-player``). Nothing here imports
or modifies the fighter worker; the only shared thing is the local game/harness source
shipped into the image, which is read-only.

Deploy with::

    # pop the BLANK MODAL_TOKEN_ID/SECRET that .env ships (they break env-var auth),
    # then deploy under the njlee007 profile:
    modal deploy modal_player_tk.py

What it mirrors (from ``modal_player.py``):
  * ``train_player_transfer`` -> ``train_tk_player``: train one fresh PPO Player on the
    payload's arena, score held-out before/after, return the improvement.
  * ``_real_ppo_transfer_result`` -> ``_real_ppo_tk_result``: the shared body.
  * ``_capture_trained_replay`` -> ``_capture_trained_tk_replay``: roll out ONE trained
    match and return a viewer-shaped replay dict (it does NOT write to disk; the local
    driver writes the returned dict, keeping the worker filesystem-free).
  * the ``.add_local_python_source(...)`` image setup, so the container can import
    ``games.target_knockback`` / ``harness.target_knockback_adapter`` / ``contracts`` /
    ``replay`` (and ``games.fighter``, which TK reuses for physics).

Return shape (same keys the fighter worker returns, so any downstream that reads
``before_winrate`` / ``after_winrate`` / ``improvement`` / ``mean_score`` works
unchanged)::

    {
      "seed": int,
      "curriculum_id": str,
      "arena_scores": [after_winrate],   # held-out AFTER win-rate (the contract score)
      "mean_score": after_winrate,
      "before_winrate": float,           # untrained-net held-out win-rate
      "after_winrate": float,            # trained held-out win-rate
      "improvement": float,              # after - before  (the REAL PPO learning signal)
      "difficulty": float,
      "status": "ppo_tk",
      "replay": {...}                    # only if payload["capture_replay_id"] is set
    }

``local_tk_worker`` is the credential-free, no-Modal fallback: the SAME body without the
Modal decorator, so the validation driver can run the whole path locally too.
"""

from __future__ import annotations

try:
    import modal
except ImportError:  # pragma: no cover - optional integration
    modal = None


# ---------------------------------------------------------------------------
# Payload -> TargetKnockbackArena (the arena under test)
# ---------------------------------------------------------------------------


def _tk_arena_from_payload(payload: dict):
    """Build the concrete ``TargetKnockbackArena`` this job trains/evaluates on.

    The payload carries the arena dials directly: ``difficulty`` (the single knob the
    Teacher turns — sets opponent strength) plus the target-zone geometry
    (``platform_width`` / ``zone_half`` / ``zone_center_frac``). Missing knobs fall back
    to the ``TargetKnockbackArena`` defaults, so a payload of just ``{"difficulty": 0.5}``
    is a valid single-dial arena. Import is INSIDE the function so the module imports
    cleanly without ``games`` on the path (e.g. at Modal deploy registration).
    """
    from games.target_knockback import TargetKnockbackArena

    defaults = TargetKnockbackArena()
    return TargetKnockbackArena(
        platform_width=float(payload.get("platform_width", defaults.platform_width)),
        zone_half=float(payload.get("zone_half", defaults.zone_half)),
        zone_center_frac=float(payload.get("zone_center_frac", defaults.zone_center_frac)),
        difficulty=float(payload.get("difficulty", defaults.difficulty)),
    )


# ---------------------------------------------------------------------------
# PPO training env + policy wrapping (the SAME SB3 wiring prove_target_knockback
# _learns.py uses — opponent re-seeded per episode so the Player must be reactive).
# ---------------------------------------------------------------------------


def _make_tk_env(arena, seed: int):
    """A single-agent Gymnasium env over ``TargetKnockbackEnv`` whose opponent is
    re-seeded every reset, exactly like ``prove_target_knockback_learns._make_env``.

    Re-seeding the opponent (and spawn) per episode is what forces the Player to learn
    a reactive policy instead of memorising one parametric-opponent rollout — the same
    reasoning the fighter trainer uses.
    """
    import gymnasium as gym
    import numpy as np

    from games.target_knockback import (
        TargetKnockbackEnv,
        parametric_target_knockback,
    )

    class _Wrapped(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self._rng = np.random.default_rng(seed)
            self._inner = TargetKnockbackEnv(
                arena,
                opponent_factory=lambda a: parametric_target_knockback(a, ego=1, seed=seed),
                seed=seed,
            )
            self.action_space = self._inner.action_space
            self.observation_space = self._inner.observation_space
            self.render_mode = None

        def reset(self, *, seed=None, options=None):
            ep_seed = int(self._rng.integers(1_000_000))
            self._inner = TargetKnockbackEnv(
                arena,
                opponent_factory=lambda a: parametric_target_knockback(a, ego=1, seed=ep_seed),
                seed=ep_seed,
            )
            return self._inner.reset(seed=seed, options=options)

        def step(self, action):
            return self._inner.step(action)

        def close(self):
            return None

    return _Wrapped()


def _make_policy_from_model(model):
    """Wrap an SB3 model as the obs->action callable the adapter scores."""

    def act(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return act


# ---------------------------------------------------------------------------
# The shared body — train one PPO Player, measure held-out before/after.
# ---------------------------------------------------------------------------


def _real_ppo_tk_result(payload: dict, seed: int) -> dict:
    """Train ONE real PPO Player on the payload's TK arena and measure its learning.

    Body shared by the Modal worker and the local fallback. Reproduces the
    ``prove_target_knockback_learns.py`` before/after protocol for a single arena and a
    single seed, scored through ``TargetKnockbackGameAdapter.evaluate`` (the SAME path
    the scripted/random references use, so the number is directly comparable):

      * BEFORE  — an untrained PPO net (identical architecture) scored on a HELD-OUT
                  arena of the same config (the adapter's fixed eval-seed set).
      * AFTER   — train the Player (SB3 PPO, short budget) on the arena, then score the
                  trained policy on the same held-out arena.
      * improvement = AFTER - BEFORE  (the real PPO learning signal).

    Supports an optional ``capture_replay_id``: if set, roll out ONE trained-Player
    held-out match and attach a viewer-shaped replay dict.
    """
    import os
    import warnings

    warnings.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from stable_baselines3 import PPO

    from harness.target_knockback_adapter import TargetKnockbackGameAdapter

    # ppo_episodes -> timesteps via the same constants prove_target_knockback_learns uses
    # (60 steps/episode, 4_000 floor), so the budget is comparable to the local proof.
    episodes = int(payload.get("ppo_episodes", 2000))
    eval_seeds = int(payload.get("eval_seeds", 50))
    steps_per_episode = 60
    min_timesteps = 4_000
    total_timesteps = max(min_timesteps, episodes * steps_per_episode)

    arena = _tk_arena_from_payload(payload)
    # Held-out arena = same config family; the adapter scores over its fixed eval-seed
    # set, which the trainer never sees (training draws its own per-episode seeds).
    held_out = type(arena)(
        platform_width=arena.platform_width,
        zone_half=arena.zone_half,
        zone_center_frac=arena.zone_center_frac,
        difficulty=arena.difficulty,
    )

    cid = payload.get("curriculum_id", "modal-tk")
    adapter = TargetKnockbackGameAdapter(eval_seeds=eval_seeds)
    held = adapter.arenas_from_configs([held_out], curriculum_id=f"{cid}-heldout")

    def _label_score(bundle: dict) -> float:
        (entry,) = bundle.values()
        return float(entry["mean_score"])

    # BEFORE: untrained net (random-init, no learning), same architecture, on held-out.
    before_env = _make_tk_env(arena, seed)
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
    train_env = _make_tk_env(arena, seed)
    after_model = PPO(
        "MlpPolicy",
        train_env,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=1024,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.005,
        device="cpu",
    )
    after_model.learn(total_timesteps=total_timesteps, progress_bar=False)
    after_policy = _make_policy_from_model(after_model)
    after_winrate = _label_score(adapter.evaluate(after_policy, held))

    improvement = after_winrate - before_winrate
    result = {
        "seed": int(seed),
        "curriculum_id": cid,
        "arena_scores": [float(after_winrate)],
        "mean_score": float(after_winrate),
        "before_winrate": float(before_winrate),
        "after_winrate": float(after_winrate),
        "improvement": float(improvement),
        "difficulty": float(arena.difficulty),
        "status": "ppo_tk",
    }

    # Optional replay capture: roll out ONE held-out match of the trained Player and
    # persist a viewer-ready replay dict (the worker returns it; the driver writes it).
    replay_id = payload.get("capture_replay_id")
    if replay_id:
        result["replay"] = _capture_trained_tk_replay(
            after_policy, held_out, replay_id=str(replay_id), seed=int(seed)
        )

    return result


def _capture_trained_tk_replay(policy, arena, *, replay_id: str, seed: int) -> dict:
    """Roll out ONE held-out Target-Knockback match of the trained Player vs the
    parametric opponent and return a viewer-ready replay dict (schema-validated).

    Mirrors the fighter's ``_capture_trained_replay``: it reads the sim through its
    public interface only (``TargetKnockbackSim`` + ``observe`` + ``step``, the same
    loop ``play_match`` uses) and feeds frames into the shared ``ReplayBuilder``. The
    builder needs a ``FighterArena`` for platform geometry — TK's arena exposes exactly
    that view via ``_as_fighter_arena()`` — and reads ``sim.f0`` / ``sim.f1``, which on
    a ``TargetKnockbackSim`` are the underlying physics bodies, so the trace is correct.
    The worker returns the dict; the local driver writes it to ``replays/<id>.json``,
    keeping the Modal worker filesystem-free.
    """
    from games.fighter import Action
    from games.target_knockback import TargetKnockbackSim, parametric_target_knockback
    from replay import ReplayBuilder, validate_replay

    opponent = parametric_target_knockback(arena, ego=1, seed=10_000 + seed)
    sim = TargetKnockbackSim(arena=arena, seed=seed)
    builder = ReplayBuilder(
        arena=arena._as_fighter_arena(),
        config=f"d{arena.difficulty}",
        p1_policy="modal_trained_ppo_tk",
        p2_policy="parametric_target_knockback",
        seed=seed,
    )
    builder.capture(sim, int(Action.IDLE), int(Action.IDLE))
    while not sim.done:
        a0 = int(policy(sim.observe(ego=0)))
        a1 = int(opponent(sim.observe(ego=1)))
        sim.step(a0, a1)
        builder.capture(sim, a0, a1)

    data = builder.to_dict(sim.winner)
    problems = validate_replay(data)
    if problems:  # pragma: no cover - defensive; builder produces valid frames
        raise ValueError(f"captured replay {replay_id!r} failed validation: {problems}")
    return {"replay_id": replay_id, "data": data}


def local_tk_worker(payload: dict, seed: int) -> dict:
    """Credential-free local fallback: the REAL TK PPO body, no Modal required."""
    return _real_ppo_tk_result(payload, seed)


# ---------------------------------------------------------------------------
# The Modal app — BRAND NEW, isolated from crucible-player.
# ---------------------------------------------------------------------------


if modal is not None:  # pragma: no branch
    app = modal.App("crucible-player-tk")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "numpy",
            "gymnasium",
            "stable-baselines3",
            "torch",
            "pydantic",
        )
        # Ship the local game + harness packages into the image so the container can
        # import games.target_knockback (+ games.fighter, which TK reuses for physics) /
        # harness.target_knockback_adapter / contracts / replay. Without this the
        # container raises ModuleNotFoundError. This is the SAME bridge fix the fighter
        # worker uses; it ships source read-only and modifies nothing.
        .add_local_python_source("games", "harness", "contracts", "replay")
    )

    @app.function(image=image, timeout=1800)
    def train_tk_player(payload: dict, seed: int) -> dict:
        """REAL remote PPO Player training on a Target-Knockback arena (no stub).

        Trains a fresh SB3 PPO Player on the payload's TK arena (difficulty + zone
        geometry), scores held-out before/after, and returns the learning improvement
        in the SAME schema the fighter worker returns. With ``capture_replay_id`` set,
        also returns one trained-Player TK replay.
        """
        return _real_ppo_tk_result(payload, seed)
