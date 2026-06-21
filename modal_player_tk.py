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
    Teacher's payload arena, score before/after on the FIXED held-out reference set,
    return the transfer improvement.
  * ``_real_ppo_transfer_result`` -> ``_real_ppo_tk_result``: the shared body.
  * ``_held_out_reference_arenas`` -> ``_held_out_reference_tk_arenas``: the FIXED,
    arena-independent held-out reference set (the anti-degeneracy yardstick — the
    validated d=0.55 learnable band at default TK geometry).
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
      "before_winrate": float,           # untrained-net win-rate on the FIXED reference
      "after_winrate": float,            # trained win-rate on the FIXED reference
      "improvement": float,              # mean(after_ref) - mean(before_ref) (TRANSFER)
      "held_out_difficulties": [...],    # the fixed reference difficulties measured on
      "difficulty": float,               # the TEACHER's emitted training-arena difficulty
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
# THE FIXED HELD-OUT REFERENCE TEST SET (the anti-degeneracy fix)
# ---------------------------------------------------------------------------
#
# CRITICAL DESIGN POINT — the TK twin of modal_player.HELD_OUT_REFERENCE_DIFFICULTIES.
# The nested-RL Teacher reward = how much a Player IMPROVES after training on the
# Teacher's generated arena. If improvement is measured on the (Teacher-chosen,
# possibly trivial) TRAINING arena, an EASY arena maxes the reward: the Player starts
# low and trivially climbs on its own easy arena, so the Teacher games the reward by
# emitting trivial arenas (e.g. d~0.1) that show big IN-DISTRIBUTION improvement with
# no real transfer. That is the degeneracy a falsifier found here.
#
# Fix: before/after improvement is ALWAYS measured on this FIXED, arena-independent
# held-out reference set — a couple of STANDARD reference difficulties at DEFAULT TK
# geometry that sit in the validated LEARNABLE BAND (where a Player has real headroom).
# The Teacher cannot move this target; it can only generate a TRAINING arena, and the
# reward is the *transfer* of that training to the standard benchmark. A trivial
# training arena teaches nothing that transfers; a genuinely good (learnable-band)
# training arena teaches transferable skill -> higher reward. Same number for every
# arena, so it is a fair, non-gameable yardstick.
#
# The band is taken straight from tk_modal_scale_validation_results.json: d=0.55 is the
# clean learnable band (improvement mean +0.160, std 0.204, 11/12 seeds improved,
# training_stabilizes=True). d=0.5 and d=0.6 bracket it (both majority_improved) for a
# small, stable reference set, mirroring how the fighter uses a couple of reference
# difficulties. Zone dials (zone_half, zone_center_frac) are held at sensible fixed
# defaults so the reference geometry is stable and Teacher-independent.
HELD_OUT_REFERENCE_TK_DIFFICULTIES: tuple[float, ...] = (0.5, 0.55, 0.6)

# Fixed default geometry for the reference arenas. These are the TargetKnockbackArena
# defaults (platform_width=10.0, zone_half=1.5, zone_center_frac=0.5) — the SAME
# geometry the d=0.55 scale validation ran at, so the held-out reference reproduces the
# validated learnable band exactly. Stated explicitly here (rather than reusing the
# Teacher's emitted geometry) so the reference is a fixed yardstick the Teacher cannot
# move by widening/narrowing its zone.
HELD_OUT_REFERENCE_PLATFORM_WIDTH: float = 10.0
HELD_OUT_REFERENCE_ZONE_HALF: float = 1.5
HELD_OUT_REFERENCE_ZONE_CENTER_FRAC: float = 0.5


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
# The FIXED held-out reference arenas (the anti-degeneracy yardstick)
# ---------------------------------------------------------------------------


def _held_out_reference_tk_arenas(payload: dict):
    """Build the FIXED held-out TK reference arenas (the standard benchmark).

    These are INDEPENDENT of the Teacher's training arena in ``payload`` — the same
    fixed set for every Teacher arena under evaluation, so improvement-on-them measures
    TRANSFER to a yardstick the Teacher cannot move. Mirrors the fighter's
    ``_held_out_reference_arenas``.

    Two layers, in priority order (so a caller can opt into a custom set, but the
    DEFAULT is the validated learnable band):

      1. ``held_out_tk_arenas`` — a list of FULL arena-spec dicts (each may carry
         ``difficulty`` plus the TK geometry dials ``platform_width`` / ``zone_half`` /
         ``zone_center_frac``; missing knobs fall back to TargetKnockbackArena
         defaults). Layered behind this explicit key so it changes nothing unless a
         caller opts in.
      2. ``held_out_difficulties`` (or the module DEFAULT
         ``HELD_OUT_REFERENCE_TK_DIFFICULTIES``) — difficulty-only arenas at the FIXED
         default reference geometry. This is the standard, non-gameable yardstick: the
         validated d=0.55 learnable band (bracketed by d=0.5/0.6) at default geometry.
    """
    from games.target_knockback import TargetKnockbackArena

    specs = payload.get("held_out_tk_arenas")
    if specs:
        defaults = TargetKnockbackArena()
        return [
            TargetKnockbackArena(
                platform_width=float(s.get("platform_width", defaults.platform_width)),
                zone_half=float(s.get("zone_half", defaults.zone_half)),
                zone_center_frac=float(s.get("zone_center_frac", defaults.zone_center_frac)),
                difficulty=float(s.get("difficulty", defaults.difficulty)),
            )
            for s in specs
        ]

    diffs = payload.get("held_out_difficulties", HELD_OUT_REFERENCE_TK_DIFFICULTIES)
    return [
        TargetKnockbackArena(
            platform_width=HELD_OUT_REFERENCE_PLATFORM_WIDTH,
            zone_half=HELD_OUT_REFERENCE_ZONE_HALF,
            zone_center_frac=HELD_OUT_REFERENCE_ZONE_CENTER_FRAC,
            difficulty=float(d),
        )
        for d in diffs
    ]


# ---------------------------------------------------------------------------
# The shared body — train one PPO Player, measure held-out before/after.
# ---------------------------------------------------------------------------


def _real_ppo_tk_result(payload: dict, seed: int) -> dict:
    """Train ONE real PPO Player on the Teacher arena, measure HELD-OUT TRANSFER.

    Body shared by the Modal worker and the local fallback. The nested-RL TK reward
    signal. The Player is TRAINED on the Teacher-generated ``payload`` arena (whatever
    difficulty/geometry the Teacher emitted), but its before/after win-rate is measured
    on the FIXED held-out reference set (``_held_out_reference_tk_arenas``), NOT on the
    training arena. That decoupling is what removes the easy-arena degeneracy: an EASY
    training arena (e.g. d~0.1) shows big IN-DISTRIBUTION improvement but its policy does
    NOT transfer to the learnable-band reference, so it can no longer game the reward.

      * BEFORE  — an untrained PPO net (identical architecture) scored on the FIXED
                  held-out reference set (the validated d=0.55 learnable band).
      * AFTER   — train the Player (SB3 PPO, short budget) on the TEACHER arena, then
                  score the trained policy on the SAME fixed held-out reference set.
      * improvement = mean(after_ref) - mean(before_ref)  (the TRANSFER learning signal).

    All scoring goes through ``TargetKnockbackGameAdapter.evaluate`` (the SAME path the
    scripted/random references use, so the number is directly comparable). Supports an
    optional ``capture_replay_id``: if set, roll out ONE trained-Player held-out match
    (on the first reference arena) and attach a viewer-shaped replay dict.
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

    # TRAIN on the Teacher's emitted arena ...
    arena = _tk_arena_from_payload(payload)
    # ... but MEASURE on the FIXED held-out reference set, independent of the training
    # arena. This is the anti-degeneracy fix: the Teacher cannot move this target, so the
    # reward reflects transfer, not in-distribution improvement on a trivial arena.
    held_out_arenas = _held_out_reference_tk_arenas(payload)

    cid = payload.get("curriculum_id", "modal-tk")
    adapter = TargetKnockbackGameAdapter(eval_seeds=eval_seeds)
    held = adapter.arenas_from_configs(held_out_arenas, curriculum_id=f"{cid}-heldout")

    def _label_score(bundle: dict) -> float:
        (entry,) = bundle.values()
        return float(entry["mean_score"])

    def _per_arena(bundle: dict) -> list[float]:
        (entry,) = bundle.values()
        return [float(s) for s in entry["per_arena"]]

    # BEFORE: untrained net (random-init, no learning), same architecture, on the FIXED
    # held-out reference set. The env is built on the TRAINING arena (it only seeds the
    # net's observation/action spaces; no learning happens), matching the fighter.
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
    before_bundle = adapter.evaluate(before_policy, held)
    before_winrate = _label_score(before_bundle)
    before_per_arena = _per_arena(before_bundle)

    # AFTER: train on the TEACHER arena, then score the trained policy on the SAME fixed
    # held-out reference set.
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
    after_bundle = adapter.evaluate(after_policy, held)
    after_winrate = _label_score(after_bundle)
    after_per_arena = _per_arena(after_bundle)

    improvement = after_winrate - before_winrate
    result = {
        "seed": int(seed),
        "curriculum_id": cid,
        # The CONTRACT score for this row is the held-out AFTER win-rate (so anything
        # built from this carries the transfer signal, not the training-arena number).
        "arena_scores": [float(after_winrate)],
        "mean_score": float(after_winrate),
        "before_winrate": float(before_winrate),
        "after_winrate": float(after_winrate),
        "improvement": float(improvement),
        # Held-out diagnostics (mirror the fighter's transfer worker), so the run log
        # SHOWS the reward came from the fixed reference set, not the training arena.
        "held_out_difficulties": [float(a.difficulty) for a in held_out_arenas],
        "held_out_before_per_arena": before_per_arena,
        "held_out_after_per_arena": after_per_arena,
        # ``difficulty`` still reports the TRAINING arena's difficulty (what the Teacher
        # emitted), so the run log shows which arena was generated.
        "difficulty": float(arena.difficulty),
        "status": "ppo_tk",
    }

    # Optional replay capture: roll out ONE held-out match of the trained Player on the
    # first reference arena and persist a viewer-ready replay dict (the worker returns
    # it; the driver writes it).
    replay_id = payload.get("capture_replay_id")
    if replay_id:
        result["replay"] = _capture_trained_tk_replay(
            after_policy, held_out_arenas[0], replay_id=str(replay_id), seed=int(seed)
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

        The nested-RL Teacher reward worker: trains a fresh SB3 PPO Player on the
        Teacher's payload TK arena (difficulty + zone geometry), measures before/after
        win-rate on the FIXED held-out reference set (NOT the training arena), and
        returns the transfer improvement in the SAME schema the fighter worker returns.
        With ``capture_replay_id`` set, also returns one trained-Player TK replay.
        """
        return _real_ppo_tk_result(payload, seed)
