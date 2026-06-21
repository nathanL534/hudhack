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
# THE FIXED HELD-OUT REFERENCE TEST SET (the anti-degeneracy fix)
# ---------------------------------------------------------------------------
#
# CRITICAL DESIGN POINT. The nested-RL Teacher reward = how much a Player IMPROVES
# after training on the Teacher's generated arena. If improvement is measured on
# the (Teacher-chosen, possibly trivial) TRAINING arena, an EASY arena maxes the
# reward — the Player starts low and trivially climbs, so the Teacher games the
# reward by emitting trivial arenas (the degeneracy we observed: d=0.2 gives
# +0.267 "improvement" on its own easy arena).
#
# Fix: before/after improvement is ALWAYS measured on this FIXED, arena-independent
# held-out reference set — a couple of STANDARD reference difficulties at default
# geometry that sit in/near the learnable band (where a Player has real headroom).
# The Teacher cannot move this target; it can only generate a TRAINING arena, and
# the reward is the *transfer* of that training to the standard benchmark. A
# trivial training arena teaches nothing that transfers; a genuinely good training
# arena teaches transferable skill -> higher reward. Same number for every arena,
# so it is a fair, non-gameable yardstick.
#
# Two reference difficulties (d=0.75, d=0.85) sit in the TURTLE-ACTIVE zone of the
# difficulty->win-rate curve (difficulty_sweep_results.json: the jump_turtle
# opponent is engaging, scripted still wins ~0.85 so there IS headroom, but only a
# Player that learned to time/bait a DODGING opponent scores — basic
# approach-and-punch draws). This band is calibrated against the degeneracy (the
# whole point): training arenas trivial/good/hard/impossible were each trained at
# the real budget and their TRANSFER to candidate reference sets measured. At an
# EASY reference (0.4-0.6), a TRIVIAL training arena's basic approach-and-punch
# policy (which trains to ~1.0 win-rate on its own easy arena) ALREADY transfers
# and maxes the reward — the degeneracy is back. At these turtle-active references
# the trivial-arena policy CANNOT transfer (it never faced a turtle, +0.05/+0.20),
# while a learnable-band arena (d~0.65) transfers real skill (+0.36) — the highest
# reward, and NOT the easiest arena. An IMPOSSIBLE arena (undefeatable turtle)
# teaches nothing transferable (+0.00). So good > trivial > impossible, and the
# easy arena does NOT max the reward. Default geometry -> a stable, arena-
# independent yardstick the Teacher cannot move.
HELD_OUT_REFERENCE_DIFFICULTIES: tuple[float, ...] = (0.75, 0.85)


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


# ---------------------------------------------------------------------------
# HELD-OUT TRANSFER worker — the nested-RL Teacher reward signal
# ---------------------------------------------------------------------------


def _held_out_reference_arenas(payload: dict):
    """Build the FIXED held-out reference arenas (the standard benchmark).

    These are INDEPENDENT of the training arena in ``payload`` — the same fixed
    set for every Teacher arena under evaluation, so improvement-on-them measures
    transfer to a yardstick the Teacher cannot move.

    Two layers, in priority order:

      1. ``held_out_arenas`` — a list of FULL arena-spec dicts (each may carry
         ``platform_width`` / ``gravity`` / ``knockback`` / ``spawn_gap`` /
         ``difficulty``; missing knobs fall back to ``FighterArena`` defaults).
         This is the STRUCTURALLY-DIVERSE eval population: a held-out set that
         varies PHYSICS as well as opponent strength, used to ask whether a
         training arena's transfer is BROAD or just gaming the narrow default-
         geometry turtle references. Layered behind this explicit key so it
         changes nothing unless a caller opts in.
      2. ``held_out_difficulties`` (or the module DEFAULT
         ``HELD_OUT_REFERENCE_DIFFICULTIES``) — difficulty-only arenas at default
         geometry. This is the ORIGINAL, unchanged behavior; it is what every
         existing reward call and test produces.
    """
    from games.fighter import FighterArena

    specs = payload.get("held_out_arenas")
    if specs:
        defaults = FighterArena()
        return [
            FighterArena(
                platform_width=float(s.get("platform_width", defaults.platform_width)),
                gravity=float(s.get("gravity", defaults.gravity)),
                knockback=float(s.get("knockback", defaults.knockback)),
                spawn_gap=float(s.get("spawn_gap", defaults.spawn_gap)),
                difficulty=float(s.get("difficulty", defaults.difficulty)),
            )
            for s in specs
        ]

    diffs = payload.get("held_out_difficulties", HELD_OUT_REFERENCE_DIFFICULTIES)
    return [FighterArena(difficulty=float(d)) for d in diffs]


def _real_ppo_transfer_result(payload: dict, seed: int) -> dict:
    """Train ONE real PPO Player on the Teacher arena, measure HELD-OUT TRANSFER.

    The nested-RL reward signal. Identical PPO training to ``_real_ppo_result``
    (SB3 PPO, short budget, on the Teacher-generated ``payload`` arena vs the
    difficulty-scaled parametric opponent), but the BEFORE/AFTER win-rate is
    scored on the FIXED held-out reference set (``_held_out_reference_arenas``),
    NOT on the training arena. That decoupling is what removes the easy-arena
    degeneracy: the Teacher trains the Player wherever it likes, but is graded on
    how much that training transfers to a standard benchmark it cannot game.

      * BEFORE — an untrained PPO net (identical architecture) scored on the fixed
                 held-out reference set.
      * AFTER  — train the Player on the Teacher arena, score the trained policy
                 on the SAME fixed held-out reference set.
      * held_out_improvement = AFTER - BEFORE  (the transfer learning signal).

    All scoring goes through ``FighterGameAdapter.evaluate`` (the reference path).
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

    episodes = int(payload.get("ppo_episodes", 1000))
    eval_seeds = int(payload.get("eval_seeds", 50))

    train_arena = _arena_from_payload(payload)
    held_out_arenas = _held_out_reference_arenas(payload)

    adapter = FighterGameAdapter(eval_seeds=eval_seeds)
    cid = payload.get("curriculum_id", "modal")
    held = adapter.arenas_from_configs(held_out_arenas, curriculum_id=f"{cid}-heldout")
    train_arenas = adapter.arenas_from_configs([train_arena], curriculum_id=cid)

    def _label_score(bundle: dict) -> float:
        (entry,) = bundle.values()
        return float(entry["mean_score"])

    def _per_arena(bundle: dict) -> list[float]:
        (entry,) = bundle.values()
        return [float(s) for s in entry["per_arena"]]

    # BEFORE: untrained net on the FIXED held-out reference set.
    from stable_baselines3 import PPO

    before_env = _MultiArenaFighterEnv([train_arena], seed=seed)
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

    # AFTER: train on the Teacher arena, then score on the SAME held-out set.
    trainer = PPOPlayerTrainer(eval_seeds=eval_seeds)
    config = PlayerConfig(
        architecture=str(payload.get("architecture", "mlp")),
        num_seeds=1,
        budget=TrainingBudget(episodes=episodes),
        seed=int(seed),
        modal_parallel=False,  # we ARE the remote worker; train in-container.
    )
    job = trainer.submit(config, train_arenas)
    train_result = job.result()  # in-distribution MatchResult (training-arena win-rate)
    after_bundle = adapter.evaluate(job.policy, held)
    after_winrate = _label_score(after_bundle)
    after_per_arena = _per_arena(after_bundle)

    held_out_improvement = after_winrate - before_winrate

    result = {
        "seed": int(seed),
        "curriculum_id": cid,
        # The CONTRACT score for this row is the held-out AFTER win-rate (so a
        # MatchResult built from this carries the transfer signal, not the
        # training-arena number).
        "arena_scores": [float(after_winrate)],
        "mean_score": float(after_winrate),
        "before_winrate": float(before_winrate),
        "after_winrate": float(after_winrate),
        "held_out_improvement": float(held_out_improvement),
        # Keep ``improvement`` as an alias so the fan-out plumbing that reads
        # ``improvement`` (modal_fanout / proxy_sweep) sees the HELD-OUT number.
        "improvement": float(held_out_improvement),
        "held_out_difficulties": [float(a.difficulty) for a in held_out_arenas],
        "held_out_before_per_arena": before_per_arena,
        "held_out_after_per_arena": after_per_arena,
        # Diagnostic: how the Player did on its OWN (training) arena, to SHOW the
        # degeneracy is gone — easy training arenas score high in-distribution but
        # transfer little to the held-out set.
        "train_arena_winrate": float(train_result.mean_score),
        "difficulty": float(train_arena.difficulty),
        "status": "ppo_transfer",
    }

    # Optional replay capture: roll out ONE held-out match of the trained Player
    # and persist it so the viewer can play the actual Modal-trained fighter.
    replay_id = payload.get("capture_replay_id")
    if replay_id:
        result["replay"] = _capture_trained_replay(
            job.policy, held_out_arenas[0], replay_id=str(replay_id), seed=int(seed)
        )

    return result


def _capture_trained_replay(policy, arena, *, replay_id: str, seed: int) -> dict:
    """Roll out ONE held-out match of the trained Player vs the parametric
    opponent and return a viewer-ready replay dict (schema-validated).

    Reads the fighter through its public interface only (FighterSim + observe +
    step, the same loop ``play_match`` / ``record_replay`` use). The caller writes
    it to ``replays/modal_trained_<id>.json`` once the row returns — keeping the
    Modal worker filesystem-free (it returns the dict, the local driver writes it).
    """
    from games.fighter import Action, FighterSim, parametric_fighter
    from replay import ReplayBuilder, validate_replay

    opponent = parametric_fighter(arena, ego=1, seed=10_000 + seed)
    sim = FighterSim(arena=arena, seed=seed)
    builder = ReplayBuilder(
        arena=arena,
        config=f"d{arena.difficulty}",
        p1_policy="modal_trained_ppo",
        p2_policy="parametric",
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


def local_transfer_worker(payload: dict, seed: int) -> dict:
    """Credential-free local fallback: the HELD-OUT TRANSFER body, no Modal."""
    return _real_ppo_transfer_result(payload, seed)


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
        # ``replay`` is needed by the held-out transfer worker's replay-capture
        # branch (it builds a viewer-ready replay of the trained Player in-container).
        .add_local_python_source("games", "harness", "contracts", "replay")
    )

    @app.function(image=image, timeout=1800)
    def train_player(payload: dict, seed: int) -> dict:
        """REAL remote PPO Player training (no stub). Returns learning improvement.

        Same-config held-out (in-distribution) improvement — kept for the
        correlation gate / difficulty sweep that already consume it.
        """
        return _real_ppo_result(payload, seed)

    @app.function(image=image, timeout=1800)
    def train_player_transfer(payload: dict, seed: int) -> dict:
        """REAL remote PPO training, scored on the FIXED HELD-OUT reference set.

        The nested-RL Teacher reward worker: trains a Player on the Teacher's
        ``payload`` arena, measures before/after win-rate on the standard held-out
        benchmark (NOT the training arena), and returns the transfer improvement.
        """
        return _real_ppo_transfer_result(payload, seed)
