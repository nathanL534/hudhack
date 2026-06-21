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


# ---------------------------------------------------------------------------
# STUDENT-vs-STUDENT head-to-head — the PRIMARY metric machinery
# ---------------------------------------------------------------------------
#
# The held-out transfer worker above (``train_player_transfer``) is the SECONDARY
# "fixed-bot before->after" diagnostic. The PRIMARY metric is a DIRECT head-to-head
# between two FROZEN Students: one trained on the BASE Teacher's curricula, one on
# the TRAINED Teacher's, fought on UNSEEN arenas with SIDE SWAPS. That needs two
# new worker bodies:
#
#   1. ``_train_student_policy`` — train ONE fresh Student on a whole curriculum
#      SET (a list of arenas the Teacher generated), serialize its policy, return
#      a tagged ``PolicyArtifact`` dict. NO scoring vs a fixed bot here — the
#      Student is graded later, by fighting the OTHER Student.
#   2. ``_head_to_head_match`` — restore two frozen Students and fight them on ONE
#      held-out arena, BOTH side assignments (trained-as-P1, then trained-as-P2),
#      over fresh seeds, returning win/loss/draw per side. No retraining per match.
#
# Both are game-agnostic: ``payload["game"]`` selects fighter (Ring-Out) or koth
# (King of the Hill). Ring-Out and KOTH have DIFFERENT obs spaces (11 vs 16) but
# the SAME 5 actions and the SAME ``play_match(arena, p0, p1, seed)`` PvP
# primitive, so a Student NEVER crosses games — each game trains its own Students
# and fights them only against same-game Students. That is the correct cross-game
# design (fresh KOTH Students from KOTH-mapped curricula), not forcing weights.


def _game_modules(game: str):
    """Return (arena_cls, play_match, obs_dim, env_cls_factory) for a game.

    ``env_cls_factory(arenas, seed)`` builds a single-agent training env over the
    arena list (used to (a) train a Student and (b) provide the obs/action spaces a
    frozen policy is restored into). All three games expose the SAME ``play_match``
    PvP signature, so head-to-head is identical across games.
    """
    if game in ("fighter", "ring-out-duel", "ring_out_duel"):
        from games.fighter import OBS_DIM, FighterArena, play_match
        from harness.ppo_trainer import _MultiArenaFighterEnv

        return FighterArena, play_match, OBS_DIM, _MultiArenaFighterEnv
    if game in ("koth", "king-of-the-hill"):
        from games.koth import OBS_DIM, KothArena, play_match
        from harness.koth_trainer import _MultiArenaKothEnv

        return KothArena, play_match, OBS_DIM, _MultiArenaKothEnv
    if game in ("target_knockback", "target-knockback", "tk"):
        # OBS_DIM = 16 (11 fighter dims + 5 target-zone dims). The TK env is the
        # structural twin of the fighter / KotH training env (re-seeded parametric
        # opponent per episode), so a fresh TK Student trains on each Teacher's TK
        # curricula and fights via the SAME ``play_match`` with side-swaps — exactly
        # like fighter / KotH, just the TK game.
        from games.target_knockback import OBS_DIM, TargetKnockbackArena, play_match
        from harness.target_knockback_trainer import _MultiArenaTkEnv

        return TargetKnockbackArena, play_match, OBS_DIM, _MultiArenaTkEnv
    raise ValueError(
        f"unknown game {game!r} for head-to-head "
        "(use 'fighter', 'koth' or 'target_knockback')"
    )


def _arenas_from_specs(specs: list[dict], arena_cls):
    """Build concrete arena objects from a list of param-dict specs.

    Missing knobs fall back to the arena class's own defaults, so a spec of just
    ``{"difficulty": 0.6}`` is valid. Only keys the dataclass accepts are passed
    (so a fighter spec with a stray ``map_size`` does not crash, and a KOTH spec's
    zone dials are honoured).
    """
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(arena_cls)}
    out = []
    for s in specs:
        kw = {k: float(v) for k, v in s.items() if k in field_names}
        out.append(arena_cls(**kw))
    return out


def _load_prior_student_artifact(prior_student_path) -> dict | None:
    """Load a prior-Student ``PolicyArtifact`` dict from disk (no torch/SB3 needed).

    Returns the raw artifact DICT, or ``None`` when no path is given OR the file is
    absent — in which case ``resolve_league`` simply drops the ``prior_student`` league
    entry, so the default styles-only behavior is unaffected. The dict is turned into a
    runnable ``Policy`` later by ``_restore_prior_student_policy`` (which needs an env).
    """
    if not prior_student_path:
        return None
    import json
    import os

    if not os.path.exists(prior_student_path):
        return None
    with open(prior_student_path) as fh:
        return json.load(fh)


def _restore_prior_student_policy(artifact: dict | None, arena_cls, arenas, seed: int):
    """Restore a prior-Student artifact into an ``obs -> action`` ``Policy`` callable.

    ``games.opponents_league.make_opponent`` types ``prior_student`` as a Policy (a
    callable), not an artifact dict — so the frozen earlier Student must be restored
    BEFORE it is threaded into the trainer. A throwaway fighter env (built from the
    league arenas) supplies the obs/action spaces ``restore_policy`` loads the weights
    into. Returns ``None`` when there is no artifact (the league then has no
    ``prior_student`` entry, so this policy is never selected).
    """
    if artifact is None:
        return None
    from harness.ppo_trainer import _MultiArenaFighterEnv
    from output.stage6.policy import DEFAULT_NET_ARCH, restore_policy

    restore_env = _MultiArenaFighterEnv(arenas, seed=int(seed))
    return restore_policy(
        artifact, restore_env, seed=int(seed)  # net_arch read from artifact
    )


def _train_student_policy(payload: dict, seed: int) -> dict:
    """Train ONE fresh Student on a whole curriculum SET; return a frozen artifact.

    The Student trains (SB3 PPO, the SAME budget knobs as the transfer worker) on
    the LIST of arenas in ``payload["curriculum_arenas"]`` — the arenas the Teacher
    generated for this replicate. It is then serialized to a ``PolicyArtifact`` dict
    tagged with (teacher, curriculum_id, seed, game). NO fixed-bot scoring: the
    Student is graded ONLY by the later head-to-head, so this worker cannot leak the
    training opponent into the primary metric.
    """
    import warnings

    warnings.filterwarnings("ignore")

    from contracts import PlayerConfig, TrainingBudget
    from harness.ppo_trainer import PPOPlayerTrainer
    from output.stage6.policy import DEFAULT_NET_ARCH, serialize_policy

    game = str(payload.get("game", "fighter"))
    # The trainer builds its own env; we need the arena class + obs_dim only.
    arena_cls, _play, obs_dim, _env_cls = _game_modules(game)

    episodes = int(payload.get("ppo_episodes", 1000))
    eval_seeds = int(payload.get("eval_seeds", 50))
    teacher = str(payload.get("teacher", ""))
    curriculum_id = str(payload.get("curriculum_id", "curriculum"))
    # Stage-6 Student net size (default [64,64]). Bigger nets let the Student actually
    # learn decisive play instead of plateauing at "stand still". Applied IDENTICALLY
    # to base and trained Students (it rides in the fairness-invariant payload), and
    # recorded on the artifact so head-to-head restores at the matching width.
    net_arch = tuple(int(h) for h in (payload.get("net_arch") or DEFAULT_NET_ARCH))

    specs = payload.get("curriculum_arenas") or [payload]
    arenas = _arenas_from_specs(specs, arena_cls)
    if not arenas:
        raise ValueError("no curriculum arenas to train the Student on")

    config = PlayerConfig(
        architecture=str(payload.get("architecture", "mlp")),
        num_seeds=1,
        budget=TrainingBudget(episodes=episodes),
        seed=int(seed),
        modal_parallel=False,
    )

    # OPPONENT LEAGUE (fighter Student path ONLY, default OFF). Resolve here so the
    # Teacher inner-reward workers (which never set these payload keys) are untouched.
    # ``active_league`` is None unless the payload explicitly opts in.
    active_league = None
    prior_student_artifact = None
    if game in ("fighter", "ring-out-duel", "ring_out_duel") and payload.get(
        "opponent_league_enabled"
    ):
        from games.opponents_league import DEFAULT_LEAGUE, resolve_league

        prior_student_artifact = _load_prior_student_artifact(
            payload.get("prior_student_path")
        )
        league_spec = payload.get("opponent_league") or DEFAULT_LEAGUE
        active_league = resolve_league(
            league_spec, has_prior_student=prior_student_artifact is not None
        )

    prior_student = None  # only the fighter league branch restores one
    if game in ("koth", "king-of-the-hill"):
        from harness.koth_trainer import KothPlayerTrainer
        from harness.koth_adapter import KothGameAdapter

        adapter = KothGameAdapter(eval_seeds=eval_seeds)
        trainer = KothPlayerTrainer(eval_seeds=eval_seeds, net_arch=net_arch)
    elif game in ("target_knockback", "target-knockback", "tk"):
        from harness.target_knockback_trainer import TargetKnockbackPlayerTrainer
        from harness.target_knockback_adapter import TargetKnockbackGameAdapter

        adapter = TargetKnockbackGameAdapter(eval_seeds=eval_seeds)
        trainer = TargetKnockbackPlayerTrainer(eval_seeds=eval_seeds)
    else:
        from harness.fighter_adapter import FighterGameAdapter

        adapter = FighterGameAdapter(eval_seeds=eval_seeds)
        # Restore the frozen prior Student into a runnable Policy (only when the
        # league is active AND a prior-student artifact was supplied). ``make_opponent``
        # expects a callable, not an artifact dict — so restore it here.
        if active_league is not None:
            prior_student = _restore_prior_student_policy(
                prior_student_artifact, arena_cls, arenas, int(seed)
            )
        # Explicit Stage-6-only anti-camping settings. The nested Teacher reward
        # workers instantiate PPOPlayerTrainer with its original defaults.
        # ``opponent_league`` defaults to None => the ORIGINAL single-parametric
        # opponent training env, unchanged.
        trainer = PPOPlayerTrainer(
            eval_seeds=eval_seeds,
            ent_coef=float(payload.get("student_ent_coef", 0.03)),
            min_timesteps=int(payload.get("student_min_timesteps", 60_000)),
            anti_camping_reward=bool(payload.get("student_anti_camping", True)),
            opponent_league=active_league,
            prior_student=prior_student,
            net_arch=net_arch,
        )

    train_arenas = adapter.arenas_from_configs(arenas, curriculum_id=curriculum_id)
    job = trainer.submit(config, train_arenas)
    train_result = job.result()

    # Audit which opponent styles the Student actually trained against (empty when
    # the league is OFF — the original single-parametric path records nothing).
    opp_counts = dict(getattr(job, "opp_counts", {}) or {})

    artifact = serialize_policy(
        job.model,  # both trainers attach the SB3 model to the finished job
        teacher=teacher,
        curriculum_id=curriculum_id,
        seed=int(seed),
        game=game,
        obs_dim=int(obs_dim),
        net_arch=net_arch,
        extra={"train_arena_winrate": float(train_result.mean_score),
               "n_curriculum_arenas": len(arenas),
               "opponent_league_enabled": active_league is not None,
               "opponent_ids": sorted(opp_counts),
               "opponent_episode_counts": opp_counts},
    )
    return {
        "status": "student_policy",
        "seed": int(seed),
        "teacher": teacher,
        "curriculum_id": curriculum_id,
        "game": game,
        "train_arena_winrate": float(train_result.mean_score),
        "opponent_league_enabled": active_league is not None,
        "opponent_ids": sorted(opp_counts),
        "opponent_episode_counts": opp_counts,
        "policy": artifact.as_dict(),
    }


def _head_to_head_match(payload: dict, seed: int) -> dict:
    """Fight two FROZEN Students on ONE held-out arena, BOTH side assignments.

    ``payload`` carries:
      * ``trained_policy`` / ``base_policy`` — two ``PolicyArtifact`` dicts.
      * ``arena`` — the held-out arena spec (param dict).
      * ``game`` — fighter | koth.
      * ``match_seeds`` — the fresh seeds to play per side (paired across sides).

    For each match seed it plays TWO matches with IDENTICAL stochastic conditions:
      (A) trained Student as P0 vs base Student as P1,
      (B) base Student as P0 vs trained Student as P1   (the SIDE SWAP).
    A side-independent advantage shows up only if the trained Student wins from
    BOTH sides. Returns per-seed, per-side outcomes from the TRAINED Student's
    point of view (win / loss / draw), never aggregated here — the curriculum-level
    aggregation happens upstream so the CI is over independent replicates.
    """
    import warnings

    warnings.filterwarnings("ignore")

    from output.stage6.policy import DEFAULT_NET_ARCH, PolicyArtifact, restore_policy

    game = str(payload.get("game", "fighter"))
    arena_cls, play_match, obs_dim, env_cls = _game_modules(game)

    arena = _arenas_from_specs([payload["arena"]], arena_cls)[0]
    if game in ("fighter", "ring-out-duel", "ring_out_duel"):
        import dataclasses
        arena = dataclasses.replace(arena, decisive_timeout=True)
    match_seeds = [int(s) for s in payload.get("match_seeds", [seed])]

    # A throwaway env only supplies obs/action spaces to restore the frozen nets.
    restore_env = env_cls([arena], seed=int(seed))
    trained = restore_policy(
        PolicyArtifact.from_dict(payload["trained_policy"]),
        restore_env, seed=int(seed),  # net_arch read from artifact
    )
    base = restore_policy(
        PolicyArtifact.from_dict(payload["base_policy"]),
        restore_env, seed=int(seed) + 1,  # net_arch read from artifact
    )

    def _outcome_for_trained(winner, trained_is_p0: bool) -> str:
        if winner is None:
            return "draw"
        trained_won = (winner == 0) == trained_is_p0
        return "win" if trained_won else "loss"

    per_seed = []
    for ms in match_seeds:
        # SIDE A: trained as P0, base as P1 — matched seed.
        w_a = play_match(arena, trained, base, seed=ms)
        # SIDE B: base as P0, trained as P1 — SAME seed, identical stochastic conds.
        w_b = play_match(arena, base, trained, seed=ms)
        per_seed.append({
            "match_seed": ms,
            "trained_p0": _outcome_for_trained(w_a, trained_is_p0=True),
            "trained_p1": _outcome_for_trained(w_b, trained_is_p0=False),
            "raw_winner_p0side": w_a,
            "raw_winner_p1side": w_b,
        })

    out = {
        "status": "head_to_head",
        "seed": int(seed),
        "game": game,
        "arena": payload["arena"],
        "trained_checksum": payload["trained_policy"].get("checksum"),
        "base_checksum": payload["base_policy"].get("checksum"),
        "per_seed": per_seed,
    }

    # Optional replay capture: roll out ONE representative match per requested
    # outcome category (trained win as P0, base win, draw, side-swapped) so the
    # demo can play the actual Student-vs-Student match. Best-effort; never gates.
    if payload.get("capture_replays"):
        out["replays"] = _capture_h2h_replays(
            game, arena, trained, base, match_seeds, payload,
        )
    return out


def _capture_h2h_replays(game, arena, trained, base, match_seeds, payload) -> list[dict]:
    """Capture representative Student-vs-Student replays with full metadata.

    One replay each (when available among the played seeds): a trained-Student win
    (trained as P0), a base-Student win, a side-swapped match (trained as P1), and a
    draw. Each carries source-Teacher / curriculum / policy-seed / arena / side /
    outcome metadata. Reuses ``record_replay.record_match`` (the single home for the
    sim+capture loop) so capture stays byte-identical to the canonical recorder.
    Only the fighter has a viewer replay; KOTH / Target Knockback do not fabricate
    frames."""
    if game not in ("fighter", "ring-out-duel", "ring_out_duel"):
        # KOTH / TK have no fighter-style viewer replay; do not fabricate — and do
        # NOT import the fighter-only recorder, so a non-fighter head-to-head never
        # depends on ``record_replay`` being on the path.
        return []

    from record_replay import record_match

    tp = payload["trained_policy"]
    bp = payload["base_policy"]
    want = {"trained_win": None, "base_win": None, "draw": None, "side_swap": None}
    # record_match returns the replay dict; meta.winner is "p1"(P0) / "p2"(P1) / "draw".
    _winner_of = {"p1": 0, "p2": 1, "draw": None}

    def _roll(p0, p1, seed, p0_label, p1_label):
        data = record_match(arena, p0, p1, config=f"h2h-{game}",
                            p1_policy=p0_label, p2_policy=p1_label, seed=seed)
        return data, _winner_of[data["meta"]["winner"]]

    for ms in match_seeds:
        # trained as P0 vs base as P1.
        data_a, w_a = _roll(trained, base, ms, "trained_student", "base_student")
        meta_common = {
            "source_teacher_trained": tp.get("teacher"), "source_teacher_base": bp.get("teacher"),
            "trained_curriculum_id": tp.get("curriculum_id"), "base_curriculum_id": bp.get("curriculum_id"),
            "trained_policy_seed": tp.get("seed"), "base_policy_seed": bp.get("seed"),
            "arena": payload["arena"], "match_seed": ms,
        }
        if w_a == 0 and want["trained_win"] is None:
            want["trained_win"] = {**meta_common, "side": "trained_as_P0", "outcome": "trained_win", "data": data_a}
        elif w_a == 1 and want["base_win"] is None:
            want["base_win"] = {**meta_common, "side": "trained_as_P0", "outcome": "base_win", "data": data_a}
        elif w_a is None and want["draw"] is None:
            want["draw"] = {**meta_common, "side": "trained_as_P0", "outcome": "draw", "data": data_a}
        # side swap: trained as P1.
        if want["side_swap"] is None:
            data_b, w_b = _roll(base, trained, ms, "base_student", "trained_student")
            want["side_swap"] = {**meta_common, "side": "trained_as_P1",
                                 "outcome": ("trained_win" if w_b == 1 else
                                             "base_win" if w_b == 0 else "draw"),
                                 "data": data_b}
        if all(v is not None for v in want.values()):
            break

    return [v for v in want.values() if v is not None]


def local_train_student_worker(payload: dict, seed: int) -> dict:
    """Credential-free local fallback: train a frozen Student policy, no Modal."""
    return _train_student_policy(payload, seed)


def local_head_to_head_worker(payload: dict, seed: int) -> dict:
    """Credential-free local fallback: fight two frozen Students, no Modal."""
    return _head_to_head_match(payload, seed)


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
        # ``output`` is needed by the Student-vs-Student workers (output.stage6.policy
        # serialize/restore). ``games.koth`` / ``games.target_knockback`` ride in via
        # ``games``. ``record_replay`` is the top-level recorder the FIGHTER head-to-head
        # replay-capture imports (``_capture_h2h_replays``); without it a fighter
        # capture crashes with ModuleNotFoundError. Non-fighter games never import it.
        .add_local_python_source(
            "games", "harness", "contracts", "replay", "output", "record_replay"
        )
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

        SECONDARY metric (fixed-bot before->after). Kept BACKWARD-COMPATIBLE so the
        Stage-6 secondary transfer diagnostic still runs unchanged.
        """
        return _real_ppo_transfer_result(payload, seed)

    @app.function(image=image, timeout=1800)
    def train_student_policy(payload: dict, seed: int) -> dict:
        """PRIMARY metric: train ONE fresh Student on a curriculum SET; return its
        FROZEN policy artifact (tagged with Teacher / curriculum / seed / game).

        No fixed-bot scoring — the Student is graded later by the head-to-head, so
        the training opponent cannot leak into the primary metric.
        """
        return _train_student_policy(payload, seed)

    @app.function(image=image, timeout=900)
    def head_to_head_match(payload: dict, seed: int) -> dict:
        """PRIMARY metric: fight two FROZEN Students on ONE held-out arena, BOTH
        side assignments (side-swap), over fresh seeds. NO retraining per match.
        """
        return _head_to_head_match(payload, seed)
