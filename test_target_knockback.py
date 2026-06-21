"""test_target_knockback.py — Target Knockback tests (Teacher-training game #2).

Target Knockback is the structurally-different game on the SAME engine: you score by
knocking the OPPONENT into a marked target zone (not by ringing them out, not by
holding a zone yourself). These tests cover the properties the game depends on,
mirroring ``test_koth.py``:

  * knock-in scoring is correct (you bank a point each tick the OPPONENT is in the
    target; you score off the OTHER player's position; ring-out does NOT eliminate —
    players are clamped, not killed),
  * an airborne opponent banks no knock-in time (JUMP is a real dodge),
  * a punch knocks the opponent INTO the target (the core scoring mechanic),
  * the scripted expert clearly beats random (so PPO has something to learn),
  * the sim is deterministic under a fixed seed,
  * the observation reuses the fighter's first-11 layout and appends the target dims,
  * the ``TargetKnockbackGameAdapter`` conforms to the FROZEN ``GameAdapter``
    interface and matches the fighter adapter's method signatures parameter-for-
    parameter (a drop-in), and scores scripted high / random low through the ONE
    frozen scorer,
  * the Gym env passes SB3's env checker.

These tests do NOT train PPO (kept fast). The learnability check lives in
prove_target_knockback_learns.py.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from contracts import ArenaSpec, CurriculumSpec
from games.fighter import STAGE1_ACTIONS as FIGHTER_ACTIONS
from games.fighter import Action
from games.target_knockback import (
    OBS_DIM,
    TargetKnockbackArena,
    TargetKnockbackSim,
    parametric_target_knockback,
    play_match,
    random_policy,
    scripted_target_knockback,
)
from harness.interfaces import GameAdapter
from harness.target_knockback_adapter import TargetKnockbackGameAdapter


# ---------------------------------------------------------------------------
# Same engine: same actions, no new action enum
# ---------------------------------------------------------------------------


def test_reuses_fighter_action_set():
    # Target Knockback must use the EXACT five fighter actions — the "same engine"
    # claim. It imports them; it does not define its own.
    from games.target_knockback import STAGE1_ACTIONS as TK_ACTIONS

    assert TK_ACTIONS is FIGHTER_ACTIONS
    assert TK_ACTIONS == (
        Action.IDLE,
        Action.LEFT,
        Action.RIGHT,
        Action.JUMP,
        Action.PUNCH,
    )


# ---------------------------------------------------------------------------
# Knock-in scoring (the NEW win condition: score off the OPPONENT's position)
# ---------------------------------------------------------------------------


def test_knockin_accrues_to_the_player_whose_opponent_is_in_target():
    # The score INVERSION vs KotH: player 0 scores when the OPPONENT (f1) sits in the
    # target. Park f1 dead-centre (always in target) and f0 far at the edge (never).
    # After N idle steps, player 0 (score0) has banked N, player 1 (score1) zero.
    arena = TargetKnockbackArena(platform_width=10.0, zone_center_frac=0.5, zone_half=1.5, max_steps=50)
    sim = TargetKnockbackSim(arena=arena, seed=0)
    sim.f1.x = 5.0   # opponent dead-centre == in target -> player 0 scores
    sim.f0.x = 0.1   # player 0 far left, out of target -> player 1 scores nothing
    n = 20
    for _ in range(n):
        sim.step(int(Action.IDLE), int(Action.IDLE))
    assert sim.score0 == n   # player 0 banked N (its opponent f1 was in the target)
    assert sim.score1 == 0   # player 1 banked nothing (f0 never in target)


def test_airborne_opponent_banks_no_knockin_time():
    # An opponent mid-jump is NOT "in target" even if its x is inside — JUMP is the
    # dodge. f1 in the target x but airborne should bank player 0 nothing that tick.
    arena = TargetKnockbackArena(platform_width=10.0, zone_center_frac=0.5, zone_half=2.0, max_steps=50)
    sim = TargetKnockbackSim(arena=arena, seed=0)
    sim.f1.x = 5.0
    sim.f0.x = 0.1
    # f1 JUMPs -> leaves the ground -> no knock-in even though its x is in the zone.
    sim.step(int(Action.IDLE), int(Action.JUMP))
    assert sim.f1.y > 0.0
    assert sim.score0 == 0


def test_winner_is_most_knockin_time():
    # f1 sits in the target (player 0 scores), f0 sits outside (player 1 scores
    # nothing): run to the budget -> player 0 wins.
    arena = TargetKnockbackArena(platform_width=10.0, zone_center_frac=0.5, zone_half=1.5, max_steps=30)
    sim = TargetKnockbackSim(arena=arena, seed=0)
    sim.f1.x = 5.0
    sim.f0.x = 0.1
    while not sim.done:
        sim.step(int(Action.IDLE), int(Action.IDLE))
    assert sim.score0 > sim.score1
    assert sim.winner == 0
    assert sim.done


def test_equal_knockin_time_is_a_draw():
    # Both players outside the target the whole match -> 0 == 0 -> draw (None).
    arena = TargetKnockbackArena(platform_width=10.0, zone_center_frac=0.5, zone_half=1.0, max_steps=20)
    sim = TargetKnockbackSim(arena=arena, seed=0)
    sim.f0.x = 0.1   # far left, outside target
    sim.f1.x = 9.9   # far right, outside target
    while not sim.done:
        sim.step(int(Action.IDLE), int(Action.IDLE))
    assert sim.score0 == sim.score1
    assert sim.winner is None


def test_no_ringout_player_is_clamped_not_eliminated():
    # The key shared difference from the fighter: walking off the edge does NOT
    # eliminate you. Drive f0 hard left for many steps; it must stay alive and be
    # clamped to x >= 0, never ringing out.
    arena = TargetKnockbackArena(platform_width=10.0, max_steps=100)
    sim = TargetKnockbackSim(arena=arena, seed=0)
    for _ in range(60):
        sim.step(int(Action.LEFT), int(Action.IDLE))
    assert sim.f0.alive            # never eliminated
    assert 0.0 <= sim.f0.x <= arena.platform_width  # clamped onto the platform


def test_ringout_off_edge_does_not_end_match_or_kill():
    # Even a body knocked / walked far past the edge stays in play and the match
    # runs the full budget (no early termination from leaving [0, width]).
    arena = TargetKnockbackArena(platform_width=10.0, max_steps=40)
    sim = TargetKnockbackSim(arena=arena, seed=0)
    sim.f1.x = 9.95
    steps = 0
    while not sim.done:
        sim.step(int(Action.IDLE), int(Action.RIGHT))  # shove f1 off the right edge
        steps += 1
    assert sim.f1.alive
    assert sim.f0.alive
    assert steps == arena.max_steps  # ran to the budget, no ring-out early end


def test_punch_knocks_opponent_into_target():
    # The core scoring mechanic: stand on the far side of the opponent from the zone
    # and punch -> knockback drives the opponent INTO the target. Zone centred at 5;
    # opponent at 7 (right of zone), attacker at 8 facing left, in range -> punch
    # pushes opponent left into the zone.
    arena = TargetKnockbackArena(
        platform_width=10.0, zone_center_frac=0.5, zone_half=1.5,
        knockback=2.5, punch_range=1.4, max_steps=50,
    )
    sim = TargetKnockbackSim(arena=arena, seed=0)
    sim.f0.x = 8.0
    sim.f0.facing = -1
    sim.f1.x = 7.0  # right of zone [3.5, 6.5], in range, in front of f0
    sim.f1.facing = 1
    assert not arena.in_target(sim.f1)
    sim.step(int(Action.PUNCH), int(Action.IDLE))
    # knockback 2.5 * facing(-1) -> f1 7.0 - 2.5 = 4.5, inside the zone [3.5, 6.5].
    assert arena.in_target(sim.f1)
    assert sim.score0 == 1     # player 0 banked the knock-in this tick
    assert sim.f1.alive        # still in play (no ring-out)


# ---------------------------------------------------------------------------
# Scripted beats random
# ---------------------------------------------------------------------------


def test_scripted_beats_random_decisively():
    # The scripted expert (player 0) must clearly out-knock a random opponent.
    arena = TargetKnockbackArena(difficulty=0.3)
    wins = 0
    n = 60
    for s in range(n):
        winner = play_match(
            arena,
            scripted_target_knockback(arena, ego=0),
            random_policy(seed=10_000 + s),
            seed=s,
        )
        if winner == 0:
            wins += 1
    assert wins / n >= 0.8, f"scripted only won {wins}/{n}"


def test_random_agent_rarely_beats_scripted_opponent():
    # The PPO 'before' baseline relies on a flailing agent NOT reliably out-knocking
    # a competent scripted opponent, so any post-training win-rate is real learning.
    arena = TargetKnockbackArena(difficulty=0.3)
    agent_wins = 0
    n = 60
    for s in range(n):
        winner = play_match(
            arena,
            random_policy(seed=20_000 + s),
            scripted_target_knockback(arena, ego=1),
            seed=s,
        )
        if winner == 0:
            agent_wins += 1
    assert agent_wins <= 3, f"random agent unexpectedly won {agent_wins}/{n}"


def test_scripted_beats_random_at_every_difficulty():
    # The win-rate is decisive across the dial against a (difficulty-blind) random
    # opponent — scripted should dominate everywhere.
    n = 40
    for d in (0.0, 0.3, 0.6, 1.0):
        arena = TargetKnockbackArena(difficulty=d)
        wins = 0
        for s in range(n):
            winner = play_match(
                arena,
                scripted_target_knockback(arena, ego=0),
                random_policy(seed=10_000 + s),
                seed=s,
            )
            if winner == 0:
                wins += 1
        assert wins / n >= 0.8, f"difficulty {d}: scripted only won {wins}/{n}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_match_deterministic_under_fixed_seed():
    arena = TargetKnockbackArena(difficulty=0.3)
    w1 = play_match(arena, scripted_target_knockback(arena, 0), random_policy(seed=7), seed=3)
    w2 = play_match(arena, scripted_target_knockback(arena, 0), random_policy(seed=7), seed=3)
    assert w1 == w2


def test_parametric_opponent_deterministic_under_fixed_seed():
    arena = TargetKnockbackArena(difficulty=0.7)
    w1 = play_match(
        arena, scripted_target_knockback(arena, 0),
        parametric_target_knockback(arena, 1, seed=5), seed=2,
    )
    w2 = play_match(
        arena, scripted_target_knockback(arena, 0),
        parametric_target_knockback(arena, 1, seed=5), seed=2,
    )
    assert w1 == w2


def test_sim_trajectory_and_score_deterministic():
    arena = TargetKnockbackArena(difficulty=0.3)

    def run(seed: int):
        sim = TargetKnockbackSim(arena=arena, seed=seed)
        scripted = scripted_target_knockback(arena, ego=0)
        opp = random_policy(seed=7)
        traj = []
        while not sim.done:
            a0 = scripted(sim.observe(0))
            a1 = opp(sim.observe(1))
            sim.step(a0, a1)
            traj.append((round(sim.f0.x, 6), round(sim.f1.x, 6)))
        return traj, sim.score0, sim.score1

    assert run(5) == run(5)


def test_different_seeds_can_differ():
    arena = TargetKnockbackArena()
    s0 = TargetKnockbackSim(arena=arena, seed=0)
    s1 = TargetKnockbackSim(arena=arena, seed=99)
    # Spawn jitter is the fighter's seeded logic, so different seeds give different
    # starts (at least sometimes).
    assert (s0.f0.x, s0.f1.x) != (s1.f0.x, s1.f1.x)


# ---------------------------------------------------------------------------
# Observation shape (reuses fighter layout + appends target dims)
# ---------------------------------------------------------------------------


def test_observation_shape_reuses_fighter_layout():
    from games.fighter import FighterSim
    from games.fighter import OBS_DIM as FIGHTER_OBS_DIM

    arena = TargetKnockbackArena()
    sim = TargetKnockbackSim(arena=arena, seed=0)
    obs0 = sim.observe(ego=0)
    obs1 = sim.observe(ego=1)
    assert obs0.shape == (OBS_DIM,)
    assert obs1.shape == (OBS_DIM,)
    assert obs0.dtype == np.float32
    assert np.all(np.isfinite(obs0))
    # The first 11 dims must be the fighter's physical observation, byte-for-byte.
    fighter_obs = FighterSim(arena=arena._as_fighter_arena(), seed=0).observe(ego=0)
    assert OBS_DIM == FIGHTER_OBS_DIM + 5
    assert np.allclose(obs0[:FIGHTER_OBS_DIM], fighter_obs)


def test_obs_opp_in_target_flag_matches_geometry():
    arena = TargetKnockbackArena(platform_width=10.0, zone_center_frac=0.5, zone_half=1.5)
    sim = TargetKnockbackSim(arena=arena, seed=0)
    sim.f0.x = 0.1   # ego (player 0) out of target
    sim.f1.x = 5.0   # opponent in target -> ego scores
    obs0 = sim.observe(ego=0)
    # index 13 == opp_in_target (the quantity ego drives up), 14 == ego_in_target.
    assert obs0[13] == 1.0   # opponent is in the target
    assert obs0[14] == 0.0   # ego is not


# ---------------------------------------------------------------------------
# Adapter conforms to the frozen GameAdapter interface — DROP-IN with fighter
# ---------------------------------------------------------------------------


def test_adapter_is_a_gameadapter_subclass():
    assert issubclass(TargetKnockbackGameAdapter, GameAdapter)
    adapter = TargetKnockbackGameAdapter()
    assert isinstance(adapter, GameAdapter)
    for name in ("build", "scripted_expert", "random_policy", "evaluate"):
        assert callable(getattr(adapter, name))
    assert not getattr(TargetKnockbackGameAdapter, "__abstractmethods__", frozenset())


def test_adapter_signature_matches_fighter_adapter():
    # Drop-in means the same call shapes as the fighter adapter, parameter-for-
    # parameter — so the SAME score_curriculum / runner / trainer drive it.
    from harness.fighter_adapter import FighterGameAdapter

    for name in ("build", "scripted_expert", "random_policy", "evaluate"):
        tk_sig = inspect.signature(getattr(TargetKnockbackGameAdapter, name))
        fighter_sig = inspect.signature(getattr(FighterGameAdapter, name))
        assert list(tk_sig.parameters) == list(fighter_sig.parameters), name


def test_adapter_scores_scripted_high_random_low():
    # Learnable-band difficulty where the gap is wide (see the prove script).
    adapter = TargetKnockbackGameAdapter(eval_seeds=20)
    arenas = adapter.arenas_from_configs(
        [TargetKnockbackArena(platform_width=12.0, zone_half=1.6, difficulty=0.55)],
        curriculum_id="t",
    )
    strong = adapter.evaluate(adapter.scripted_expert(), arenas)
    weak = adapter.evaluate(adapter.random_policy(), arenas)
    assert "scripted_expert" in strong
    assert "random_policy" in weak
    assert strong["scripted_expert"]["mean_score"] > weak["random_policy"]["mean_score"]
    assert strong["scripted_expert"]["mean_score"] >= 0.8
    assert weak["random_policy"]["mean_score"] <= 0.6


def test_adapter_build_from_curriculum_spec():
    # The contract path: build() accepts a gridworld-shaped CurriculumSpec and
    # returns one Target-Knockback arena handle per ArenaSpec.
    spec = CurriculumSpec(
        game_id="target_knockback",
        curriculum_id="c1",
        arenas=[
            ArenaSpec(map_size=9, doors=0, keys=0, hazard_density=0.0, difficulty=0.2),
            ArenaSpec(map_size=15, doors=0, keys=0, hazard_density=0.0, difficulty=0.8),
        ],
    )
    adapter = TargetKnockbackGameAdapter(eval_seeds=10)
    arenas = adapter.build(spec)
    assert len(arenas) == 2
    # Higher difficulty -> SMALLER target (harder to knock into) — the mapping.
    assert arenas[1].arena.zone_half < arenas[0].arena.zone_half


def test_adapter_evaluate_through_real_scorer():
    # The ONE frozen scorer must run end-to-end on the real adapter — exactly as it
    # does on the fighter. Use a learnable-band difficulty so the gap is real.
    from harness.scoring import score_curriculum

    spec = CurriculumSpec(
        game_id="target_knockback",
        curriculum_id="score-test",
        arenas=[ArenaSpec(map_size=12, doors=0, keys=0, hazard_density=0.0, difficulty=0.5)],
    )
    adapter = TargetKnockbackGameAdapter(eval_seeds=25)
    result = score_curriculum(spec, adapter, mode="gap_proxy")
    assert result.strong_score >= result.weak_score
    assert result.p == result.weak_score
    assert 0.0 <= result.reward <= 1.0
    assert result.curriculum_id == "score-test"
    assert result.mode == "gap_proxy"


def test_difficulty_slides_random_winrate_not_a_cliff():
    # "Distinct learning outcomes": the solvability proxy (random win-rate) slides
    # smoothly across the difficulty dial rather than snapping 1->0. Check a low,
    # a mid-band, and a high difficulty give a wide, monotone-ish spread.
    adapter = TargetKnockbackGameAdapter(eval_seeds=20)

    def random_p(d: float) -> float:
        arenas = adapter.arenas_from_configs(
            [TargetKnockbackArena(platform_width=12.0, zone_half=1.6, difficulty=d)],
            curriculum_id=f"g-{d}",
        )
        return adapter.evaluate(adapter.random_policy(), arenas)["random_policy"]["mean_score"]

    p_low = random_p(0.0)
    p_mid = random_p(0.5)
    p_high = random_p(1.0)
    # Low difficulty: random wins a lot; high difficulty: random rarely wins.
    assert p_low >= 0.8
    assert p_high <= 0.3
    # A genuine slide: the mid value sits strictly between, with a wide total span.
    assert p_high < p_mid < p_low
    assert (p_low - p_high) >= 0.5


# ---------------------------------------------------------------------------
# Gym env conforms to SB3 (same wiring path as the fighter -> PPO drop-in)
# ---------------------------------------------------------------------------


def test_gym_env_passes_sb3_checker():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env

    from games.target_knockback import TargetKnockbackEnv

    arena = TargetKnockbackArena(difficulty=0.3)
    env = TargetKnockbackEnv(
        arena, opponent_factory=lambda a: scripted_target_knockback(a, ego=1), seed=0
    )
    check_env(env, warn=True)
