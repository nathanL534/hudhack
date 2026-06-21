"""test_koth.py — King-of-the-Hill tests (the held-out game).

KotH is the structurally-different game on the SAME engine. These tests cover the
properties the transfer claim depends on, mirroring ``test_fighter.py``:

  * zone scoring is correct (zone-time accumulates only while occupying the zone;
    the winner is the most zone-time; NO ring-out — players are clamped, not
    eliminated),
  * the scripted KotH expert clearly beats random (so PPO has something to learn),
  * the sim is deterministic under a fixed seed,
  * the observation reuses the fighter's first-11 layout and appends the zone dims,
  * the ``KothGameAdapter`` conforms to the FROZEN ``GameAdapter`` interface and
    the frozen scoring shape (scripted scores high, random scores low) — i.e. it
    is a drop-in for the fighter,
  * the Gym env passes SB3's env checker.

These tests do NOT train PPO (kept fast). The learnability check lives in
prove_koth_learns.py.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from contracts import ArenaSpec, CurriculumSpec
from games.fighter import STAGE1_ACTIONS as FIGHTER_ACTIONS
from games.fighter import Action
from games.koth import (
    OBS_DIM,
    KothArena,
    KothSim,
    play_match,
    random_policy,
    scripted_koth,
)
from harness.interfaces import GameAdapter
from harness.koth_adapter import KothGameAdapter


# ---------------------------------------------------------------------------
# Same engine: same actions, no new action enum
# ---------------------------------------------------------------------------


def test_koth_reuses_fighter_action_set():
    # KotH must use the EXACT five fighter actions — that is the "same engine"
    # claim. It imports them; it does not define its own.
    from games.koth import STAGE1_ACTIONS as KOTH_ACTIONS

    assert KOTH_ACTIONS is FIGHTER_ACTIONS
    assert KOTH_ACTIONS == (
        Action.IDLE,
        Action.LEFT,
        Action.RIGHT,
        Action.JUMP,
        Action.PUNCH,
    )


# ---------------------------------------------------------------------------
# Zone scoring (the NEW win condition)
# ---------------------------------------------------------------------------


def test_zone_time_accumulates_only_while_in_zone():
    # Center zone; park f0 dead-centre (always in zone) and f1 far at the edge
    # (never in zone). After N idle steps, f0 has banked N zone-time, f1 zero.
    arena = KothArena(platform_width=10.0, zone_center_frac=0.5, zone_half=1.5, max_steps=50)
    sim = KothSim(arena=arena, seed=0)
    sim.f0.x = 5.0   # platform centre == zone centre
    sim.f1.x = 0.1   # far left, outside the zone [3.5, 6.5]
    n = 20
    for _ in range(n):
        sim.step(int(Action.IDLE), int(Action.IDLE))
    assert sim.zone_time0 == n
    assert sim.zone_time1 == 0


def test_airborne_player_banks_no_zone_time():
    # Standing in the zone but mid-jump => not occupying. JUMP off the ground in
    # the zone should NOT bank zone-time that tick (occupancy requires grounded).
    arena = KothArena(platform_width=10.0, zone_center_frac=0.5, zone_half=2.0, max_steps=50)
    sim = KothSim(arena=arena, seed=0)
    sim.f0.x = 5.0
    sim.f1.x = 0.1
    # First tick: JUMP -> leaves the ground -> no zone-time even though x is in zone.
    sim.step(int(Action.JUMP), int(Action.IDLE))
    assert sim.f0.y > 0.0           # airborne
    assert sim.zone_time0 == 0      # banked nothing while airborne


def test_winner_is_most_zone_time():
    # f0 sits in the zone, f1 sits outside, run to the budget: f0 must win.
    arena = KothArena(platform_width=10.0, zone_center_frac=0.5, zone_half=1.5, max_steps=30)
    sim = KothSim(arena=arena, seed=0)
    sim.f0.x = 5.0
    sim.f1.x = 0.1
    while not sim.done:
        sim.step(int(Action.IDLE), int(Action.IDLE))
    assert sim.zone_time0 > sim.zone_time1
    assert sim.winner == 0
    assert sim.done


def test_equal_zone_time_is_a_draw():
    # Both players outside the zone the whole match -> 0 == 0 -> draw (None).
    arena = KothArena(platform_width=10.0, zone_center_frac=0.5, zone_half=1.0, max_steps=20)
    sim = KothSim(arena=arena, seed=0)
    sim.f0.x = 0.1   # far left, outside zone
    sim.f1.x = 9.9   # far right, outside zone
    while not sim.done:
        sim.step(int(Action.IDLE), int(Action.IDLE))
    assert sim.zone_time0 == sim.zone_time1
    assert sim.winner is None


def test_no_ringout_player_is_clamped_not_eliminated():
    # KotH's key difference from the fighter: walking off the edge does NOT
    # eliminate you. Drive f0 hard left for many steps; it must stay alive and be
    # clamped to x >= 0, not ring out.
    arena = KothArena(platform_width=10.0, max_steps=100)
    sim = KothSim(arena=arena, seed=0)
    for _ in range(60):
        sim.step(int(Action.LEFT), int(Action.IDLE))
    assert sim.f0.alive            # never eliminated
    assert sim.f0.x >= 0.0         # clamped onto the platform
    assert sim.f0.x <= arena.platform_width


def test_punch_knocks_rival_out_of_zone():
    # Knockback still works and serves the KotH objective: shove the rival OUT of
    # the zone. Put both in the zone with f1 on the right edge of it, f0 facing
    # right in range; punch should push f1 past the zone's right boundary.
    arena = KothArena(platform_width=10.0, zone_center_frac=0.5, zone_half=1.5,
                      knockback=3.0, punch_range=2.0, max_steps=50)
    sim = KothSim(arena=arena, seed=0)
    sim.f0.x = 5.0
    sim.f0.facing = 1
    sim.f1.x = 6.0   # inside zone [3.5, 6.5], in range, in front of f0
    sim.f1.facing = -1
    assert arena.in_zone(sim.f1)
    sim.step(int(Action.PUNCH), int(Action.IDLE))
    # +3.0 knockback pushes f1 to ~9.0, past the zone's right edge (6.5).
    assert not arena.in_zone(sim.f1)
    assert sim.f1.alive  # still in play (no ring-out)


# ---------------------------------------------------------------------------
# Scripted beats random
# ---------------------------------------------------------------------------


def test_scripted_beats_random_decisively():
    # Use a mid-band difficulty where the opponent is exploitable so the scripted
    # heuristic wins big (at the top of the dial the contester denies everyone).
    arena = KothArena(difficulty=0.3)
    wins = 0
    n = 60
    for s in range(n):
        winner = play_match(
            arena,
            scripted_koth(arena, ego=0),
            random_policy(seed=10_000 + s),
            seed=s,
        )
        if winner == 0:
            wins += 1
    assert wins / n >= 0.8, f"scripted only won {wins}/{n}"


def test_random_agent_never_beats_scripted_opponent():
    # The PPO 'before' baseline relies on this: a flailing agent can't out-hold a
    # competent scripted opponent, so any post-training win-rate is real learning.
    arena = KothArena(difficulty=0.3)
    agent_wins = 0
    n = 60
    for s in range(n):
        winner = play_match(
            arena,
            random_policy(seed=20_000 + s),
            scripted_koth(arena, ego=1),
            seed=s,
        )
        if winner == 0:
            agent_wins += 1
    assert agent_wins == 0, f"random agent unexpectedly won {agent_wins}/{n}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_match_deterministic_under_fixed_seed():
    arena = KothArena(difficulty=0.3)
    w1 = play_match(arena, scripted_koth(arena, 0), random_policy(seed=7), seed=3)
    w2 = play_match(arena, scripted_koth(arena, 0), random_policy(seed=7), seed=3)
    assert w1 == w2


def test_sim_trajectory_and_zone_time_deterministic():
    arena = KothArena(difficulty=0.3)

    def run(seed: int):
        sim = KothSim(arena=arena, seed=seed)
        scripted = scripted_koth(arena, ego=0)
        opp = random_policy(seed=7)
        traj = []
        while not sim.done:
            a0 = scripted(sim.observe(0))
            a1 = opp(sim.observe(1))
            sim.step(a0, a1)
            traj.append((round(sim.f0.x, 6), round(sim.f1.x, 6)))
        return traj, sim.zone_time0, sim.zone_time1

    assert run(5) == run(5)


def test_different_seeds_can_differ():
    arena = KothArena()
    s0 = KothSim(arena=arena, seed=0)
    s1 = KothSim(arena=arena, seed=99)
    # Spawn jitter is the fighter's seeded logic, so different seeds give
    # different starts (at least sometimes).
    assert (s0.f0.x, s0.f1.x) != (s1.f0.x, s1.f1.x)


# ---------------------------------------------------------------------------
# Observation shape (reuses fighter layout + appends zone dims)
# ---------------------------------------------------------------------------


def test_observation_shape_reuses_fighter_layout():
    from games.fighter import FighterSim
    from games.fighter import OBS_DIM as FIGHTER_OBS_DIM

    arena = KothArena()
    sim = KothSim(arena=arena, seed=0)
    obs0 = sim.observe(ego=0)
    obs1 = sim.observe(ego=1)
    assert obs0.shape == (OBS_DIM,)
    assert obs1.shape == (OBS_DIM,)
    assert obs0.dtype == np.float32
    assert np.all(np.isfinite(obs0))
    # The first 11 dims must be the fighter's physical observation, byte-for-byte
    # (this is the "same observation style / shared sub-vector" claim).
    fighter_obs = FighterSim(arena=arena._as_fighter_arena(), seed=0).observe(ego=0)
    assert OBS_DIM == FIGHTER_OBS_DIM + 5
    assert np.allclose(obs0[:FIGHTER_OBS_DIM], fighter_obs)


def test_in_zone_obs_flag_matches_geometry():
    arena = KothArena(platform_width=10.0, zone_center_frac=0.5, zone_half=1.5)
    sim = KothSim(arena=arena, seed=0)
    sim.f0.x = 5.0   # in zone
    sim.f1.x = 0.1   # out of zone
    obs0 = sim.observe(ego=0)
    # index 13 == ego_in_zone, 14 == opp_in_zone (see KothSim.observe docstring).
    assert obs0[13] == 1.0
    assert obs0[14] == 0.0


# ---------------------------------------------------------------------------
# Adapter conforms to the frozen GameAdapter interface — DROP-IN with fighter
# ---------------------------------------------------------------------------


def test_adapter_is_a_gameadapter_subclass():
    # The drop-in claim, literally: KothGameAdapter IS a GameAdapter, with the
    # same four abstract methods implemented.
    assert issubclass(KothGameAdapter, GameAdapter)
    adapter = KothGameAdapter()
    assert isinstance(adapter, GameAdapter)
    for name in ("build", "scripted_expert", "random_policy", "evaluate"):
        assert callable(getattr(adapter, name))
    # No unimplemented abstract methods remain (instantiation already proves it,
    # but make the invariant explicit).
    assert not getattr(KothGameAdapter, "__abstractmethods__", frozenset())


def test_adapter_signature_matches_fighter_adapter():
    # Drop-in means the same call shapes as the fighter adapter.
    from harness.fighter_adapter import FighterGameAdapter

    for name in ("build", "scripted_expert", "random_policy", "evaluate"):
        koth_sig = inspect.signature(getattr(KothGameAdapter, name))
        fighter_sig = inspect.signature(getattr(FighterGameAdapter, name))
        assert list(koth_sig.parameters) == list(fighter_sig.parameters), name


def test_adapter_scores_scripted_high_random_low():
    # Mid-band difficulty where the gap is wide (see prove_koth_learns.py).
    adapter = KothGameAdapter(eval_seeds=20)
    arenas = adapter.arenas_from_configs([KothArena(difficulty=0.3)], curriculum_id="t")
    strong = adapter.evaluate(adapter.scripted_expert(), arenas)
    weak = adapter.evaluate(adapter.random_policy(), arenas)
    assert "scripted_expert" in strong
    assert "random_policy" in weak
    assert strong["scripted_expert"]["mean_score"] > weak["random_policy"]["mean_score"]
    assert strong["scripted_expert"]["mean_score"] >= 0.8
    assert weak["random_policy"]["mean_score"] <= 0.4


def test_adapter_build_from_curriculum_spec():
    # The contract path: build() accepts a gridworld-shaped CurriculumSpec and
    # returns one KotH arena handle per ArenaSpec.
    spec = CurriculumSpec(
        game_id="koth",
        curriculum_id="c1",
        arenas=[
            ArenaSpec(map_size=9, doors=0, keys=0, hazard_density=0.0, difficulty=0.2),
            ArenaSpec(map_size=15, doors=0, keys=0, hazard_density=0.0, difficulty=0.8),
        ],
    )
    adapter = KothGameAdapter(eval_seeds=10)
    arenas = adapter.build(spec)
    assert len(arenas) == 2
    # Higher difficulty -> SMALLER zone (harder to hold) — the parametric mapping.
    assert arenas[1].arena.zone_half < arenas[0].arena.zone_half


def test_adapter_evaluate_through_real_scorer():
    # The ONE frozen scorer must run end-to-end on the real KotH adapter — exactly
    # as it does on the fighter. Use full-strength difficulty=1.0 where the
    # contester denies random any win, so random scores low.
    from harness.scoring import score_curriculum

    spec = CurriculumSpec(
        game_id="koth",
        curriculum_id="score-test",
        arenas=[ArenaSpec(map_size=9, doors=0, keys=0, hazard_density=0.0, difficulty=1.0)],
    )
    adapter = KothGameAdapter(eval_seeds=20)
    result = score_curriculum(spec, adapter, mode="gap_proxy")
    assert result.weak_score <= 0.2  # full-strength contester: random can't out-hold it
    assert result.strong_score >= result.weak_score
    assert result.p == result.weak_score
    assert 0.0 <= result.reward <= 1.0
    assert result.curriculum_id == "score-test"
    assert result.mode == "gap_proxy"


# ---------------------------------------------------------------------------
# Gym env conforms to SB3 (same wiring path as the fighter -> PPO drop-in)
# ---------------------------------------------------------------------------


def test_gym_env_passes_sb3_checker():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env

    from games.koth import KothEnv

    arena = KothArena(difficulty=0.3)
    env = KothEnv(arena, opponent_factory=lambda a: scripted_koth(a, ego=1), seed=0)
    check_env(env, warn=True)
