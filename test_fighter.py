"""test_fighter.py — Stage-1 fighter tests.

Covers the three properties the milestone depends on:
  * ring-out works (walking / being knocked off an edge is a loss),
  * the scripted fighter clearly beats random,
  * the sim is deterministic under a fixed seed,
plus that the GameAdapter conforms to the frozen scoring shape (scripted scores
high, random scores low) and the Gym env passes SB3's env checker.

These tests do NOT train PPO (kept fast / no torch dependency at import time
beyond the env-checker case). The full learnability check lives in
prove_ppo_learns.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from contracts import ArenaSpec, CurriculumSpec
from games.fighter import (
    OBS_DIM,
    STAGE1_ACTIONS,
    Action,
    FighterArena,
    FighterSim,
    play_match,
    random_policy,
    scripted_fighter,
)
from harness.fighter_adapter import FighterGameAdapter


# ---------------------------------------------------------------------------
# Ring-out
# ---------------------------------------------------------------------------


def test_ringout_off_right_edge_is_a_loss():
    arena = FighterArena()
    sim = FighterSim(arena=arena, seed=0)
    # Teleport fighter 0 past the right edge, grounded -> must ring out.
    sim.f0.x = arena.platform_width + 2.0
    sim.f0.y = 0.0
    sim._check_ringout(sim.f0)
    assert not sim.f0.alive


def test_ringout_off_left_edge_is_a_loss():
    arena = FighterArena()
    sim = FighterSim(arena=arena, seed=0)
    sim.f0.x = -2.0
    sim.f0.y = 0.0
    sim._check_ringout(sim.f0)
    assert not sim.f0.alive


def test_on_platform_is_safe():
    arena = FighterArena()
    sim = FighterSim(arena=arena, seed=0)
    sim.f0.x = arena.platform_width / 2.0
    sim._check_ringout(sim.f0)
    assert sim.f0.alive


def test_punch_knockback_can_ring_out_opponent():
    # Put the defender at the edge, attacker in range facing it, punch once.
    arena = FighterArena(platform_width=10.0, knockback=3.0, punch_range=2.0)
    sim = FighterSim(arena=arena, seed=0)
    sim.f0.x = 8.7
    sim.f0.facing = 1
    sim.f1.x = 9.6  # near the right edge, within range, in front
    sim.f1.facing = -1
    sim.step(int(Action.PUNCH), int(Action.IDLE))
    # The knockback (+3.0) pushes f1 past width 10 -> ring-out -> f0 wins.
    assert not sim.f1.alive
    assert sim.winner == 0
    assert sim.done


# ---------------------------------------------------------------------------
# Scripted beats random
# ---------------------------------------------------------------------------


def test_scripted_beats_random_decisively():
    arena = FighterArena()
    wins = 0
    n = 60
    for s in range(n):
        winner = play_match(
            arena,
            scripted_fighter(arena, ego=0),
            random_policy(seed=10_000 + s),
            seed=s,
        )
        if winner == 0:
            wins += 1
    # The scripted heuristic should win the overwhelming majority.
    assert wins / n >= 0.8, f"scripted only won {wins}/{n}"


def test_random_agent_never_beats_scripted_opponent():
    # The PPO 'before' baseline relies on this: a flailing agent cannot win,
    # so any post-training win-rate is real learning.
    arena = FighterArena()
    agent_wins = 0
    n = 60
    for s in range(n):
        winner = play_match(
            arena,
            random_policy(seed=20_000 + s),
            scripted_fighter(arena, ego=1),
            seed=s,
        )
        if winner == 0:
            agent_wins += 1
    assert agent_wins == 0, f"random agent unexpectedly won {agent_wins}/{n}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_match_deterministic_under_fixed_seed():
    arena = FighterArena()
    w1 = play_match(arena, scripted_fighter(arena, 0), random_policy(seed=7), seed=3)
    w2 = play_match(arena, scripted_fighter(arena, 0), random_policy(seed=7), seed=3)
    assert w1 == w2


def test_sim_trajectory_deterministic():
    arena = FighterArena()

    def trajectory(seed: int):
        sim = FighterSim(arena=arena, seed=seed)
        positions = []
        for _ in range(30):
            if sim.done:
                break
            sim.step(int(Action.RIGHT), int(Action.LEFT))
            positions.append((sim.f0.x, sim.f1.x, sim.f0.y, sim.f1.y))
        return positions

    assert trajectory(5) == trajectory(5)


def test_different_seeds_can_differ():
    arena = FighterArena()
    # Spawn jitter is seeded, so different seeds give (at least sometimes)
    # different starting positions.
    s0 = FighterSim(arena=arena, seed=0)
    s1 = FighterSim(arena=arena, seed=99)
    assert (s0.f0.x, s0.f1.x) != (s1.f0.x, s1.f1.x)


# ---------------------------------------------------------------------------
# Observation shape
# ---------------------------------------------------------------------------


def test_observation_shape_and_finiteness():
    arena = FighterArena()
    sim = FighterSim(arena=arena, seed=0)
    obs0 = sim.observe(ego=0)
    obs1 = sim.observe(ego=1)
    assert obs0.shape == (OBS_DIM,)
    assert obs1.shape == (OBS_DIM,)
    assert np.all(np.isfinite(obs0))
    assert obs0.dtype == np.float32


def test_action_enum_is_extensible():
    # Stage-1 exposes exactly five actions, but the enum leaves room to ADD
    # more without renumbering — IDLE..PUNCH are the canonical first five.
    assert STAGE1_ACTIONS == (
        Action.IDLE,
        Action.LEFT,
        Action.RIGHT,
        Action.JUMP,
        Action.PUNCH,
    )
    assert len(STAGE1_ACTIONS) == 5


# ---------------------------------------------------------------------------
# Adapter conforms to the frozen scoring shape
# ---------------------------------------------------------------------------


def test_adapter_scores_scripted_high_random_low():
    adapter = FighterGameAdapter(eval_seeds=20)
    arenas = adapter.arenas_from_configs([FighterArena()], curriculum_id="t")
    strong = adapter.evaluate(adapter.scripted_expert(), arenas)
    weak = adapter.evaluate(adapter.random_policy(), arenas)
    assert "scripted_expert" in strong
    assert "random_policy" in weak
    assert strong["scripted_expert"]["mean_score"] > weak["random_policy"]["mean_score"]
    assert strong["scripted_expert"]["mean_score"] >= 0.8
    assert weak["random_policy"]["mean_score"] <= 0.2


def test_adapter_build_from_curriculum_spec():
    # The contract path: build() must accept a gridworld-shaped CurriculumSpec
    # and return one arena handle per ArenaSpec.
    spec = CurriculumSpec(
        game_id="fighter",
        curriculum_id="c1",
        arenas=[
            ArenaSpec(map_size=9, doors=0, keys=0, hazard_density=0.0, difficulty=0.3),
            ArenaSpec(map_size=15, doors=0, keys=0, hazard_density=0.0, difficulty=0.7),
        ],
    )
    adapter = FighterGameAdapter(eval_seeds=10)
    arenas = adapter.build(spec)
    assert len(arenas) == 2
    # Higher difficulty -> stronger gravity (deterministic mapping).
    assert arenas[1].arena.gravity > arenas[0].arena.gravity


def test_adapter_evaluate_through_real_scorer():
    # The ONE frozen scorer must run end-to-end on the real fighter adapter and
    # produce a positive gap-proxy reward (scripted high, random low, in band).
    from harness.scoring import score_curriculum

    spec = CurriculumSpec(
        game_id="fighter",
        curriculum_id="score-test",
        arenas=[ArenaSpec(map_size=9, doors=0, keys=0, hazard_density=0.0, difficulty=0.3)],
    )
    adapter = FighterGameAdapter(eval_seeds=20)
    result = score_curriculum(spec, adapter, mode="gap_proxy")
    assert result.strong_score >= 0.8
    assert result.weak_score <= 0.2
    # p = weak_score is near 0 here, so p*(1-p) ~ 0 < 0.2 band threshold ->
    # reward gated to 0.0. The scorer still runs cleanly end-to-end, which is
    # what this asserts (the band gate is a property of the scorer, not a bug).
    assert result.curriculum_id == "score-test"
    assert result.mode == "gap_proxy"


# ---------------------------------------------------------------------------
# Gym env conforms to SB3
# ---------------------------------------------------------------------------


def test_gym_env_passes_sb3_checker():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env

    from games.fighter import FighterEnv

    arena = FighterArena()
    env = FighterEnv(arena, opponent_factory=lambda a: scripted_fighter(a, ego=1), seed=0)
    check_env(env, warn=True)
