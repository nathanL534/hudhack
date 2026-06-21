"""test_opponent_league.py — pure-Python unit tests for the opponent league.

No gymnasium / SB3 needed: ``games.fighter`` imports fine without them (its
gym dependency is guarded), and ``games.opponents_league`` only touches NumPy.
These tests GATE the league's three guarantees with real assertions:

  1. resolve_league drops prior_student when absent and renormalises (sum ~1.0);
     keeps it (and renormalises) when present.
  2. select_opponent_id is DETERMINISTIC: a fixed-seed rng yields an identical
     opponent-id SEQUENCE across two independent runs.
  3. Over many draws, aggressive / turtle / random are ALL sampled (each appears).
  4. make_opponent returns a WORKING obs->action callable for each style, scored
     on a real FighterSim observation and asserted to return a valid Action int.
  5. A missing prior_student is handled cleanly: it is never selected, and asking
     for it directly raises rather than crashing elsewhere.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from games.fighter import STAGE1_ACTIONS, Action, FighterArena, FighterSim
from games.opponents_league import (
    DEFAULT_LEAGUE,
    PRIOR_STUDENT_ID,
    make_opponent,
    resolve_league,
    select_opponent_id,
)

_VALID_ACTIONS = {int(a) for a in STAGE1_ACTIONS}


def _sample_obs(ego: int = 1, seed: int = 7) -> np.ndarray:
    """A real observation from a fresh FighterSim (ego=1, the opponent's POV)."""
    sim = FighterSim(arena=FighterArena(), seed=seed)
    return sim.observe(ego=ego)


# ---------------------------------------------------------------------------
# 1. resolve_league: drop + renormalise
# ---------------------------------------------------------------------------


def test_resolve_drops_prior_student_when_absent_and_renormalises():
    active = resolve_league(DEFAULT_LEAGUE, has_prior_student=False)

    ids = [m["id"] for m in active]
    assert PRIOR_STUDENT_ID not in ids, "prior_student must be dropped when absent"
    # DEFAULT_LEAGUE has 4 members; dropping prior_student leaves 3.
    assert sorted(ids) == ["aggressive", "random", "turtle"]

    total = sum(m["weight"] for m in active)
    assert math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9), f"weights must sum ~1.0, got {total}"
    # Equal input weights -> equal normalised weights.
    for m in active:
        assert math.isclose(m["weight"], 1.0 / 3.0, abs_tol=1e-9)


def test_resolve_keeps_prior_student_when_present_and_renormalises():
    active = resolve_league(DEFAULT_LEAGUE, has_prior_student=True)

    ids = [m["id"] for m in active]
    assert PRIOR_STUDENT_ID in ids, "prior_student must be kept when present"
    assert len(active) == 4

    total = sum(m["weight"] for m in active)
    assert math.isclose(total, 1.0, abs_tol=1e-9), f"weights must sum ~1.0, got {total}"
    for m in active:
        assert math.isclose(m["weight"], 0.25, abs_tol=1e-9)


def test_resolve_renormalises_uneven_weights():
    league = [
        {"id": "aggressive", "weight": 3},
        {"id": "turtle", "weight": 1},
        {"id": "random", "weight": 0},
    ]
    active = resolve_league(league, has_prior_student=False)
    by_id = {m["id"]: m["weight"] for m in active}

    assert math.isclose(sum(by_id.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(by_id["aggressive"], 0.75, abs_tol=1e-9)
    assert math.isclose(by_id["turtle"], 0.25, abs_tol=1e-9)
    assert math.isclose(by_id["random"], 0.0, abs_tol=1e-9)


def test_resolve_does_not_mutate_input():
    league = [dict(m) for m in DEFAULT_LEAGUE]
    snapshot = [dict(m) for m in league]
    resolve_league(league, has_prior_student=False)
    assert league == snapshot, "resolve_league must not mutate its input league"


def test_resolve_empty_after_drop_raises():
    # A league that is ONLY prior_student collapses to empty when it is dropped.
    with pytest.raises(ValueError):
        resolve_league([{"id": PRIOR_STUDENT_ID, "weight": 1}], has_prior_student=False)


# ---------------------------------------------------------------------------
# 2. select_opponent_id: determinism
# ---------------------------------------------------------------------------


def test_select_is_deterministic_same_seed_same_sequence():
    active = resolve_league(DEFAULT_LEAGUE, has_prior_student=False)

    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(12345)
    seq_a = [select_opponent_id(active, rng_a) for _ in range(200)]
    seq_b = [select_opponent_id(active, rng_b) for _ in range(200)]

    assert seq_a == seq_b, "same seed must produce an identical opponent-id sequence"


def test_select_different_seeds_diverge():
    # Not a hard guarantee in general, but with 200 draws over 3 ids the chance of
    # two distinct seeds matching exactly is astronomically small — a useful guard
    # that determinism comes from the rng STATE, not from a constant return.
    active = resolve_league(DEFAULT_LEAGUE, has_prior_student=False)
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    long_a = [select_opponent_id(active, rng_a) for _ in range(200)]
    long_b = [select_opponent_id(active, rng_b) for _ in range(200)]
    assert long_a != long_b, "distinct seeds should not yield identical 200-draw sequences"


def test_select_only_returns_active_ids():
    active = resolve_league(DEFAULT_LEAGUE, has_prior_student=False)
    allowed = {m["id"] for m in active}
    rng = np.random.default_rng(99)
    for _ in range(300):
        assert select_opponent_id(active, rng) in allowed


# ---------------------------------------------------------------------------
# 3. coverage: every non-prior style is sampled over many draws
# ---------------------------------------------------------------------------


def test_select_samples_all_three_base_styles():
    active = resolve_league(DEFAULT_LEAGUE, has_prior_student=False)
    rng = np.random.default_rng(2024)
    drawn = {select_opponent_id(active, rng) for _ in range(500)}

    for style in ("aggressive", "turtle", "random"):
        assert style in drawn, f"{style} was never sampled over 500 equal-weight draws"


def test_prior_student_never_selected_when_dropped():
    active = resolve_league(DEFAULT_LEAGUE, has_prior_student=False)
    rng = np.random.default_rng(555)
    drawn = {select_opponent_id(active, rng) for _ in range(500)}
    assert PRIOR_STUDENT_ID not in drawn, "prior_student must never be selected once dropped"


# ---------------------------------------------------------------------------
# 4. make_opponent: each style is a working obs->action policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["aggressive", "turtle", "random"])
def test_make_opponent_returns_valid_action(style):
    arena = FighterArena()
    obs = _sample_obs(ego=1, seed=11)
    policy = make_opponent(style, arena, ego=1, seed=42)
    assert callable(policy)

    action = policy(obs)
    assert isinstance(action, int), f"{style} policy must return a plain int action"
    assert action in _VALID_ACTIONS, f"{style} returned {action}, not a valid Action"
    # The returned int must be a real Action member.
    assert Action(action) in STAGE1_ACTIONS


def test_make_opponent_prior_student_passes_policy_through():
    arena = FighterArena()
    obs = _sample_obs(ego=1, seed=3)

    # A trivial frozen "prior Student": always punch. Stands in for a real one.
    def frozen_prior(_obs: np.ndarray) -> int:
        return int(Action.PUNCH)

    policy = make_opponent(
        PRIOR_STUDENT_ID, arena, ego=1, seed=0, prior_student=frozen_prior
    )
    assert policy is frozen_prior, "prior_student style must pass the frozen policy through"
    assert policy(obs) == int(Action.PUNCH)


def test_make_opponent_drives_a_full_match():
    """End-to-end smoke: a league opponent can actually play a full FighterSim
    match against a fixed reference policy without crashing, and the match
    terminates (winner is 0, 1, or None)."""
    from games.fighter import play_match, scripted_fighter

    arena = FighterArena()
    opponent = make_opponent("turtle", arena, ego=1, seed=17)
    winner = play_match(arena, scripted_fighter(arena, ego=0), opponent, seed=5)
    assert winner in (0, 1, None)


# ---------------------------------------------------------------------------
# 5. missing prior_student handled cleanly
# ---------------------------------------------------------------------------


def test_make_opponent_prior_student_missing_raises_cleanly():
    arena = FighterArena()
    with pytest.raises(ValueError):
        make_opponent(PRIOR_STUDENT_ID, arena, ego=1, seed=0, prior_student=None)


def test_make_opponent_unknown_style_raises():
    arena = FighterArena()
    with pytest.raises(ValueError):
        make_opponent("nonexistent_style", arena, ego=1, seed=0)


def test_full_path_no_prior_student_is_safe():
    """The realistic 'first Student, no predecessor' path: resolve drops
    prior_student, selection never yields it, and every selected style builds a
    working policy — all with no prior_student passed anywhere."""
    arena = FighterArena()
    obs = _sample_obs(ego=1, seed=8)
    active = resolve_league(DEFAULT_LEAGUE, has_prior_student=False)
    rng = np.random.default_rng(31337)

    for _ in range(100):
        oid = select_opponent_id(active, rng)
        assert oid != PRIOR_STUDENT_ID
        policy = make_opponent(oid, arena, ego=1, seed=1)  # no prior_student
        action = policy(obs)
        assert action in _VALID_ACTIONS
