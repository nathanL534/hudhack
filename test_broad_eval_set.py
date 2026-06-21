"""Credential-free tests for the diverse held-out eval population.

These cover the NEW eval-set builder (output/broad_eval_set.py) and the worker's
opt-in ``held_out_arenas`` payload key (modal_player._held_out_reference_arenas).
No PPO / Modal cost: arena construction is pure.
"""

from games.fighter import FighterArena
from modal_player import (
    HELD_OUT_REFERENCE_DIFFICULTIES,
    _held_out_reference_arenas,
)
from output.broad_eval_set import (
    DIFFICULTIES,
    PHYSICS_VARIANTS,
    assert_disjoint_from_training,
    build_broad_eval_arenas,
    payload_arenas,
)

TRAINING_DIFFICULTIES = (0.25, 0.55, 0.85)


def test_full_grid_is_variants_times_difficulties():
    arenas = build_broad_eval_arenas(grid="full")
    assert len(arenas) == len(PHYSICS_VARIANTS) * len(DIFFICULTIES)
    # In the target 9-15 range the task asks for.
    assert 9 <= len(arenas) <= 15


def test_diagonal_grid_covers_every_geometry_and_difficulty():
    arenas = build_broad_eval_arenas(grid="diagonal")
    assert len(arenas) == len(PHYSICS_VARIANTS)
    # Each diagonal arena pairs a distinct geometry with a difficulty.
    widths = {a["platform_width"] for a in arenas}
    assert len(widths) >= 3  # several distinct platform widths
    diffs = {a["difficulty"] for a in arenas}
    assert diffs.issubset(set(DIFFICULTIES))


def test_population_is_structurally_diverse_in_physics():
    """The whole point: the eval set must vary PHYSICS, not just difficulty."""
    arenas = build_broad_eval_arenas(grid="full")
    widths = {a["platform_width"] for a in arenas}
    gravities = {a["gravity"] for a in arenas}
    knockbacks = {a["knockback"] for a in arenas}
    spawn_gaps = {a["spawn_gap"] for a in arenas}
    # Every physics axis takes at least two distinct values across the set.
    assert len(widths) >= 2
    assert len(gravities) >= 2
    assert len(knockbacks) >= 2
    assert len(spawn_gaps) >= 2
    # And opponent strength varies too.
    assert {a["difficulty"] for a in arenas} == set(DIFFICULTIES)


def test_physics_within_plausible_schema_bounds():
    for a in build_broad_eval_arenas(grid="full"):
        assert 6.0 <= a["platform_width"] <= 16.0
        assert 0.3 <= a["gravity"] <= 0.85
        assert 1.0 <= a["knockback"] <= 4.5
        assert 2.0 <= a["spawn_gap"] <= 7.0
        assert 0.0 <= a["difficulty"] <= 1.0


def test_no_eval_arena_equals_a_training_arena():
    arenas = build_broad_eval_arenas(grid="full")
    # Must not raise: held-out set is disjoint from the width-11/default-physics
    # training arenas at the training difficulties.
    assert_disjoint_from_training(arenas, TRAINING_DIFFICULTIES)


def test_disjoint_guard_actually_fires_on_a_collision():
    import pytest

    defaults = FighterArena()
    colliding = [
        {
            "name": "collision",
            "platform_width": 11.0,  # = 6.0 + 0.5*map_size(10), the training geometry
            "gravity": defaults.gravity,
            "knockback": defaults.knockback,
            "spawn_gap": defaults.spawn_gap,
            "difficulty": 0.55,  # a training difficulty
        }
    ]
    with pytest.raises(ValueError):
        assert_disjoint_from_training(colliding, TRAINING_DIFFICULTIES)


def test_builder_is_deterministic():
    a = build_broad_eval_arenas(grid="full")
    b = build_broad_eval_arenas(grid="full")
    assert a == b


def test_payload_arenas_strip_name_only():
    arenas = build_broad_eval_arenas(grid="full")
    payload = payload_arenas(arenas)
    assert all("name" not in p for p in payload)
    for src, p in zip(arenas, payload):
        for k in ("platform_width", "gravity", "knockback", "spawn_gap", "difficulty"):
            assert p[k] == src[k]


# --- worker opt-in: held_out_arenas overrides difficulty-only behavior --------


def test_worker_uses_diverse_held_out_arenas_when_provided():
    arenas = payload_arenas(build_broad_eval_arenas(grid="full"))
    built = _held_out_reference_arenas({"held_out_arenas": arenas})
    assert len(built) == len(arenas)
    assert all(isinstance(a, FighterArena) for a in built)
    # The worker honored the FULL physics, not just difficulty.
    for spec, fa in zip(arenas, built):
        assert fa.platform_width == spec["platform_width"]
        assert fa.gravity == spec["gravity"]
        assert fa.knockback == spec["knockback"]
        assert fa.spawn_gap == spec["spawn_gap"]
        assert fa.difficulty == spec["difficulty"]


def test_worker_default_behavior_unchanged_without_the_key():
    """Default reward path is untouched: no key -> the two turtle references."""
    built = _held_out_reference_arenas({})
    assert [a.difficulty for a in built] == list(HELD_OUT_REFERENCE_DIFFICULTIES)
    # Default geometry, exactly as before.
    defaults = FighterArena()
    for a in built:
        assert a.platform_width == defaults.platform_width
        assert a.gravity == defaults.gravity
        assert a.knockback == defaults.knockback
        assert a.spawn_gap == defaults.spawn_gap


def test_worker_held_out_difficulties_still_works():
    built = _held_out_reference_arenas({"held_out_difficulties": [0.5, 0.9]})
    assert [a.difficulty for a in built] == [0.5, 0.9]


def test_held_out_arenas_takes_priority_over_difficulties():
    arenas = payload_arenas(build_broad_eval_arenas(grid="diagonal"))
    built = _held_out_reference_arenas(
        {"held_out_arenas": arenas, "held_out_difficulties": [0.1]}
    )
    assert len(built) == len(arenas)  # the full-spec list wins
