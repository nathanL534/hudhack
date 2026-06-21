"""test_anti_collapse.py — unit tests for the STRONG anti-collapse mechanism.

Covers the pure host-side logic (training/anti_collapse.py) that GRPO cannot normalize
away: canonicalization, greedy farthest-point selection with the unique floor + skip
signal, the novelty bonus, the behavior gate, and the combined adjusted-reward builder
(duplicate-zeroing + behavior-gated improvement + novelty). These are the guardrails
that stop the Teacher from collapsing arena diversity the way leagueB128/256 did.
"""

import pytest

from training.anti_collapse import (
    KNOB_ORDER,
    adjusted_group_rewards,
    behavior_pass,
    canonical_tuple,
    greedy_farthest_select,
    normalize_knobs,
    novelty_bonuses,
)

_ARENAS = [
    {"difficulty": 0.1, "platform_width": 8.0, "gravity": 0.2, "knockback": 0.5, "spawn_gap": 1.0},
    {"difficulty": 0.9, "platform_width": 30.0, "gravity": 1.2, "knockback": 6.0, "spawn_gap": 12.0},
    {"difficulty": 0.5, "platform_width": 20.0, "gravity": 0.6, "knockback": 3.0, "spawn_gap": 6.0},
    {"difficulty": 0.3, "platform_width": 12.0, "gravity": 0.9, "knockback": 1.5, "spawn_gap": 9.0},
]


def _good_behavior():
    return {"movement_fraction": 0.3, "punch_fraction": 0.5, "real_ringout_win_rate": 0.4}


def test_canonical_collapses_float_jitter_and_rejects_missing():
    a = dict(_ARENAS[0]); a["difficulty"] = 0.10000001
    assert canonical_tuple(a) == canonical_tuple(_ARENAS[0])
    assert canonical_tuple(None) is None
    assert canonical_tuple({"difficulty": 0.1}) is None  # missing knobs -> None


def test_normalize_is_min_max_per_knob():
    upper = {"difficulty": 1.0, "platform_width": 30.0, "gravity": 1.2, "knockback": 6.0, "spawn_gap": 12.0}
    v = normalize_knobs(upper)  # all knobs at their upper bound
    assert all(abs(x - 1.0) < 1e-9 for x in v)
    lower = {"difficulty": 0.0, "platform_width": 8.0, "gravity": 0.2, "knockback": 0.5, "spawn_gap": 1.0}
    v0 = normalize_knobs(lower)  # all knobs at their lower bound
    assert all(abs(x - 0.0) < 1e-9 for x in v0)


def test_select_meets_unique_floor():
    cands = [_ARENAS[i % 4] for i in range(16)]  # 4 distinct, 16 total
    canon = [canonical_tuple(c) for c in cands]
    norm = [normalize_knobs(c) for c in cands]
    sel, n_unique = greedy_farthest_select(norm, canon, target=8, min_unique=4)
    assert sel is not None
    assert len(sel) == 8
    assert n_unique >= 4


def test_select_skips_when_below_unique_floor():
    cands = [_ARENAS[i % 3] for i in range(16)]  # only 3 distinct
    canon = [canonical_tuple(c) for c in cands]
    norm = [normalize_knobs(c) for c in cands]
    sel, n_unique = greedy_farthest_select(norm, canon, target=8, min_unique=4)
    assert sel is None  # caller must retry/skip
    assert n_unique == 3


def test_select_ignores_invalid_candidates():
    cands = [_ARENAS[0], None, _ARENAS[1], {"difficulty": 0.5}, _ARENAS[2], _ARENAS[3]]
    canon = [canonical_tuple(c) for c in cands]
    norm = [normalize_knobs(c) if c and canonical_tuple(c) else [0.0] * 5 for c in cands]
    sel, n_unique = greedy_farthest_select(norm, canon, target=8, min_unique=4)
    assert sel is not None
    # Only the 4 valid distinct arenas are selectable.
    assert n_unique == 4
    assert all(canon[i] is not None for i in sel)


def test_novelty_rewards_spread():
    spread = novelty_bonuses([normalize_knobs(_ARENAS[0]), normalize_knobs(_ARENAS[1])])
    clustered = novelty_bonuses([normalize_knobs(_ARENAS[2]), normalize_knobs(_ARENAS[2])])
    assert spread[0] > clustered[0]
    assert novelty_bonuses([[0.0] * 5]) == [0.0]  # singleton has no novelty


@pytest.mark.parametrize(
    "diag,expected",
    [
        ({"movement_fraction": 0.3, "punch_fraction": 0.5, "real_ringout_win_rate": 0.4}, True),
        ({"movement_fraction": 0.05, "punch_fraction": 0.5, "real_ringout_win_rate": 0.4}, False),  # camps
        ({"movement_fraction": 0.3, "punch_fraction": 0.95, "real_ringout_win_rate": 0.4}, False),  # spams
        ({"movement_fraction": 0.3, "punch_fraction": 0.5, "real_ringout_win_rate": 0.1}, False),  # timeout-only
        (None, False),  # no probe -> never credited
    ],
)
def test_behavior_gate(diag, expected):
    assert behavior_pass(diag) is expected


def test_behavior_gate_boundaries_are_inclusive():
    # The thresholds are >=/<=, so exact-boundary values PASS.
    assert behavior_pass({"movement_fraction": 0.10, "punch_fraction": 0.90, "real_ringout_win_rate": 0.20})


def test_adjusted_rewards_zero_duplicates_gate_and_add_novelty():
    params = [_ARENAS[0], _ARENAS[1], _ARENAS[0]]  # third is an exact dup of the first
    canon = [canonical_tuple(p) for p in params]
    improvements = [0.3, 0.3, 0.3]
    behavior = [
        _good_behavior(),  # passes
        {"movement_fraction": 0.0, "punch_fraction": 0.5, "real_ringout_win_rate": 0.4},  # camps -> gated 0
        _good_behavior(),  # would pass, but it's a DUP
    ]
    adjusted, diags = adjusted_group_rewards(params, canon, improvements, behavior, novelty_weight=0.10)

    assert adjusted[2] == 0.0  # exact duplicate -> 0 (overrides gate + novelty)
    assert diags[2]["is_exact_duplicate"] is True
    assert diags[1]["behavior_gated_improvement"] == 0.0  # camping Student earns no improvement
    assert diags[1]["behavior_pass"] is False
    assert adjusted[0] > 0.3  # passing Student keeps improvement + gains novelty on top
    assert diags[0]["novelty_bonus"] > 0.0


def test_adjusted_signed_improvement_survives_negative():
    # A negative-transfer arena (hurt the Student) keeps its sign through the gate, so
    # GRPO is pushed AWAY from it rather than flooring it to 0.
    params = [_ARENAS[0], _ARENAS[1]]
    canon = [canonical_tuple(p) for p in params]
    improvements = [-0.2, 0.3]
    behavior = [_good_behavior(), _good_behavior()]
    adjusted, diags = adjusted_group_rewards(params, canon, improvements, behavior, novelty_weight=0.0)
    assert diags[0]["behavior_gated_improvement"] == pytest.approx(-0.2)
    assert adjusted[0] < adjusted[1]  # the harmful arena ranks below the helpful one


def test_knob_order_is_the_five_ring_out_knobs():
    assert KNOB_ORDER == ("difficulty", "platform_width", "gravity", "knockback", "spawn_gap")
