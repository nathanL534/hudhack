"""test_stage6_primary.py — focused tests for the Stage-6 PRIMARY metric.

The headline metric is a DIRECT Student-vs-Student head-to-head (replacing the
fixed-bot before->after diagnostic, which is now SECONDARY). These tests cover the
load-bearing pieces with NO Modal and NO PPO where possible (stubbed workers), plus
ONE real-PPO serialize/restore test (cheap budget) that proves a frozen policy
reproduces its actions:

  * policy serialize/restore: a restored policy reproduces identical actions.
  * head-to-head aggregation: paired advantage, bootstrap/t CI, side-bias.
  * anti-circularity checks: every PRIMARY failure mode + the smoke exemptions.
  * equal training budgets: both Students get the identical payload knobs.
  * held-out disjointness: overlap counting + the disjointness check.
  * curriculum-level CI: the CI is over replicates, not matches.
  * base-vs-base null behavior: identical Students -> zero advantage, flagged.
  * KOTH cross-game: mapping is valid + supported flag; UNSUPPORTED path documented.
  * fabricated-metric prevention: the dashboard JSON carries only real numbers.
  * report/dashboard schema: the primary headline renders + dashboard keys present.

Run:  .venv/bin/python -m pytest test_stage6_primary.py -q
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from output.stage6 import aggregate, anti_gaming
from output.stage6.config import EvalConfig, head_to_head_smoke_config
from output.stage6.evaluator import EvalRequest, run_full_eval
from output.stage6.games import get_game
from output.stage6.handles import ResolvedTeacher


# ---------------------------------------------------------------------------
# policy serialize / restore (one cheap real-PPO test)
# ---------------------------------------------------------------------------


def test_policy_serialize_restore_reproduces_actions():
    import numpy as np
    from stable_baselines3 import PPO

    from games.fighter import FighterArena
    from harness.ppo_trainer import _MultiArenaFighterEnv
    from output.stage6.policy import restore_policy, serialize_policy

    arena = FighterArena(difficulty=0.5)
    env = _MultiArenaFighterEnv([arena], seed=3)
    model = PPO("MlpPolicy", env, seed=3, verbose=0,
                policy_kwargs={"net_arch": [64, 64]}, device="cpu")
    model.learn(total_timesteps=2000)

    art = serialize_policy(model, teacher="base", curriculum_id="c0", seed=3)
    assert art.checksum and art.teacher == "base" and art.obs_dim == 11
    # round-trips through dict (the Modal return path).
    art2 = type(art).from_dict(art.as_dict())
    assert art2.checksum == art.checksum

    restored = restore_policy(art, env, seed=99)
    original = lambda obs: int(model.policy.predict(obs, deterministic=True)[0])
    agree = sum(1 for _ in range(150)
                if restored(env.observation_space.sample()) is not None)
    assert agree == 150  # callable on every obs
    # exact action agreement on the same observations
    obss = [env.observation_space.sample() for _ in range(100)]
    assert all(restored(o) == original(o) for o in obss)


def test_policy_tag_excludes_weights():
    from output.stage6.policy import PolicyArtifact

    art = PolicyArtifact(state_dict_b64="QUJD", arch="mlp:64-64", checksum="x",
                         teacher="trained", curriculum_id="c1", seed=7)
    tag = art.tag()
    assert "state_dict_b64" not in tag
    assert tag["teacher"] == "trained" and tag["bytes"] == len("QUJD")


# ---------------------------------------------------------------------------
# head-to-head aggregation
# ---------------------------------------------------------------------------


def test_paired_advantage_rates_sum_to_one():
    adv = aggregate.paired_advantage(trained_wins=6, base_wins=2, draws=2)
    assert adv["trained_win_rate"] == 0.6
    assert adv["base_win_rate"] == 0.2
    assert adv["draw_rate"] == 0.2
    assert adv["advantage"] == pytest.approx(0.4)
    assert abs(adv["trained_win_rate"] + adv["base_win_rate"] + adv["draw_rate"] - 1.0) < 1e-9


def test_bootstrap_ci_brackets_mean_and_single_sample():
    xs = [0.1, 0.2, 0.3, 0.25, 0.15]
    ci = aggregate.bootstrap_ci(xs, seed=1)
    assert ci.low <= aggregate.mean(xs) <= ci.high
    single = aggregate.bootstrap_ci([0.3])
    assert single.low == 0.3 and single.high == 0.3 and single.half_width == 0.0


# ---------------------------------------------------------------------------
# anti-circularity checks (PRIMARY)
# ---------------------------------------------------------------------------


def test_ci_lower_above_zero_check():
    assert anti_gaming.check_ci_lower_above_zero(0.2, 0.05).passed
    assert not anti_gaming.check_ci_lower_above_zero(0.2, -0.01).passed
    assert not anti_gaming.check_ci_lower_above_zero(0.2, 0.0).passed  # strictly > 0


def test_no_side_bias_check():
    assert anti_gaming.check_no_side_bias(0.6, 0.55).passed             # both sides win, small gap
    assert not anti_gaming.check_no_side_bias(0.6, 0.0).passed          # only P0 wins
    assert not anti_gaming.check_no_side_bias(0.9, 0.5).passed          # gap > 0.25


def test_multi_arena_benefit_check():
    assert anti_gaming.check_multi_arena_benefit([0.1, 0.2, -0.05]).passed   # >1 positive
    assert not anti_gaming.check_multi_arena_benefit([0.3, -0.1, -0.2]).passed  # only 1 positive
    assert not anti_gaming.check_multi_arena_benefit([0.3]).passed          # single family


def test_distinct_students_check_smoke_vs_real():
    assert not anti_gaming.check_distinct_students("A", "A", is_smoke=False).passed
    assert anti_gaming.check_distinct_students("A", "B", is_smoke=False).passed
    smoke = anti_gaming.check_distinct_students("A", "A", is_smoke=True)
    assert smoke.passed and smoke.severity == "skip"


def test_disjoint_held_out_check():
    assert anti_gaming.check_disjoint_held_out(0).passed
    assert not anti_gaming.check_disjoint_held_out(1).passed


def test_enough_replicates_check():
    real = anti_gaming.check_enough_replicates(1, is_smoke=False)
    assert not real.passed and real.severity == "fail"
    assert anti_gaming.check_enough_replicates(4, is_smoke=False).passed
    smoke = anti_gaming.check_enough_replicates(1, is_smoke=True)
    assert not smoke.passed and smoke.severity == "warn"   # warn, not fail, in smoke


def test_run_head_to_head_checks_passing_run():
    rep = anti_gaming.run_head_to_head_checks(
        advantage_mean=0.2, ci_low=0.05,
        trained_p0_winrate=0.6, trained_p1_winrate=0.55,
        per_arena_advantages=[0.1, 0.3, 0.2],
        base_checksum="A", trained_checksum="B",
        held_out_overlap_count=0, n_replicates=4, is_smoke=False,
    )
    assert rep.passed and not rep.as_dict()["failed"]


def test_run_head_to_head_checks_failing_run():
    rep = anti_gaming.run_head_to_head_checks(
        advantage_mean=0.2, ci_low=-0.1,                     # H1 fail
        trained_p0_winrate=0.6, trained_p1_winrate=0.0,      # H2 fail (one-sided)
        per_arena_advantages=[0.3, -0.1],                    # H3 fail (1 positive)
        base_checksum="SAME", trained_checksum="SAME",       # H4 fail
        held_out_overlap_count=2,                            # H5 fail
        n_replicates=1, is_smoke=False,                      # H6 fail
    )
    failed = rep.as_dict()["failed"]
    for name in ("ci_lower_above_zero", "no_side_bias", "multi_arena_benefit",
                 "distinct_students", "disjoint_held_out", "enough_replicates"):
        assert name in failed


# ---------------------------------------------------------------------------
# held-out disjointness
# ---------------------------------------------------------------------------


def test_count_held_out_overlap():
    from output.stage6.head_to_head import count_held_out_overlap

    keys = ("difficulty", "platform_width")
    held = [{"difficulty": 0.4, "platform_width": 7.0},
            {"difficulty": 0.6, "platform_width": 16.0}]
    curricula = [{"difficulty": 0.4, "platform_width": 7.0}]   # equals held[0]
    assert count_held_out_overlap(held, curricula, keys) == 1
    assert count_held_out_overlap(held, [{"difficulty": 0.9, "platform_width": 9.0}], keys) == 0


# ---------------------------------------------------------------------------
# end-to-end orchestration with STUB workers (no Modal, no PPO)
# ---------------------------------------------------------------------------


class _StubTeacher:
    def __init__(self, arenas):
        self._arenas = arenas
        self._i = 0

    def generate(self, game):
        a = self._arenas[self._i % len(self._arenas)]
        self._i += 1
        return dict(a)


def _stub_resolver(base_arenas, trained_arenas):
    def _resolve(label, handle, **kw):
        arenas = base_arenas if label.startswith("base") else trained_arenas
        return ResolvedTeacher(label=label, handle=handle, resolved_id=f"stub::{handle}",
                               kind="fireworks", _build=lambda: _StubTeacher(arenas))
    return _resolve


def _stub_train(advantage_by_teacher):
    """Stub train worker: returns a tagged policy whose checksum encodes the teacher."""
    def _run(payloads, seeds, *, backend):
        rows = []
        for p, s in zip(payloads, seeds):
            teacher = p["teacher"]
            rows.append({
                "status": "student_policy", "seed": s, "teacher": teacher,
                "curriculum_id": p["curriculum_id"], "game": p["game"],
                "train_arena_winrate": 0.5,
                "policy": {"state_dict_b64": "X", "arch": "mlp:64-64",
                           "checksum": f"cs-{teacher}-{p['curriculum_id']}",
                           "teacher": teacher, "curriculum_id": p["curriculum_id"],
                           "seed": s, "obs_dim": 11},
            })
        return rows
    return _run


def _stub_h2h(trained_win, base_win, draw):
    """Stub h2h worker: deterministic win/loss/draw counts per arena, side-symmetric."""
    def _run(payloads, seeds, *, backend):
        rows = []
        for p, s in zip(payloads, seeds):
            per_seed = []
            for ms in p["match_seeds"]:
                # cycle outcomes deterministically; symmetric across sides.
                idx = (ms + sum(ord(c) for c in str(p["arena"]))) % (trained_win + base_win + draw)
                out = "win" if idx < trained_win else ("loss" if idx < trained_win + base_win else "draw")
                per_seed.append({"trained_p0": out, "trained_p1": out,
                                 "raw_winner_p0side": 0, "raw_winner_p1side": 1})
            rows.append({"status": "head_to_head", "game": p["game"], "arena": p["arena"],
                         "trained_checksum": p["trained_policy"]["checksum"],
                         "base_checksum": p["base_policy"]["checksum"], "per_seed": per_seed})
        return rows
    return _run


def _fighter_arenas():
    base = [{"difficulty": 0.3, "platform_width": 11.0, "gravity": 0.6, "knockback": 2.5, "spawn_gap": 4.0}]
    trained = [{"difficulty": 0.6, "platform_width": 11.0, "gravity": 0.6, "knockback": 2.5, "spawn_gap": 4.0}]
    return base, trained


def test_end_to_end_primary_trained_wins():
    base, trained = _fighter_arenas()
    cfg = EvalConfig(n_replicates=4, curriculum_arenas=1, match_seeds_per_arena=6,
                     head_to_head_grid="diagonal", ppo_episodes=10, eval_seeds=2)
    req = EvalRequest(base_handle="b", trained_handle="t", game="fighter",
                      backend="local", config=cfg, is_smoke=False, verbose=False,
                      run_secondary=False)
    summary = run_full_eval(
        req, resolve=_stub_resolver(base, trained),
        run_train=_stub_train({}), run_h2h=_stub_h2h(trained_win=6, base_win=0, draw=0),
    )
    p = summary["primary"]
    assert p["trained_student_win_rate"] == 1.0
    assert p["mean_paired_advantage"] == pytest.approx(1.0)
    assert p["ci_lower_bound"] > 0.0
    # distinct stub checksums (base vs trained) -> distinct_students passes
    assert "distinct_students" not in p["anti_circularity"]["failed"]
    # CI is over the 4 replicates, NOT over matches.
    assert p["ci_t"]["n"] == 4
    assert summary["overall_pass"] is True
    assert summary["headline_metric"] == "primary_student_vs_student_head_to_head"


def test_end_to_end_primary_null_zero_advantage():
    """Identical curricula AND identical training seed -> identical Students -> the
    distinct_students check FAILS and advantage is ~0 (the base-vs-base null)."""
    same = [{"difficulty": 0.5, "platform_width": 11.0, "gravity": 0.6, "knockback": 2.5, "spawn_gap": 4.0}]
    cfg = EvalConfig(n_replicates=3, curriculum_arenas=1, match_seeds_per_arena=6,
                     head_to_head_grid="diagonal", ppo_episodes=10, eval_seeds=2)
    req = EvalRequest(base_handle="b", trained_handle="t", game="fighter",
                      backend="local", config=cfg, is_smoke=False, verbose=False,
                      run_secondary=False)

    # A train stub that gives BOTH teachers the SAME checksum (identical Students).
    def _identical_train(payloads, seeds, *, backend):
        rows = []
        for p, s in zip(payloads, seeds):
            rows.append({"status": "student_policy", "seed": s, "teacher": p["teacher"],
                         "curriculum_id": p["curriculum_id"], "game": p["game"],
                         "train_arena_winrate": 0.5,
                         "policy": {"state_dict_b64": "X", "arch": "mlp:64-64",
                                    "checksum": "IDENTICAL", "teacher": p["teacher"],
                                    "curriculum_id": p["curriculum_id"], "seed": s, "obs_dim": 11}})
        return rows

    summary = run_full_eval(
        req, resolve=_stub_resolver(same, same),
        run_train=_identical_train, run_h2h=_stub_h2h(trained_win=0, base_win=0, draw=6),
    )
    p = summary["primary"]
    assert p["mean_paired_advantage"] == pytest.approx(0.0)
    assert "distinct_students" in p["anti_circularity"]["failed"]
    assert summary["overall_pass"] is False


def test_equal_training_budgets_for_both_students():
    """Both Students must get the IDENTICAL payload budget knobs (fairness)."""
    base, trained = _fighter_arenas()
    cfg = EvalConfig(n_replicates=1, curriculum_arenas=1, match_seeds_per_arena=2,
                     head_to_head_grid="diagonal", ppo_episodes=321, eval_seeds=11)
    captured = {}

    def _capture_train(payloads, seeds, *, backend):
        for p in payloads:
            captured[p["teacher"]] = (p["ppo_episodes"], p["eval_seeds"], p["architecture"])
        return _stub_train({})(payloads, seeds, backend=backend)

    req = EvalRequest(base_handle="b", trained_handle="t", game="fighter",
                      backend="local", config=cfg, is_smoke=True, verbose=False,
                      run_secondary=False)
    run_full_eval(req, resolve=_stub_resolver(base, trained),
                  run_train=_capture_train, run_h2h=_stub_h2h(1, 0, 1))
    assert captured["base"] == captured["trained"] == (321, 11, "mlp")


# ---------------------------------------------------------------------------
# KOTH cross-game
# ---------------------------------------------------------------------------


def test_koth_mapping_valid_and_monotone():
    from output.stage6.koth_cross_game import KOTH_BOUNDS, map_fighter_arena_to_koth

    easy = map_fighter_arena_to_koth({"difficulty": 0.2, "platform_width": 12.0})
    hard = map_fighter_arena_to_koth({"difficulty": 0.9, "platform_width": 12.0})
    # harder difficulty -> SMALLER zone (harder to hold) and more off-centre.
    assert hard["zone_half"] < easy["zone_half"]
    assert hard["zone_center_frac"] > easy["zone_center_frac"]
    # every mapped value is within KOTH bounds (total + clamped mapping).
    for k, (lo, hi) in KOTH_BOUNDS.items():
        assert lo <= easy[k] <= hi and lo <= hard[k] <= hi


def test_koth_supported_flag():
    from output.stage6.koth_cross_game import koth_supported

    ok, reason = koth_supported()
    assert ok is True and "koth" in reason.lower()


def test_koth_cross_game_with_stubs():
    """The cross-game runs the SAME head-to-head machinery on game='koth'."""
    from output.stage6.koth_cross_game import run_koth_cross_game

    base = ResolvedTeacher("base", "b", "stub::b", "fireworks",
                           _build=lambda: _StubTeacher([{"difficulty": 0.3, "platform_width": 12.0}]))
    trained = ResolvedTeacher("trained", "t", "stub::t", "fireworks",
                              _build=lambda: _StubTeacher([{"difficulty": 0.7, "platform_width": 12.0}]))
    cfg = head_to_head_smoke_config()
    res = run_koth_cross_game(base, trained, config=cfg, backend="local", is_smoke=True,
                              verbose=False, run_train=_stub_train({}),
                              run_h2h=_stub_h2h(trained_win=4, base_win=1, draw=1))
    assert res["supported"] is True
    assert res["label"] == "cross_game_generalization"
    assert res["game"] == "koth"
    assert res["trained_student_win_rate"] > res["base_student_win_rate"]


# ---------------------------------------------------------------------------
# report / dashboard schema + fabricated-metric prevention
# ---------------------------------------------------------------------------


def _full_result_for_report():
    base, trained = _fighter_arenas()
    cfg = EvalConfig(n_replicates=3, curriculum_arenas=1, match_seeds_per_arena=4,
                     head_to_head_grid="diagonal", ppo_episodes=10, eval_seeds=2)
    req = EvalRequest(base_handle="b", trained_handle="t", game="fighter",
                      backend="local", config=cfg, is_smoke=False, verbose=False,
                      run_secondary=False)
    return run_full_eval(req, resolve=_stub_resolver(base, trained),
                         run_train=_stub_train({}), run_h2h=_stub_h2h(4, 1, 1))


def test_report_and_dashboard_schema(tmp_path):
    from output.stage6 import report as report_mod

    summary = _full_result_for_report()
    arts = report_mod.generate_report(summary, tmp_path)
    md = Path(arts["report_md"]).read_text()
    assert "Base Teacher's Student VS Trained Teacher's Student" in md
    assert "Secondary transfer diagnostic" not in md  # secondary skipped here
    # dashboard exists and carries ONLY real numbers (no 0.314 / +89% fabrications).
    import json
    dash = json.loads(Path(arts["dashboard"]).read_text())
    for key in ("trained_student_win_rate", "base_student_win_rate", "mean_paired_advantage",
                "ci_low", "ci_high", "n_replicates", "primary_pass"):
        assert key in dash
    assert dash["headline_metric"] == "base_student_vs_trained_student"


def test_dashboard_has_no_fabricated_constants():
    """Guard against the old fabricated 0.314 / +89% sneaking into the dashboard."""
    import json

    from output.stage6 import report as report_mod

    summary = _full_result_for_report()
    out = Path(report_mod.__file__).parent  # write to a temp via the function instead
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        arts = report_mod.generate_report(summary, Path(d))
        dash_text = Path(arts["dashboard"]).read_text()
    # The numbers must all come from the (stubbed but real) result, never the
    # demo constants. 0.314 was the fabricated trained score; it must not appear.
    assert "0.314" not in dash_text
    parsed = json.loads(dash_text)
    # The win-rate is whatever the stubbed (but real) result computed — the point
    # is it is a REAL number from the result, not the fabricated demo constant.
    assert isinstance(parsed["trained_student_win_rate"], float)
    assert 0.0 <= parsed["trained_student_win_rate"] <= 1.0
