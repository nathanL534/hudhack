"""test_stage6.py — focused unit tests for the Stage-6 final-evaluation architecture.

Covers the load-bearing logic with NO Modal credits and NO Fireworks calls:

  * aggregation: mean / sample_std / confidence interval / diversity.
  * anti-gaming checks: every failure mode + the smoke exemption.
  * optional-game handling: Target Knockback reports "not installed" cleanly;
    KOTH is a probe (not a Teacher game).
  * handle resolution: Fireworks id vs local-adapter path dispatch (no network).
  * config fairness invariant: one config builds identical payloads for both models.
  * CLI validation: required-arg and bad-value paths.
  * end-to-end orchestration with STUB teachers + a STUB worker (no Modal/Fireworks):
    proves the single fan-out, aggregation, anti-gaming, and structured JSON all
    wire up — and that base-vs-base is correctly flagged as same-model in a real run.

Run:  .venv/bin/python -m pytest test_stage6.py -q
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from output.stage6 import aggregate, anti_gaming
from output.stage6.config import EvalConfig, smoke_config
from output.stage6.evaluator import EvalRequest, run_eval
from output.stage6.games import (
    GameNotInstalled,
    get_game,
    probe_games,
    teacher_games,
)
from output.stage6.handles import (
    ResolvedTeacher,
    _looks_like_fireworks,
    _looks_like_local_path,
    resolve_local_adapter,
)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def test_mean_and_std():
    assert aggregate.mean([]) == 0.0
    assert aggregate.mean([2.0, 4.0]) == 3.0
    assert aggregate.sample_std([5.0]) == 0.0           # n<2 -> 0
    # std of [2,4,4,4,5,5,7,9] is 2.138... (sample, n-1)
    s = aggregate.sample_std([2, 4, 4, 4, 5, 5, 7, 9])
    assert abs(s - 2.13809) < 1e-3


def test_confidence_interval_known_value():
    # n=5, mean=5, sample std=2 -> se=2/sqrt(5)=0.8944, t(4,95%)=2.776
    xs = [3.0, 4.0, 5.0, 6.0, 7.0]
    ci = aggregate.mean_confidence_interval(xs)
    assert abs(ci.mean - 5.0) < 1e-9
    assert abs(ci.std - math.sqrt(2.5)) < 1e-9        # sample std = sqrt(2.5)=1.5811
    expected_half = 2.776 * (math.sqrt(2.5) / math.sqrt(5))
    assert abs(ci.half_width - expected_half) < 1e-3
    assert ci.low < ci.mean < ci.high


def test_confidence_interval_single_sample_zero_width():
    ci = aggregate.mean_confidence_interval([0.42])
    assert ci.mean == 0.42 and ci.low == 0.42 and ci.high == 0.42
    assert ci.half_width == 0.0


def test_confidence_interval_large_n_uses_normal():
    xs = [float(i % 3) for i in range(40)]  # df=39 -> z=1.96
    ci = aggregate.mean_confidence_interval(xs)
    se = aggregate.sample_std(xs) / math.sqrt(len(xs))
    assert abs(ci.half_width - 1.96 * se) < 1e-9


def test_parameter_diversity_distinct_vs_collapsed():
    keys = ("difficulty", "platform_width")
    distinct = [
        {"difficulty": 0.4, "platform_width": 10.0},
        {"difficulty": 0.6, "platform_width": 14.0},
        {"difficulty": 0.8, "platform_width": 18.0},
    ]
    div = aggregate.parameter_diversity(distinct, keys)
    assert div["n_unique"] == 3 and div["unique_fraction"] == 1.0
    assert div["collapsed"] is False
    assert div["per_param_std"]["difficulty"] > 0

    collapsed = [{"difficulty": 0.5, "platform_width": 12.0}] * 4
    cdiv = aggregate.parameter_diversity(collapsed, keys)
    assert cdiv["n_unique"] == 1 and cdiv["collapsed"] is True
    assert cdiv["mean_param_std"] == 0.0

    assert aggregate.parameter_diversity([], keys)["collapsed"] is True


# ---------------------------------------------------------------------------
# anti-gaming checks
# ---------------------------------------------------------------------------


def test_transfer_followed_reward():
    assert anti_gaming.check_transfer_followed_reward(0.1, 0.2).passed       # improved
    assert not anti_gaming.check_transfer_followed_reward(0.2, 0.1).passed   # regressed
    # training rose but transfer didn't -> FAIL
    c = anti_gaming.check_transfer_followed_reward(0.3, 0.2, train_signal_rose=True)
    assert not c.passed


def test_arena_collapse_check():
    good = anti_gaming.check_arena_collapse({"collapsed": False, "n_unique": 5, "n_arenas": 5})
    bad = anti_gaming.check_arena_collapse({"collapsed": True, "n_unique": 1, "n_arenas": 5})
    assert good.passed and not bad.passed


def test_clamp_fraction_check_boundary():
    assert anti_gaming.check_clamp_fraction(0.10).passed       # exactly at limit OK
    assert anti_gaming.check_clamp_fraction(0.05).passed
    assert not anti_gaming.check_clamp_fraction(0.11).passed   # over 10% fails


def test_seed_noise_check():
    assert anti_gaming.check_seed_noise(0.20, 0.10).passed        # std < improvement
    assert not anti_gaming.check_seed_noise(0.10, 0.25).passed    # std > improvement
    # negative improvement with small std is a real (bad) effect, not noise.
    assert anti_gaming.check_seed_noise(-0.20, 0.05).passed


def test_train_game_only_check():
    skip = anti_gaming.check_train_game_only(0.2, None)
    assert skip.passed and skip.severity == "skip"
    overfit = anti_gaming.check_train_game_only(0.2, {"koth": -0.1})
    assert not overfit.passed
    genuine = anti_gaming.check_train_game_only(0.2, {"koth": 0.05})
    assert genuine.passed


def test_same_model_check_smoke_vs_real():
    real = anti_gaming.check_distinct_models("A", "A", is_smoke=False)
    assert not real.passed                                # same model in a real run fails
    distinct = anti_gaming.check_distinct_models("A", "B", is_smoke=False)
    assert distinct.passed
    smoke = anti_gaming.check_distinct_models("A", "A", is_smoke=True)
    assert smoke.passed and smoke.severity == "skip"      # base vs base is fine in smoke


def test_run_all_aggregates_failures():
    rep = anti_gaming.run_all(
        base_mean=0.3, trained_mean=0.1,                  # regressed -> G1 fail
        trained_diversity={"collapsed": True, "n_unique": 1, "n_arenas": 5},  # G2 fail
        clamp_fraction=0.5,                                # G3 fail
        seed_std=0.4, base_resolved_id="A", trained_resolved_id="A",
        is_smoke=False,
    )
    assert rep.any_failed and not rep.passed
    assert "reward_up_but_no_transfer" in rep.as_dict()["failed"]
    assert "arena_collapse" in rep.as_dict()["failed"]
    assert "same_model" in rep.as_dict()["failed"]


# ---------------------------------------------------------------------------
# game registry / optional-game handling
# ---------------------------------------------------------------------------


def test_fighter_is_installed_teacher_game():
    g = get_game("fighter")
    assert g.installed and g.role == "teacher"
    assert "fighter" in teacher_games()
    # alias resolves
    assert get_game("ring-out-duel").name == "fighter"


def test_target_knockback_installed_and_routes():
    # Game-2 is now merged onto this branch, so the install probe flips True and
    # TK routes as a real Teacher game through the identical evaluator path.
    g = get_game("target_knockback")
    assert g.installed is True
    assert g.role == "teacher"
    # the 7 real TK knobs are registered (5 shared fighter knobs + 2 zone dials)
    assert g.param_keys == (
        "difficulty",
        "platform_width",
        "gravity",
        "knockback",
        "spawn_gap",
        "zone_half",
        "zone_center_frac",
    )
    # teacher_game() builds cleanly now that games.target_knockback exports the schema
    tk_game = g.teacher_game()
    assert tk_game.name == "target_knockback"
    # included as a Teacher training game now that it's installed
    assert "target_knockback" in teacher_games()


def test_koth_is_probe_only():
    g = get_game("koth")
    assert g.role == "probe"
    assert "koth" in probe_games()
    assert "koth" not in teacher_games()              # never a Teacher training game


# ---------------------------------------------------------------------------
# handle resolution
# ---------------------------------------------------------------------------


def test_handle_classification():
    assert _looks_like_fireworks("accounts/fireworks/models/qwen3-4b")
    assert _looks_like_fireworks("fw:qwen3-4b-rft")
    assert not _looks_like_local_path("accounts/fireworks/models/qwen3-4b")


def test_resolve_local_adapter_records_path(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    rt = resolve_local_adapter("trained", f"local:{adapter}")
    assert rt.kind == "local"
    assert str(adapter) in rt.resolved_id
    assert _looks_like_local_path(str(adapter))       # bare existing path -> local


def test_resolve_local_adapter_missing_path_fails():
    with pytest.raises(RuntimeError):
        resolve_local_adapter("trained", "local:/no/such/adapter/path/xyz")


def test_offline_handle_builds_keyless_teacher(monkeypatch):
    """The cheap-smoke 'offline' handle resolves with NO Fireworks key, fighter-valid."""
    from output.stage6.games import get_game
    from output.stage6.handles import resolve_handle

    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    rt = resolve_handle("base", "offline")
    assert rt.kind == "offline"
    teacher = rt.build()                              # no network, no key
    params = teacher.generate(get_game("fighter").teacher_game())
    assert set(params) >= {"difficulty", "platform_width", "gravity"}


# ---------------------------------------------------------------------------
# fairness invariant
# ---------------------------------------------------------------------------


def test_config_builds_identical_payloads_for_both_models():
    cfg = EvalConfig(ppo_episodes=500, eval_seeds=20, arenas_per_model=2)
    params = {"difficulty": 0.5, "platform_width": 12.0, "gravity": 0.6,
              "knockback": 2.5, "spawn_gap": 4.0}
    held = [{"difficulty": 0.4, "platform_width": 7.0}]
    keys = ("difficulty", "platform_width", "gravity", "knockback", "spawn_gap")
    pb = cfg.train_payload(params, held, param_keys=keys, curriculum_id="base-a0-s1")
    pt = cfg.train_payload(params, held, param_keys=keys, curriculum_id="trained-a0-s1")
    # Everything except curriculum_id is identical (the invariant).
    for k in ("ppo_episodes", "eval_seeds", "architecture", "held_out_arenas",
              "difficulty", "platform_width", "gravity", "knockback", "spawn_gap"):
        assert pb[k] == pt[k]
    assert pb["ppo_episodes"] == 500 and pb["eval_seeds"] == 20
    assert cfg.fingerprint() == EvalConfig(ppo_episodes=500, eval_seeds=20,
                                           arenas_per_model=2).fingerprint()


def test_smoke_config_is_small():
    c = smoke_config()
    assert c.arenas_per_model == 2 and c.player_seeds == (1, 2)


# ---------------------------------------------------------------------------
# end-to-end orchestration with stubs (no Modal, no Fireworks)
# ---------------------------------------------------------------------------


class _StubTeacher:
    """Emits a fixed list of param dicts (round-robin) — no network."""

    def __init__(self, arenas):
        self._arenas = arenas
        self._i = 0

    def generate(self, game):
        a = self._arenas[self._i % len(self._arenas)]
        self._i += 1
        return dict(a)


def _stub_resolver(base_arenas, trained_arenas):
    def _resolve(label, handle, **kw):
        arenas = base_arenas if label == "base" else trained_arenas
        rid = f"stub::{handle}"
        return ResolvedTeacher(
            label=label, handle=handle, resolved_id=rid, kind="fireworks",
            _build=lambda *, gen_seed=None: _StubTeacher(arenas),
        )
    return _resolve


def _stub_worker_factory(improvement_by_model):
    """A worker that returns a deterministic transfer row keyed by curriculum label."""
    def _run(payloads, seeds, *, backend):
        rows = []
        for p, s in zip(payloads, seeds):
            cid = p["curriculum_id"]
            model = cid.split("-")[0]
            impr = improvement_by_model[model]
            rows.append({
                "status": "ppo_transfer",
                "held_out_improvement": impr,
                "before_winrate": 0.1,
                "after_winrate": 0.1 + impr,
                "train_arena_winrate": 0.2,
                "seed": s,
            })
        return rows
    return _run


def test_end_to_end_trained_beats_base(tmp_path):
    cfg = EvalConfig(arenas_per_model=2, player_seeds=(1, 2), ppo_episodes=10, eval_seeds=2)
    base_arenas = [
        {"difficulty": 0.4, "platform_width": 10.0, "gravity": 0.6, "knockback": 2.0, "spawn_gap": 4.0},
        {"difficulty": 0.6, "platform_width": 14.0, "gravity": 0.7, "knockback": 3.0, "spawn_gap": 5.0},
    ]
    trained_arenas = [
        {"difficulty": 0.5, "platform_width": 12.0, "gravity": 0.6, "knockback": 2.5, "spawn_gap": 4.5},
        {"difficulty": 0.7, "platform_width": 16.0, "gravity": 0.8, "knockback": 3.5, "spawn_gap": 6.0},
    ]
    req = EvalRequest(
        base_handle="base-x", trained_handle="trained-y", game="fighter",
        backend="local", config=cfg, is_smoke=False, verbose=False,
    )
    summary = run_eval(
        req,
        resolve=_stub_resolver(base_arenas, trained_arenas),
        run_jobs=_stub_worker_factory({"base": 0.10, "trained": 0.30}),
    )
    assert summary["trained_beats_base"] is True
    assert summary["delta"] == pytest.approx(0.20, abs=1e-6)
    assert summary["n_jobs"] == 2 * 2 * 2     # 2 models x 2 arenas x 2 seeds
    # distinct stub ids -> same_model check passes -> overall pass
    assert summary["overall_pass"] is True
    # per-model CI present and structured-output fields exist
    for m in summary["per_model"]:
        assert "ci" in m and "diversity" in m and "generation" in m
    assert summary["config"]["fingerprint"]
    assert summary["json_validation"]["clamp_fraction"] == 0.0


def test_end_to_end_same_model_flags_failure():
    """Base and trained resolving to the SAME id in a real run must FAIL (G6)."""
    cfg = EvalConfig(arenas_per_model=1, player_seeds=(1,), ppo_episodes=10, eval_seeds=2)
    arenas = [{"difficulty": 0.5, "platform_width": 12.0, "gravity": 0.6,
               "knockback": 2.5, "spawn_gap": 4.5}]

    def _same_resolver(label, handle, **kw):
        return ResolvedTeacher(label=label, handle=handle, resolved_id="SAME",
                               kind="fireworks",
                               _build=lambda *, gen_seed=None: _StubTeacher(arenas))

    req = EvalRequest(base_handle="a", trained_handle="b", game="fighter",
                      backend="local", config=cfg, is_smoke=False, verbose=False)
    summary = run_eval(req, resolve=_same_resolver,
                       run_jobs=_stub_worker_factory({"base": 0.1, "trained": 0.2}))
    assert "same_model" in summary["anti_gaming"]["failed"]
    assert summary["overall_pass"] is False    # even though trained_beats_base


def test_hud_traces_disabled_by_default():
    from output.stage6 import hud_traces
    assert hud_traces.capture_traces({"base": [{"difficulty": 0.5}]}, {}, enabled=False) == []


def test_run_eval_rejects_probe_game():
    req = EvalRequest(base_handle="a", trained_handle="b", game="koth",
                      backend="local", config=EvalConfig(), is_smoke=False, verbose=False)
    with pytest.raises(ValueError):
        run_eval(req, resolve=_stub_resolver([{}], [{}]),
                 run_jobs=_stub_worker_factory({"base": 0.1, "trained": 0.2}))


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------


def _cli_main(argv):
    from output.eval_base_vs_trained import main

    return main(argv)


def test_cli_requires_trained_for_real_run():
    assert _cli_main(["--game", "fighter"]) == 2          # missing --trained


def test_cli_rejects_bad_arenas():
    assert _cli_main(["--trained", "x", "--arenas", "0"]) == 2


def test_cli_report_only_requires_result():
    assert _cli_main(["--report-only"]) == 2


def test_cli_list_games_ok(capsys):
    assert _cli_main(["--list-games"]) == 0
    out = capsys.readouterr().out
    assert "fighter" in out and "target_knockback" in out and "koth" in out


def test_cli_unknown_game():
    assert _cli_main(["--trained", "x", "--game", "nope"]) == 2
