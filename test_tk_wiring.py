"""test_tk_wiring.py — proves Target Knockback is WIRED into the shared pipeline.

These tests cover the 6 wiring items WITHOUT Modal or Fireworks credentials (the
heavy PPO fan-out and the live RFT are exercised by the dry round-trip + the deployed
crucible-player-tk app; here we prove the SEAMS line up):

  1. shared schema/router      -> games_registry registers TK with a distinct game,
                                  isolated worker, isolated bridge.
  2. Modal PPO reward path     -> output.nested_reward_tk targets crucible-player-tk and
                                  reads the TK worker schema (improvement / ppo_tk).
  3. HUD task/grader           -> training.tk_hud_teacher_env serves a SEPARATE env and
                                  grades a TK answer through the shared grading core.
  4. EP / RFT dataset routing  -> the clean TK dataset row shape + the TK RFT evaluator
                                  point at the TK bridge, not the fighter bridge.
  5. replay capture            -> the wired nested-reward path persists a viewer-valid
                                  TK replay (driven via the local worker).
  6. round-trip parsing        -> a TK answer validates+clamps through the shared core
                                  and the nested TK reward agrees Modal/HUD/EP code paths
                                  (local backend, fully in-process).
"""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# 1. Shared schema / router (games_registry) — TK plugs in, isolated from Ring-Out
# ---------------------------------------------------------------------------


def test_registry_has_both_games_with_distinct_routing():
    import games_registry as reg

    assert set(reg.games()) >= {"ring_out", "target_knockback"}
    ring = reg.get("ring_out")
    tk = reg.get("target_knockback")

    # The two reward paths coexist precisely because each names its OWN worker + bridge.
    assert ring.modal_app == "crucible-player"
    assert tk.modal_app == "crucible-player-tk"
    assert ring.modal_app != tk.modal_app
    assert ring.bridge_app != tk.bridge_app
    assert ring.bridge_url_env != tk.bridge_url_env


def test_registry_unknown_game_raises():
    import games_registry as reg

    with pytest.raises(KeyError):
        reg.get("no_such_game")


def test_tk_schema_extends_not_forks_ring_out():
    # "extend, don't fork": TK carries the SAME fighter knobs PLUS the TK zone dials.
    import games_registry as reg

    ring = set(reg.get("ring_out").bounds)
    tk = set(reg.get("target_knockback").bounds)
    assert ring.issubset(tk)  # every Ring-Out knob is present in TK
    assert {"zone_half", "zone_center_frac"}.issubset(tk)  # plus the TK-specific dials


# ---------------------------------------------------------------------------
# 2. Modal PPO reward path — TK nested reward targets the ISOLATED worker
# ---------------------------------------------------------------------------


def test_nested_reward_tk_targets_isolated_worker():
    import output.nested_reward_tk as nr

    assert nr.MODAL_APP == "crucible-player-tk"
    assert nr.MODAL_FN == "train_tk_player"
    # It must NEVER point at the fighter worker.
    assert nr.MODAL_APP != "crucible-player"


def test_nested_reward_tk_reads_tk_worker_schema():
    # The reward fans seeds out and reads the TK worker's schema. Drive it with a fake
    # worker (no Modal) to prove it reads `improvement` and asserts status `ppo_tk`.
    import output.nested_reward_tk as nr

    rows = [
        {"status": "ppo_tk", "improvement": 0.30, "before_winrate": 0.40, "after_winrate": 0.70},
        {"status": "ppo_tk", "improvement": 0.20, "before_winrate": 0.45, "after_winrate": 0.65},
    ]

    def fake_run(payloads, seeds, *, backend):
        return rows

    orig = nr._run_seeds
    nr._run_seeds = fake_run
    try:
        sink: dict = {}
        reward = nr.tk_teacher_reward(
            {"difficulty": 0.55, "platform_width": 12.0, "zone_half": 1.6},
            backend="local", seeds=(1, 2), _detail_sink=sink,
        )
    finally:
        nr._run_seeds = orig

    assert reward == pytest.approx(0.25, abs=1e-9)  # mean(0.30, 0.20)
    assert sink["mean_after"] == pytest.approx(0.675, abs=1e-9)


def test_nested_reward_tk_rejects_fighter_worker_rows():
    # A fighter worker row (status ppo_transfer) must be REJECTED, never mis-scored.
    import output.nested_reward_tk as nr

    def fake_run(payloads, seeds, *, backend):
        return [{"status": "ppo_transfer", "held_out_improvement": 0.9}]

    orig = nr._run_seeds
    nr._run_seeds = fake_run
    try:
        with pytest.raises(AssertionError):
            nr.tk_teacher_reward({"difficulty": 0.55}, backend="local", seeds=(1,))
    finally:
        nr._run_seeds = orig


def test_tk_arena_payload_threads_geometry_verbatim():
    import output.nested_reward_tk as nr

    p = nr.tk_arena_payload(
        {"difficulty": 0.55, "platform_width": 14.0, "zone_half": 2.0, "zone_center_frac": 0.45,
         "gravity": 0.7, "knockback": 3.0, "spawn_gap": 5.0},
        episodes=400, eval_seeds=20, curriculum_id="t",
    )
    for k in ("platform_width", "zone_half", "zone_center_frac", "gravity", "knockback", "spawn_gap"):
        assert k in p
    assert p["difficulty"] == 0.55
    assert p["ppo_episodes"] == 400


# ---------------------------------------------------------------------------
# 3. HUD task / grader — separate env, shared grading core
# ---------------------------------------------------------------------------


def test_tk_hud_env_is_separate_from_fighter():
    import training.tk_hud_teacher_env as tkenv

    assert tkenv.env.name == "crucible-teacher-tk"
    # The TK bounds come from the registry and include the zone dials.
    assert {"zone_half", "zone_center_frac"}.issubset(tkenv.TK_BOUNDS)


def test_tk_hud_gap_scorer_is_load_bearing():
    # The default served scorer runs the REAL TK sim (gap proxy) on local CPU.
    from training.tk_hud_teacher_env import tk_gap_scorer

    reward = tk_gap_scorer(
        {"difficulty": 0.55, "platform_width": 12.0, "zone_half": 1.6, "zone_center_frac": 0.5}
    )
    assert 0.0 <= reward <= 1.0


def test_tk_hud_template_smoke_grades_an_answer():
    import asyncio

    from training.tk_hud_teacher_env import TK_BOUNDS, run_template_smoke, tk_gap_scorer

    answer = json.dumps(
        {"difficulty": 0.55, "platform_width": 12.0, "gravity": 0.6, "knockback": 2.5,
         "spawn_gap": 4.0, "zone_half": 1.6, "zone_center_frac": 0.5}
    )
    reward = asyncio.run(run_template_smoke(answer, bounds=TK_BOUNDS, scorer=tk_gap_scorer))
    assert 0.0 <= reward <= 1.0


def test_tk_params_to_curriculum_uses_tk_game_id():
    from training.tk_hud_teacher_env import params_to_tk_curriculum

    spec = params_to_tk_curriculum({"difficulty": 0.55, "platform_width": 12.0})
    assert spec.game_id == "target_knockback"
    assert len(spec.arenas) == 1
    assert spec.arenas[0].difficulty == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# 4. EP / RFT dataset routing — clean rows, TK bridge (not the fighter bridge)
# ---------------------------------------------------------------------------


def test_tk_dataset_rows_are_clean():
    from training.make_tk_dataset import build_rows

    rows = build_rows(2)
    assert len(rows) == 2
    for row in rows:
        # Exactly the accepted shape — NO stale runtime fields.
        assert set(row) == {"messages", "input_metadata"}
        assert row["input_metadata"] == {"completion_params": {}}
        assert row["messages"][0]["role"] == "user"
        for bad in ("rollout_status", "execution_metadata", "created_at"):
            assert bad not in row
        # The prompt is the registry TK prompt (single source of truth).
        assert "Target Knockback" in row["messages"][0]["content"]


def test_tk_dataset_row_is_valid_jsonl():
    from training.make_tk_dataset import build_rows

    line = json.dumps(build_rows(1)[0])
    reparsed = json.loads(line)
    assert reparsed["messages"][0]["content"].startswith("You design RL training curricula")


def test_tk_rft_evaluator_points_at_tk_bridge(monkeypatch):
    # The TK evaluator reads CRUCIBLE_TK_BRIDGE_URL — NOT the fighter's CRUCIBLE_BRIDGE_URL.
    import importlib

    monkeypatch.setenv("CRUCIBLE_TK_BRIDGE_URL", "http://localhost:9999")
    monkeypatch.setenv("CRUCIBLE_BRIDGE_URL", "http://fighter-should-not-be-used:1")
    import training.rft_evaluator_tk as ev

    importlib.reload(ev)
    try:
        assert ev.BRIDGE_URL == "http://localhost:9999"
        assert "Target Knockback" in ev.TK_PROMPT
    finally:
        importlib.reload(ev)  # restore module-level default for other tests


def test_tk_bridge_app_name_is_isolated():
    # Import the bridge module's registry-sourced name without deploying anything.
    import games_registry as reg

    assert reg.get("target_knockback").bridge_app == "crucible-ep-bridge-tk"


# ---------------------------------------------------------------------------
# 5. Replay capture — the wired path persists a viewer-valid TK replay
# ---------------------------------------------------------------------------


def test_tk_worker_replay_is_viewer_valid():
    # Drive the LOCAL TK worker with a tiny budget + capture flag and validate the
    # returned replay against the shared replay schema (the same path the dry loop and
    # the deployed worker use to produce viewer-discoverable replays).
    pytest.importorskip("stable_baselines3")
    from modal_player_tk import local_tk_worker
    from replay import validate_replay

    payload = {
        "difficulty": 0.5,
        "platform_width": 12.0,
        "zone_half": 1.6,
        "ppo_episodes": 50,   # tiny — we are testing the replay plumbing, not learning
        "eval_seeds": 4,
        "curriculum_id": "tk-replay-test",
        "capture_replay_id": "tk_wiring_replay_test",
    }
    row = local_tk_worker(payload, seed=1)
    assert row["status"] == "ppo_tk"
    assert "replay" in row
    assert validate_replay(row["replay"]["data"]) == []
    assert row["replay"]["data"]["meta"]["p1_policy"] == "modal_trained_ppo_tk"


# ---------------------------------------------------------------------------
# 6. Round-trip parsing — shared core agrees with the TK reward code paths
# ---------------------------------------------------------------------------


def test_tk_answer_validates_and_clamps_through_shared_core():
    from training.hud_teacher_env import score_teacher_answer
    from training.tk_hud_teacher_env import TK_BOUNDS, tk_gap_scorer

    # Out-of-range knockback must clamp into [0.5, 6.0]; the zone dials must survive.
    answer = json.dumps(
        {"difficulty": 0.55, "platform_width": 12.0, "gravity": 0.6, "knockback": 99.0,
         "spawn_gap": 4.0, "zone_half": 1.6, "zone_center_frac": 0.5}
    )
    _, clamped = score_teacher_answer(answer, bounds=TK_BOUNDS, scorer=tk_gap_scorer)
    assert clamped["knockback"] == 6.0  # clamped to the high bound
    assert clamped["zone_half"] == 1.6
    assert clamped["zone_center_frac"] == 0.5


def test_tk_sidecar_publish_lookup_roundtrip(tmp_path, monkeypatch):
    # HUD == EP agreement relies on the sidecar returning the SAME published number.
    monkeypatch.setenv("CRUCIBLE_TK_SIDECAR_DIR", str(tmp_path))
    import importlib

    import training.tk_dry_loop_sidecar as side

    importlib.reload(side)
    try:
        side.reset()
        params = {"difficulty": 0.55, "platform_width": 12.0, "zone_half": 1.6}
        side.publish(params, 0.4242)
        assert side.served_tk_nested_scorer(params) == pytest.approx(0.4242)
    finally:
        importlib.reload(side)
