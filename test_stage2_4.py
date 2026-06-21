"""Credential-free tests for Stages 2–4 scaffolding."""

from pathlib import Path

import pytest

from engine.games import GAME_1
from eval.proxy_sweep import run_proxy_sweep
from harness.modal_fanout import FanoutConfig, PlayerFanout
from training.fireworks_teacher import FireworksTeacher
from training.reward_service import RewardRequest, RewardService


def test_proxy_sweep_detects_positive_correlation(tmp_path: Path):
    candidates = [{"difficulty": value} for value in (0.1, 0.3, 0.6, 0.9)]
    report = run_proxy_sweep(
        candidates,
        proxy_fn=lambda p: p["difficulty"],
        learning_fn=lambda p, seed: p["difficulty"] + seed * 0.001,
        seeds=[1, 2, 3],
        output_path=tmp_path / "sweep.json",
    )
    assert report.pearson_r > 0.99
    assert report.spearman_r > 0.99
    assert (tmp_path / "sweep.json").exists()


def test_fireworks_teacher_parses_and_clamps_json():
    teacher = FireworksTeacher(
        transport=lambda payload: {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"difficulty": 4, "obstacle_density": -1, '
                            '"goal_distance": 0.7, "ignored": 9}'
                        )
                    }
                }
            ]
        }
    )
    params = teacher.generate(GAME_1)
    assert params["difficulty"] == 1.0
    assert params["obstacle_density"] == 0.0
    assert params["goal_distance"] == pytest.approx(0.7)
    assert "ignored" not in params


def test_reward_service_preserves_rollout_id_and_clamps():
    service = RewardService(lambda params: params["raw_reward"])
    response = service.score(
        RewardRequest(params={"raw_reward": 1.7}, rollout_id="rollout-123")
    )
    assert response.reward == 1.0
    assert response.rollout_id == "rollout-123"


def test_local_fanout_is_seed_ordered():
    worker = lambda payload, seed: {
        "seed": seed,
        "score": payload["offset"] + seed,
    }
    fanout = PlayerFanout(
        FanoutConfig(backend="local", max_workers=3),
        local_worker=worker,
    )
    results = fanout.run({"offset": 10}, [4, 1, 9])
    assert [row["seed"] for row in results] == [4, 1, 9]
    assert [row["score"] for row in results] == [14, 11, 19]

