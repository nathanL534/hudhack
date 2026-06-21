"""Credential-free tests for Stages 2–4 scaffolding."""

import asyncio
from pathlib import Path

import pytest

from engine.games import GAME_1
from eval.proxy_sweep import run_proxy_sweep
from harness.modal_fanout import (
    FanoutConfig,
    FanoutPlayerTrainer,
    PlayerFanout,
    aggregate_experiment_result,
)
from training.fireworks_teacher import FireworksTeacher
from training.reward_service import (
    RewardRequest,
    RewardService,
    _extract_rollout_id,
    _parse_curriculum,
    create_fastapi_app,
)


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


def test_fireworks_teacher_offline_mode_needs_no_key():
    """offline() returns a valid, schema-clamped params dict with no API key."""
    teacher = FireworksTeacher.offline()
    params = teacher.generate(GAME_1)
    assert set(params) == set(GAME_1.param_schema)
    for key, (low, high) in GAME_1.param_schema.items():
        assert low <= params[key] <= high


def test_fireworks_teacher_strict_rejects_junk_only_response():
    """A response with no known parameter keys is rejected, not silently defaulted."""
    teacher = FireworksTeacher(
        transport=lambda payload: {
            "choices": [{"message": {"content": '{"totally_unknown": 1}'}}]
        },
        max_retries=0,
    )
    with pytest.raises(ValueError):
        teacher.generate(GAME_1)


def test_fireworks_teacher_to_curriculum_validates_against_contract():
    """The Teacher -> CurriculumSpec bridge emits a contract-valid curriculum."""
    from contracts import CurriculumSpec

    teacher = FireworksTeacher.offline({"difficulty": 0.7, "obstacle_density": 0.3})
    curriculum = teacher.to_curriculum(GAME_1, curriculum_id="c-1", n_arenas=2)
    assert isinstance(curriculum, CurriculumSpec)
    assert curriculum.curriculum_id == "c-1"
    assert len(curriculum.arenas) == 2
    # Round-trips through the frozen contract (proves strict validation passed).
    assert CurriculumSpec.model_validate_json(curriculum.model_dump_json()).game_id == GAME_1.name


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


def test_fanout_player_trainer_conforms_to_interfaces_and_aggregates():
    """The local-fallback fan-out conforms to PlayerTrainer/TrainingJob and
    aggregates per-seed MatchResults into a frozen ExperimentResult."""
    from contracts import ExperimentResult, MatchResult, PlayerConfig, TrainingBudget
    from harness.interfaces import PlayerTrainer, TrainingJob
    from modal_player import local_smoke_worker

    trainer = FanoutPlayerTrainer(local_smoke_worker)
    assert isinstance(trainer, PlayerTrainer)

    matches = []
    for seed in (0, 1, 2, 3):
        config = PlayerConfig(
            architecture="mlp",
            num_seeds=1,
            budget=TrainingBudget(episodes=1),
            seed=seed,
        )
        job = trainer.submit(config, arenas=[])
        assert isinstance(job, TrainingJob)
        assert job.is_done()
        match = job.result()
        assert isinstance(match, MatchResult)
        assert match.seed == seed
        matches.append(match)

    result = aggregate_experiment_result(
        matches, reward=0.42, curricula_ids=["c-1"]
    )
    assert isinstance(result, ExperimentResult)
    assert result.n_seeds_completed == 4
    assert result.reward == 0.42
    assert "mlp" in result.per_seed_scores
    assert len(result.per_seed_scores["mlp"]) == 4


# ---------------------------------------------------------------------------
# Stage 3: async /init reward service (the silent-corruption fix)
# ---------------------------------------------------------------------------


def test_init_handler_is_async():
    """The /init route must be a coroutine so it never blocks the event loop."""
    service = RewardService(lambda params: params.get("raw_reward", 0.0))
    app = create_fastapi_app(service)
    init_route = next(r for r in app.routes if getattr(r, "path", None) == "/init")
    assert asyncio.iscoroutinefunction(init_route.endpoint)


def test_rollout_id_read_from_metadata_envelope():
    """rollout_id is read from top level OR the metadata envelope, via ScoringContext."""
    assert _extract_rollout_id({"rollout_id": "top"}) == "top"
    assert _extract_rollout_id({"metadata": {"rollout_id": "meta"}}) == "meta"
    assert _extract_rollout_id({}) is None


def test_init_runs_scoring_in_background_and_result_is_collectable(caplog):
    """/init schedules a per-rollout background task; /result collects the reward
    once it lands.  Every background log line carries the rollout_id so n=4
    concurrent rollouts can never cross-tag (the silent-corruption guard)."""
    from fastapi.testclient import TestClient

    service = RewardService(lambda params: params["raw_reward"])
    client = TestClient(create_fastapi_app(service))

    with caplog.at_level("INFO", logger="crucible.reward_service"):
        accepted = client.post(
            "/init",
            json={"params": {"raw_reward": 0.7}, "rollout_id": "rollout-async-1"},
        )
        assert accepted.status_code == 202

        # Poll /result until the background task lands (it runs off the loop).
        for _ in range(50):
            result = client.get("/result/rollout-async-1").json()
            if result.get("status") != "pending":
                break

    assert result["rollout_id"] == "rollout-async-1"
    assert result["reward"] == 0.7
    # Every background log line is tagged with this rollout_id, not a bare message.
    rollout_logs = [r for r in caplog.records if r.name == "crucible.reward_service"]
    assert rollout_logs, "background task should emit tagged logs"
    assert all("rollout-async-1" in r.getMessage() for r in rollout_logs)


def test_init_returns_fast_202_and_validates_curriculum():
    """/init returns a fast 202 for a valid request and 422 for a malformed curriculum."""
    from fastapi.testclient import TestClient

    service = RewardService(lambda params: params.get("raw_reward", 0.5))
    client = TestClient(create_fastapi_app(service))

    ok = client.post(
        "/init",
        json={
            "params": {"raw_reward": 0.8},
            "metadata": {"rollout_id": "rollout-xyz"},
        },
    )
    assert ok.status_code == 202
    assert ok.json()["rollout_id"] == "rollout-xyz"

    bad = client.post(
        "/init",
        json={
            "params": {"raw_reward": 0.8},
            # map_size=999 violates ArenaSpec (le=64) -> 422 before any scoring.
            "curriculum": {
                "game_id": "g",
                "curriculum_id": "c",
                "arenas": [
                    {
                        "map_size": 999,
                        "doors": 0,
                        "keys": 0,
                        "hazard_density": 0.0,
                        "difficulty": 0.5,
                    }
                ],
            },
        },
    )
    assert bad.status_code == 422


def test_parse_curriculum_accepts_valid_and_rejects_invalid():
    """The frozen CurriculumSpec is the validation authority at parse time."""
    from pydantic import ValidationError

    spec = _parse_curriculum(
        {
            "curriculum": {
                "game_id": "g",
                "curriculum_id": "c",
                "arenas": [
                    {
                        "map_size": 9,
                        "doors": 1,
                        "keys": 1,
                        "hazard_density": 0.1,
                        "difficulty": 0.4,
                    }
                ],
            }
        }
    )
    assert spec is not None
    assert spec.curriculum_id == "c"
    assert _parse_curriculum({}) is None  # bare params request, no curriculum

    with pytest.raises(ValidationError):
        _parse_curriculum({"curriculum": {"game_id": "g", "curriculum_id": "c", "arenas": []}})

