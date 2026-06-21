"""test_phase1.py — Phase 1 acceptance asserts.

Proves: curriculum validates, 2 arenas built, 4 seeds complete, the reward is
the REAL formula's output (not a hardcoded value), and ExperimentResult
round-trips through JSON.
"""

from __future__ import annotations

import json

import pytest

from contracts import (
    CurriculumSpec,
    ExperimentResult,
    PlayerConfig,
    ScoringResult,
    TrainingBudget,
)
from harness.fakes import FakeGameAdapter, FakePlayerTrainer, FakeTeacher
from harness.runner import DefaultExperimentRunner
from harness.scoring import LEARNABLE_BAND_THRESHOLD, score_curriculum

NUM_SEEDS = 4


@pytest.fixture
def teacher():
    return FakeTeacher()


@pytest.fixture
def game():
    return FakeGameAdapter()


@pytest.fixture
def player_config():
    return PlayerConfig(
        architecture="mlp",
        num_seeds=NUM_SEEDS,
        budget=TrainingBudget(episodes=10),
        seed=0,
        modal_parallel=False,
    )


def test_curriculum_validates(teacher):
    spec = teacher.generate(game_spec={"game_id": "gridworld"})
    assert isinstance(spec, CurriculumSpec)
    # .validate() returns a validated instance and does not raise.
    validated = spec.validate()
    assert isinstance(validated, CurriculumSpec)
    assert validated.curriculum_id == spec.curriculum_id


def test_curriculum_spec_rejects_invalid():
    # Out-of-range hazard_density must be rejected by the frozen contract.
    with pytest.raises(Exception):
        CurriculumSpec(
            game_id="gridworld",
            curriculum_id="bad",
            arenas=[
                {"map_size": 9, "doors": 2, "keys": 2, "hazard_density": 5.0, "difficulty": 0.5}
            ],
        )


def test_two_arenas_built(teacher, game):
    spec = teacher.generate(game_spec={}).validate()
    arenas = game.build(spec)
    assert len(arenas) == 2


def test_json_schema_exposed():
    schema = CurriculumSpec.model_json_schema()
    assert "properties" in schema
    assert "arenas" in schema["properties"]


def test_reward_is_real_formula(teacher, game):
    """The reward must equal the real formula's output, recomputed by hand."""
    spec = teacher.generate(game_spec={}).validate()
    scoring = score_curriculum(spec, game, mode="gap_proxy")
    assert isinstance(scoring, ScoringResult)

    # Recompute the formula independently from the fake scores.
    strong = game.evaluate(game.scripted_expert(), game.build(spec))["scripted_expert"][
        "mean_score"
    ]
    weak = game.evaluate(game.random_policy(), game.build(spec))["random_policy"]["mean_score"]
    p = weak
    expected = (strong - weak) if (p * (1.0 - p)) > LEARNABLE_BAND_THRESHOLD else 0.0

    assert scoring.reward == pytest.approx(expected)
    assert scoring.weak_score == pytest.approx(weak)
    assert scoring.strong_score == pytest.approx(strong)
    assert scoring.p == pytest.approx(p)
    # Sanity: the fake is configured to be in-band and non-zero.
    assert scoring.reward > 0.0


def test_four_seeds_complete(teacher, game, player_config):
    runner = DefaultExperimentRunner(FakePlayerTrainer())
    result = runner.run(teacher, game, player_config)
    assert result.n_seeds_completed == NUM_SEEDS
    assert len(result.per_seed_scores["mlp"]) == NUM_SEEDS


def test_runner_reward_matches_scorer(teacher, game, player_config):
    runner = DefaultExperimentRunner(FakePlayerTrainer())
    result = runner.run(teacher, game, player_config)
    spec = teacher.generate(game_spec={}).validate()
    scoring = score_curriculum(spec, game, mode="gap_proxy")
    assert result.reward == pytest.approx(scoring.reward)


def test_experiment_result_roundtrips_json(teacher, game, player_config):
    runner = DefaultExperimentRunner(FakePlayerTrainer())
    result = runner.run(teacher, game, player_config)

    blob = result.model_dump_json()
    # Valid JSON.
    parsed = json.loads(blob)
    assert "per_seed_scores" in parsed

    # Round-trips back into the frozen type, value-identical.
    restored = ExperimentResult.model_validate_json(blob)
    assert restored == result


def test_modal_flag_is_reserved_not_implemented(teacher, game):
    config = PlayerConfig(
        architecture="mlp",
        num_seeds=2,
        budget=TrainingBudget(episodes=5),
        seed=0,
        modal_parallel=True,
    )
    runner = DefaultExperimentRunner(FakePlayerTrainer())
    with pytest.raises(NotImplementedError):
        runner.run(teacher, game, config)


def test_ppo_improvement_mode_reserved(teacher, game):
    spec = teacher.generate(game_spec={}).validate()
    with pytest.raises(NotImplementedError):
        score_curriculum(spec, game, mode="ppo_improvement")
