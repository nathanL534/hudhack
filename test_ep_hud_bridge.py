from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from eval_protocol import InitRequest
from fastapi.testclient import TestClient
from eval_protocol.models import EvaluationRow

from training.ep_remote_server import create_ep_app, execute_rollout
from training.fireworks_rft_eval import apply_hud_reward
from training.hud_teacher_env import parse_teacher_params, run_template_smoke


BOUNDS = {
    "opponent_strength": (0.0, 1.0),
    "arena_width": (8.0, 30.0),
}


def make_request(rollout_id: str = "rollout-1") -> InitRequest:
    return InitRequest.model_validate(
        {
            "completion_params": {"model": "accounts/fireworks/models/qwen3-4b", "temperature": 0.8},
            "messages": [{"role": "user", "content": "Design a fighter curriculum"}],
            "model_base_url": "https://training-model.invalid/v1",
            "api_key": "test-key",
            "metadata": {
                "invocation_id": "inv-1",
                "experiment_id": "exp-1",
                "rollout_id": rollout_id,
                "run_id": "run-1",
                "row_id": "row-1",
            },
        }
    )


class FakeCompletions:
    async def create(self, **kwargs):
        assert kwargs["model"] == "accounts/fireworks/models/qwen3-4b"
        content = json.dumps({"opponent_strength": 0.4, "arena_width": 999})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeClient:
    chat = SimpleNamespace(completions=FakeCompletions())


def fake_client_factory(base_url: str, api_key: str):
    assert base_url == "https://training-model.invalid/v1"
    assert api_key == "test-key"
    return FakeClient()


def test_hud_template_prompt_answer_reward_round_trip():
    reward = asyncio.run(
        run_template_smoke(
            json.dumps({"opponent_strength": 0.25, "arena_width": 16}),
            bounds=BOUNDS,
            scorer=lambda params: params["opponent_strength"],
        )
    )
    assert reward == 0.25


def test_eval_protocol_rollout_calls_teacher_and_reports_hud_reward():
    reports = []
    result = asyncio.run(
        execute_rollout(
            make_request(),
            bounds=BOUNDS,
            scorer=lambda params: params["opponent_strength"],
            client_factory=fake_client_factory,
            reporter=lambda rollout_id, extras: reports.append((rollout_id, extras)),
        )
    )

    assert result.reward == 0.4
    assert result.params["arena_width"] == 30.0
    assert reports[0][0] == "rollout-1"
    assert reports[0][1]["hud_reward"] == 0.4
    assert reports[0][1]["messages"][-1]["role"] == "assistant"


def test_real_init_contract_accepts_and_finishes_background_rollout():
    reports = []
    app = create_ep_app(
        bounds=BOUNDS,
        scorer=lambda params: params["opponent_strength"],
        client_factory=fake_client_factory,
        reporter=lambda rollout_id, extras: reports.append((rollout_id, extras)),
        live_tracing=False,
    )
    with TestClient(app) as client:
        response = client.post("/init", json=make_request().model_dump(mode="json"))
        assert response.status_code == 202
        assert response.json()["rollout_id"] == "rollout-1"

        for _ in range(50):
            if "rollout-1" in app.state.results:
                break
            import time

            time.sleep(0.01)

    result = app.state.results["rollout-1"]
    assert result.reward == 0.4
    assert reports[0][1]["teacher_params"]["arena_width"] == 30.0


def test_debug_result_exposes_reward_without_request_or_credentials():
    app = create_ep_app(
        bounds=BOUNDS,
        scorer=lambda params: params["opponent_strength"],
        client_factory=fake_client_factory,
        reporter=lambda *_: None,
        live_tracing=False,
    )
    with TestClient(app) as client:
        client.post("/init", json=make_request().model_dump(mode="json"))
        for _ in range(50):
            response = client.get("/debug/result/rollout-1")
            if response.json()["status"] == "finished":
                break
            import time

            time.sleep(0.01)

    body = response.json()
    assert body == {
        "status": "finished",
        "rollout_id": "rollout-1",
        "reward": 0.4,
        "params": {"opponent_strength": 0.4, "arena_width": 30.0},
    }


def test_eval_protocol_evaluator_reads_hud_reward():
    row = EvaluationRow()
    row.execution_metadata.extra = {"hud_reward": 0.73}
    result = apply_hud_reward(row)
    assert result.evaluation_result.score == 0.73
    assert result.evaluation_result.is_score_valid is True


def test_grader_rejects_non_object_json():
    """Strict JSON validation: a non-object answer is rejected, not silently coerced."""
    with pytest.raises(ValueError, match="JSON object"):
        parse_teacher_params("[1, 2, 3]", BOUNDS)


def test_grader_rejects_missing_parameter():
    """Every declared bound key must be present; a missing one raises, never defaults."""
    with pytest.raises(ValueError, match="missing parameter 'arena_width'"):
        parse_teacher_params(json.dumps({"opponent_strength": 0.4}), BOUNDS)


def test_concurrent_rollouts_do_not_cross_tag_rewards():
    """Two concurrent /init rollouts with different ids must each get THEIR OWN reward.

    This is the silent-corruption guard: rewards are keyed by rollout_id end to
    end (results dict + reporter), so rollout A's reward can never be filed under
    rollout B's id even when both run as overlapping background tasks.
    """
    reports: list[tuple[str, dict]] = []
    # rollout_id -> the opponent_strength its Teacher "returns" (== its reward).
    strength_by_id = {"rollout-a": 0.20, "rollout-b": 0.90}

    def client_factory(base_url: str, api_key: str):
        # api_key carries the rollout_id (set per request below) so the fake
        # Teacher can answer differently per rollout.
        rollout_id = api_key

        class _Completions:
            async def create(self, **kwargs):
                # Yield the event loop so both rollouts are genuinely in flight.
                await asyncio.sleep(0.02)
                strength = strength_by_id[rollout_id]
                content = json.dumps({"opponent_strength": strength, "arena_width": 16})
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        return SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    app = create_ep_app(
        bounds=BOUNDS,
        scorer=lambda params: params["opponent_strength"],
        client_factory=client_factory,
        reporter=lambda rollout_id, extras: reports.append((rollout_id, extras)),
        live_tracing=False,
    )

    with TestClient(app) as client:
        for rid in ("rollout-a", "rollout-b"):
            req = make_request(rollout_id=rid).model_dump(mode="json")
            req["api_key"] = rid  # smuggle the rollout_id into the fake client
            resp = client.post("/init", json=req)
            assert resp.status_code == 202
            assert resp.json()["rollout_id"] == rid

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if {"rollout-a", "rollout-b"} <= set(app.state.results):
                break
            time.sleep(0.01)

    # Each rollout's stored reward equals ITS OWN Teacher answer — no cross-tag.
    assert app.state.results["rollout-a"].reward == 0.20
    assert app.state.results["rollout-b"].reward == 0.90
    assert app.state.results["rollout-a"].rollout_id == "rollout-a"
    assert app.state.results["rollout-b"].rollout_id == "rollout-b"
    # The reporter is also keyed by rollout_id, with matching reward per id.
    reported = {rid: extras["hud_reward"] for rid, extras in reports}
    assert reported == {"rollout-a": 0.20, "rollout-b": 0.90}
