# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from nemo_gym.judge import JudgeError, judge_failsafe
from nemo_gym.openai_utils import NeMoGymChatCompletion
from nemo_gym.verifier_fixture import exercise_verifier_fixture, load_verifier_fixture
from resources_servers.healthbench_rubric.app import (
    VERIFIER_FIXTURE,
    HealthBenchRubricVerifyRequest,
    _calculate_score,
    _extract_last_assistant_text,
    _finite_reward,
    _parse_json_object,
)
from resources_servers.healthbench_rubric.tests.fixture_support import make_server


def request(text="fixture correct"):
    data = load_verifier_fixture(VERIFIER_FIXTURE.cases_path)[0].request
    data["response"]["output"][0]["content"][0]["text"] = text
    return HealthBenchRubricVerifyRequest.model_validate(data)


def completion(text='{"criteria_met": true, "explanation": "ok"}'):
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "judge-123",
            "object": "chat.completion",
            "created": 0,
            "model": "scripted",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
    )


async def test_onboarding_fixture():
    results = await exercise_verifier_fixture(VERIFIER_FIXTURE, reward_range=(0, 1), determinism="stochastic")
    assert [r.kind for r in results] == ["full_reward", "zero_reward", "malformed"]


@pytest.mark.parametrize("text,expected", [(" ", 0.0), ("answer", 1.0)])
async def test_empty_answer_does_not_call_judge_or_get_length_bonus(text, expected):
    server = make_server(
        length_adjustment_center=2000,
        length_adjustment_penalty_per_500_chars=0.0147,
        reward_metric="overall_score_length_adjusted",
    )
    with patch("resources_servers.healthbench_rubric.app.call_judge", AsyncMock(return_value=completion())) as judge:
        result = await server.verify(request(text))
    assert result.reward == expected
    assert judge.await_count == (0 if not text.strip() else 1)
    assert not result.mask_sample
    if not text.strip():
        assert result.overall_score_length_adjusted == 0
        assert result.failure_kind == "healthbench_rubric:empty_response"
    else:
        assert result.overall_score_length_adjusted > 1
        assert result.grades[0]["judgement_id"] == "judge-123"
        assert result.grades[0]["usage"]["total_tokens"] == 14


async def test_negative_points_and_raw_metrics_are_preserved():
    server = make_server()
    body = request()
    body.rubrics = [{"criterion": "positive", "points": 1}, {"criterion": "negative", "points": -2}]
    with patch(
        "resources_servers.healthbench_rubric.app.call_judge",
        AsyncMock(side_effect=[completion('{"criteria_met":false}'), completion()]),
    ):
        result = await server.verify(body)
    assert result.overall_score == -2
    assert result.reward == 0
    assert result.overall_score_length_adjusted is None


async def test_retry_malformed_grade_and_keep_valid_false():
    server = make_server()
    with patch(
        "resources_servers.healthbench_rubric.app.call_judge",
        AsyncMock(side_effect=[completion("bad"), completion('{"criteria_met":false}')]),
    ) as judge:
        result = await server.verify(request())
    assert judge.await_count == 2
    assert result.reward == 0
    assert not result.mask_sample
    assert result.grades[0]["criteria_met"] is False


@pytest.mark.parametrize("text", ["bad", '{"criteria_met": "false"}', "{}"])
async def test_exhausted_parse_retries_are_masked(text):
    server = make_server(judge_max_attempts=2)
    with patch(
        "resources_servers.healthbench_rubric.app.call_judge", AsyncMock(return_value=completion(text))
    ) as judge:
        result = await judge_failsafe(server.verify)(request())
    data = result.model_dump() if hasattr(result, "model_dump") else json.loads(result.body)
    assert judge.await_count == 2
    assert data["mask_sample"] is True
    assert data["failure_kind"] == "judge_failed"
    assert data["_ng_failure_class"] == "judge_failed"
    assert "bounded retries" in data["failure_reason"]


@pytest.mark.parametrize("error", [JudgeError("endpoint unavailable"), TimeoutError("late")])
async def test_judge_failure_is_masked(error):
    server = make_server()
    with patch("resources_servers.healthbench_rubric.app.call_judge", AsyncMock(side_effect=error)):
        result = await judge_failsafe(server.verify)(request())
    data = result.model_dump() if hasattr(result, "model_dump") else json.loads(result.body)
    assert data["mask_sample"] and data["failure_kind"] == "judge_failed"
    assert data["failure_reason"]
    assert data["evaluation_completed"] is False


def test_score_clips_only_after_averaging_and_excludes_masked_rows():
    server = make_server()
    rows = [
        [{"overall_score": -1}, {"overall_score": 1}],
        [{"overall_score": 1}, {"overall_score": 100, "mask_sample": True}],
    ]
    metrics = server.compute_metrics(rows)
    assert metrics["overall_score"] == pytest.approx(1 / 3)
    assert server.compute_repeat_metrics(rows) == metrics
    assert server.get_key_metrics(metrics | {"mean/reward": 0.5}) == metrics
    assert server.compute_metrics([]) == {}
    assert server.compute_metrics([[{"overall_score_length_adjusted": 2}]]) == {"overall_score_length_adjusted": 1}


@pytest.mark.parametrize(
    "overrides",
    [
        {"judge_endpoint_max_concurrency": 0},
        {"judge_timeout_seconds": float("nan")},
        {"judge_max_attempts": 0},
        {"length_adjustment_center": 2000},
        {"reward_metric": "overall_score_length_adjusted"},
        {"reward_metric": "unknown"},
    ],
)
def test_invalid_config_fails_before_startup(overrides):
    with pytest.raises(ValidationError):
        make_server(**overrides)


@pytest.mark.parametrize(
    "rubrics",
    [
        [],
        [{"criterion": "x", "points": 0}],
        [{"criterion": "x", "points": float("nan")}],
        [{"points": 1}],
        [{"criterion": "x", "points": True}],
    ],
)
def test_malformed_rubrics_fail_before_paid_calls(rubrics):
    data = request().model_dump()
    data["rubrics"] = rubrics
    with pytest.raises(ValidationError):
        HealthBenchRubricVerifyRequest.model_validate(data)


def test_conversation_must_end_in_user():
    data = request().model_dump()
    data["conversation"] = []
    with pytest.raises(ValidationError, match="end with a user"):
        HealthBenchRubricVerifyRequest.model_validate(data)


def test_last_answer_and_exact_whitespace():
    body = request(" first ")
    last = body.response.output[0].model_copy(deep=True)
    last.content[0].text = " final \n"
    body.response.output.append(last)
    assert _extract_last_assistant_text(body) == " final \n"
    body.response.output = []
    assert _extract_last_assistant_text(body) == ""


def test_length_adjustment_reference_formula():
    server = make_server(length_adjustment_center=2000, length_adjustment_penalty_per_500_chars=0.0147)
    assert server._calculate_length_adjusted_score(0.6, "x" * 2500) == pytest.approx(0.5853)


@pytest.mark.parametrize("text", ['```json\n{"criteria_met":true}\n```', 'prefix {"criteria_met":true} suffix'])
def test_supported_json_wrappers(text):
    assert _parse_json_object(text)["criteria_met"] is True


@pytest.mark.parametrize("text", ["[]", "no object", "{invalid}"])
def test_invalid_json(text):
    with pytest.raises(ValueError):
        _parse_json_object(text)


def test_score_rejects_invalid_points_and_grade_count():
    with pytest.raises(ValueError, match="positive"):
        _calculate_score([{"points": -1}], [{"criteria_met": True}])
    with pytest.raises(ValueError):
        _calculate_score([{"points": 1}], [])
    with pytest.raises(ValueError, match="finite"):
        _finite_reward(float("inf"))


async def test_professional_uses_reference_responses_judge_settings():
    server = make_server(
        judge_system_prompt=None,
        judge_responses_create_params={
            "input": [],
            "reasoning": {"effort": "low"},
            "max_output_tokens": 4096,
        },
    )
    judge_reply = request('{"criteria_met":true,"explanation":"ok"}').response
    with patch("resources_servers.healthbench_rubric.app.call_judge", AsyncMock(return_value=judge_reply)) as judge:
        result = await server.verify(request())
    kwargs = judge.call_args.kwargs
    assert kwargs["url_path"] == "/v1/responses"
    assert kwargs["json"].reasoning["effort"] == "low"
    assert kwargs["json"].input[0]["role"] == "user"
    assert result.reward == 1


async def test_chat_judge_matches_reference_system_and_sampling():
    server = make_server()
    with patch("resources_servers.healthbench_rubric.app.call_judge", AsyncMock(return_value=completion())) as judge:
        await server.verify(request())
    params = judge.call_args.kwargs["json"]
    assert params.messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert "[1]" in params.messages[1]["content"]
    assert params.temperature == 0.5
    assert params.max_tokens == 2048


async def test_reverify_clears_previous_failure_fields_and_replaces_scores():
    data = request().model_dump() | {
        "reward": 0.0,
        "overall_score": 0.0,
        "evaluation_completed": False,
        "mask_sample": True,
        "failure_kind": "judge_failed",
        "failure_reason": "old",
        "_ng_failure_class": "judge_failed",
        "_ng_failure_judge_error": "old",
    }
    server = make_server()
    with patch("resources_servers.healthbench_rubric.app.call_judge", AsyncMock(return_value=completion())):
        result = await server.verify(HealthBenchRubricVerifyRequest.model_validate(data))
    assert result.reward == 1
    assert result.evaluation_completed and not result.mask_sample
    assert result.failure_kind is None
    assert "_ng_failure_class" not in result.model_dump()
