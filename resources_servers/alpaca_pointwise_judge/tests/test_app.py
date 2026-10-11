# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, judge_failsafe
from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming, NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.alpaca_pointwise_judge import app
from resources_servers.alpaca_pointwise_judge.app import (
    AlpacaPointwiseJudgeConfig,
    AlpacaPointwiseJudgeServer,
    AlpacaPointwiseVerifyRequest,
    PointwiseJudgment,
    final_answer_only,
)
from resources_servers.alpaca_pointwise_judge.task_data import LANGUAGES, AlpacaTask, TaskData


def metadata(code="hi", row_id="item-0", counts=None):
    return dict(
        question="२ + २ कितना होता है?",
        row_id=row_id,
        row_index=0,
        language_code=code,
        language_name=LANGUAGES[code],
        expected_counts=counts or {code: 1},
        evaluation_set_id="test-set",
        judge_pass_stage="first_judge_pass",
        human_evaluation_pending=True,
        human_review_reason="random_sample",
    )


def judgment(score=4):
    return dict(
        overall_score=score,
        dimension_scores=dict(
            instruction_following=score,
            correctness=score,
            usefulness=score,
            language_quality=score,
        ),
        rationale="The answer follows the instruction.",
        detected_response_language="Hindi",
    )


def completion(content=None, *, finish_reason="stop", choices=True):
    return NeMoGymChatCompletion.model_validate(
        dict(
            id="judge",
            object="chat.completion",
            created=0,
            model="frontier",
            choices=[dict(index=0, message=dict(role="assistant", content=content), finish_reason=finish_reason)]
            if choices
            else [],
        )
    )


def response(text="चार।"):
    return NeMoGymResponse.model_validate(
        dict(
            id="candidate",
            created_at=0,
            model="gemma",
            object="response",
            output=[
                dict(type="reasoning", id="r", summary=[dict(type="summary_text", text="PRIVATE REASONING")]),
                dict(
                    type="message",
                    id="m",
                    role="assistant",
                    status="completed",
                    content=[dict(type="output_text", text=text, annotations=[])],
                ),
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
    )


def body(text="चार।"):
    return AlpacaPointwiseVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "२ + २ कितना होता है?"}]},
        response=response(text),
        verifier_metadata=metadata(),
    )


@pytest.fixture
def server():
    return AlpacaPointwiseJudgeServer(
        config=AlpacaPointwiseJudgeConfig(
            host="127.0.0.1",
            port=8080,
            name="alpaca",
            entrypoint="app.py",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_chat_completions_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(
                messages=[],
                temperature=0,
                max_tokens=4096,
                reasoning_effort="low",
            ),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.parametrize("score", [True, False, "4", 0, 6, 2.5])
def test_invalid_scores(score):
    value = judgment()
    value["overall_score"] = score
    with pytest.raises(ValidationError):
        PointwiseJudgment.model_validate(value)
    value = judgment()
    value["dimension_scores"]["correctness"] = score
    with pytest.raises(ValidationError):
        PointwiseJudgment.model_validate(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rationale", ""),
        ("rationale", "   "),
        ("rationale", "word " * 121),
        ("detected_response_language", ""),
        ("detected_response_language", " "),
    ],
)
def test_invalid_text(field, value):
    item = judgment()
    item[field] = value
    with pytest.raises(ValidationError):
        PointwiseJudgment.model_validate(item)


def test_schema_forbids_extra_and_missing_fields():
    for field in judgment():
        value = judgment()
        value.pop(field)
        with pytest.raises(ValidationError):
            PointwiseJudgment.model_validate(value)
    with pytest.raises(ValidationError):
        PointwiseJudgment.model_validate({**judgment(), "extra": 1})


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<think>secret</think>answer", "answer"),
        ("orphan secret</think>answer", "answer"),
        ("preface<think>secret</think>answer", "answer"),
        ("first</think>intermediate<think>more</think>final", "final"),
        ("<thinking>secret</thinking>answer", "answer"),
        ("<think>unfinished", ""),
        ("<thinking>unfinished", ""),
        ("answer", "answer"),
        ("", ""),
    ],
)
def test_reasoning_removed(text, expected):
    assert final_answer_only(text) == expected


@pytest.mark.parametrize("score", [1, 3, 5])
async def test_real_gym_request_and_graded_reward(server, monkeypatch, score):
    mock = AsyncMock(return_value=completion(json.dumps(judgment(score))))
    monkeypatch.setattr(app, "call_judge", mock)
    result = await server.verify(body("<think>secret</think>चार।"))
    assert result.reward == (score - 1) / 4
    assert result.judgment.overall_score == score
    assert result.judge_attempts == 1
    assert result.candidate_answer == "चार।"
    assert result.verifier_metadata.human_evaluation_pending
    params = mock.call_args.kwargs["json"]
    input_data = json.loads(params.messages[1]["content"])
    assert input_data == dict(
        language_code="hi", language_name="Hindi", instruction="२ + २ कितना होता है?", response="चार।"
    )
    assert "PRIVATE REASONING" not in params.messages[1]["content"]
    assert params.response_format["json_schema"]["strict"] is True
    assert params.temperature == 0
    assert params.max_tokens == 4096
    assert params.reasoning_effort == "low"
    assert server.config.judge_chat_completions_create_params.messages == []


async def test_empty_and_refusal_are_judged(server, monkeypatch):
    mock = AsyncMock(return_value=completion(json.dumps(judgment(1))))
    monkeypatch.setattr(app, "call_judge", mock)
    result = await server.verify(body(""))
    assert result.empty_final_answer
    assert result.reward == 0
    assert not result.mask_sample
    request = body()
    raw_response = request.response.model_dump()
    raw_response["output"][-1]["content"] = [{"type": "refusal", "refusal": "I cannot help with that."}]
    request.response = NeMoGymResponse.model_validate(raw_response)
    result = await server.verify(request)
    assert result.candidate_answer == "I cannot help with that."


@pytest.mark.parametrize(
    "invalid",
    [
        completion("not json"),
        completion(""),
        completion(None),
        completion(json.dumps(judgment()), finish_reason="length"),
        completion(choices=False),
        completion(json.dumps({**judgment(), "overall_score": True})),
    ],
)
async def test_invalid_judge_retries_and_routes_to_failure(server, monkeypatch, invalid):
    mock = AsyncMock(return_value=invalid)
    monkeypatch.setattr(app, "call_judge", mock)
    failure = await judge_failsafe(server.verify)(body())
    data = json.loads(failure.body)
    assert mock.await_count == 3
    assert data["mask_sample"] is True
    assert data["_ng_failure_class"] == "judge_failed"
    assert data["response"]["id"] == "candidate"


async def test_judge_retry_success(server, monkeypatch):
    mock = AsyncMock(side_effect=[completion("broken"), completion(json.dumps(judgment()))])
    monkeypatch.setattr(app, "call_judge", mock)
    result = await server.verify(body())
    assert result.judge_attempts == 2
    assert result.judgment.overall_score == 4


async def test_transport_failure_is_not_a_model_loss(server, monkeypatch):
    monkeypatch.setattr(app, "call_judge", AsyncMock(side_effect=JudgeError("HTTP 429")))
    failure = await judge_failsafe(server.verify)(body())
    data = json.loads(failure.body)
    assert data["mask_sample"] and data["_ng_failure_class"] == "judge_failed"


@pytest.mark.parametrize("previous_failure", [False, True])
async def test_reverification_replaces_previous_measurement(server, monkeypatch, previous_failure):
    monkeypatch.setattr(app, "call_judge", AsyncMock(return_value=completion(json.dumps(judgment(5)))))
    previous = body().model_dump() | dict(
        id="hi/item-0",
        reward=0,
        judgment=judgment(1),
        judge_completion=completion(json.dumps(judgment(1))).model_dump(),
        judge_attempts=3,
        candidate_answer="stale",
        empty_final_answer=True,
        protocol="old",
        mask_sample=previous_failure,
        failure_kind="judge_failed" if previous_failure else None,
        failure_reason="HTTP 429" if previous_failure else None,
        instance_config={"mask_sample": previous_failure, "other": "preserve"},
    )
    if previous_failure:
        previous.update(_ng_failure_class="judge_failed", _ng_failure_judge_error="HTTP 429")
    result = await server.verify(AlpacaPointwiseVerifyRequest.model_validate(previous))
    assert result.judgment.overall_score == 5
    assert result.reward == 1 and result.judge_attempts == 1
    assert result.candidate_answer == "चार।" and not result.empty_final_answer
    assert result.protocol == "indic_alpaca_pointwise_v1"
    assert not result.mask_sample and result.failure_kind is None and result.failure_reason is None
    data = result.model_dump()
    assert data["id"] == "hi/item-0"
    assert data["instance_config"] == {"mask_sample": False, "other": "preserve"}
    assert "_ng_failure_class" not in data and "_ng_failure_judge_error" not in data


def metric_row(code="hi", row_id="item-0", score=4, counts=None):
    return dict(
        verifier_metadata=metadata(code, row_id, counts),
        judgment=judgment(score),
        mask_sample=False,
        empty_final_answer=False,
        reward=(score - 1) / 4,
    )


def test_metrics_average_repeats_and_languages(server):
    counts = {"hi": 2, "ta": 1}
    tasks = [
        [metric_row(score=1, counts=counts), metric_row(score=5, counts=counts)],
        [metric_row(row_id="item-1", score=5, counts=counts)],
        [metric_row("ta", score=2, counts=counts)],
    ]
    result = server.compute_metrics(tasks)
    assert result["alpaca_pointwise/hi/mean_overall_score"] == 4
    assert result["alpaca_pointwise/ta/mean_overall_score"] == 2
    assert result["alpaca_pointwise/macro_mean_overall_score"] == 3
    assert result["alpaca_pointwise/macro_mean_correctness"] == 3
    assert result["alpaca_pointwise/hi/mean_correctness"] == 4
    assert result["alpaca_pointwise/hi/standard_error"] == 1
    assert result["alpaca_pointwise/complete"] is True
    assert server.get_key_metrics({**result, "mean/reward": 0.5}) == result


def test_missing_language_or_masked_results_are_not_complete(server):
    counts = {"hi": 2, "ta": 1}
    failed = metric_row(row_id="item-1", counts=counts)
    failed.update(mask_sample=True, judgment=None)
    result = server.compute_metrics([[], [metric_row(counts=counts)], [failed]])
    assert result["alpaca_pointwise/hi/mean_overall_score"] == 4
    assert result["alpaca_pointwise/ta/judged"] == 0
    assert result["alpaca_pointwise/hi/invalid_judgments"] == 1
    assert not result["alpaca_pointwise/complete"]
    assert "alpaca_pointwise/macro_mean_overall_score" not in result
    assert "alpaca_pointwise/macro_mean_correctness" not in result
    assert not server.compute_metrics([])["alpaca_pointwise/complete"]


@pytest.mark.parametrize(
    "change,match",
    [
        ({}, "Duplicate"),
        ({"row_id": "item-1", "evaluation_set_id": "another"}, "different prepared"),
        ({"row_id": "item-1", "expected_counts": {"hi": 2}}, "Inconsistent"),
    ],
)
def test_incompatible_metric_inputs_rejected(server, change, match):
    first, second = metric_row(), metric_row()
    second["verifier_metadata"].update(change)
    with pytest.raises(ValueError, match=match):
        server.compute_metrics([[first], [second]])


def test_repeat_identity_is_validated(server):
    first, second = metric_row(), metric_row()
    second["verifier_metadata"]["row_id"] = "different"
    with pytest.raises(ValueError, match="same task"):
        server.compute_metrics([[first, second]])


@pytest.mark.parametrize(
    "change",
    [
        {"language_name": "English"},
        {"language_code": "sa"},
        {"expected_counts": {"ta": 1}},
        {"expected_counts": {"hi": 1, "en": 1}},
        {"expected_counts": {"hi": 0}},
        {"question": "  "},
        {"row_id": "  "},
    ],
)
def test_task_data_validation(change):
    with pytest.raises(ValidationError):
        AlpacaTask.model_validate({**metadata(), **change})
    assert TaskData(verifier_metadata=metadata()).verifier_metadata.language_code == "hi"
