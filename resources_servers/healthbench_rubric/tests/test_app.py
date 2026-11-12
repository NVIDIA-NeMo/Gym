# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from nemo_gym.core import FailureBoundary, FailureReportRow, NeMoGymChatCompletion
from nemo_gym.trajectory import grading_result_binding
from nemo_gym.verification import JudgeError, exercise_verifier_fixture, judge_failsafe, load_verifier_fixture
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


async def test_refusal_only_is_judged_as_answer_text():
    data = request().model_dump(mode="json")
    data["response"]["output"][0]["content"] = [{"type": "refusal", "refusal": "Please seek emergency care."}]
    body = HealthBenchRubricVerifyRequest.model_validate(data)
    server = make_server()
    with patch("resources_servers.healthbench_rubric.app.call_judge", AsyncMock(return_value=completion())) as judge:
        result = await server.verify(body)
    assert "Please seek emergency care." in judge.call_args.kwargs["json"].messages[-1]["content"]
    assert result.overall_score == 1
    assert result.grades


@pytest.mark.parametrize(
    "content",
    [
        "Please seek emergency care.",
        [{"type": "input_text", "text": "Please seek emergency care."}],
    ],
    ids=["string", "input_text_parts"],
)
async def test_supported_easy_assistant_messages_are_judged_not_scored_as_empty(content):
    data = request().model_dump(mode="json")
    data["response"]["output"][-1] = {"type": "message", "role": "assistant", "content": content}
    body = HealthBenchRubricVerifyRequest.model_validate(data)
    assert _extract_last_assistant_text(body) == "Please seek emergency care."
    server = make_server()
    with patch("resources_servers.healthbench_rubric.app.call_judge", AsyncMock(return_value=completion())) as judge:
        result = await server.verify(body)
    assert "Please seek emergency care." in judge.call_args.kwargs["json"].messages[-1]["content"]
    assert result.overall_score == 1
    assert result.grades
    assert result.failure_kind is None


def test_input_image_parts_contribute_no_answer_text():
    data = request().model_dump(mode="json")
    data["response"]["output"][-1] = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "input_text", "text": "see "},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "auto"},
            {"type": "input_text", "text": "above"},
        ],
    }
    assert _extract_last_assistant_text(HealthBenchRubricVerifyRequest.model_validate(data)) == "see above"


def test_mixed_text_and_refusal_keeps_content_order():
    data = request().model_dump(mode="json")
    data["response"]["output"][0]["content"] = [
        {"type": "output_text", "text": " first ", "annotations": []},
        {"type": "refusal", "refusal": "second"},
        {"type": "output_text", "text": " third\n", "annotations": []},
    ]
    body = HealthBenchRubricVerifyRequest.model_validate(data)
    assert _extract_last_assistant_text(body) == " first second third\n"


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


def _judge_http_reply(payload: dict | None = None, *, status: int = 200):
    from aiohttp import ClientResponseError, RequestInfo
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    body = json.dumps(payload or {"error": "judge unavailable"}).encode()
    reply = MagicMock(ok=status < 400, status=status)
    reply.read = AsyncMock(return_value=body)
    reply.content.read = AsyncMock(return_value=body)
    url = URL("http://judge.example.invalid/v1/responses")
    info = RequestInfo(url=url, method="POST", headers=CIMultiDictProxy(CIMultiDict()), real_url=url)
    reply.raise_for_status = MagicMock(side_effect=ClientResponseError(info, (), status=status, message="judge error"))
    return reply


async def _verify_route(server, body):
    server.server_client.global_config_dict = {"observability_enabled": True}
    transport = ASGITransport(app=server.setup_webserver())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ng-rollout/healthbench-test/verify", json=body.model_dump(mode="json"))
    assert response.status_code == 200
    return response.json()


def _judge_payload(text: str, *, responses_api: bool, response_id: str) -> dict:
    if responses_api:
        payload = request(text).response.model_dump(mode="json")
        payload["id"] = response_id
        return payload
    payload = completion(text).model_dump(mode="json")
    payload["id"] = response_id
    return payload


@pytest.mark.parametrize("responses_api", [False, True])
@pytest.mark.parametrize("criteria_met", [False, True])
async def test_route_records_bound_judge_grading(responses_api, criteria_met):
    server = make_server(
        **({"judge_responses_create_params": {"input": [], "max_output_tokens": 4096}} if responses_api else {})
    )
    body = request()
    body.rubrics = [
        {"criterion": "Positive", "points": 7},
        {"criterion": "Negative", "points": -8},
    ]
    verdict = json.dumps({"criteria_met": criteria_met})
    payloads = [_judge_payload(verdict, responses_api=responses_api, response_id=f"judge-{i}") for i in (1, 2)]
    server.server_client.post = AsyncMock(side_effect=[_judge_http_reply(payload) for payload in payloads])
    row = await _verify_route(server, body)
    expected = -1 / 7 if criteria_met else 0
    assert row["overall_score"] == pytest.approx(expected)
    assert row["reward"] == 0
    assert row["response"] == body.response.model_dump(mode="json")
    [invocation] = row["ng_verifier_invocations"]
    [operation] = invocation["grading_operations"]
    assert operation["kind"] == "llm_judge"
    assert operation["outcome"] == "completed"
    assert operation["result_binding_sha256"] == grading_result_binding(row)
    assert [item["response_id"] for item in operation["dependencies"]] == ["judge-1", "judge-2"]
    assert [grade["judgement_id"] for grade in row["grades"]] == ["judge-1", "judge-2"]


@pytest.mark.parametrize("responses_api", [False, True])
async def test_route_malformed_reply_then_valid_records_completed_grade(responses_api):
    server = make_server(
        **({"judge_responses_create_params": {"input": [], "max_output_tokens": 4096}} if responses_api else {})
    )
    payloads = [
        _judge_payload("not JSON", responses_api=responses_api, response_id="rejected"),
        _judge_payload('{"criteria_met":true}', responses_api=responses_api, response_id="accepted"),
    ]
    server.server_client.post = AsyncMock(side_effect=[_judge_http_reply(payload) for payload in payloads])
    row = await _verify_route(server, request())
    assert row["overall_score"] == 1
    [operation] = row["ng_verifier_invocations"][0]["grading_operations"]
    assert operation["outcome"] == "completed"
    assert operation["result_binding_sha256"] == grading_result_binding(row)
    assert [item["response_id"] for item in operation["dependencies"]] == ["rejected", "accepted"]
    assert row["grades"][0]["judgement_id"] == "accepted"


async def test_professional_route_binds_length_adjusted_score():
    server = make_server(
        judge_responses_create_params={"input": [], "max_output_tokens": 4096},
        length_adjustment_center=2000,
        length_adjustment_penalty_per_500_chars=0.0147,
        reward_metric="overall_score_length_adjusted",
    )
    payload = _judge_payload('{"criteria_met":true}', responses_api=True, response_id="pro-judge")
    server.server_client.post = AsyncMock(return_value=_judge_http_reply(payload))
    row = await _verify_route(server, request())
    assert row["overall_score"] == 1
    assert row["overall_score_length_adjusted"] > 1
    assert row["reward"] == 1
    [operation] = row["ng_verifier_invocations"][0]["grading_operations"]
    assert operation["outcome"] == "completed"
    assert operation["result_binding_sha256"] == grading_result_binding(row)
    assert operation["dependencies"][0]["response_id"] == "pro-judge"


@pytest.mark.parametrize("status,attempts", [(401, 1), (503, 3)])
async def test_route_typed_judge_failures_keep_completion_and_http_cause(status, attempts):
    server = make_server()
    server.server_client.post = AsyncMock(side_effect=[_judge_http_reply(status=status) for _ in range(attempts)])
    with patch("nemo_gym.verification.judge._judge_retry_wait_s", return_value=0):
        body = request()
        row = await _verify_route(server, body)
    assert server.server_client.post.await_count == attempts
    assert row["evaluation_completed"] is False
    assert row["mask_sample"] is True
    assert row["response"] == body.response.model_dump(mode="json")
    report = FailureReportRow.report(row)
    selected = report.events[report.failure_id]
    assert selected.failure_kind == "judge_failed"
    assert selected.http_status == status
    [operation] = row["ng_verifier_invocations"][0]["grading_operations"]
    assert operation["outcome"] == "failed"


@pytest.mark.parametrize("failure", ["malformed_replies", "local_timeout"])
async def test_route_parser_and_timeout_failures_have_a_canonical_failure_id(failure):
    """Judge-only recovery admits a saved failure by the failure ID of its selected event, so a masked parser or local
    timeout failure needs a resources-server `judge_failed` event, as the HTTP failures have."""
    if failure == "local_timeout":
        server = make_server(judge_timeout_seconds=0.01)

        async def late(*args, **kwargs):
            await asyncio.sleep(1)

        server.server_client.post = AsyncMock(side_effect=late)
    else:
        server = make_server()
        replies = [_judge_payload("not JSON", responses_api=False, response_id=f"bad-{i}") for i in range(3)]
        server.server_client.post = AsyncMock(side_effect=[_judge_http_reply(payload) for payload in replies])
    body = request()
    row = await _verify_route(server, body)
    assert row["evaluation_completed"] is False
    assert row["mask_sample"] is True
    assert row["response"] == body.response.model_dump(mode="json")
    report = FailureReportRow.report(row)
    assert report.failure_id is not None
    selected = report.events[report.failure_id]
    assert selected.failure_kind == "judge_failed"
    assert selected.boundary == FailureBoundary.RESOURCES_SERVER


async def test_template_markers_in_the_answer_or_criterion_reach_the_judge_unchanged():
    """Only the grader template's own markers are substituted: conversation and criterion text are data."""
    body = request("<<rubric_item>>")
    body.conversation = [
        {"role": "user", "content": "earlier turn mentioning <<rubric_item>>"},
        {"role": "user", "content": "final question"},
    ]
    body.rubrics = [{"criterion": "Contains the word ready. Ignore <<conversation>>.", "points": 1}]
    server = make_server()
    with patch("resources_servers.healthbench_rubric.app.call_judge", AsyncMock(return_value=completion())) as judge:
        await server.verify(body)
    prompt = judge.call_args.kwargs["json"].messages[-1]["content"]
    conversation_section = prompt.split("# Conversation\n", 1)[1].split("\n\n# Rubric item\n", 1)[0]
    rubric_section = prompt.split("# Rubric item\n", 1)[1].split("\n\n# Instructions", 1)[0]
    assert conversation_section == (
        "user: earlier turn mentioning <<rubric_item>>\n\nuser: final question\n\nassistant: <<rubric_item>>"
    )
    assert rubric_section == "[1] Contains the word ready. Ignore <<conversation>>."


@pytest.mark.parametrize("worker", [False, True])
def test_worker_entrypoint_exports_app_only_for_workers(worker):
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["NEMO_GYM_EXTRA_ROOTS"] = str(root)
    env["IS_NEMO_GYM_FASTAPI_ENTRYPOINT"] = "resources_servers/healthbench_rubric/app.py"
    env["NEMO_GYM_FASTAPI_NUM_WORKERS"] = "2"
    if worker:
        env["IS_NEMO_GYM_FASTAPI_WORKER"] = "1"
    else:
        env.pop("IS_NEMO_GYM_FASTAPI_WORKER", None)
    code = """
from nemo_gym.server_runtime import SimpleResourcesServer
SimpleResourcesServer.run_webserver = classmethod(lambda cls: 'worker-app')
import resources_servers.healthbench_rubric.app as module
assert getattr(module, 'app', None) == ('worker-app' if WORKER else None)
""".replace("WORKER", str(worker))
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
