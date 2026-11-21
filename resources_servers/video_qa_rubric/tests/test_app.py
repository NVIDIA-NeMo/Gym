# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The verifier with a scripted judge behind a mocked server client: no API key, no network."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.client import ServerClient
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, AggregateMetricsRequest, ModelServerRef
from nemo_gym.core import NgFailureEvent
from nemo_gym.server_runtime import verifier_observability
from nemo_gym.trajectory import VerifierInvocation, grading_result_binding
from nemo_gym.verification import JudgeError
from resources_servers.video_mcqa.tests.helpers import grade_failed_run_and_replay, response
from resources_servers.video_qa_rubric.app import (
    VideoQARubricResourcesServer,
    VideoQARubricResourcesServerConfig,
    VideoQARubricVerifyRequest,
)
from resources_servers.video_qa_rubric.rubric import JUDGE_SYSTEM_PROMPT, JUDGMENT_SCHEMA, JUDGMENT_SCHEMA_NAME
from resources_servers.video_qa_rubric.task import prepare_single_turn


JUDGE = ModelServerRef(type="responses_api_models", name="rubric_judge")
RUBRIC = {
    "criteria": [
        {"id": "c1", "requirement": "Names the object: an orange square.", "weight_points": 60},
        {"id": "c2", "requirement": "Gives the direction: left to right.", "weight_points": 40},
    ],
    "contradictions": [
        {"id": "k1", "condition": "Says it moves right to left.", "penalty_points": 0, "score_cap_points": 30}
    ],
}
ROW = prepare_single_turn(
    "video_qa_rubric:invented-1",
    "What moves in the video, and in which direction?",
    RUBRIC,
    video_url="https://example.invalid/square.mp4",
)
CALL = {"id": "call", "type": "function_call", "call_id": "call", "name": "watch_video", "arguments": "{}"}
TOOL_RESULT = {"type": "function_call_output", "call_id": "call", "output": "frames", "status": "completed"}
THINKING = {"id": "thinking", "type": "reasoning", "summary": [{"type": "summary_text", "text": "Maybe a circle."}]}


def judgment(c1: float = 60, c2: float = 40, *, applied: bool = False, score: float | None = None) -> str:
    total = min(30, c1 + c2) if applied else c1 + c2
    return json.dumps(
        {
            "score_0_100": total if score is None else score,
            "reason": "invented",
            "criterion_scores": [
                {"id": "c1", "awarded_points": c1, "reason": "invented"},
                {"id": "c2", "awarded_points": c2, "reason": "invented"},
            ],
            "applied_contradictions": [{"id": "k1", "penalty_points": 0, "score_cap_points": 30, "reason": "invented"}]
            if applied
            else [],
        }
    )


def judge_reply(content: str | None, *, finish_reason: str = "stop", reasoning: str | None = None) -> bytes:
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return json.dumps(
        {
            "id": "judge-reply",
            "object": "chat.completion",
            "created": 0,
            "model": "judge",
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 120, "total_tokens": 1020},
        }
    ).encode()


def http_error(status: int) -> ClientResponseError:
    return ClientResponseError(MagicMock(), (), status=status, message="fixture")


def scripted_judge(*replies: str | bytes | Exception) -> tuple[MagicMock, list[dict]]:
    """A server client whose judge answers with REPLIES in turn (judgment text, a whole reply, or an exception)."""
    calls: list[dict] = []
    queue = iter(replies)

    async def post(*, server_name: str, url_path: str, json: object, **kwargs: object) -> SimpleNamespace:
        calls.append({"server_name": server_name, "url_path": url_path, "json": json})
        reply = next(queue)
        if isinstance(reply, Exception):
            raise reply
        body = reply if isinstance(reply, bytes) else judge_reply(reply)
        return SimpleNamespace(ok=True, read=AsyncMock(return_value=body))

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    client.post = post
    return client, calls


def make_server(*replies: str | bytes | Exception, **options: object) -> tuple[VideoQARubricResourcesServer, list]:
    client, calls = scripted_judge(*replies)
    config = VideoQARubricResourcesServerConfig(
        name="video_qa_rubric",
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        judge_model_server=JUDGE,
        judge_chat_completion_create_params={"messages": [], "max_completion_tokens": 4096},
        judge_retry_base_delay_seconds=0,
        **options,
    )
    return VideoQARubricResourcesServer(config=config, server_client=client), calls


def request(text: str | None = "An orange square moves to the right.", *, setting: str = "single_turn", **updates):
    row = copy.deepcopy(ROW) | {"setting": setting}
    return VideoQARubricVerifyRequest.model_validate(row | {"response": response(text, **updates)})


def request_with(*items: dict, setting: str = "agentic") -> VideoQARubricVerifyRequest:
    return request(None, setting=setting, output=list(items))


def message(text: str) -> dict:
    return response(text)["output"][0]


async def test_a_graded_answer_earns_its_share_of_the_rubric_points() -> None:
    server, calls = make_server(judgment(60, 10))
    result = await server.verify(request("<think>A circle?</think>An orange square moves to the right."))
    assert (result.outcome, result.reward, result.rubric_score, result.rubric_points) == ("graded", 0.7, 0.7, 70.0)
    assert not result.mask_sample and result.evaluation_completed and result.judge_attempts == 1
    assert result.judgment["criterion_scores"][1] == {"id": "c2", "awarded_points": 10, "reason": "invented"}
    assert result.judge_usage["prompt_tokens"] == 900 and result.failure_kind is None
    (call,) = calls
    assert (call["server_name"], call["url_path"]) == ("rubric_judge", "/v1/chat/completions")
    # The configured judge parameters stay; the server adds the messages and the strict judgment schema.
    params = call["json"]
    assert set(params.model_dump(exclude_unset=True)) == {"messages", "max_completion_tokens", "response_format"}
    assert params.response_format == {
        "type": "json_schema",
        "json_schema": {"name": JUDGMENT_SCHEMA_NAME, "strict": True, "schema": JUDGMENT_SCHEMA},
    }
    system, user = params.messages
    assert system == {"role": "system", "content": JUDGE_SYSTEM_PROMPT}
    graded = json.loads(user["content"].split("\n", 1)[1])
    assert graded == {
        "question": ROW["verifier_metadata"]["question"],
        "rubric": RUBRIC,
        "candidate_final_answer": "An orange square moves to the right.",
    }
    # The judge sees no reasoning and no video.
    assert "circle" not in user["content"] and "example.invalid" not in user["content"]


async def test_a_contradiction_caps_the_score() -> None:
    server, _ = make_server(judgment(60, 40, applied=True))
    result = await server.verify(request("An orange square moves to the left."))
    assert (result.reward, result.rubric_points) == (0.3, 30.0)


async def test_the_judges_reasoning_is_not_part_of_its_judgment() -> None:
    server, _ = make_server(
        judge_reply(judgment(60, 40), reasoning="The answer names the square."),
        judge_reply("<think>Both criteria hold.</think>" + judgment(60, 40)),
    )
    for _ in range(2):
        assert (await server.verify(request())).reward == 1.0


async def test_a_rejected_judgment_is_asked_for_again(caplog) -> None:
    server, calls = make_server(
        judgment(60, 40, score=90),  # the total does not follow from the points
        "I would give it 100 points.",  # not JSON
        judge_reply(judgment(60, 40), finish_reason="length"),  # cut off
        http_error(503),  # an outage is retried too
        judgment(60, 40),
        judge_max_attempts=5,
    )
    with caplog.at_level("WARNING", logger="resources_servers.video_qa_rubric.app"):
        result = await server.verify(request())
    assert (result.reward, result.judge_attempts, len(calls)) == (1.0, 5, 5)
    # Each failed attempt is logged with its reason, never with the answer.
    reasons = [record.getMessage() for record in caplog.records]
    assert len(reasons) == 4 and "differs from the rubric's arithmetic" in reasons[0]
    assert not any("orange square" in reason for reason in reasons)


async def test_a_judge_that_fails_every_attempt_raises_a_judge_failure() -> None:
    server, calls = make_server(judgment(score=99), http_error(500), judge_max_attempts=2)
    with pytest.raises(JudgeError, match="failed 2 attempts"):
        await server.verify(request())
    assert len(calls) == 2


async def test_a_rejected_key_is_not_retried() -> None:
    server, calls = make_server(http_error(401), judgment())
    with pytest.raises(NgFailureEvent) as failure:
        await server.verify(request())
    assert failure.value.failure_kind == "judge_failed" and len(calls) == 1


@pytest.mark.parametrize("reply", ["no judgment", http_error(401)], ids=["unusable_reply", "rejected_call"])
def test_judge_failures_are_masked_rows_for_rejudging_not_wrong_answers(reply) -> None:
    server, calls = make_server(reply, judge_max_attempts=1)
    client = TestClient(server.setup_webserver())
    result = client.post("/verify", json=request().model_dump())
    assert result.status_code == 200 and len(calls) == 1
    row = result.json()
    assert (row["reward"], row["mask_sample"], row["failure_kind"]) == (0.0, True, "judge_failed")
    assert row["_ng_failure_class"] == "judge_failed"
    # The policy's answer is kept, so recovery judges it again without sampling the policy.
    assert row["response"]["output"][0]["content"][0]["text"] == "An orange square moves to the right."


async def observed_verify(server: VideoQARubricResourcesServer, body: VideoQARubricVerifyRequest):
    invocation = VerifierInvocation(invocation_id="verify-1")
    token = verifier_observability._active_verifier.set(invocation)
    try:
        result = await server.verify(body)
    finally:
        verifier_observability._active_verifier.reset(token)
    return result, invocation.grading_operations


async def test_a_graded_answer_is_bound_to_the_judges_replies() -> None:
    server, _ = make_server(judgment(score=1), judgment(60, 40))
    result, operations = await observed_verify(server, request())
    (operation,) = operations
    assert (operation.kind, operation.outcome, result.judge_attempts) == ("llm_judge", "completed", 2)
    assert [ref.model_ref.name for ref in operation.dependencies] == ["rubric_judge", "rubric_judge"]
    assert operation.result_binding_sha256 == grading_result_binding(result)
    # No judge call, no judge grading to record.
    result, operations = await observed_verify(server, request(""))
    assert (result.outcome, operations) == ("empty_answer", None)


async def test_no_judge_call_without_an_answer() -> None:
    server, calls = make_server()
    for text in (None, "", "  ", "<think>still thinking"):
        result = await server.verify(request(text))
        assert (result.outcome, result.reward, result.rubric_score, result.mask_sample) == (
            "empty_answer",
            0,
            0,
            False,
        )
    failed = await server.verify(request(status="failed", error={"code": "server_error", "message": "fixture"}))
    assert (failed.outcome, failed.reward, failed.mask_sample, failed.evaluation_completed) == (
        "infrastructure_failure",
        0.0,
        True,
        False,
    )
    assert failed.rubric_score is None and failed.failure_kind == "video_qa_rubric:infrastructure_failure"
    assert calls == []


@pytest.mark.parametrize("penalty", [None, -0.5])
async def test_a_single_turn_tool_call_is_an_execution_failure_worth_zero(penalty) -> None:
    server, calls = make_server(empty_answer_reward=penalty)
    result = await server.verify(request_with(message("An orange square."), CALL, setting="single_turn"))
    assert (result.outcome, result.rubric_score, result.reward, result.mask_sample) == (
        "execution_failure",
        0,
        0,
        False,
    )
    assert calls == []
    # An agent may call tools.
    server, calls = make_server(judgment())
    assert (await server.verify(request_with(CALL, TOOL_RESULT, message("An orange square.")))).reward == 1.0


@pytest.mark.parametrize("penalty", [None, -0.5])
@pytest.mark.parametrize(
    "updates",
    [
        {"status": "failed", "error": {"code": "invalid_prompt", "message": "fixture: the model refused the request"}},
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
    ],
    ids=["failed_run", "incomplete_run"],
)
async def test_a_failed_or_incomplete_agent_run_scores_zero_also_when_replayed(penalty, updates) -> None:
    # Gym's agents verify a failed run (execution_failure_reward_zero) as an empty response and require exactly 0.
    server, calls = make_server(empty_answer_reward=penalty)
    agentic = copy.deepcopy(ROW) | {"setting": "agentic"}
    failed = response("An orange square moves to the right.", **updates)
    saved, replayed = await grade_failed_run_and_replay(server, VideoQARubricVerifyRequest, agentic, failed)
    # The saved row keeps the answer; its status decides the score on replay too.
    for result in (saved, replayed.model_dump()):
        assert (result["outcome"], result["reward"], result["mask_sample"]) == ("execution_failure", 0, False)
    direct = await server.verify(request(setting="agentic", **updates))
    assert direct.failure_kind == "video_qa_rubric:execution_failure" and direct.failure_reason
    assert calls == []
    # A completed run that an agent verifies as an empty response gets the same zero (a replay of its saved row then
    # grades the saved answer, as video_mcqa does).
    server, calls = make_server(judgment(60, 0), empty_answer_reward=penalty)
    saved, replayed = await grade_failed_run_and_replay(server, VideoQARubricVerifyRequest, agentic, response("Red."))
    assert (saved["outcome"], saved["reward"], saved["mask_sample"]) == ("empty_answer", 0, False)
    assert (replayed.outcome, replayed.reward, len(calls)) == ("graded", 0.6, 1)


async def test_a_failed_model_call_is_masked_only_in_the_single_turn_setting() -> None:
    server, calls = make_server()
    for setting, outcome, masked in (
        ("single_turn", "infrastructure_failure", True),
        ("agentic", "execution_failure", False),
    ):
        result = await server.verify(
            request(setting=setting, status="failed", error={"code": "server_error", "message": "fixture"})
        )
        assert (result.outcome, result.mask_sample, result.reward) == (outcome, masked, 0)
    assert calls == []


@pytest.mark.parametrize("penalty", [None, -0.5])
@pytest.mark.parametrize(
    ("items", "graded"),
    [
        # The last assistant message is the answer, whatever comes after it.
        ((message("An orange square."), THINKING), True),
        ((message("An orange square."), CALL, TOOL_RESULT, THINKING), True),
        ((CALL, TOOL_RESULT, THINKING, message("An orange square moves right.")), True),
        # No assistant message, or one with nothing after its reasoning: no answer.
        ((CALL, TOOL_RESULT), False),
        ((THINKING,), False),
        ((CALL, TOOL_RESULT, message("<think>still thinking")), False),
    ],
)
async def test_the_last_assistant_message_is_graded_and_only_no_answer_gets_empty_answer_reward(
    items, graded, penalty
) -> None:
    server, calls = make_server(judgment(), empty_answer_reward=penalty)
    result = await server.verify(request_with(*items))
    no_answer = ("empty_answer", 0.0 if penalty is None else penalty)
    assert (result.outcome, result.reward) == (("graded", 1.0) if graded else no_answer)
    assert result.evaluation_completed and not result.mask_sample and len(calls) == graded


async def test_a_response_without_output_scores_zero_even_with_empty_answer_reward() -> None:
    server, calls = make_server(empty_answer_reward=-0.5)
    result = await server.verify(request_with())
    assert (result.outcome, result.reward, result.mask_sample, calls) == ("empty_answer", 0, False, [])


async def test_a_replayed_row_is_graded_again_from_its_saved_answer() -> None:
    server, calls = make_server(judgment(60, 40), judgment(30, 40))
    first = await server.verify(request())
    replayed = await server.verify(VideoQARubricVerifyRequest.model_validate(first.model_dump()))
    assert (first.reward, replayed.reward, len(calls)) == (1.0, 0.7, 2)


def test_rows_with_a_malformed_rubric_are_refused() -> None:
    row = request().model_dump()
    row["verifier_metadata"]["rubric"]["criteria"][1]["weight_points"] = 30
    with pytest.raises(ValidationError, match="add up to 100"):
        VideoQARubricVerifyRequest.model_validate(row)
    server, _ = make_server()
    assert TestClient(server.setup_webserver()).post("/verify", json=row).status_code == 422


async def test_aggregate_metrics_report_the_rubric_score_of_measured_attempts() -> None:
    server, _ = make_server(judgment(60, 40), judgment(30, 20), judgment(60, 20), empty_answer_reward=-0.5)
    results = [await server.verify(request()) for _ in range(3)]
    results += [
        await server.verify(request("")),
        await server.verify(request(status="failed", error={"code": "server_error", "message": "fixture"})),
    ]
    rows = [
        result.model_dump(mode="json") | {TASK_INDEX_KEY_NAME: index, ROLLOUT_INDEX_KEY_NAME: 0}
        for index, result in enumerate(results)
    ]
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    agent = metrics.agent_metrics
    # Scores 1.0, 0.5, 0.8 and 0 (no answer); the failed model call is masked and left out.
    assert agent["mean/rubric_score"] == pytest.approx(2.3 / 4)
    assert (agent["full_score_rate"], agent["at_least_0.8_rate"]) == (0.25, 0.5)
    assert (agent["outcomes/graded"], agent["outcomes/empty_answer"], agent["outcomes/infrastructure_failure"]) == (
        3,
        1,
        0,
    )
    # The training reward of the unanswered attempt (-0.5) is not its rubric score (0).
    assert agent["mean/reward"] == pytest.approx((2.3 - 0.5) / 4)
    assert set(metrics.key_metrics) >= {"mean/rubric_score", "mean/reward", "full_score_rate"}
