# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest

from nemo_gym.core import NeMoGymResponse
from nemo_gym.server_runtime.base_server import ServerClient
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.agentic_mme.app import (
    VERIFIER_FIXTURE,
    AgenticMMEConfig,
    AgenticMMEServer,
    AgenticMMEVerifyRequest,
)


def response(text: str = "", output: list | None = None) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "test",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": output
            if output is not None
            else [
                {
                    "id": "answer",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        }
    )


@pytest.fixture
def server() -> AgenticMMEServer:
    return AgenticMMEServer(
        config=AgenticMMEConfig(name="agentic_mme", host="localhost", port=1, entrypoint=""),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "golden", "reward"),
    [
        ("<answer>blue</answer>", {"value": "blue", "match_type": "exact"}, 1),
        ("Blue", {"value": "blue", "match_type": "exact"}, 0),
        ("The color is BLUE.", {"value": "blue"}, 1),
        ("2017", {"value": "1"}, 0),
        ("Answer is 1.", {"value": 1}, 1),
        ("0", {"value": 0}, 1),
        ("4.01", {"value": 4, "match_type": "numeric", "tolerance": 0.02}, 1),
        ("4.1", {"value": 4, "match_type": "numeric"}, 0),
        ("four", {"value": 4, "match_type": "numeric"}, 0),
        ("NaN", {"value": 4, "match_type": "numeric"}, 0),
        ("", {"value": "blue"}, 0),
        ("<answer></answer>", {"value": "blue"}, 0),
        ("<think><answer>blue</answer></think>red", {"value": "blue"}, 0),
        ("<thinking>blue</thinking><answer>red</answer>", {"value": "blue"}, 0),
        ("<think>blue", {"value": "blue"}, 0),
        ("blue</think>red", {"value": "blue"}, 0),
        ("<answer>blue", {"value": "blue"}, 1),
        ("<answer>YOUR_SHORT_ANSWER</answer> so <answer>blue</answer>", {"value": "blue", "match_type": "exact"}, 1),
        ("<answer>blue</answer> On reflection, <answer>red</answer>", {"value": "blue"}, 0),
        ("<answer>red</answer> Final: <answer>blue", {"value": "blue", "match_type": "exact"}, 1),
        ("None", {"value": None}, 0),
        ("anything", {"value": ""}, 0),
        ("anything", {"value": " "}, 0),
        ("True", {"value": True}, 0),
        ("inf", {"value": float("inf")}, 0),
        ("blue", {"value": ["blue"]}, 0),
        ("blue", {"value": "blue", "match_type": "unsupported"}, 0),
        ("blue", None, 0),
    ],
)
async def test_grading(server, text, golden, reward) -> None:
    body = AgenticMMEVerifyRequest(
        responses_create_params={"input": "question"},
        response=response(text),
        verifier_metadata={"golden_answer": golden},
    )
    result = await server.verify(body)
    assert result.reward == reward
    assert result.evaluation_track == "task_only"
    assert bool(result.failure_reason) == (reward == 0)


@pytest.mark.asyncio
async def test_terminal_tool_and_incomplete_do_not_score(server) -> None:
    call = {"type": "function_call", "name": "crop", "call_id": "a", "arguments": "{}"}
    output = response("blue").model_dump()["output"] + [call]
    body = AgenticMMEVerifyRequest(
        responses_create_params={"input": "q"},
        response=response(output=output),
        verifier_metadata={"golden_answer": {"value": "blue"}},
    )
    assert (await server.verify(body)).reward == 0
    body.response = NeMoGymResponse.model_validate(
        response("blue").model_dump() | {"incomplete_details": {"reason": "max_output_tokens"}}
    )
    assert (await server.verify(body)).failure_reason == "incomplete_response"


def completion(content: str):
    from nemo_gym.core import NeMoGymChatCompletion

    return NeMoGymChatCompletion.model_validate(
        {
            "id": "judge",
            "created": 0,
            "model": "judge",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        }
    )


@pytest.fixture
def judged_server() -> AgenticMMEServer:
    return AgenticMMEServer(
        config=AgenticMMEConfig(
            name="agentic_mme",
            host="localhost",
            port=1,
            entrypoint="",
            judge_model_server={"type": "responses_api_models", "name": "judge"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def judged_request(
    text: str, value: str = "44.6 million", golden: dict | None = None, metadata=None
) -> AgenticMMEVerifyRequest:
    return AgenticMMEVerifyRequest(
        responses_create_params={
            "input": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Image 0."},
                        {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "auto"},
                        {"type": "input_text", "text": "How much was it sold for?"},
                    ],
                },
            ]
        },
        response=response(text),
        verifier_metadata={"golden_answer": golden or {"value": value, "match_type": "exact"}} | (metadata or {}),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdict", "reward"),
    [
        ('{"verdict": "equivalent", "confidence": 0.9, "reason": "same amount"}', 1.0),
        ('```json\n{"verdict": "different", "reason": "other number"}\n```', 0.0),
        ('{"verdict": "unsure"}', 0.0),
        ("not json", 0.0),
    ],
)
async def test_judge_decides_reward(judged_server, monkeypatch, verdict, reward) -> None:
    seen = {}

    async def fake_call_judge(client, *, server_name, url_path, json, response_model):
        seen.update(server_name=server_name, url_path=url_path, messages=json["messages"])
        return completion(verdict)

    monkeypatch.setattr("resources_servers.agentic_mme.app.call_judge", fake_call_judge)
    # String match fails (exact), so any reward comes from the judge.
    result = await judged_server.verify(judged_request("<answer>$44.6M</answer>"))
    assert result.reward == reward
    assert result.string_match_reward == 0.0
    assert seen["server_name"] == "judge" and seen["url_path"] == "/v1/chat/completions"
    user = seen["messages"][1]["content"]
    assert "How much was it sold for?" in user and "44.6 million" in user and "$44.6M" in user
    assert "base64" not in user
    assert result.failure_reason == (None if reward else "incorrect_answer")


@pytest.mark.asyncio
async def test_judge_error_goes_to_failures_sidecar(judged_server, monkeypatch) -> None:
    from nemo_gym.verification import JudgeError, judge_failsafe

    async def failing_call_judge(*args, **kwargs):
        raise JudgeError("judge down")

    monkeypatch.setattr("resources_servers.agentic_mme.app.call_judge", failing_call_judge)
    request = judged_request("<answer>44.6 million</answer>")
    with pytest.raises(JudgeError):
        await judged_server.verify(request)
    # The base server wraps /verify in judge_failsafe: the row is masked and kept for regrading.
    routed = json.loads((await judge_failsafe(judged_server.verify)(body=request)).body)
    assert routed["reward"] == 0.0 and routed["mask_sample"] is True
    assert routed["failure_kind"] == "judge_failed" and routed["_ng_failure_class"] == "judge_failed"
    assert routed["response"]["output"][0]["content"][0]["text"] == "<answer>44.6 million</answer>"


@pytest.mark.asyncio
async def test_judge_skipped_without_answer(judged_server, monkeypatch) -> None:
    async def unexpected_call(*args, **kwargs):
        raise AssertionError("judge must not be called without an answer")

    monkeypatch.setattr("resources_servers.agentic_mme.app.call_judge", unexpected_call)
    result = await judged_server.verify(judged_request("<answer></answer>"))
    assert result.reward == 0.0 and result.failure_reason == "missing_answer"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("golden", "metadata", "shown"),
    [
        ({"value": "7556", "match_type": "numeric", "tolerance": 150.0}, None, "within ±150 of the reference"),
        ({"value": "7556", "match_type": "numeric"}, None, None),
        ({"value": "7556", "match_type": "contains", "tolerance": 150.0}, None, None),
        # Prepared rows: the released golden answer (contains) plus the author's tolerance for the judge.
        ({"value": "7556"}, {"answer_tolerance": 150.0}, "within ±150 of the reference"),
    ],
    ids=["numeric_golden", "numeric_no_tolerance", "contains_ignores_tolerance", "prepared_row"],
)
async def test_judge_sees_numeric_tolerance(judged_server, monkeypatch, golden, metadata, shown) -> None:
    seen = {}

    async def fake_call_judge(client, *, server_name, url_path, json, response_model):
        seen["user"] = json["messages"][1]["content"]
        return completion('{"verdict": "equivalent"}')

    monkeypatch.setattr("resources_servers.agentic_mme.app.call_judge", fake_call_judge)
    result = await judged_server.verify(judged_request("<answer>7600 km</answer>", golden=golden, metadata=metadata))
    assert result.reward == 1.0
    if shown:
        assert shown in seen["user"]
    else:
        assert "Tolerance:" not in seen["user"]


async def test_verifier_fixture() -> None:
    await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
    )
