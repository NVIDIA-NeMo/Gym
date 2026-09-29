# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise terminal rewards through the real verifier, not source-text guards."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.server_utils import ServerClient
from resources_servers.image_tools.app import (
    FailureCode,
    ImageToolsPivotResourcesServer,
    ImageToolsPivotResourcesServerConfig,
    ImageToolsPivotVerifyRequest,
)
from resources_servers.string_match.app import (
    StringMatchResourcesServer,
    StringMatchResourcesServerConfig,
    StringMatchVerifyRequest,
)
from responses_api_agents.tool_simulation_agent.app import ToolSimulationAgentVerifyRequest


@pytest.fixture
def server() -> ImageToolsPivotResourcesServer:
    return ImageToolsPivotResourcesServer(
        config=ImageToolsPivotResourcesServerConfig(host="localhost", port=0, entrypoint="app.py"),
        server_client=MagicMock(spec=ServerClient),
    )


def make_request(text: str, **row: Any) -> ImageToolsPivotVerifyRequest:
    return ImageToolsPivotVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "Answer the question."},
            "response": {
                "id": "test-response",
                "created_at": 1,
                "model": "unit-model",
                "object": "response",
                "output": [
                    {
                        "id": "test-message",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
            "expected_action": {"name": "__answer__", "arguments": {"answer": "4"}},
            **row,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,mode,gold,case_sensitive,reward",
    [
        ("4", "last_line", "4", False, 1.0),
        ("4", "full_response", "4", False, 1.0),
        ("4", "boxed", "4", False, 0.0),
        ("4", "final_answer", "4", False, 0.0),
        (r"\boxed{4.0}", "boxed", "4", False, 1.0),
        (r"\boxed{\text{4}}", "boxed", "4", False, 1.0),
        (r"\boxed{\frac{1}{2}}", "boxed", r"\frac{1}{2}", False, 1.0),
        (r"\boxed{1} then \boxed{4}", "boxed", "4", False, 1.0),
        ("Answer: 1\nAnswer: 4", "final_answer", "4", False, 1.0),
        (r"\boxed{1} Answer: 4", "final_answer", "4", False, 1.0),
        (r"\boxed{4}", "final_answer", "4", False, 1.0),
        ("Answer: left.", "final_answer", "left", False, 1.0),
        ("Answer: d", "final_answer", "D", False, 1.0),
        ("Answer: d", "final_answer", "D", True, 0.0),
        ("Answer: D", "final_answer", "D", True, 1.0),
        ("Answer: 4.0", "final_answer", "4", True, 0.0),
        (r'\boxed{"6"}', "boxed", "6", False, 1.0),
        (r"\boxed{Cricket  Ball}", "boxed", "cricket ball", False, 1.0),
        (r"\boxed{(5,16)}", "boxed", "(5,17)", False, 0.0),
        (r"\boxed{6}", "boxed", "9", False, 0.0),
        # Shared grading can return partial credit; don't coerce it to bool.
        (r"\boxed{100.1}", "boxed", "100", False, 0.98),
    ],
)
async def test_terminal_reward_matches_string_match(
    server: ImageToolsPivotResourcesServer,
    text: str,
    mode: str,
    gold: str,
    case_sensitive: bool,
    reward: float,
) -> None:
    body = make_request(
        text,
        extraction_mode=mode,
        case_sensitive=case_sensitive,
        expected_action={"name": "__answer__", "arguments": {"answer": gold}},
    )
    result = await server.verify(body)
    reference = StringMatchResourcesServer(
        config=StringMatchResourcesServerConfig(name="string_match", host="localhost", port=0, entrypoint="app.py"),
        server_client=MagicMock(spec=ServerClient),
    )
    shared_result = await reference.verify(
        StringMatchVerifyRequest.model_validate({**body.model_dump(), "expected_answer": gold})
    )
    assert result.reward == shared_result.reward == reward
    assert result.expected_tool_name == "__answer__"
    assert result.tool_family == "answer"
    assert result.num_rollout_tool_calls == 0
    if reward:
        assert result.failure_reason == FailureCode.NONE
    elif shared_result.extracted_answer is None:
        assert result.failure_reason == FailureCode.ANSWER_MISSING
    else:
        assert result.failure_reason == FailureCode.ANSWER_INCORRECT


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["boxed", "final_answer", "last_line", "full_response"])
@pytest.mark.parametrize("answer", ["", "\n"])
async def test_reasoning_only_is_not_an_answer(server: ImageToolsPivotResourcesServer, mode: str, answer: str) -> None:
    result = await server.verify(make_request(r"<think>\boxed{4}</think>" + answer, extraction_mode=mode))
    assert result.reward == 0.0
    assert result.failure_reason == FailureCode.ANSWER_MISSING


@pytest.mark.asyncio
async def test_reasoning_is_removed_before_extraction(server: ImageToolsPivotResourcesServer) -> None:
    result = await server.verify(make_request("<think>wrong guess</think>\n4", extraction_mode="full_response"))
    assert result.reward == 1.0
    assert result.model_output == "4"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments,text",
    [
        ({}, r"\boxed{}"),
        ({"answer": None}, "Answer: None"),
        ({"answer": ""}, r"\boxed{}"),
        ({"answer": " \t\n"}, r"\boxed{}"),
        ({"answer": []}, "Answer: []"),
        ({"answer": {}}, "Answer: {}"),
        ({"answer": False}, "Answer: False"),
        ({"answer": float("nan")}, "Answer: nan"),
        ({"answer": float("inf")}, "Answer: inf"),
        (None, r"\boxed{}"),
        ([], r"\boxed{}"),
        ("not JSON", r"\boxed{}"),
        ('{"answer": null}', "Answer: None"),
    ],
)
async def test_invalid_terminal_action_is_never_rewarded(
    server: ImageToolsPivotResourcesServer, arguments: Any, text: str
) -> None:
    result = await server.verify(make_request(text, expected_action={"name": "__answer__", "arguments": arguments}))
    assert result.reward == 0.0
    assert result.failure_reason == FailureCode.EXPECTED_ACTION_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("gold", [0, 0.0, "0"])
async def test_zero_is_a_valid_expected_answer(server: ImageToolsPivotResourcesServer, gold: Any) -> None:
    result = await server.verify(
        make_request(r"\boxed{0}", expected_action={"name": "__answer__", "arguments": {"answer": gold}})
    )
    assert result.reward == 1.0
    assert result.failure_reason == FailureCode.NONE


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["expected_action", "metadata", "expected_answer"])
async def test_terminal_action_locations(server: ImageToolsPivotResourcesServer, location: str) -> None:
    action = {"name": "__answer__", "arguments": json.dumps({"answer": "4"})}
    row = {"expected_action": None}
    row[location] = {"expected_action": action} if location == "metadata" else action
    if location == "expected_answer":
        row[location] = json.dumps(action)
    result = await server.verify(make_request(r"\boxed{4}", **row))
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_tool_call_rejected_even_with_correct_answer(server: ImageToolsPivotResourcesServer) -> None:
    text = '<tool_call>{"name":"image_rotate_tool","arguments":{"img_idx":0,"degrees":90}}</tool_call>\nAnswer: 4'
    result = await server.verify(make_request(text))
    assert result.num_rollout_tool_calls == 1
    assert result.reward == 0.0
    assert result.failure_reason == FailureCode.TOOL_CALL_WHEN_ANSWER_EXPECTED


@pytest.mark.asyncio
async def test_agent_schema_preserves_answer_options(server: ImageToolsPivotResourcesServer) -> None:
    body = make_request("4", extraction_mode="full_response", case_sensitive=True)
    forwarded = ToolSimulationAgentVerifyRequest.model_validate(body.model_dump()).model_dump()
    assert forwarded["extraction_mode"] == "full_response"
    assert forwarded["case_sensitive"] is True
    result = await server.verify(ImageToolsPivotVerifyRequest.model_validate(forwarded))
    assert result.reward == 1.0


def test_invalid_extraction_mode_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extraction_mode"):
        make_request("4", extraction_mode="unsupported")
