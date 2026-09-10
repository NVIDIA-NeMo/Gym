# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.processors.user_assistant import (
    UserAssistantProcessor,
    UserAssistantProcessorConfig,
    UserAssistantRunRequest,
)
from nemo_gym.server_utils import ServerClient


def _http_response(payload: dict, *, cookies: dict | None = None) -> MagicMock:
    response = MagicMock(status=200, ok=True, cookies=cookies or {})
    response.content.read = AsyncMock(return_value=json.dumps(payload).encode())
    response.read = AsyncMock(return_value=json.dumps(payload))
    return response


def _model_response(response_id: str, text: str) -> dict:
    return {
        "id": response_id,
        "created_at": 1,
        "model": "model",
        "object": "response",
        "output": [
            {
                "id": f"{response_id}-message",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _user_tool_response() -> dict:
    response = _model_response("user-response", "I would like a vegetarian meal.")
    response["output"] = [
        {
            "arguments": '{"diet":"vegetarian","max_price":20}',
            "call_id": "save-preference",
            "name": "save_preference",
            "type": "function_call",
            "id": "function-call",
            "status": "completed",
        },
        {
            "call_id": "save-preference",
            "output": '{"saved":true}',
            "type": "function_call_output",
            "id": "function-output",
            "status": "completed",
        },
        *response["output"],
    ]
    return response


def _processor(*, max_turns: int = 4) -> UserAssistantProcessor:
    config = UserAssistantProcessorConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="conversation",
        assistant_agent=AgentServerRef(type="responses_api_agents", name="assistant"),
        user_agent=AgentServerRef(type="responses_api_agents", name="user"),
        resources_server=ResourcesServerRef(type="resources_servers", name="environment"),
        max_turns=max_turns,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    return UserAssistantProcessor(config=config, server_client=client)


def _request() -> UserAssistantRunRequest:
    return UserAssistantRunRequest(
        responses_create_params={
            "input": [{"role": "developer", "content": "Resolve the user's request."}],
            "tools": [
                {
                    "type": "function",
                    "name": "read_preferences",
                    "description": "Read saved preferences.",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": True,
                }
            ],
        },
        user_responses_create_params={
            "input": [{"role": "developer", "content": "Ask for a vegetarian meal."}],
            "tools": [
                {
                    "type": "function",
                    "name": "save_preference",
                    "description": "Save a preference.",
                    "parameters": {
                        "type": "object",
                        "properties": {"diet": {"type": "string"}},
                        "required": ["diet"],
                    },
                    "strict": True,
                }
            ],
        },
    )


@pytest.mark.asyncio
async def test_alternates_independent_agents_and_preserves_attribution() -> None:
    processor = _processor()
    calls = []

    async def post(**kwargs):
        calls.append(kwargs)
        path = kwargs["url_path"]
        if path == "/seed_session":
            return _http_response({}, cookies={"environment": "seeded"})
        if kwargs["server_name"] == "assistant":
            assistant_turn = len([call for call in calls if call["server_name"] == "assistant"])
            return _http_response(
                _model_response(
                    f"assistant-response-{assistant_turn}",
                    (
                        "What kind of meal would you like?"
                        if assistant_turn == 1
                        else "I recommend the vegetarian curry for $16."
                    ),
                ),
                cookies={"environment": "assistant-state", "assistant_agent": "assistant-session"},
            )
        if kwargs["server_name"] == "user":
            return _http_response(
                _user_tool_response(),
                cookies={"environment": "user-updated-state", "user_agent": "user-session"},
            )
        if path == "/episode_status":
            terminated = len([call for call in calls if call["url_path"] == "/episode_status"]) == 3
            return _http_response(
                {
                    "terminated": terminated,
                    "reason": "request_resolved" if terminated else None,
                    "state": {"diet": "vegetarian"} if terminated else {},
                },
                cookies=kwargs["cookies"],
            )
        if path == "/verify":
            return _http_response(kwargs["json"] | {"reward": 1.0})
        raise AssertionError(kwargs)

    processor.server_client.post = AsyncMock(side_effect=post)
    result = await processor.run(MagicMock(cookies={}), _request())

    assert result.reward == 1.0
    assert result.termination_reason == "request_resolved"
    assert [turn.participant for turn in result.assistant_trajectory] == ["assistant", "assistant"]
    assert [turn.participant for turn in result.user_trajectory] == ["user"]
    assert "What kind of meal would you like?" in result.response.output_text
    assert "vegetarian curry" in result.response.output_text
    assert [event.kind for event in result.episode_trajectory] == [
        "response_item",
        "state",
        "response_item",
        "response_item",
        "response_item",
        "state",
        "response_item",
        "state",
        "termination",
    ]
    user_event_types = [
        event.data["type"]
        for event in result.episode_trajectory
        if event.kind == "response_item" and event.participant == "user"
    ]
    assert user_event_types == ["function_call", "function_call_output", "message"]

    assistant_calls = [call for call in calls if call["server_name"] == "assistant"]
    assistant_call = assistant_calls[0]
    user_call = next(call for call in calls if call["server_name"] == "user")
    assert assistant_call["json"].tools[0]["name"] == "read_preferences"
    assert user_call["json"].tools[0]["name"] == "save_preference"
    assert user_call["json"].input[-1].content == "What kind of meal would you like?"
    assert assistant_call["cookies"] == {"environment": "seeded"}
    assert user_call["cookies"] == {"environment": "assistant-state"}
    assert "assistant_agent" not in user_call["cookies"]
    assert assistant_calls[1]["cookies"] == {
        "environment": "user-updated-state",
        "assistant_agent": "assistant-session",
    }
    assert "user_agent" not in assistant_calls[1]["cookies"]
    status_calls = [call for call in calls if call["url_path"] == "/episode_status"]
    assert status_calls[1]["cookies"] == {"environment": "user-updated-state"}


@pytest.mark.asyncio
async def test_max_turns_is_an_explicit_termination_reason() -> None:
    processor = _processor(max_turns=1)
    responses = iter(
        [
            _http_response({}, cookies={"environment": "seeded"}),
            _http_response(
                _model_response("assistant-response", "How can I help?"),
                cookies={"environment": "assistant", "assistant_agent": "session"},
            ),
            _http_response({"terminated": False, "state": {}}, cookies={"environment": "assistant"}),
        ]
    )

    async def post(**kwargs):
        if kwargs["url_path"] == "/verify":
            return _http_response(kwargs["json"] | {"reward": 0.0})
        return next(responses)

    processor.server_client.post = AsyncMock(side_effect=post)

    result = await processor.run(MagicMock(cookies={}), _request())

    assert result.termination_reason == "max_turns"
    assert result.turns_completed == 1
    assert result.episode_trajectory[-1].kind == "termination"
    assert result.episode_trajectory[-1].data == {"reason": "max_turns"}


@pytest.mark.asyncio
async def test_incomplete_agent_response_has_a_structured_termination_reason() -> None:
    processor = _processor()
    incomplete_response = _model_response("assistant-response", "")
    incomplete_response["output"] = []
    incomplete_response["incomplete_details"] = {"reason": "max_output_tokens"}
    responses = iter(
        [
            _http_response({}, cookies={"environment": "seeded"}),
            _http_response(incomplete_response, cookies={"environment": "seeded"}),
        ]
    )

    async def post(**kwargs):
        if kwargs["url_path"] == "/verify":
            return _http_response(kwargs["json"] | {"reward": 0.0})
        return next(responses)

    processor.server_client.post = AsyncMock(side_effect=post)
    result = await processor.run(MagicMock(cookies={}), _request())

    assert result.termination_reason == "assistant_max_output_tokens"
    assert result.episode_trajectory[-1].data == {"reason": "assistant_max_output_tokens"}
