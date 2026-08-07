# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi import HTTPException
from starlette.responses import Response

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.conversational_tool_use.simulation.app import (
    NG_FAILURE_CLASS_KEY,
    TRANSIENT_FAILURE_CLASS,
    ConversationalToolUseAgent,
    ConversationalToolUseAgentConfig,
    ConversationalToolUseAgentRunRequest,
    ConversationalToolUseAgentVerifyResponse,
)
from responses_api_agents.conversational_tool_use.simulation.prompt import agent_system_message


POLICY = "Follow policy."
TOP_LEVEL_TOOLS = [
    {
        "name": "lookup",
        "doc": "Look up state.",
        "params": {"type": "object", "properties": {}},
        "returns": {"type": "object", "properties": {}},
    }
]


def make_agent(max_agent_steps: int = 50, *, observability: bool = False) -> ConversationalToolUseAgent:
    config = ConversationalToolUseAgentConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="app.py",
        name="conversational_tool_use_agent",
        domain="agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="simulation_resource_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_agent_steps=max_agent_steps,
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {"observability_enabled": observability}
    return ConversationalToolUseAgent(config=config, server_client=server_client)


def assistant_message(message_id: str, text: str) -> dict:
    return {
        "id": message_id,
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


def response_payload(output: list[dict]) -> dict:
    return {
        "id": "resp_test",
        "created_at": 0.0,
        "model": "dummy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


class RequestStub:
    def __init__(self, *, cookies: dict | None = None, path_params: dict | None = None) -> None:
        self.cookies = cookies or {}
        self.path_params = path_params or {}


class JsonResponseStub:
    ok = True

    def __init__(self, payload: dict, cookies: dict | None = None) -> None:
        self.payload = payload
        self.cookies = cookies or {}

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def function_call(call_id: str, name: str, arguments: str) -> dict:
    return {
        "arguments": arguments,
        "call_id": call_id,
        "name": name,
        "type": "function_call",
        "id": call_id,
        "status": "completed",
    }


def http_error(status: int) -> ClientResponseError:
    request_info = MagicMock()
    request_info.real_url = "http://test.invalid/v1/responses"
    return ClientResponseError(
        request_info=request_info,
        history=(),
        status=status,
        message="upstream request failed",
    )


def test_canonicalize_run_transcript_moves_leading_user_items_to_input() -> None:
    agent = make_agent()
    responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
        input=[NeMoGymEasyInputMessage(role="system", content="Follow policy.")],
        parallel_tool_calls=False,
    )

    canonical_params, canonical_response = agent._canonicalize_run_transcript(
        responses_create_params,
        response_payload(
            [
                {"role": "user", "content": "I need help with my subscription.", "type": "message"},
                assistant_message("msg_1", "I can help."),
                {"role": "user", "content": "Thanks.", "type": "message"},
                assistant_message("msg_2", "Done. ###STOP###"),
            ]
        ),
    )

    assert [item.role for item in canonical_params.input] == ["system", "user"]
    assert canonical_params.input[-1].content == "I need help with my subscription."
    assert [item["role"] for item in canonical_response["output"]] == ["assistant", "user", "assistant"]
    assert canonical_response["output"][0]["content"][0]["text"] == "I can help."


def test_canonicalize_run_transcript_moves_user_only_terminal_output_to_input() -> None:
    agent = make_agent()
    responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
        input=[NeMoGymEasyInputMessage(role="system", content="Follow policy.")],
    )

    canonical_params, canonical_response = agent._canonicalize_run_transcript(
        responses_create_params,
        response_payload(
            [
                {
                    "role": "user",
                    "content": "###STOP###",
                    "type": "message",
                }
            ]
        ),
    )

    assert [item.role for item in canonical_params.input] == ["system", "user"]
    assert canonical_params.input[-1].content == "###STOP###"
    assert canonical_response["output"] == []


def test_normalize_response_input_items_preserves_default_message_type() -> None:
    agent = make_agent()

    normalized = agent._normalize_response_input_items([NeMoGymEasyInputMessage(role="user", content="hello")])

    assert normalized == [{"content": "hello", "role": "user", "type": "message"}]


def test_normalize_response_input_items_adds_function_call_output_type() -> None:
    agent = make_agent()

    normalized = agent._normalize_response_input_items(
        [NeMoGymFunctionCallOutput(call_id="call_1", output='{"ok": true}')]
    )

    assert normalized == [{"call_id": "call_1", "output": '{"ok": true}', "type": "function_call_output"}]


def materialized_params() -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[NeMoGymEasyInputMessage(role="system", content=agent_system_message(POLICY))],
        tools=[
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up state.",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            }
        ],
        parallel_tool_calls=False,
    )


def materialized_run_request(**kwargs) -> ConversationalToolUseAgentRunRequest:
    return ConversationalToolUseAgentRunRequest(
        responses_create_params=materialized_params(),
        profile="general",
        policy=POLICY,
        tools=TOP_LEVEL_TOOLS,
        **kwargs,
    )


def test_run_request_accepts_training_rows_without_profile() -> None:
    request = ConversationalToolUseAgentRunRequest(
        responses_create_params=materialized_params(),
        policy=POLICY,
        tools=TOP_LEVEL_TOOLS,
    )

    assert request.profile is None


def typed_trajectory_result() -> dict:
    return {
        "profile": "general",
        "source_artifacts": {},
        "trajectory": {
            "messages": [],
            "prefill_message_count": 0,
            "continuation_start_index": 0,
            "terminal_state": "complete",
            "generation_invalid_reason": None,
            "terminal_error": None,
            "agent_verification_result": None,
            "user_verification_result": None,
            "environment_verification_result": None,
        },
    }


def test_scored_verify_response_requires_typed_result() -> None:
    payload = materialized_run_request().model_dump(mode="json") | {
        "response": response_payload([]),
        "reward": 1.0,
    }

    with pytest.raises(ValueError, match="require a typed result"):
        ConversationalToolUseAgentVerifyResponse.model_validate(payload)


def test_materialized_responses_create_params_validation_accepts_system_prompt_and_tools() -> None:
    agent = make_agent()

    agent._validate_materialized_responses_create_params(materialized_run_request())


def test_initial_user_message_is_materialized_into_policy_input() -> None:
    agent = make_agent()
    synchronized = agent._materialize_initial_user_message(
        materialized_run_request(initial_user_message="Start from this request.")
    )

    items = agent._input_items(synchronized.responses_create_params)
    assert [agent._item_role(item) for item in items] == ["system", "user"]
    assert getattr(items[-1], "content") == "Start from this request."


def test_prefilled_history_resumes_after_assistant_message() -> None:
    agent = make_agent()
    params = materialized_params()
    params.input = agent._input_items(params) + [
        NeMoGymEasyInputMessage(role="user", content="I need help."),
        NeMoGymEasyInputMessage(role="assistant", content="What is your account ID?"),
    ]

    resume_state = agent._prefilled_resume_state(params)

    assert resume_state == ("user", [])


def test_prefilled_history_resumes_pending_tool_calls() -> None:
    agent = make_agent()
    params = materialized_params()
    params.input = agent._input_items(params) + [
        NeMoGymEasyInputMessage(role="user", content="Look up my account."),
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
        },
    ]

    next_actor, pending_calls = agent._prefilled_resume_state(params)

    assert next_actor == "environment"
    assert [(call.call_id, call.name, call.arguments) for call in pending_calls] == [("call_1", "lookup", "{}")]


def test_materialized_responses_create_params_validation_rejects_missing_system_prompt() -> None:
    agent = make_agent()
    body = materialized_run_request()
    body.responses_create_params.input = []

    with pytest.raises(HTTPException) as exc_info:
        agent._validate_materialized_responses_create_params(body)

    assert exc_info.value.status_code == 400
    assert "system prompt" in exc_info.value.detail


def test_materialized_responses_create_params_validation_rejects_missing_tools() -> None:
    agent = make_agent()
    body = materialized_run_request()
    body.responses_create_params.tools = []

    with pytest.raises(HTTPException) as exc_info:
        agent._validate_materialized_responses_create_params(body)

    assert exc_info.value.status_code == 400
    assert "materialized policy tools" in exc_info.value.detail


def test_materialized_policy_prompt_must_match_top_level_policy() -> None:
    agent = make_agent()
    body = materialized_run_request()
    body.policy = "A different policy."

    with pytest.raises(HTTPException, match="top-level policy"):
        agent._validate_materialized_responses_create_params(body)


def test_materialized_tools_must_match_top_level_tools() -> None:
    agent = make_agent()
    body = materialized_run_request()
    body.tools[0]["doc"] = "A different tool."

    with pytest.raises(HTTPException, match="top-level tool definitions"):
        agent._validate_materialized_responses_create_params(body)


def test_max_agent_steps_is_required_positive_primary_agent_call_cap() -> None:
    default_agent = make_agent()
    assert default_agent.config.max_agent_steps == 50

    agent = make_agent(max_agent_steps=7)
    assert agent.config.max_agent_steps == 7

    with pytest.raises(ValueError):
        make_agent(max_agent_steps=0)


def test_empty_response_can_represent_terminal_before_agent_call() -> None:
    agent = make_agent()

    response = agent._empty_response()

    assert response.id == "conversational_tool_use_no_agent_response"
    assert response.model == "policy_model"
    assert response.output == []
    assert response.parallel_tool_calls is False


async def test_responses_resumes_after_prefilled_assistant_with_user_simulator() -> None:
    agent = make_agent()
    requests = []

    async def route_post(**kwargs):
        requests.append(kwargs)
        if kwargs["url_path"] == "/next_user_message":
            return JsonResponseStub({"message": "My account ID is A-1.", "should_continue": True})
        if kwargs["server_name"] == "policy_model":
            input_items = kwargs["json"]["input"]
            assert input_items[-1]["role"] == "user"
            assert input_items[-1]["content"] == "My account ID is A-1."
            return JsonResponseStub(response_payload([assistant_message("msg_2", "Thanks.")]))
        if kwargs["url_path"] == "/record_agent_outputs":
            return JsonResponseStub({"should_continue": False})
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = materialized_params()
    body.input = agent._input_items(body) + [
        NeMoGymEasyInputMessage(role="user", content="I need help."),
        NeMoGymEasyInputMessage(role="assistant", content="What is your account ID?"),
    ]

    response = await agent.responses(RequestStub(), Response(), body)

    assert [request["url_path"] for request in requests] == [
        "/next_user_message",
        "/v1/responses",
        "/record_agent_outputs",
    ]
    assert any(agent._item_role(item) == "user" for item in response.output)


async def test_responses_forwards_inbound_rollout_prefix_to_policy_model() -> None:
    agent = make_agent()

    async def route_post(**kwargs):
        if kwargs["server_name"] == "policy_model":
            assert kwargs["url_path"] == "/ng-rollout/7-2/v1/responses"
            return JsonResponseStub(response_payload([assistant_message("msg_1", "Done.")]))
        if kwargs["url_path"] == "/record_agent_outputs":
            return JsonResponseStub({"should_continue": False})
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = materialized_params()
    body.input = agent._input_items(body) + [NeMoGymEasyInputMessage(role="user", content="Help me.")]

    await agent.responses(
        RequestStub(path_params={"rollout_id": "7-2"}),
        Response(),
        body,
    )


async def test_responses_executes_prefilled_pending_tool_call_before_policy() -> None:
    agent = make_agent()
    requests = []

    async def route_post(**kwargs):
        requests.append(kwargs)
        if kwargs["url_path"] == "/execute_agent_tool_call":
            return JsonResponseStub(
                {
                    "output": '{"status":"active"}',
                    "schema_valid": True,
                    "should_continue": True,
                }
            )
        if kwargs["server_name"] == "policy_model":
            assert kwargs["json"]["input"][-1]["type"] == "function_call_output"
            assert kwargs["json"]["input"][-1]["call_id"] == "call_1"
            return JsonResponseStub(response_payload([assistant_message("msg_2", "Your account is active.")]))
        if kwargs["url_path"] == "/record_agent_outputs":
            return JsonResponseStub({"should_continue": False})
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = materialized_params()
    body.input = agent._input_items(body) + [
        NeMoGymEasyInputMessage(role="user", content="Check my account."),
        NeMoGymResponseFunctionToolCall(
            call_id="call_1",
            name="lookup",
            arguments="{}",
        ),
    ]

    await agent.responses(RequestStub(), Response(), body)

    assert [request["url_path"] for request in requests] == [
        "/execute_agent_tool_call",
        "/v1/responses",
        "/record_agent_outputs",
    ]


async def test_responses_executes_all_parallel_function_calls_in_one_model_turn() -> None:
    agent = make_agent()
    tool_requests = []

    async def route_post(**kwargs):
        if kwargs["server_name"] == "policy_model":
            return JsonResponseStub(
                response_payload(
                    [
                        function_call("call_1", "lookup_order", '{"order_id":"ORD-1"}'),
                        function_call("call_2", "lookup_order", '{"order_id":"ORD-2"}'),
                    ]
                )
            )
        if kwargs["url_path"] == "/record_agent_outputs":
            assert [output["tool_call_id"] for output in kwargs["json"]["outputs"]] == [
                "call_1",
                "call_2",
            ]
            assert [output["response_output_index"] for output in kwargs["json"]["outputs"]] == [0, 1]
            return JsonResponseStub({"should_continue": True})
        tool_requests.append(kwargs["json"])
        return JsonResponseStub(
            {
                "output": {"status": "ok"},
                "schema_valid": True,
                "should_continue": len(tool_requests) < 2,
            }
        )

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            NeMoGymEasyInputMessage(role="system", content="Follow policy."),
            NeMoGymEasyInputMessage(role="user", content="I need help."),
        ],
        tools=[
            {
                "type": "function",
                "name": "lookup_order",
                "description": "Look up an order.",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
                "strict": True,
            }
        ],
        parallel_tool_calls=True,
    )

    model_response = await agent.responses(RequestStub(), Response(), body)

    function_outputs = [output for output in model_response.output if output.type == "function_call_output"]
    assert [request["tool_call_id"] for request in tool_requests] == ["call_1", "call_2"]
    assert [request["tool_name"] for request in tool_requests] == ["lookup_order", "lookup_order"]
    assert [output.call_id for output in function_outputs] == ["call_1", "call_2"]


async def test_responses_stops_parallel_tool_execution_after_terminal_result() -> None:
    agent = make_agent()
    tool_requests = []

    async def route_post(**kwargs):
        if kwargs["server_name"] == "policy_model":
            return JsonResponseStub(
                response_payload(
                    [
                        function_call("call_1", "lookup_order", '{"order_id":"ORD-1"}'),
                        function_call("call_2", "lookup_order", '{"order_id":"ORD-2"}'),
                    ]
                )
            )
        if kwargs["url_path"] == "/record_agent_outputs":
            return JsonResponseStub({"should_continue": True})
        tool_requests.append(kwargs["json"])
        return JsonResponseStub(
            {
                "output": {"status": "terminal"},
                "schema_valid": True,
                "should_continue": False,
            }
        )

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            NeMoGymEasyInputMessage(role="system", content="Follow policy."),
            NeMoGymEasyInputMessage(role="user", content="I need help."),
        ],
        tools=[
            {
                "type": "function",
                "name": "lookup_order",
                "description": "Look up an order.",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
                "strict": True,
            }
        ],
        parallel_tool_calls=True,
    )

    model_response = await agent.responses(RequestStub(), Response(), body)

    function_calls = [output for output in model_response.output if output.type == "function_call"]
    function_outputs = [output for output in model_response.output if output.type == "function_call_output"]
    assert [request["tool_call_id"] for request in tool_requests] == ["call_1"]
    assert [output.call_id for output in function_calls] == ["call_1", "call_2"]
    assert [output.call_id for output in function_outputs] == ["call_1"]


async def test_resumed_parallel_tool_execution_stops_after_terminal_result() -> None:
    agent = make_agent()
    tool_requests = []

    async def route_post(**kwargs):
        if kwargs["url_path"] == "/execute_agent_tool_call":
            tool_requests.append(kwargs["json"])
            return JsonResponseStub(
                {
                    "output": {"status": "terminal"},
                    "schema_valid": True,
                    "should_continue": False,
                }
            )
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = materialized_params()
    body.parallel_tool_calls = True
    body.input = agent._input_items(body) + [
        NeMoGymEasyInputMessage(role="user", content="Check both orders."),
        function_call("call_1", "lookup", "{}"),
        function_call("call_2", "lookup", "{}"),
    ]

    model_response = await agent.responses(RequestStub(), Response(), body)

    function_outputs = [output for output in model_response.output if output.type == "function_call_output"]
    assert [request["tool_call_id"] for request in tool_requests] == ["call_1"]
    assert [output.call_id for output in function_outputs] == ["call_1"]


async def test_responses_records_mixed_parallel_output_in_provider_order_before_tool_results() -> None:
    agent = make_agent()
    resource_requests = []

    async def route_post(**kwargs):
        if kwargs["server_name"] == "policy_model":
            return JsonResponseStub(
                response_payload(
                    [
                        function_call("call_1", "lookup_order", '{"order_id":"ORD-1"}'),
                        assistant_message("msg_1", "I need to check that."),
                        function_call("call_2", "lookup_order", '{"order_id":"ORD-2"}'),
                    ]
                )
            )
        resource_requests.append(kwargs)
        if kwargs["url_path"] == "/record_agent_outputs":
            return JsonResponseStub({"should_continue": True})
        return JsonResponseStub(
            {
                "output": {"status": "ok"},
                "schema_valid": True,
                "should_continue": len(resource_requests) < 3,
            }
        )

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            NeMoGymEasyInputMessage(role="system", content="Follow policy."),
            NeMoGymEasyInputMessage(role="user", content="I need help."),
        ],
        tools=[
            {
                "type": "function",
                "name": "lookup_order",
                "description": "Look up an order.",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
                "strict": True,
            }
        ],
        parallel_tool_calls=True,
    )

    model_response = await agent.responses(RequestStub(), Response(), body)

    function_calls = [output for output in model_response.output if output.type == "function_call"]
    function_outputs = [output for output in model_response.output if output.type == "function_call_output"]
    assert [request["url_path"] for request in resource_requests] == [
        "/record_agent_outputs",
        "/execute_agent_tool_call",
        "/execute_agent_tool_call",
    ]
    assert [output.call_id for output in function_calls] == ["call_1", "call_2"]
    assert [output.call_id for output in function_outputs] == ["call_1", "call_2"]
    assert [output["type"] for output in resource_requests[0]["json"]["outputs"]] == [
        "function_call",
        "message",
        "function_call",
    ]
    assert [output["response_output_index"] for output in resource_requests[0]["json"]["outputs"]] == [0, 1, 2]
    raw_response_calls = [
        output["call_id"]
        for output in resource_requests[0]["json"]["response"]["output"]
        if output["type"] == "function_call"
    ]
    assert raw_response_calls == ["call_1", "call_2"]


async def test_responses_drops_whitespace_only_message_adjacent_to_tool_call() -> None:
    agent = make_agent()
    resource_requests = []

    async def route_post(**kwargs):
        if kwargs["server_name"] == "policy_model":
            return JsonResponseStub(
                response_payload(
                    [
                        function_call("call_1", "lookup_order", '{"order_id":"ORD-1"}'),
                        assistant_message("msg_empty", "\n\n"),
                    ]
                )
            )
        resource_requests.append(kwargs)
        if kwargs["url_path"] == "/record_agent_outputs":
            return JsonResponseStub({"should_continue": True})
        return JsonResponseStub(
            {
                "output": {"status": "ok"},
                "schema_valid": True,
                "should_continue": False,
            }
        )

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            NeMoGymEasyInputMessage(role="system", content="Follow policy."),
            NeMoGymEasyInputMessage(role="user", content="I need help."),
        ],
        tools=[
            {
                "type": "function",
                "name": "lookup_order",
                "description": "Look up an order.",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
                "strict": True,
            }
        ],
        parallel_tool_calls=True,
    )

    model_response = await agent.responses(RequestStub(), Response(), body)

    assert [output.type for output in model_response.output] == ["function_call", "function_call_output"]
    assert [request["url_path"] for request in resource_requests] == [
        "/record_agent_outputs",
        "/execute_agent_tool_call",
    ]
    assert [output["type"] for output in resource_requests[0]["json"]["outputs"]] == ["function_call"]
    assert [output["type"] for output in resource_requests[0]["json"]["response"]["output"]] == ["function_call"]


async def test_responses_records_mixed_single_output_and_executes_first_tool_call_only() -> None:
    agent = make_agent()
    resource_requests = []

    async def route_post(**kwargs):
        if kwargs["server_name"] == "policy_model":
            return JsonResponseStub(
                response_payload(
                    [
                        assistant_message("msg_1", "I need to check that."),
                        function_call("call_1", "lookup_order", '{"order_id":"ORD-1"}'),
                        function_call("call_2", "lookup_order", '{"order_id":"ORD-2"}'),
                    ]
                )
            )
        resource_requests.append(kwargs)
        if kwargs["url_path"] == "/record_agent_outputs":
            return JsonResponseStub({"should_continue": True})
        return JsonResponseStub(
            {
                "output": '  {"status":"ok"}\n',
                "schema_valid": True,
                "should_continue": False,
            }
        )

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            NeMoGymEasyInputMessage(role="system", content="Follow policy."),
            NeMoGymEasyInputMessage(role="user", content="I need help."),
        ],
        tools=[
            {
                "type": "function",
                "name": "lookup_order",
                "description": "Look up an order.",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
                "strict": True,
            }
        ],
        parallel_tool_calls=False,
    )

    model_response = await agent.responses(RequestStub(), Response(), body)

    function_calls = [output for output in model_response.output if output.type == "function_call"]
    function_outputs = [output for output in model_response.output if output.type == "function_call_output"]
    assert [request["url_path"] for request in resource_requests] == [
        "/record_agent_outputs",
        "/execute_agent_tool_call",
    ]
    assert [output.call_id for output in function_calls] == ["call_1"]
    assert [output.call_id for output in function_outputs] == ["call_1"]
    assert function_outputs[0].output == '  {"status":"ok"}\n'
    assert [output.type for output in model_response.output] == ["message", "function_call", "function_call_output"]
    assert [output["response_output_index"] for output in resource_requests[0]["json"]["outputs"]] == [0, 1]
    raw_response_calls = [
        output["call_id"]
        for output in resource_requests[0]["json"]["response"]["output"]
        if output["type"] == "function_call"
    ]
    assert raw_response_calls == ["call_1"]


async def test_nonparallel_next_policy_request_contains_only_selected_tool_call() -> None:
    agent = make_agent()
    model_requests = []

    async def route_post(**kwargs):
        if kwargs["server_name"] == "policy_model":
            model_requests.append(kwargs["json"])
            if len(model_requests) == 1:
                return JsonResponseStub(
                    response_payload(
                        [
                            function_call("call_1", "lookup_order", '{"order_id":"ORD-1"}'),
                            function_call("call_2", "lookup_order", '{"order_id":"ORD-2"}'),
                        ]
                    )
                )
            return JsonResponseStub(response_payload([assistant_message("msg_2", "The lookup is complete.")]))
        if kwargs["url_path"] == "/execute_agent_tool_call":
            return JsonResponseStub({"output": '{"status":"ok"}', "schema_valid": True, "should_continue": True})
        if kwargs["url_path"] == "/record_agent_outputs":
            return JsonResponseStub({"should_continue": True})
        if kwargs["url_path"] == "/next_user_message":
            return JsonResponseStub({"message": "###STOP###", "should_continue": False, "terminal_state": "complete"})
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            NeMoGymEasyInputMessage(role="system", content="Follow policy."),
            NeMoGymEasyInputMessage(role="user", content="Look up both orders."),
        ],
        tools=[
            {
                "type": "function",
                "name": "lookup_order",
                "description": "Look up an order.",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
                "strict": True,
            }
        ],
        parallel_tool_calls=False,
    )

    await agent.responses(RequestStub(), Response(), body)

    second_request_calls = [item["call_id"] for item in model_requests[1]["input"] if item["type"] == "function_call"]
    second_request_outputs = [
        item["call_id"] for item in model_requests[1]["input"] if item["type"] == "function_call_output"
    ]
    assert second_request_calls == ["call_1"]
    assert second_request_outputs == ["call_1"]
    assert [item["output"] for item in model_requests[1]["input"] if item["type"] == "function_call_output"] == [
        '{"status":"ok"}'
    ]


async def test_final_agent_text_allows_user_stop_without_step_limit_failure() -> None:
    agent = make_agent(max_agent_steps=1)
    resource_paths = []

    async def route_post(**kwargs):
        if kwargs["server_name"] == "policy_model":
            return JsonResponseStub(response_payload([assistant_message("msg_1", "Your request is complete.")]))
        resource_paths.append(kwargs["url_path"])
        if kwargs["url_path"] == "/record_agent_outputs":
            return JsonResponseStub({"should_continue": True})
        if kwargs["url_path"] == "/next_user_message":
            return JsonResponseStub({"message": "###STOP###", "should_continue": False, "terminal_state": "complete"})
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = materialized_params()
    body.input.append(NeMoGymEasyInputMessage(role="user", content="Please finish this request."))

    response = await agent.responses(RequestStub(), Response(), body)

    assert resource_paths == ["/record_agent_outputs", "/next_user_message"]
    assert [output.type for output in response.output] == ["message"]


async def test_final_agent_text_records_step_limit_after_nonterminal_user_reply() -> None:
    agent = make_agent(max_agent_steps=1)
    resource_requests = []

    async def route_post(**kwargs):
        if kwargs["server_name"] == "policy_model":
            return JsonResponseStub(response_payload([assistant_message("msg_1", "What else can I help with?")]))
        resource_requests.append(kwargs)
        if kwargs["url_path"] == "/record_agent_outputs":
            return JsonResponseStub({"should_continue": True})
        if kwargs["url_path"] == "/next_user_message":
            return JsonResponseStub({"message": "I still need help.", "should_continue": True})
        if kwargs["url_path"] == "/record_agent_step_limit":
            return JsonResponseStub({"should_continue": False, "terminal_state": "incomplete"})
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = materialized_params()
    body.input.append(NeMoGymEasyInputMessage(role="user", content="Please help."))

    response = await agent.responses(RequestStub(), Response(), body)

    assert [request["url_path"] for request in resource_requests] == [
        "/record_agent_outputs",
        "/next_user_message",
        "/record_agent_step_limit",
    ]
    assert resource_requests[-1]["json"]["max_agent_steps"] == 1
    assert resource_requests[-1]["json"]["tool_calls"] == []
    assert [getattr(output, "role", None) for output in response.output] == ["assistant", "user"]


async def test_final_agent_tool_calls_terminate_without_tool_simulation_or_dummy_outputs() -> None:
    agent = make_agent(max_agent_steps=1)
    resource_requests = []

    async def route_post(**kwargs):
        if kwargs["server_name"] == "policy_model":
            return JsonResponseStub(
                response_payload(
                    [
                        function_call("call_1", "lookup_order", '{"order_id":"ORD-1"}'),
                        function_call("call_2", "lookup_order", '{"order_id":"ORD-2"}'),
                    ]
                )
            )
        resource_requests.append(kwargs)
        if kwargs["url_path"] == "/record_agent_outputs":
            return JsonResponseStub({"should_continue": True})
        if kwargs["url_path"] == "/record_agent_step_limit":
            return JsonResponseStub({"should_continue": False, "terminal_state": "incomplete"})
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            NeMoGymEasyInputMessage(role="system", content="Follow policy."),
            NeMoGymEasyInputMessage(role="user", content="Look up both orders."),
        ],
        tools=[
            {
                "type": "function",
                "name": "lookup_order",
                "description": "Look up an order.",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
                "strict": True,
            }
        ],
        parallel_tool_calls=True,
    )

    response = await agent.responses(RequestStub(), Response(), body)

    assert [request["url_path"] for request in resource_requests] == [
        "/record_agent_outputs",
        "/record_agent_step_limit",
    ]
    assert [output["tool_call_id"] for output in resource_requests[0]["json"]["outputs"]] == [
        "call_1",
        "call_2",
    ]
    assert resource_requests[1]["json"]["tool_calls"] == []
    assert [output.type for output in response.output] == ["function_call", "function_call"]
    assert not any(output.type == "function_call_output" for output in response.output)


async def test_run_routes_rollout_5xx_to_transient_failure_sidecar_contract() -> None:
    agent = make_agent()

    async def route_post(**kwargs):
        if kwargs["url_path"] == "/seed_session":
            return JsonResponseStub({}, cookies={"session_id": "session-1"})
        if kwargs["url_path"] == "/v1/responses":
            raise http_error(503)
        if kwargs["url_path"] == "/discard_session":
            assert kwargs["cookies"] == {"session_id": "session-1"}
            return JsonResponseStub({"discarded": True})
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)

    result = await agent.run(
        RequestStub(),
        materialized_run_request(),
    )
    result_payload = result.model_dump(mode="json")

    assert result.reward == 0.0
    assert result_payload[NG_FAILURE_CLASS_KEY] == TRANSIENT_FAILURE_CLASS
    assert result_payload["error_class"] == TRANSIENT_FAILURE_CLASS
    assert result_payload["error_stage"] == "rollout"
    assert "503" in result_payload["error_message"]
    assert result.instance_config == {"mask_sample": True}
    assert result.response.id == "conversational_tool_use_rollout_failed"
    assert agent.server_client.post.await_count == 3


async def test_run_correlates_self_and_resource_model_calls_with_rollout_identity() -> None:
    agent = make_agent(observability=True)
    requests = []

    async def route_post(**kwargs):
        requests.append(kwargs)
        if kwargs["url_path"] == "/seed_session":
            assert kwargs["json"]["rollout_id"] == "7-2"
            return JsonResponseStub({}, cookies={"session_id": "session-1"})
        if kwargs["url_path"] == "/ng-rollout/7-2/v1/responses":
            return JsonResponseStub(response_payload([assistant_message("msg_1", "Done.")]))
        if kwargs["url_path"] == "/verify":
            return JsonResponseStub(
                kwargs["json"]
                | {
                    "reward": 1.0,
                    "result": typed_trajectory_result(),
                }
            )
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = materialized_run_request(
        **{
            TASK_INDEX_KEY_NAME: 7,
            ROLLOUT_INDEX_KEY_NAME: 2,
        }
    )

    result = await agent.run(RequestStub(), body)

    assert result.reward == 1.0
    assert [request["url_path"] for request in requests] == [
        "/seed_session",
        "/ng-rollout/7-2/v1/responses",
        "/verify",
    ]


async def test_run_routes_seed_rate_limit_to_transient_failure_sidecar_contract() -> None:
    agent = make_agent()
    agent.server_client.post = AsyncMock(side_effect=http_error(429))

    result = await agent.run(
        RequestStub(),
        materialized_run_request(),
    )
    result_payload = result.model_dump(mode="json")

    assert result_payload[NG_FAILURE_CLASS_KEY] == TRANSIENT_FAILURE_CLASS
    assert result_payload["error_stage"] == "seed_session"
    assert "429" in result_payload["error_message"]
    assert result.instance_config == {"mask_sample": True}
    assert agent.server_client.post.await_count == 1


async def test_run_routes_verify_5xx_to_transient_failure_sidecar_contract() -> None:
    agent = make_agent()

    async def route_post(**kwargs):
        if kwargs["url_path"] == "/seed_session":
            return JsonResponseStub({}, cookies={"session_id": "session-1"})
        if kwargs["url_path"] == "/v1/responses":
            return JsonResponseStub(response_payload([assistant_message("msg_1", "Done.")]))
        if kwargs["url_path"] == "/verify":
            raise http_error(502)
        if kwargs["url_path"] == "/discard_session":
            assert kwargs["cookies"] == {"session_id": "session-1"}
            return JsonResponseStub({"discarded": True})
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)

    result = await agent.run(
        RequestStub(),
        materialized_run_request(),
    )
    result_payload = result.model_dump(mode="json")

    assert result_payload[NG_FAILURE_CLASS_KEY] == TRANSIENT_FAILURE_CLASS
    assert result_payload["error_stage"] == "verify"
    assert "502" in result_payload["error_message"]
    assert result.instance_config == {"mask_sample": True}
    assert agent.server_client.post.await_count == 4


async def test_run_preserves_verified_semantic_zero_as_scored_result() -> None:
    agent = make_agent()

    async def route_post(**kwargs):
        if kwargs["url_path"] == "/seed_session":
            return JsonResponseStub({}, cookies={"session_id": "session-1"})
        if kwargs["url_path"] == "/v1/responses":
            return JsonResponseStub(response_payload([assistant_message("msg_1", "Incorrect answer.")]))
        if kwargs["url_path"] == "/verify":
            return JsonResponseStub(
                kwargs["json"]
                | {
                    "reward": 0.0,
                    "result": typed_trajectory_result() | {"judge_label": "fail"},
                }
            )
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)

    result = await agent.run(
        RequestStub(),
        materialized_run_request(),
    )
    result_payload = result.model_dump(mode="json")

    assert result.reward == 0.0
    assert NG_FAILURE_CLASS_KEY not in result_payload
    assert result_payload["result"]["judge_label"] == "fail"
    assert result_payload["result"]["profile"] == "general"
    assert result.instance_config == {"mask_sample": False}


async def test_run_does_not_convert_non_retryable_4xx_to_transient_failure() -> None:
    agent = make_agent()

    async def route_post(**kwargs):
        if kwargs["url_path"] == "/seed_session":
            return JsonResponseStub({}, cookies={"session_id": "session-1"})
        if kwargs["url_path"] == "/v1/responses":
            raise http_error(400)
        if kwargs["url_path"] == "/discard_session":
            return JsonResponseStub({"discarded": True})
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)

    with pytest.raises(ClientResponseError) as exc_info:
        await agent.run(
            RequestStub(),
            materialized_run_request(),
        )

    assert exc_info.value.status == 400
