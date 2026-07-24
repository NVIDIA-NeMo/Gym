# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi import HTTPException
from starlette.responses import Response

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.synthetic_tool_use_agent.app import (
    NG_FAILURE_CLASS_KEY,
    TRANSIENT_FAILURE_CLASS,
    SyntheticToolUseAgent,
    SyntheticToolUseAgentConfig,
    SyntheticToolUseAgentRunRequest,
)


def make_agent(max_agent_steps: int = 50) -> SyntheticToolUseAgent:
    config = SyntheticToolUseAgentConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="app.py",
        name="synthetic_tool_use_agent",
        domain="agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="synthetic_resource_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_agent_steps=max_agent_steps,
    )
    return SyntheticToolUseAgent(config=config, server_client=MagicMock(spec=ServerClient))


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
    cookies: dict = {}


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
        input=[NeMoGymEasyInputMessage(role="system", content="Follow policy.")],
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


def test_materialized_responses_create_params_validation_accepts_system_prompt_and_tools() -> None:
    agent = make_agent()

    agent._validate_materialized_responses_create_params(materialized_params())


def test_initial_user_message_is_materialized_into_policy_input() -> None:
    agent = make_agent()
    synchronized = agent._materialize_initial_user_message(
        SyntheticToolUseAgentRunRequest(
            responses_create_params=materialized_params(),
            initial_user_message="Start from this request.",
        )
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
    params = materialized_params()
    params.input = []

    with pytest.raises(HTTPException) as exc_info:
        agent._validate_materialized_responses_create_params(params)

    assert exc_info.value.status_code == 400
    assert "system prompt" in exc_info.value.detail


def test_materialized_responses_create_params_validation_rejects_missing_tools() -> None:
    agent = make_agent()
    params = materialized_params()
    params.tools = []

    with pytest.raises(HTTPException) as exc_info:
        agent._validate_materialized_responses_create_params(params)

    assert exc_info.value.status_code == 400
    assert "materialized policy tools" in exc_info.value.detail


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

    assert response.id == "synthetic_tool_use_no_agent_response"
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
        if kwargs["url_path"] == "/record_agent_message":
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
        "/record_agent_message",
    ]
    assert any(agent._item_role(item) == "user" for item in response.output)


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
        if kwargs["url_path"] == "/record_agent_message":
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
        "/record_agent_message",
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


async def test_responses_treats_mixed_parallel_output_as_tool_calls_only() -> None:
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
        return JsonResponseStub(
            {
                "output": {"status": "ok"},
                "schema_valid": True,
                "should_continue": len(resource_requests) < 2,
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
        "/execute_agent_tool_call",
        "/execute_agent_tool_call",
    ]
    assert [output.call_id for output in function_calls] == ["call_1", "call_2"]
    assert [output.call_id for output in function_outputs] == ["call_1", "call_2"]
    raw_response_calls = [
        output["call_id"]
        for output in resource_requests[0]["json"]["response"]["output"]
        if output["type"] == "function_call"
    ]
    assert raw_response_calls == ["call_1", "call_2"]


async def test_responses_treats_mixed_single_output_as_first_tool_call_only() -> None:
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
    assert [request["url_path"] for request in resource_requests] == ["/execute_agent_tool_call"]
    assert [output.call_id for output in function_calls] == ["call_1"]
    assert [output.call_id for output in function_outputs] == ["call_1"]
    assert function_outputs[0].output == '  {"status":"ok"}\n'
    assert [output.type for output in model_response.output] == ["message", "function_call", "function_call_output"]
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
        if kwargs["url_path"] == "/record_agent_message":
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
        if kwargs["url_path"] == "/record_agent_message":
            return JsonResponseStub({"should_continue": True})
        if kwargs["url_path"] == "/next_user_message":
            return JsonResponseStub({"message": "###STOP###", "should_continue": False, "terminal_state": "complete"})
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)
    body = materialized_params()
    body.input.append(NeMoGymEasyInputMessage(role="user", content="Please finish this request."))

    response = await agent.responses(RequestStub(), Response(), body)

    assert resource_paths == ["/record_agent_message", "/next_user_message"]
    assert [output.type for output in response.output] == ["message"]


async def test_final_agent_text_records_step_limit_after_nonterminal_user_reply() -> None:
    agent = make_agent(max_agent_steps=1)
    resource_requests = []

    async def route_post(**kwargs):
        if kwargs["server_name"] == "policy_model":
            return JsonResponseStub(response_payload([assistant_message("msg_1", "What else can I help with?")]))
        resource_requests.append(kwargs)
        if kwargs["url_path"] == "/record_agent_message":
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
        "/record_agent_message",
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

    assert [request["url_path"] for request in resource_requests] == ["/record_agent_step_limit"]
    assert [call["tool_call_id"] for call in resource_requests[0]["json"]["tool_calls"]] == ["call_1", "call_2"]
    assert [output.type for output in response.output] == ["function_call", "function_call"]
    assert not any(output.type == "function_call_output" for output in response.output)


async def test_run_routes_rollout_5xx_to_transient_failure_sidecar_contract() -> None:
    agent = make_agent()

    async def route_post(**kwargs):
        if kwargs["url_path"] == "/seed_session":
            return JsonResponseStub({}, cookies={"session_id": "session-1"})
        if kwargs["url_path"] == "/v1/responses":
            raise http_error(503)
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)

    result = await agent.run(
        RequestStub(),
        SyntheticToolUseAgentRunRequest(responses_create_params=materialized_params()),
    )
    result_payload = result.model_dump(mode="json")

    assert result.reward == 0.0
    assert result_payload[NG_FAILURE_CLASS_KEY] == TRANSIENT_FAILURE_CLASS
    assert result_payload["error_class"] == TRANSIENT_FAILURE_CLASS
    assert result_payload["error_stage"] == "rollout"
    assert "503" in result_payload["error_message"]
    assert result.instance_config == {"mask_sample": True}
    assert result.response.id == "synthetic_tool_use_rollout_failed"
    assert agent.server_client.post.await_count == 2


async def test_run_routes_seed_rate_limit_to_transient_failure_sidecar_contract() -> None:
    agent = make_agent()
    agent.server_client.post = AsyncMock(side_effect=http_error(429))

    result = await agent.run(
        RequestStub(),
        SyntheticToolUseAgentRunRequest(responses_create_params=materialized_params()),
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
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)

    result = await agent.run(
        RequestStub(),
        SyntheticToolUseAgentRunRequest(responses_create_params=materialized_params()),
    )
    result_payload = result.model_dump(mode="json")

    assert result_payload[NG_FAILURE_CLASS_KEY] == TRANSIENT_FAILURE_CLASS
    assert result_payload["error_stage"] == "verify"
    assert "502" in result_payload["error_message"]
    assert result.instance_config == {"mask_sample": True}
    assert agent.server_client.post.await_count == 3


async def test_run_preserves_verified_semantic_zero_as_scored_result() -> None:
    agent = make_agent()

    async def route_post(**kwargs):
        if kwargs["url_path"] == "/seed_session":
            return JsonResponseStub({}, cookies={"session_id": "session-1"})
        if kwargs["url_path"] == "/v1/responses":
            return JsonResponseStub(response_payload([assistant_message("msg_1", "Incorrect answer.")]))
        if kwargs["url_path"] == "/verify":
            return JsonResponseStub(kwargs["json"] | {"reward": 0.0, "result": {"judge_label": "fail"}})
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)

    result = await agent.run(
        RequestStub(),
        SyntheticToolUseAgentRunRequest(responses_create_params=materialized_params()),
    )
    result_payload = result.model_dump(mode="json")

    assert result.reward == 0.0
    assert NG_FAILURE_CLASS_KEY not in result_payload
    assert result_payload["result"] == {"judge_label": "fail"}
    assert result.instance_config == {"mask_sample": False}


async def test_run_does_not_convert_non_retryable_4xx_to_transient_failure() -> None:
    agent = make_agent()

    async def route_post(**kwargs):
        if kwargs["url_path"] == "/seed_session":
            return JsonResponseStub({}, cookies={"session_id": "session-1"})
        if kwargs["url_path"] == "/v1/responses":
            raise http_error(400)
        raise AssertionError(f"Unexpected request: {kwargs}")

    agent.server_client.post = AsyncMock(side_effect=route_post)

    with pytest.raises(ClientResponseError) as exc_info:
        await agent.run(
            RequestStub(),
            SyntheticToolUseAgentRunRequest(responses_create_params=materialized_params()),
        )

    assert exc_info.value.status == 400
