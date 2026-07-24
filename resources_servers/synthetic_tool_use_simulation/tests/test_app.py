# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientConnectionError, ClientResponseError
from fastapi import HTTPException

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.synthetic_tool_use_simulation.app import (
    AgentToolCallRequest,
    MessageType,
    RecordAgentMessageRequest,
    RecordAgentStepLimitRequest,
    Source,
    SyntheticMessage,
    SyntheticToolCallRequest,
    SyntheticToolUseSeedSessionRequest,
    SyntheticToolUseSimulationConfig,
    SyntheticToolUseSimulationServer,
    SyntheticToolUseVerifyRequest,
    TrajectoryEvaluationType,
    VerificationFailureLabel,
    VerificationResult,
    VerificationType,
)


class RequestStub:
    def __init__(self, session_id: str = "session_1") -> None:
        self.session = {SESSION_ID_KEY: session_id}


def make_server(
    enable_llm_judge: bool = False, enforce_transfer_ground_truth: bool = False
) -> SyntheticToolUseSimulationServer:
    config = SyntheticToolUseSimulationConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="app.py",
        name="synthetic_tool_use_simulation",
        domain="agent",
        enable_llm_judge=enable_llm_judge,
        enforce_transfer_ground_truth=enforce_transfer_ground_truth,
        judge_model_server={"type": "responses_api_models", "name": "judge_model"} if enable_llm_judge else None,
    )
    return SyntheticToolUseSimulationServer(config=config, server_client=MagicMock(spec=ServerClient))


def make_verify_request() -> SyntheticToolUseVerifyRequest:
    return SyntheticToolUseVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "response": {
                "id": "resp_test",
                "created_at": 0,
                "model": "dummy",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        }
    )


def make_judge_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "judge_response",
            "created_at": 0,
            "model": "judge_model",
            "object": "response",
            "output": [
                {
                    "id": "judge_message",
                    "content": [{"annotations": [], "text": text, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def http_error(status: int) -> ClientResponseError:
    request_info = MagicMock()
    request_info.real_url = "http://judge.invalid/v1/responses"
    return ClientResponseError(
        request_info=request_info,
        history=(),
        status=status,
        message="judge provider request failed",
    )


def test_retry_defaults() -> None:
    server = make_server()

    assert server.config.generation_attempts == 3
    assert server.config.judge_provider_attempts == 3
    assert server.config.enforce_transfer_ground_truth is False


async def test_seed_session_starts_from_prefilled_policy_user_message() -> None:
    server = make_server()
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "scenario fallback"},
            responses_create_params={
                "input": [
                    {"role": "system", "content": "Follow policy."},
                    {"role": "user", "content": "Use this exact initial request."},
                ]
            },
        ),
    )

    state = server.session_id_to_state["session_1"]
    assert [(message.source, message.content) for message in state.messages] == [
        (Source.USER, "Use this exact initial request.")
    ]


async def test_seed_session_rejects_conflicting_initial_user_messages() -> None:
    server = make_server()
    request = RequestStub()
    with pytest.raises(HTTPException) as exc_info:
        await server.seed_session(
            request,
            SyntheticToolUseSeedSessionRequest(
                domain_name="subscription support",
                policy="policy",
                customer_scenario={"reason_for_contact": "scenario fallback"},
                initial_user_message="First request.",
                responses_create_params={
                    "input": [
                        {"role": "system", "content": "Follow policy."},
                        {"role": "user", "content": "Different request."},
                    ]
                },
            ),
        )

    assert exc_info.value.status_code == 400


async def test_seed_session_hydrates_full_prefilled_history() -> None:
    server = make_server()
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            tools=[
                {
                    "name": "lookup_subscription",
                    "params": {"type": "object", "properties": {}},
                    "returns": {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                    },
                }
            ],
            customer_scenario={"reason_for_contact": "scenario fallback"},
            responses_create_params={
                "input": [
                    {"role": "system", "content": "Follow policy."},
                    {"role": "user", "content": "Check my subscription."},
                    {"role": "assistant", "content": "I will look it up."},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup_subscription",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": '{"status":"active"}',
                    },
                    {"role": "assistant", "content": "Your subscription is active."},
                    {"role": "user", "content": "Can you cancel it?"},
                ]
            },
        ),
    )

    state = server.session_id_to_state["session_1"]
    assert [(message.type, message.source) for message in state.messages] == [
        (MessageType.TEXT, Source.USER),
        (MessageType.TEXT, Source.AGENT),
        (MessageType.TOOL_CALL, Source.AGENT),
        (MessageType.TOOL_EXECUTION, Source.ENVIRONMENT),
        (MessageType.TEXT, Source.AGENT),
        (MessageType.TEXT, Source.USER),
    ]
    assert state.messages[3].schema_valid is True
    assert state.prefill_message_count == 6
    sidecar = server._trajectory_result(state)
    assert sidecar["trajectory"]["prefill_message_count"] == 6
    assert sidecar["trajectory"]["continuation_start_index"] == 6
    resume = await server.session_resume(request)
    assert resume.next_actor == "agent"
    assert resume.pending_tool_calls == []


async def test_resumed_pending_tool_call_is_not_recorded_twice(monkeypatch) -> None:
    server = make_server()
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            tools=[
                {
                    "name": "lookup_subscription",
                    "params": {"type": "object", "properties": {}},
                    "returns": {"type": "object"},
                }
            ],
            customer_scenario={"reason_for_contact": "scenario fallback"},
            responses_create_params={
                "input": [
                    {"role": "system", "content": "Follow policy."},
                    {"role": "user", "content": "Check my subscription."},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup_subscription",
                        "arguments": "{}",
                    },
                ]
            },
        ),
    )

    async def resumed_tool_result(state, pending_tool_call=None):
        assert pending_tool_call is not None
        assert pending_tool_call.tool_call_id == "call_1"
        return "{}", None

    monkeypatch.setattr(server, "_generate_tool_result", resumed_tool_result)
    result = await server.execute_agent_tool_call(
        request,
        AgentToolCallRequest(
            tool_name="lookup_subscription",
            tool_call_id="call_1",
            arguments="{}",
        ),
    )

    state = server.session_id_to_state["session_1"]
    assert result.schema_valid is True
    assert [message.type for message in state.messages].count(MessageType.TOOL_CALL) == 1
    assert state.messages[-1].type == MessageType.TOOL_EXECUTION


async def test_user_stop_completes_and_agent_stop_does_not(monkeypatch) -> None:
    server = make_server()
    request = RequestStub()

    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={
                "reason_for_contact": "I need help canceling my subscription.",
                "outside_policy_scope": False,
            },
        ),
    )

    user_message = await server.next_user_message(request)
    assert "canceling" in user_message.message

    terminal = await server.record_agent_message(request, RecordAgentMessageRequest(content="Done. ###STOP###"))
    assert terminal.should_continue is True
    assert terminal.terminal_state is None

    async def stop_user_message(state):
        return "###STOP###", None

    monkeypatch.setattr(server, "_generate_user_message", stop_user_message)
    final_user = await server.next_user_message(request)
    assert final_user.should_continue is False
    assert final_user.terminal_state == "complete"

    verified = await server.verify(request, make_verify_request())
    assert verified.reward == 1.0
    assert verified.terminal_state == "complete"
    assert verified.num_user_messages == 2
    assert verified.num_agent_messages == 1
    assert verified.failure_labels == []
    assert verified.result["trajectory"]["prefill_message_count"] == 0
    assert verified.result["trajectory"]["continuation_start_index"] == 0
    assert [message["source"] for message in verified.result["trajectory"]["messages"]] == ["user", "agent", "user"]
    assert verified.result["trajectory"]["messages"][0]["text"] == user_message.message


async def test_first_user_stop_is_invalid(monkeypatch) -> None:
    server = make_server()
    request = RequestStub()

    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "I need help."},
        ),
    )

    async def stop_user_message(state):
        return "###STOP###", None

    monkeypatch.setattr(server, "_generate_user_message", stop_user_message)
    user_message = await server.next_user_message(request)
    assert user_message.should_continue is False
    assert user_message.terminal_state == "incomplete"

    verified = await server.verify(request, make_verify_request())
    assert verified.reward == 0.0
    assert verified.trajectory_invalid_reasons == ["invalid_user_message"]
    assert verified.failure_labels == [VerificationFailureLabel.USER_FAILURE]


async def test_tool_schema_failure_is_recorded() -> None:
    server = make_server()
    request = RequestStub()

    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            tools=[
                {
                    "name": "lookup_subscription",
                    "description": "Look up a subscription.",
                    "parameters": {"type": "object", "properties": {"email": {"type": "string"}}},
                    "returns": {"type": "object", "required": ["status"]},
                }
            ],
            customer_scenario={"reason_for_contact": "Look up my subscription."},
        ),
    )

    result = await server.route_tool_call(
        "lookup_subscription",
        SyntheticToolCallRequest.model_validate({"email": "alex@example.com"}),
        request,
    )
    assert result.schema_valid is False
    assert result.error is not None

    verified = await server.verify(request, make_verify_request())
    assert verified.reward == 0.0
    assert verified.num_tool_schema_failures == 1
    assert verified.trajectory_invalid_reasons
    assert verified.failure_labels == [VerificationFailureLabel.TOOL_FAILURE]
    assert verified.result["trajectory"]["messages"][-1]["source"] == "environment"
    assert verified.result["trajectory"]["messages"][-1]["schema_valid"] is False


async def test_agent_step_limit_records_tool_calls_without_tool_simulation(monkeypatch) -> None:
    server = make_server()
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            tools=[
                {
                    "name": "lookup_subscription",
                    "doc": "Look up a subscription.",
                    "params": {
                        "type": "object",
                        "properties": {"email": {"type": "string"}},
                        "required": ["email"],
                    },
                    "returns": {"type": "object"},
                }
            ],
            customer_scenario={"reason_for_contact": "Look up my subscription."},
        ),
    )

    async def unexpected_tool_simulation(state):
        raise AssertionError("tool simulator must not run at the agent-step limit")

    monkeypatch.setattr(server, "_generate_tool_result", unexpected_tool_simulation)
    result = await server.record_agent_step_limit(
        request,
        RecordAgentStepLimitRequest(
            max_agent_steps=50,
            tool_calls=[
                {
                    "tool_name": "lookup_subscription",
                    "tool_call_id": "call_1",
                    "arguments": '{"email":"alex@example.com"}',
                }
            ],
            response={"id": "resp_limit"},
        ),
    )

    state = server.session_id_to_state["session_1"]
    assert result.should_continue is False
    assert result.terminal_state == "incomplete"
    assert [message.type for message in state.messages] == [MessageType.TOOL_CALL]
    assert state.messages[0].deserialized_arguments == {"email": "alex@example.com"}
    assert state.generation_invalid_reason == "excessive_length"
    assert state.failure_labels == [VerificationFailureLabel.TRAJECTORY_FAILURE]
    assert state.terminal_error == "The maximum number of agent steps of 50 has been reached"
    verified = await server.verify(request, make_verify_request())
    assert verified.instance_config == {"mask_sample": False}


async def test_agent_step_limit_after_text_and_user_reply_preserves_messages() -> None:
    server = make_server()
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "I still need help."},
        ),
    )
    state = server.session_id_to_state["session_1"]
    state.messages = [
        SyntheticMessage(type=MessageType.TEXT, source=Source.AGENT, content="What else can I help with?"),
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="I still need help."),
    ]

    result = await server.record_agent_step_limit(
        request,
        RecordAgentStepLimitRequest(max_agent_steps=1),
    )

    assert result.terminal_state == "incomplete"
    assert len(state.messages) == 2
    assert state.generation_invalid_reason == "excessive_length"


async def test_tool_argument_enum_is_validated_by_jsonschema() -> None:
    server = make_server()
    request = RequestStub()

    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            tools=[
                {
                    "name": "update_subscription",
                    "description": "Update a subscription.",
                    "parameters": {
                        "type": "object",
                        "properties": {"action": {"type": "string", "enum": ["cancel"]}},
                        "required": ["action"],
                    },
                    "returns": {"type": "object"},
                }
            ],
            customer_scenario={"reason_for_contact": "Cancel my subscription."},
        ),
    )

    result = await server.route_tool_call(
        "update_subscription",
        SyntheticToolCallRequest.model_validate({"action": "refund"}),
        request,
    )

    assert result.should_continue is False
    assert result.terminal_state == "incomplete"
    assert result.error is not None
    verified = await server.verify(request, make_verify_request())
    assert verified.reward == 0.0
    assert verified.trajectory_invalid_reasons == ["invalid_agent_tool_call"]
    assert verified.failure_labels == [VerificationFailureLabel.AGENT_FAILURE]
    assert verified.instance_config == {"mask_sample": False}
    assert verified.result["trajectory"]["messages"][-1]["tool_name"] == "update_subscription"


async def test_user_generation_error_records_invalid_reason(monkeypatch) -> None:
    server = make_server()
    request = RequestStub()

    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "I need help."},
        ),
    )

    async def raise_generation_error(state):
        raise RuntimeError("model server returned no text")

    monkeypatch.setattr(server, "_generate_user_message", raise_generation_error)

    user_message = await server.next_user_message(request)

    assert user_message.should_continue is False
    assert user_message.terminal_state == "incomplete"
    state = server.session_id_to_state["session_1"]
    assert state.messages == []
    assert state.generation_invalid_reason == "message_generation_error"
    assert state.terminal_error == "model server returned no text"

    verified = await server.verify(request, make_verify_request())
    assert verified.reward == 0.0
    assert verified.trajectory_invalid_reasons == ["message_generation_error"]
    assert verified.failure_labels == [VerificationFailureLabel.USER_FAILURE]
    assert verified.instance_config == {"mask_sample": True}


async def test_tool_generation_error_records_invalid_reason(monkeypatch) -> None:
    server = make_server()
    request = RequestStub()

    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            tools=[
                {
                    "name": "lookup_subscription",
                    "doc": "Look up a subscription.",
                    "params": {"type": "object", "properties": {"email": {"type": "string"}}},
                    "returns": {"type": "object"},
                }
            ],
            customer_scenario={"reason_for_contact": "Look up my subscription."},
        ),
    )

    async def raise_generation_error(state):
        raise RuntimeError("tool simulator returned no text")

    monkeypatch.setattr(server, "_generate_tool_result", raise_generation_error)

    result = await server.execute_agent_tool_call(
        request,
        AgentToolCallRequest(
            tool_name="lookup_subscription",
            tool_call_id="call_1",
            arguments='{"email":"alex@example.com"}',
        ),
    )

    assert result.should_continue is False
    assert result.terminal_state == "incomplete"
    assert result.error == "tool simulator returned no text"
    state = server.session_id_to_state["session_1"]
    assert [message.type for message in state.messages] == [MessageType.TOOL_CALL]
    assert state.generation_invalid_reason == "message_generation_error"

    verified = await server.verify(request, make_verify_request())
    assert verified.trajectory_invalid_reasons == ["message_generation_error"]
    assert verified.failure_labels == [VerificationFailureLabel.TOOL_FAILURE]
    assert verified.instance_config == {"mask_sample": True}


async def test_none_tool_schemas_skip_validation(monkeypatch) -> None:
    server = make_server()
    request = RequestStub()

    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            tools=[
                {
                    "name": "freeform_tool",
                    "doc": "Execute a freeform tool.",
                    "params": None,
                    "returns": None,
                }
            ],
            customer_scenario={"reason_for_contact": "Use the freeform tool."},
        ),
    )

    async def raw_tool_result(state):
        return "not json", None

    monkeypatch.setattr(server, "_generate_tool_result", raw_tool_result)

    result = await server.execute_agent_tool_call(
        request,
        AgentToolCallRequest(
            tool_name="freeform_tool",
            tool_call_id="call_1",
            arguments="not json",
        ),
    )

    assert result.schema_valid is True
    assert result.output == "not json"
    state = server.session_id_to_state["session_1"]
    assert len(state.messages) == 2
    assert state.messages[0].deserialized_arguments is None
    assert state.messages[1].deserialized_execution_result is None
    assert state.terminal_state is None


async def test_valid_tool_result_returns_exact_raw_simulator_content(monkeypatch) -> None:
    server = make_server()
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            tools=[
                {
                    "name": "lookup_subscription",
                    "doc": "Look up a subscription.",
                    "params": {"type": "object", "properties": {}},
                    "returns": {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                    },
                }
            ],
            customer_scenario={"reason_for_contact": "Look up my subscription."},
        ),
    )
    raw_content = '```json\n{ "status": "active" }\n```'

    async def raw_tool_result(state):
        return raw_content, {"id": "tool_simulator_response"}

    monkeypatch.setattr(server, "_generate_tool_result", raw_tool_result)
    result = await server.execute_agent_tool_call(
        request,
        AgentToolCallRequest(
            tool_name="lookup_subscription",
            tool_call_id="call_1",
            arguments="{}",
        ),
    )

    state = server.session_id_to_state["session_1"]
    assert result.schema_valid is True
    assert result.output == raw_content
    assert state.messages[-1].content == raw_content
    assert state.messages[-1].deserialized_execution_result == {"status": "active"}


async def test_llm_judge_uses_termination_order(monkeypatch) -> None:
    server = make_server(enable_llm_judge=True)
    request = RequestStub()

    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "Look up my subscription."},
        ),
    )
    state = server.session_id_to_state["session_1"]
    state.messages = [
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="I need help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.AGENT, content="I can help."),
        SyntheticMessage(
            type=MessageType.TOOL_CALL,
            source=Source.AGENT,
            tool_call_id="call_1",
            tool_name="lookup_subscription",
            arguments='{"email":"alex@example.com"}',
        ),
        SyntheticMessage(
            type=MessageType.TOOL_EXECUTION,
            source=Source.ENVIRONMENT,
            tool_call_id="call_1",
            tool_name="lookup_subscription",
            arguments='{"email":"alex@example.com"}',
            content='{"status":"active"}',
        ),
        SyntheticMessage(type=MessageType.TEXT, source=Source.AGENT, content="You are active."),
    ]
    state.terminal_state = "complete"

    calls = []

    async def fake_judge(evaluation_type, user_message, state=None):
        calls.append(evaluation_type)
        return VerificationResult(reward=1, explanation="ok", judge_response='{"success":true,"explanation":"ok"}')

    monkeypatch.setattr(server, "_generate_judge_evaluation", fake_judge)

    verified = await server.verify(request, make_verify_request())
    assert verified.reward == 1.0
    assert calls == [
        Source.USER,
        Source.ENVIRONMENT,
        Source.AGENT,
        Source.AGENT,
        Source.AGENT,
        TrajectoryEvaluationType.AGENT_CONVERSATION,
    ]
    assert verified.failure_labels == []
    assert verified.judge_diagnostics["verification_type"] == "message"
    agent_verification = verified.result["trajectory"]["agent_verification_result"]
    assert agent_verification["overall_reward"] == 1
    assert agent_verification["conversation_verification_result"]["reward"] == 1


@pytest.mark.parametrize(
    ("outside_policy_scope", "terminal_message", "expected_transfer"),
    [
        (False, "###TRANSFER###", False),
        (True, "###STOP###", True),
    ],
)
async def test_transfer_ground_truth_mismatch_zeroes_reward_and_skips_judge(
    monkeypatch, outside_policy_scope: bool, terminal_message: str, expected_transfer: bool
) -> None:
    server = make_server(enable_llm_judge=True, enforce_transfer_ground_truth=True)
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={
                "reason_for_contact": "Look up my subscription.",
                "outside_policy_scope": outside_policy_scope,
            },
        ),
    )
    state = server.session_id_to_state["session_1"]
    state.messages = [
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="I need help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.AGENT, content="I will handle this."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content=terminal_message),
    ]
    state.terminal_state = "complete"

    judge = AsyncMock()
    monkeypatch.setattr(server, "_verify_messages", judge)

    verified = await server.verify(request, make_verify_request())

    assert verified.reward == 0.0
    assert verified.failure_labels == [VerificationFailureLabel.AGENT_FAILURE]
    assert verified.instance_config == {"mask_sample": False}
    assert verified.transfer_ground_truth_enforced is True
    assert verified.transfer_ground_truth_mismatch is True
    assert verified.judge_skipped_for_transfer_mismatch is True
    judge.assert_not_awaited()
    assert verified.judge_diagnostics["transfer_ground_truth_enforcement"] == {
        "expected_transfer": expected_transfer,
        "observed_transfer": not expected_transfer,
        "mismatch": True,
        "judge_skipped": True,
    }
    agent_verification = verified.result["trajectory"]["agent_verification_result"]
    assert agent_verification["overall_reward"] == 0
    assert agent_verification["conversation_verification_result"]["reward"] == 0


async def test_matching_transfer_ground_truth_still_runs_judge(monkeypatch) -> None:
    server = make_server(enable_llm_judge=True, enforce_transfer_ground_truth=True)
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={
                "reason_for_contact": "Look up my subscription.",
                "outside_policy_scope": False,
            },
        ),
    )
    state = server.session_id_to_state["session_1"]
    state.messages = [
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="I need help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.AGENT, content="I can help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="###STOP###"),
    ]
    state.terminal_state = "complete"

    judge = AsyncMock(return_value=(1.0, [], [], {"verification_type": "message"}))
    monkeypatch.setattr(server, "_verify_messages", judge)

    verified = await server.verify(request, make_verify_request())

    assert verified.reward == 1.0
    assert verified.failure_labels == []
    assert verified.transfer_ground_truth_enforced is True
    assert verified.transfer_ground_truth_mismatch is False
    assert verified.judge_skipped_for_transfer_mismatch is False
    judge.assert_awaited_once_with(state)
    assert verified.judge_diagnostics["transfer_ground_truth_enforcement"] == {
        "expected_transfer": False,
        "observed_transfer": False,
        "mismatch": False,
        "judge_skipped": False,
    }


async def test_llm_judge_labels_tool_response_failure_and_short_circuits(monkeypatch) -> None:
    server = make_server(enable_llm_judge=True)
    request = RequestStub()

    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "Look up my subscription."},
        ),
    )
    state = server.session_id_to_state["session_1"]
    state.messages = [
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="I need help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.AGENT, content="I can help."),
        SyntheticMessage(
            type=MessageType.TOOL_EXECUTION,
            source=Source.ENVIRONMENT,
            tool_call_id="call_1",
            tool_name="lookup_subscription",
            arguments='{"email":"alex@example.com"}',
            content='{"status":"wrong"}',
        ),
        SyntheticMessage(type=MessageType.TEXT, source=Source.AGENT, content="You are active."),
    ]
    state.terminal_state = "complete"

    calls = []

    async def fake_judge(evaluation_type, user_message, state=None):
        calls.append(evaluation_type)
        reward = 0 if evaluation_type == Source.ENVIRONMENT else 1
        return VerificationResult(reward=reward, explanation="checked", judge_response="{}")

    monkeypatch.setattr(server, "_generate_judge_evaluation", fake_judge)

    verified = await server.verify(request, make_verify_request())
    assert verified.reward == 0.0
    assert calls == [Source.USER, Source.ENVIRONMENT]
    assert verified.trajectory_invalid_reasons == ["no_reward_environment_message"]
    assert verified.failure_labels == [VerificationFailureLabel.TOOL_FAILURE]
    assert verified.num_tool_failures == 1
    assert verified.instance_config == {"mask_sample": True}
    assert verified.result["trajectory"]["agent_verification_result"]["trajectory_invalid_reasons"] == [
        "no_reward_environment_message"
    ]


async def test_combined_verification_type_uses_single_judge_shape(monkeypatch) -> None:
    server = make_server(enable_llm_judge=True)
    server.config.verification_type = VerificationType.COMPLETE_TRAJECTORY_COMBINED_EVALUATION
    request = RequestStub()

    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "Look up my subscription."},
        ),
    )
    state = server.session_id_to_state["session_1"]
    state.messages = [
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="I need help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.AGENT, content="I cannot help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="###STOP###"),
    ]
    state.terminal_state = "complete"

    calls = []

    async def fake_combined_judge(state, user_message):
        calls.append(user_message)
        return (
            VerificationResult(reward=1, explanation="customer ok"),
            VerificationResult(reward=0, explanation="agent bad", judge_response="{}"),
            VerificationResult(reward=1, explanation="tools ok"),
        )

    monkeypatch.setattr(server, "_generate_user_agent_environment_evaluation", fake_combined_judge)

    verified = await server.verify(request, make_verify_request())

    assert len(calls) == 1
    assert verified.reward == 0.0
    assert verified.trajectory_invalid_reasons == []
    assert verified.failure_labels == [VerificationFailureLabel.AGENT_FAILURE]
    assert verified.instance_config == {"mask_sample": False}
    assert verified.judge_diagnostics["verification_type"] == "complete_trajectory_combined_evaluation"
    assert verified.judge_diagnostics["conversation_verification_result"]["reward"] == 0
    trajectory_result = verified.result["trajectory"]
    assert trajectory_result["user_verification_result"]["reward"] == 1
    assert trajectory_result["environment_verification_result"]["reward"] == 1
    assert trajectory_result["agent_verification_result"]["overall_reward"] == 0
    assert trajectory_result["agent_verification_result"]["conversation_verification_result"]["reward"] == 0


async def test_judge_provider_retries_transient_errors_with_bounded_backoff(monkeypatch) -> None:
    server = make_server(enable_llm_judge=True)
    server.config.judge_provider_attempts = 4
    server.config.judge_provider_retry_initial_backoff_seconds = 0.25
    server.config.judge_provider_retry_max_backoff_seconds = 0.6
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "I need help."},
        ),
    )
    state = server.session_id_to_state["session_1"]
    call_model = AsyncMock(
        side_effect=[
            http_error(429),
            ClientConnectionError("connection reset"),
            http_error(503),
            make_judge_response('{"success":true,"explanation":"ok"}'),
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(server, "_call_model", call_model)
    monkeypatch.setattr("resources_servers.synthetic_tool_use_simulation.app.asyncio.sleep", sleep)

    result = await server._generate_judge_evaluation(Source.USER, "Evaluate this message.", state)

    assert result.reward == 1
    assert call_model.await_count == 4
    assert [call.args[0] for call in sleep.await_args_list] == [0.25, 0.5, 0.6]


async def test_malformed_judge_output_uses_only_semantic_attempts(monkeypatch) -> None:
    server = make_server(enable_llm_judge=True)
    server.config.generation_attempts = 2
    server.config.judge_provider_attempts = 5
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "I need help."},
        ),
    )
    state = server.session_id_to_state["session_1"]
    call_model = AsyncMock(
        side_effect=[
            make_judge_response("not valid json"),
            make_judge_response('{"success":true,"explanation":"ok"}'),
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(server, "_call_model", call_model)
    monkeypatch.setattr("resources_servers.synthetic_tool_use_simulation.app.asyncio.sleep", sleep)

    result = await server._generate_judge_evaluation(Source.USER, "Evaluate this message.", state)

    assert result.reward == 1
    assert call_model.await_count == 2
    sleep.assert_not_awaited()


async def test_combined_judge_uses_provider_retry_path(monkeypatch) -> None:
    server = make_server(enable_llm_judge=True)
    server.config.judge_provider_attempts = 2
    server.config.judge_provider_retry_initial_backoff_seconds = 0
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "I need help."},
        ),
    )
    state = server.session_id_to_state["session_1"]
    combined_evaluation = {
        "customer_success": True,
        "customer_explanation": "ok",
        "representative_success": True,
        "representative_explanation": "ok",
        "tool_results_success": True,
        "tool_results_explanation": "ok",
    }
    call_model = AsyncMock(
        side_effect=[
            http_error(503),
            make_judge_response(json.dumps(combined_evaluation)),
        ]
    )
    monkeypatch.setattr(server, "_call_model", call_model)

    user_result, agent_result, environment_result = await server._generate_user_agent_environment_evaluation(
        state, "Evaluate this trajectory."
    )

    assert user_result is not None and user_result.reward == 1
    assert agent_result.reward == 1
    assert environment_result is not None and environment_result.reward == 1
    assert call_model.await_count == 2


async def test_exhausted_judge_provider_errors_return_retryable_http_failure(monkeypatch) -> None:
    server = make_server(enable_llm_judge=True)
    server.config.judge_provider_attempts = 2
    server.config.judge_provider_retry_initial_backoff_seconds = 0
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "I need help."},
        ),
    )
    state = server.session_id_to_state["session_1"]
    state.messages = [
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="I need help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.AGENT, content="I can help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="###STOP###"),
    ]
    state.terminal_state = "complete"
    call_model = AsyncMock(side_effect=http_error(503))
    monkeypatch.setattr(server, "_call_model", call_model)

    with pytest.raises(HTTPException) as exc_info:
        await server.verify(request, make_verify_request())

    assert exc_info.value.status_code == 503
    assert "failed after 2 attempts" in str(exc_info.value.detail)
    assert call_model.await_count == 2
    assert state.verification_reward is None


async def test_nonretryable_judge_provider_errors_fail_without_retry(monkeypatch) -> None:
    server = make_server(enable_llm_judge=True)
    server.config.judge_provider_attempts = 5
    server.config.judge_provider_retry_initial_backoff_seconds = 0
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "I need help."},
        ),
    )
    state = server.session_id_to_state["session_1"]
    state.messages = [
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="I need help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.AGENT, content="I can help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="###STOP###"),
    ]
    state.terminal_state = "complete"
    call_model = AsyncMock(side_effect=http_error(400))
    monkeypatch.setattr(server, "_call_model", call_model)

    with pytest.raises(HTTPException) as exc_info:
        await server.verify(request, make_verify_request())

    assert exc_info.value.status_code == 400
    assert call_model.await_count == 1
    assert state.verification_reward is None


async def test_judge_exception_is_synchronized_into_internal_sidecar(monkeypatch) -> None:
    server = make_server(enable_llm_judge=True)
    request = RequestStub()
    await server.seed_session(
        request,
        SyntheticToolUseSeedSessionRequest(
            domain_name="subscription support",
            policy="policy",
            customer_scenario={"reason_for_contact": "I need help."},
        ),
    )
    state = server.session_id_to_state["session_1"]
    state.messages = [
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="I need help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.AGENT, content="I can help."),
        SyntheticMessage(type=MessageType.TEXT, source=Source.USER, content="###STOP###"),
    ]
    state.terminal_state = "complete"

    async def raise_judge_error(state):
        raise RuntimeError("judge provider unavailable")

    monkeypatch.setattr(server, "_verify_messages", raise_judge_error)

    verified = await server.verify(request, make_verify_request())

    assert verified.reward == 0.0
    assert verified.trajectory_invalid_reasons == ["verification_generation_error"]
    assert verified.failure_labels == [VerificationFailureLabel.VERIFICATION_GENERATION_ERROR]
    assert verified.instance_config == {"mask_sample": True}
    assert verified.judge_generation_error == "RuntimeError: judge provider unavailable"
    agent_verification = verified.result["trajectory"]["agent_verification_result"]
    assert agent_verification["trajectory_invalid_reasons"] == ["verification_generation_error"]
    assert agent_verification["conversation_verification_result"]["generation_error"] == (
        "RuntimeError: judge provider unavailable"
    )


def test_compute_metrics_reports_transfer_enforcement_rates() -> None:
    server = make_server()
    tasks = [
        [
            {
                "transferred": True,
                "transfer_ground_truth_mismatch": True,
                "judge_skipped_for_transfer_mismatch": True,
            },
            {
                "transferred": False,
                "transfer_ground_truth_mismatch": False,
                "judge_skipped_for_transfer_mismatch": False,
            },
        ]
    ]

    metrics = server.compute_metrics(tasks)

    assert metrics["synthetic_tool_use/transfer_rate"] == 0.5
    assert metrics["synthetic_tool_use/transfer_ground_truth_mismatch_rate"] == 0.5
    assert metrics["synthetic_tool_use/transfer_mismatch_judge_skip_rate"] == 0.5


def test_get_key_metrics_returns_headline_metric_dict() -> None:
    server = make_server()
    metrics = {
        "mean/reward": 0.75,
        "reward/std": 0.25,
        "synthetic_tool_use/transfer_rate": 0.1,
        "synthetic_tool_use/transfer_ground_truth_mismatch_rate": 0.05,
        "synthetic_tool_use/transfer_mismatch_judge_skip_rate": 0.05,
        "synthetic_tool_use/tool_schema_failure_rate": 0.2,
        "synthetic_tool_use/user_failure_rate": 0.3,
        "synthetic_tool_use/tool_failure_rate": 0.4,
        "synthetic_tool_use/other_metric": 99,
    }

    assert server.get_key_metrics(metrics) == {
        "mean/reward": 0.75,
        "synthetic_tool_use/transfer_rate": 0.1,
        "synthetic_tool_use/transfer_ground_truth_mismatch_rate": 0.05,
        "synthetic_tool_use/transfer_mismatch_judge_skip_rate": 0.05,
        "synthetic_tool_use/tool_schema_failure_rate": 0.2,
        "synthetic_tool_use/user_failure_rate": 0.3,
        "synthetic_tool_use/tool_failure_rate": 0.4,
    }
