# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity between a UserSim-prepared task and Gym's external execution path."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from pydantic import ConfigDict
from usersim.engine.core.episode_runtime import ProbeEpisodeRuntime
from usersim.engine.external import (
    AssistantLoopPolicy as NativeAssistantLoopPolicy,
)
from usersim.engine.external import (
    ConversationRuntime,
    HostRoleModel,
    ProbeToolSession,
    materialize_episode_inputs,
)
from usersim.engine.generator import ConversationSimulatorGenerator

from environment_servers.usersim.app import (
    UserSimEnvironmentServer,
    UserSimEnvironmentServerConfig,
    _apply_activation_parameters,
    _to_responses_input_items,
)
from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.base_responses_api_agent import AgentToolLoopPolicy
from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, BaseServerConfig, ServerClient
from nemo_gym.tool_access import ContextualToolCallRequest
from resources_servers.usersim.app import UserSimResourcesServer, UserSimResourcesServerConfig
from resources_servers.usersim.episode_contracts import (
    ActivationRequest,
    ActivationResult,
    AssistantTurnRequest,
    CompletedAssistantTurn,
    CompletedTurnEvidence,
)
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


USERSIM_REVISION = "b3381ae021baac2a6fb314b5f08a55845243017d"  # pragma: allowlist secret


def _message_value(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _arguments_for_schema(schema: dict[str, Any]) -> Any:
    if enum := schema.get("enum"):
        return enum[0]
    if schema.get("type") == "object":
        required = set(schema.get("required", []))
        return {
            name: _arguments_for_schema(value)
            for name, value in schema.get("properties", {}).items()
            if name in required
        }
    return {
        "array": [],
        "boolean": True,
        "integer": 1,
        "number": 1,
        "string": "example",
    }.get(schema.get("type"), "example")


class _ReplayableModel:
    model_name = "nvidia/nemotron-3-super-120b-a12b"

    def __init__(self, role: str) -> None:
        self.role = role

    async def acompletion(self, messages: list[Any], **kwargs: Any) -> SimpleNamespace:
        tool_calls = None
        has_tool_result = any(str(_message_value(message, "role")) == "tool" for message in messages)
        if self.role == "assistant" and kwargs.get("tools") and not has_tool_result:
            function = kwargs["tools"][0]["function"]
            tool_calls = [
                {
                    "id": "call-parity",
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": json.dumps(_arguments_for_schema(function.get("parameters", {})), sort_keys=True),
                    },
                }
            ]
            content = ""
        else:
            content = {
                "api": '{"ok":true}',
                "assistant": "Scripted assistant response.",
                "judge": "<explanation>valid scripted turn</explanation><rating>success</rating>",
                "summary": "yes",
                "user": "Could you explain that a little more?",
            }[self.role]
        return SimpleNamespace(
            message=SimpleNamespace(
                content=content,
                reasoning_content=f"{self.role} reasoning",
                tool_calls=tool_calls,
            ),
            usage=None,
        )


def _chat_response(response: SimpleNamespace) -> dict[str, Any]:
    result = {
        "role": "assistant",
        "content": response.message.content or "",
        "reasoning_content": response.message.reasoning_content,
        "tool_calls": deepcopy(response.message.tool_calls),
    }
    return {name: value for name, value in result.items() if value is not None}


class _HTTPResponse:
    ok = True
    status = 200
    cookies: dict[str, Any] = {}

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.content = self

    async def read(self) -> bytes:
        return self.body


class _ParityClient(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    resources: UserSimResourcesServer
    resources_request: Any
    model_calls: int = 0

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> _HTTPResponse:
        if server_name == "resources":
            value = await self.resources.invoke_probe_tool(
                self.resources_request,
                url_path.removeprefix("/"),
                ContextualToolCallRequest.model_validate(kwargs["json"]),
            )
            return _HTTPResponse(value.model_dump_json().encode())

        self.model_calls += 1
        body = kwargs["json"]
        has_tool_result = any(getattr(item, "type", None) == "function_call_output" for item in body.input)
        if body.tools and not has_tool_result:
            tool = body.tools[0]
            tool_name = _message_value(tool, "name")
            tool_parameters = _message_value(tool, "parameters", {})
            output = [
                {
                    "id": "reasoning-parity",
                    "summary": [{"text": "assistant reasoning", "type": "summary_text"}],
                    "type": "reasoning",
                },
                {
                    "id": "function-parity",
                    "call_id": "call-parity",
                    "name": tool_name,
                    "arguments": json.dumps(_arguments_for_schema(tool_parameters), sort_keys=True),
                    "type": "function_call",
                    "status": "completed",
                },
            ]
        else:
            output = [
                {
                    "id": "reasoning-parity",
                    "summary": [{"text": "assistant reasoning", "type": "summary_text"}],
                    "type": "reasoning",
                },
                {
                    "id": "message-parity",
                    "content": [{"annotations": [], "text": "Scripted assistant response.", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ]
        response = {
            "id": f"response-{self.model_calls}",
            "created_at": 1,
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "object": "response",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        return _HTTPResponse(json.dumps(response).encode())

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


class _ExactCapClient(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_requests: list[Any]
    resource_requests: list[ContextualToolCallRequest]
    round_count: int = 0

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> _HTTPResponse:
        if server_name == "model":
            body = kwargs["json"]
            self.model_requests.append(body)
            call_index = len(self.model_requests)
            if body.tools:
                output = [
                    {
                        "id": f"reasoning-{call_index}",
                        "summary": [{"text": f"reasoning {call_index}", "type": "summary_text"}],
                        "type": "reasoning",
                    },
                    {
                        "id": f"function-{call_index}",
                        "call_id": f"call-{call_index}",
                        "name": "kb_search",
                        "arguments": f'{{"query":"round {call_index}"}}',
                        "type": "function_call",
                        "status": "completed",
                    },
                ]
            else:
                output = [
                    {
                        "id": "final-message",
                        "content": [{"annotations": [], "text": "final", "type": "output_text"}],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                ]
            return _HTTPResponse(
                json.dumps(
                    {
                        "id": f"response-{call_index}",
                        "created_at": 1,
                        "model": "model",
                        "object": "response",
                        "output": output,
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                        "usage": {
                            "input_tokens": 10 + call_index,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 5 + call_index,
                            "output_tokens_details": {"reasoning_tokens": 1},
                            "total_tokens": 15 + 2 * call_index,
                        },
                    }
                ).encode()
            )

        assert (server_name, url_path) == ("resources", "/kb_search")
        self.round_count += 1
        contextual_request = ContextualToolCallRequest.model_validate(kwargs["json"])
        self.resource_requests.append(contextual_request)
        assistant_response = contextual_request.assistant_response
        raw_call = assistant_response["tool_calls"][0]
        return _HTTPResponse(
            json.dumps(
                {
                    "output": '{"results":[]}' if self.round_count == 1 else None,
                    "tool_call_context": {
                        "turn_id": "turn-1",
                        "round_id": contextual_request.round_id,
                        "raw_tool_call": raw_call,
                    },
                    "limit_reached": self.round_count == 2,
                }
            ).encode()
        )


class _GymBridge:
    def __init__(self, agent: SimpleAgent, assistant_tools: list[dict[str, Any]]) -> None:
        self.agent = agent
        self.assistant_tools = assistant_tools
        self.resources_cookies: dict[str, str] = {}
        self.models = {role: _ReplayableModel(role) for role in ("user", "judge", "summary")}

    async def invoke(self, activation: ActivationRequest) -> ActivationResult:
        response = await self.models[activation.role].acompletion(
            activation.messages,
            **activation.parameters,
        )
        return ActivationResult(
            activation_id=activation.activation_id,
            response=_chat_response(response),
        )

    async def invoke_assistant(self, activation: AssistantTurnRequest) -> CompletedAssistantTurn:
        values: dict[str, Any] = {
            "input": [item for message in activation.messages for item in _to_responses_input_items(message)]
        }
        _apply_activation_parameters(values, activation.parameters, assistant_tools=activation.tools)
        response, _, _, _ = await self.agent._create_episode(
            NeMoGymResponseCreateParamsNonStreaming.model_validate(values),
            model_url_path="/v1/responses",
            resources_server_cookies={},
            tool_loop_policy=AgentToolLoopPolicy.model_validate(activation.loop_policy.model_dump(mode="json")),
            tool_call_context=activation.tool_context.model_dump(mode="json"),
        )
        trace = response.model_extra.pop("_ng_completed_turn")
        evidence = (
            CompletedTurnEvidence.model_validate(trace["resources_context"])
            if trace["resources_context"] is not None
            else CompletedTurnEvidence.from_unexecuted_turn(activation.tool_context)
        )
        return CompletedAssistantTurn(
            turn_id=activation.turn_id,
            transcript=trace["transcript"],
            model_calls=trace["model_calls"],
            evidence=evidence,
        )


def _normalized_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(result)
    for name in ("conversation_messages", "conversation_metadata", "simulation_outcome", "simulation_traces"):
        if isinstance(normalized.get(name), str):
            normalized[name] = json.loads(normalized[name])
    normalized["simulation_outcome"].pop("wall_clock_s", None)
    normalized["simulation_outcome"].pop("wall_clock_s_by_alias", None)
    return normalized


async def test_semantic_exact_cap_model_sequence_matches_native_policy() -> None:
    native_policy = NativeAssistantLoopPolicy(
        mode="multi",
        max_model_calls=3,
        max_tool_calls=1,
        final_synthesis=True,
    )
    expected_tools = [
        native_policy.tools_enabled(model_calls=0, tool_calls=0, limit_reached=False),
        native_policy.tools_enabled(model_calls=1, tool_calls=1, limit_reached=False),
        native_policy.tools_enabled(model_calls=2, tool_calls=1, limit_reached=True),
    ]
    client = _ExactCapClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({}),
        model_requests=[],
        resource_requests=[],
    )
    agent = SimpleAgent(
        config=SimpleAgentConfig(
            host="assistant",
            port=8001,
            entrypoint="app.py",
            name="assistant",
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            model_server=ModelServerRef(type="responses_api_models", name="model"),
        ),
        server_client=client,
    )

    response, _, _, _ = await agent._create_episode(
        NeMoGymResponseCreateParamsNonStreaming(
            input="question",
            tools=[
                {
                    "type": "function",
                    "name": "kb_search",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": False,
                }
            ],
        ),
        model_url_path="/v1/responses",
        tool_loop_policy=AgentToolLoopPolicy.model_validate(native_policy.to_dict()),
        tool_call_context={"turn_id": "turn-1"},
    )

    assert [bool(request.tools) for request in client.model_requests] == expected_tools == [True, True, False]
    assert client.round_count == 2
    assert [request.arguments for request in client.resource_requests] == [
        {"query": "round 1"},
        {"query": "round 2"},
    ]
    trace = response.model_extra["_ng_completed_turn"]
    assert trace["resources_context"]["round_id"] == "round-000002"
    assert [call["usage"] for call in trace["model_calls"]] == [
        {"input_tokens": 11, "output_tokens": 6},
        {"input_tokens": 12, "output_tokens": 7},
        {"input_tokens": 13, "output_tokens": 8},
    ]
    assert [call["response"].get("reasoning_content") for call in trace["model_calls"][:2]] == [
        "reasoning 1",
        "reasoning 2",
    ]


@pytest.mark.parametrize("probe_type", ["tool_calling", "financial_services", "safety_agentic"])
async def test_prepared_task_matches_standalone_generator_through_gym_stack(monkeypatch, probe_type: str) -> None:
    monkeypatch.setattr("usersim.engine.core.episode_input.get_code_sha", lambda: USERSIM_REVISION)
    [resolved_row] = materialize_episode_inputs(
        locale="en_US",
        num_rows=1,
        probe_mix={probe_type: 1.0},
        random_seed=42,
    )
    direct_models = {
        "api_response_model": _ReplayableModel("api"),
        **{f"{role}_model": _ReplayableModel(role) for role in ("assistant", "judge", "summary", "user")},
    }
    direct_runtime = ProbeEpisodeRuntime.from_resolved_row(deepcopy(resolved_row), models=direct_models)
    provider = SimpleNamespace(
        model_registry=SimpleNamespace(get_model=lambda *, model_alias: direct_models[model_alias])
    )
    standalone = await ConversationSimulatorGenerator(direct_runtime.config, provider).agenerate(
        deepcopy(resolved_row)
    )

    resources = UserSimResourcesServer(
        config=UserSimResourcesServerConfig(
            host="resources",
            port=8000,
            entrypoint="app.py",
            name="resources",
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    monkeypatch.setattr(
        resources,
        "_create_tool_session",
        lambda row: ProbeToolSession.from_resolved_row(
            row,
            models={"api_response_model": _ReplayableModel("api")} if probe_type == "tool_calling" else {},
        ),
    )
    resources_request = SimpleNamespace(session={SESSION_ID_KEY: "parity-session"})
    seed = await resources.seed_session(
        resources_request,
        ResourcesSeedSessionRequest(
            resources_session_id="resources-session",
            episode_id=EpisodeId(rollout_id="parity", attempt=0),
            task_id=TaskId(taskset="usersim:example", task_id=probe_type),
            task_data={"resolved_row": resolved_row},
        ),
    )
    client = _ParityClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({}),
        resources=resources,
        resources_request=resources_request,
    )
    agent = SimpleAgent(
        config=SimpleAgentConfig(
            host="assistant",
            port=8001,
            entrypoint="app.py",
            name="assistant",
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            model_server=ModelServerRef(type="responses_api_models", name="model"),
        ),
        server_client=client,
    )
    environment = UserSimEnvironmentServer(
        config=UserSimEnvironmentServerConfig(
            host="environment",
            port=8002,
            entrypoint="app.py",
            name="environment",
            cleanup_timeout_seconds=10,
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            user_agent=AgentServerRef(type="responses_api_agents", name="user"),
            assistant_agent=AgentServerRef(type="responses_api_agents", name="assistant"),
            judge_model=ModelServerRef(type="responses_api_models", name="judge"),
            summary_model=ModelServerRef(type="responses_api_models", name="summary"),
            resources_tool_transports=["direct_http"],
        ),
        server_client=client,
    )
    host_models = {
        f"{role}_model": HostRoleModel(model_name=_ReplayableModel.model_name)
        for role in ("assistant", "judge", "summary", "user")
    }
    conversation = ConversationRuntime.from_resolved_row(deepcopy(resolved_row), models=host_models)
    gym_result = await environment._run_usersim(conversation, _GymBridge(agent, seed.assistant_tools))

    standalone_result = {name: standalone[name] for name in gym_result}
    assert _normalized_result(gym_result) == _normalized_result(standalone_result)
    assert resolved_row["trajectory_id"] == standalone["trajectory_id"]
    assert resolved_row["usersim_provenance"] == standalone["usersim_provenance"]
    traces = _normalized_result(gym_result)["simulation_traces"]
    assert traces
    assert all(isinstance(trace["turn_idx"], int) and isinstance(trace["call_idx"], int) for trace in traces)
    assert client.model_calls >= 2
