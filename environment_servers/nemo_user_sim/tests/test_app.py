# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
import usersim.engine.external as usersim_external
from aiohttp import ClientConnectionError, ClientResponseError
from omegaconf import OmegaConf
from pydantic import ConfigDict

from environment_servers.nemo_user_sim.app import (
    UserSimEnvironmentServer,
    UserSimEnvironmentServerConfig,
    _AgentSession,
    _apply_activation_parameters,
    _assistant_response_validation_error,
    _ConversationBridge,
    _GymEmbeddingFacade,
    _GymModelFacade,
    _is_retryable_dependency_error,
    _is_scorable_assistant_activation_error,
    _ParticipantActivationError,
    _to_responses_input_items,
)
from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.rollout_collection import _episode_record
from nemo_gym.server_utils import SESSION_ID_KEY, BaseServerConfig, ServerClient
from resources_servers.nemo_user_sim.app import UserSimResourcesServer, UserSimResourcesServerConfig
from resources_servers.nemo_user_sim.episode_contracts import (
    UserSimEpisodeRequest,
    UserSimEpisodeResult,
    UserSimSimulationResult,
    UserSimTaskInput,
    UserSimVerification,
)


def _server() -> UserSimEnvironmentServer:
    return UserSimEnvironmentServer(
        config=UserSimEnvironmentServerConfig(
            name="nemo_user_sim_environment",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            cleanup_timeout_seconds=10,
            user_agent=AgentServerRef(type="responses_api_agents", name="nemo_user_sim_user"),
            assistant_agent=AgentServerRef(type="responses_api_agents", name="nemo_user_sim_assistant"),
            judge_model=ModelServerRef(type="responses_api_models", name="support_model"),
            summary_model=ModelServerRef(type="responses_api_models", name="support_model"),
            tool_simulation_model=ModelServerRef(type="responses_api_models", name="support_model"),
            embedding_model=ModelServerRef(type="responses_api_models", name="embedding_model"),
            resources_server=ResourcesServerRef(type="resources_servers", name="nemo_user_sim_resources"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def test_aliases_route_participants_and_support_roles() -> None:
    config = _server().config

    assert config.target_for_alias("user_model").name == "nemo_user_sim_user"
    assert config.target_for_alias("assistant_model").name == "nemo_user_sim_assistant"
    assert config.target_for_alias("judge_model").name == "support_model"
    assert config.target_for_alias("summary_model").name == "support_model"
    assert config.target_for_alias("api_response_model").name == "support_model"
    assert config.target_for_alias("embedding_model").name == "embedding_model"


def test_episode_result_projects_verification_into_gym_scoring_contract() -> None:
    usersim_result = UserSimSimulationResult(
        conversation_messages=[
            {"role": "user", "content": "Help me."},
            {"role": "assistant", "content": "Here is help."},
        ],
        conversation_status=True,
        simulation_outcome={"status": "completed"},
    )
    verification = UserSimVerification(
        reward=0.75,
        mask_sample=False,
        failure_kind=None,
        failure_reason=None,
        reward_components={"assistant_quality": 0.75},
        scenario_completed=True,
        verifier_data={"assistant_eval": {"axes": {}}},
        usersim_result=usersim_result,
    )

    result = UserSimEpisodeResult.from_verification(
        verification=verification,
        usersim_result=usersim_result,
        invocations=[],
    )
    record = _episode_record(
        {
            "episode_id": {"rollout_id": "0-0", "attempt": 0},
            "task_id": {"taskset": "nemo_user_sim:example", "task_id": "0"},
            "result": result.model_dump(mode="json"),
        }
    )

    assert record["reward"] == 0.75
    assert record["mask_sample"] is False
    assert record["failure_kind"] is None
    assert record["failure_reason"] is None
    assert record["reward_components"] == {"assistant_quality": 0.75}
    assert record["verification"]["verifier_data"]["assistant_eval"] == {"axes": {}}
    assert record["verification"]["usersim_result"] is None
    assert record["usersim_result"]["conversation_status"] is True


def test_assistant_tool_calls_and_results_convert_to_responses_items() -> None:
    assert _to_responses_input_items(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        }
    ) == [
        {
            "type": "function_call",
            "call_id": "call-weather",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        }
    ]
    assert _to_responses_input_items(
        SimpleNamespace(role="tool", tool_call_id="call-weather", content='{"temperature":72}')
    ) == [
        {
            "type": "function_call_output",
            "call_id": "call-weather",
            "output": '{"temperature":72}',
        }
    ]


def test_activation_parameters_preserve_usersim_tools_and_sampling_controls() -> None:
    values = {"input": []}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    _apply_activation_parameters(
        values,
        {"temperature": 0.2, "max_tokens": 128, "tools": tools},
        assistant_tools=tools,
    )

    assert values["temperature"] == 0.2
    assert values["max_output_tokens"] == 128
    assert values["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {"type": "object", "properties": {}},
            "strict": None,
        }
    ]


def test_activation_parameters_clear_tools_and_leave_json_schema_strict_absent() -> None:
    values = {"input": [], "tools": [{"type": "function", "name": "row_tool"}]}

    _apply_activation_parameters(values, {"tools": []}, assistant_tools=[])
    _apply_activation_parameters(
        values,
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": {"type": "object"}},
            }
        },
        assistant_tools=None,
    )

    assert values["tools"] == []
    assert values["text"]["format"] == {
        "type": "json_schema",
        "name": "result",
        "schema": {"type": "object"},
    }


@pytest.mark.asyncio
async def test_environment_drives_probe_runtime_at_activation_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class ActivationRequest:
        activation_id: str
        role: str
        model_alias: str
        messages: tuple[dict, ...]
        parameters: dict
        tools: tuple[dict, ...] = ()

    @dataclass
    class ActivationResult:
        activation_id: str
        response: dict

    @dataclass
    class EpisodeLifecycleComplete:
        result: dict

    @dataclass
    class HostRoleModel:
        model_name: str

    class EpisodeContractError(Exception):
        pass

    activations = [
        ActivationRequest(
            activation_id="user-0",
            role="user",
            model_alias="user_model",
            messages=({"role": "system", "content": "Act as the user."},),
            parameters={},
        ),
        ActivationRequest(
            activation_id="assistant-0",
            role="assistant",
            model_alias="assistant_model",
            messages=({"role": "user", "content": "Check the weather."},),
            parameters={},
            tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ),
        ),
    ]

    class FakeRuntime:
        models: dict[str, object]
        instance: "FakeRuntime"

        def __init__(self, models: dict[str, object]) -> None:
            self.models = models
            self.index = 0
            self.results: list[ActivationResult] = []
            self.closed = False
            FakeRuntime.instance = self

        @classmethod
        def from_resolved_row(cls, _row: dict, *, models: dict[str, object]) -> "FakeRuntime":
            return cls(models)

        async def advance(self, result: ActivationResult | None = None):
            if result is not None:
                self.results.append(result)
                if result.activation_id == "assistant-0":
                    bridge.invocations.append("tool_simulation")
                self.index += 1
            if self.index < len(activations):
                return activations[self.index]
            return EpisodeLifecycleComplete(
                result={
                    "trajectory_id": "trajectory-0",
                    "conversation_messages": [
                        {"role": "user", "content": "Check the weather."},
                        {"role": "assistant", "content": "It is sunny."},
                    ],
                    "conversation_status": True,
                    "simulation_outcome": {"status": "completed"},
                }
            )

        async def evidence(self) -> dict:
            return {"activation_count": len(self.results)}

        async def close(self) -> None:
            self.closed = True

    external = ModuleType("usersim.engine.external")
    external.ActivationRequest = ActivationRequest
    external.ActivationResult = ActivationResult
    external.EpisodeContractError = EpisodeContractError
    external.EpisodeLifecycleComplete = EpisodeLifecycleComplete
    external.HostRoleModel = HostRoleModel
    external.ProbeEpisodeRuntime = FakeRuntime
    monkeypatch.setitem(sys.modules, "usersim.engine.external", external)
    monkeypatch.setattr("environment_servers.nemo_user_sim.app._configured_model_name", lambda _server, alias: alias)
    server, _ = _lifecycle_server()

    async def invoke_activation(activation: ActivationRequest) -> ActivationResult:
        bridge.invocations.append(activation.role)
        if activation.role == "user":
            return ActivationResult(
                activation_id="user-0",
                response={"role": "assistant", "content": "Check the weather."},
            )
        return ActivationResult(
            activation_id="assistant-0",
            response={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-weather",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
        )

    bridge = SimpleNamespace(
        environment_server=server,
        invocations=[],
        invoke_activation=AsyncMock(side_effect=invoke_activation),
        record_state=MagicMock(),
    )
    result = await server._run_usersim(bridge, {"trajectory_id": "trajectory-0"})

    assert result["conversation_status"] is True
    assert [call.args[0].role for call in bridge.invoke_activation.await_args_list] == ["user", "assistant"]
    assert isinstance(FakeRuntime.instance.models["api_response_model"], _GymModelFacade)
    assert isinstance(FakeRuntime.instance.models["embedding_model"], _GymEmbeddingFacade)
    assert len(FakeRuntime.instance.results) == 2
    assert bridge.invocations == ["user", "assistant", "tool_simulation"]
    bridge.record_state.assert_called_once_with(1, {"activation_count": 2})
    assert FakeRuntime.instance.closed is True


@pytest.mark.asyncio
async def test_real_probe_runtime_routes_tools_disabled_response_to_policy_failure() -> None:
    assert usersim_external.ProbeEpisodeRuntime is not None
    examples_path = Path(__file__).parents[3] / "resources_servers/nemo_user_sim/data/example.jsonl"
    example = orjson.loads(next(line for line in examples_path.read_bytes().splitlines() if line))
    server, client = _lifecycle_server()
    client.responses = [
        _Response(
            _responses_body(
                [
                    {
                        "type": "message",
                        "id": "user-message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Please help me plan a respectful event.",
                                "annotations": [],
                            }
                        ],
                    }
                ]
            )
        ),
        _Response(
            _responses_body(
                [
                    {
                        "type": "message",
                        "id": "judge-message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "<rating>success</rating>",
                                "annotations": [],
                            }
                        ],
                    }
                ]
            )
        ),
        _Response(
            _responses_body(
                [
                    {
                        "type": "function_call",
                        "id": "unoffered-function",
                        "call_id": "call-unoffered",
                        "name": "unoffered_tool",
                        "arguments": "{}",
                        "status": "completed",
                    }
                ]
            )
        ),
    ]
    request = UserSimEpisodeRequest(
        episode_id=EpisodeId(rollout_id="real-runtime", attempt=0),
        task=MaterializedTask(
            task_id=TaskId(taskset="nemo_user_sim:example", task_id=example["task_id"]),
            task_input=UserSimTaskInput.model_validate(example),
        ),
    )
    bridge = _ConversationBridge(
        server,
        request,
        request.task.task_input,
        {},
        {
            "user_model": _AgentSession(
                alias="user_model",
                target=server.config.user_agent,
                session_id="user-session",
                cookies={},
            ),
            "assistant_model": _AgentSession(
                alias="assistant_model",
                target=server.config.assistant_agent,
                session_id="assistant-session",
                cookies={},
            ),
        },
    )
    with pytest.raises(_ParticipantActivationError, match="returned tool calls with tools disabled") as error:
        await server._run_usersim(bridge, example["resolved_row"])

    assert error.value.alias == "assistant_model"
    assert isinstance(error.value.error, ValueError)
    assert [invocation.role for invocation in bridge.invocations] == ["user", "judge", "assistant"]


def test_assistant_response_validation_rejects_unoffered_tool_name() -> None:
    activation = SimpleNamespace(
        activation_id="assistant-0",
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "offered_tool",
                    "description": "The only offered tool.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
    )
    result = SimpleNamespace(
        response={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-unoffered",
                    "type": "function",
                    "function": {"name": "unoffered_tool", "arguments": "{}"},
                }
            ],
        }
    )

    error = _assistant_response_validation_error(activation, result)

    assert error is not None
    assert str(error) == "Assistant called unoffered tool 'unoffered_tool'; available tools: ['offered_tool']"


@pytest.mark.asyncio
async def test_runtime_contract_error_after_valid_assistant_response_is_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = usersim_external.ActivationRequest(
        activation_id="assistant-0",
        role="assistant",
        model_alias="assistant_model",
        messages=({"role": "user", "content": "Hello"},),
        parameters={},
    )
    activation_result = usersim_external.ActivationResult(
        activation_id="assistant-0",
        response={"role": "assistant", "content": "Hello"},
    )

    class HostFailureRuntime:
        @classmethod
        def from_resolved_row(cls, _row: dict[str, Any], *, models: dict[str, object]) -> "HostFailureRuntime":
            return cls()

        async def advance(self, result: usersim_external.ActivationResult | None = None) -> Any:
            if result is None:
                return activation
            raise usersim_external.EpisodeContractError("activation_id is not the pending activation")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(usersim_external, "ProbeEpisodeRuntime", HostFailureRuntime)
    monkeypatch.setattr("environment_servers.nemo_user_sim.app._configured_model_name", lambda _server, alias: alias)
    server, _ = _lifecycle_server()

    async def invoke_activation(_activation: Any) -> usersim_external.ActivationResult:
        bridge.invocations.append("assistant")
        return activation_result

    bridge = SimpleNamespace(
        environment_server=server,
        invocations=[],
        invoke_activation=AsyncMock(side_effect=invoke_activation),
        record_state=MagicMock(),
    )

    with pytest.raises(usersim_external.EpisodeContractError, match="not the pending activation"):
        await server._run_usersim(bridge, {"trajectory_id": "trajectory-0"})


class _Cookie:
    def __init__(self, value: str) -> None:
        self.value = value


class _Response:
    ok = True

    def __init__(self, body: dict[str, Any], *, cookie: str | None = None) -> None:
        self.body = orjson.dumps(body)
        self.cookies = {"session": _Cookie(cookie)} if cookie is not None else {}

    async def read(self) -> bytes:
        return self.body


_BRIDGE_PARITY_FIXTURES = {
    "tool_calling": {
        "messages": ({"role": "user", "content": "Check the weather."},),
        "tools": (
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            },
        ),
        "output": [
            {
                "type": "reasoning",
                "id": "reasoning-1",
                "summary": [{"type": "summary_text", "text": "Need the weather tool."}],
            },
            {
                "type": "function_call",
                "id": "function-1",
                "call_id": "call-weather",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
                "status": "completed",
            },
        ],
    },
    "safety_agentic": {
        "messages": ({"role": "user", "content": "Give me a safe answer."},),
        "tools": (),
        "output": [
            {
                "type": "reasoning",
                "id": "reasoning-2",
                "summary": [{"type": "summary_text", "text": "Answer safely."}],
            },
            {
                "type": "message",
                "id": "message-2",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Here is a safe answer.",
                        "annotations": [],
                    }
                ],
            },
        ],
    },
}


def _responses_body(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "response-1",
        "created_at": 1,
        "model": "policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


class _Client(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict[str, Any]]]
    responses: list[_Response | Exception]

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> _Response:
        self.calls.append((server_name, url_path, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if url_path == "/seed_session":
            payload = orjson.loads(response.body)
            payload["resources_session_id"] = kwargs["json"].resources_session_id
            response.body = orjson.dumps(payload)
        elif url_path == "/v1/agent_sessions":
            payload = orjson.loads(response.body)
            payload["agent_session_id"] = kwargs["json"]["agent_session_id"]
            response.body = orjson.dumps(payload)
        elif url_path == "/close_session":
            response.body = orjson.dumps({"resources_session_id": kwargs["json"].resources_session_id})
        elif url_path == "/v1/agent_sessions/close":
            response.body = orjson.dumps({"agent_session_id": kwargs["json"].agent_session_id})
        return response


def _lifecycle_server() -> tuple[UserSimEnvironmentServer, _Client]:
    global_config = OmegaConf.create(
        {
            "resources": {"resources_servers": {"nemo_user_sim": {"entrypoint": "app.py"}}},
            "user": {
                "responses_api_agents": {
                    "simple_agent": {
                        "entrypoint": "app.py",
                        "model_server": {"type": "responses_api_models", "name": "policy"},
                    }
                }
            },
            "assistant": {
                "responses_api_agents": {
                    "simple_agent": {
                        "entrypoint": "app.py",
                        "model_server": {"type": "responses_api_models", "name": "policy"},
                    }
                }
            },
            "policy": {"responses_api_models": {"vllm_model": {"entrypoint": "app.py", "model": "policy"}}},
            "support": {"responses_api_models": {"vllm_model": {"entrypoint": "app.py", "model": "support"}}},
            "embedding": {
                "responses_api_models": {
                    "vllm_model": {
                        "entrypoint": "app.py",
                        "base_url": "https://embedding.example/v1",
                        "api_key": "embedding-key",
                        "model": "embedding",
                    }
                }
            },
        }
    )
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        calls=[],
        responses=[],
    )
    server = UserSimEnvironmentServer(
        config=UserSimEnvironmentServerConfig(
            name="nemo-user-sim-environment",
            host="environment",
            port=8005,
            entrypoint="app.py",
            cleanup_timeout_seconds=10,
            user_agent=AgentServerRef(type="responses_api_agents", name="user"),
            assistant_agent=AgentServerRef(type="responses_api_agents", name="assistant"),
            judge_model=ModelServerRef(type="responses_api_models", name="support"),
            summary_model=ModelServerRef(type="responses_api_models", name="support"),
            tool_simulation_model=ModelServerRef(type="responses_api_models", name="support"),
            embedding_model=ModelServerRef(type="responses_api_models", name="embedding"),
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        ),
        server_client=client,
    )
    return server, client


def _resolved_row() -> dict[str, Any]:
    return {
        "persona": {"first_name": "Example"},
        "probe_type": "general_open_ended",
        "probe_family": "general_open_ended",
        "probe_variant": "default",
        "conversation_language": "English",
        "trajectory_id": "trajectory",
        "usersim_config": {"max_turns": 1},
        "usersim_provenance": {
            "code_sha": "a5f676bf6dc5a73914c8a0860f97c10dd2c214ee",  # pragma: allowlist secret
            "nemotron_personas_version": "synthetic",
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_type", ["tool_calling", "safety_agentic"])
async def test_committed_activation_fixtures_round_trip_through_real_bridge(
    probe_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class ActivationUsage:
        input_tokens: int
        output_tokens: int

    @dataclass
    class ActivationResult:
        activation_id: str
        response: dict[str, Any]
        usage: ActivationUsage | None = None

    external = ModuleType("usersim.engine.external")
    external.ActivationResult = ActivationResult
    external.ActivationUsage = ActivationUsage
    monkeypatch.setitem(sys.modules, "usersim.engine.external", external)
    server, client = _lifecycle_server()
    fixture = _BRIDGE_PARITY_FIXTURES[probe_type]
    client.responses = [_Response(_responses_body(fixture["output"]), cookie="assistant-cookie")]
    request = _request()
    task = UserSimTaskInput.model_validate(
        {
            "resolved_row": request.task.task_input.resolved_row,
            "role_request_params": {
                "assistant": {
                    "input": [],
                    "tools": [
                        {
                            "type": "function",
                            "name": "stale_row_tool",
                            "description": "Must be replaced by activation tools.",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                }
            },
        }
    )
    bridge = _ConversationBridge(
        server,
        request,
        task,
        {},
        {
            "assistant_model": _AgentSession(
                alias="assistant_model",
                target=server.config.assistant_agent,
                session_id="assistant-session",
                cookies={"session": "assistant-cookie"},
            )
        },
    )
    activation = SimpleNamespace(
        activation_id=f"{probe_type}-activation",
        model_alias="assistant_model",
        messages=fixture["messages"],
        parameters={},
        tools=fixture["tools"],
    )

    result = await bridge.invoke_activation(activation)

    assert result.response["reasoning_content"]
    assert len(bridge.invocations) == 1
    assert bridge.invocations[0].role == "assistant"
    sent_tools = client.calls[0][2]["json"].tools
    if probe_type == "tool_calling":
        assert [tool["name"] for tool in sent_tools] == ["get_weather"]
        assert result.response["tool_calls"][0]["function"]["name"] == "get_weather"
    else:
        assert sent_tools == []
        assert result.response["content"] == "Here is a safe answer."


@pytest.mark.asyncio
async def test_embedding_facade_routes_to_configured_model_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    server, _ = _lifecycle_server()
    response = _Response(
        {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }
    )
    request = AsyncMock(return_value=response)
    monkeypatch.setattr("environment_servers.nemo_user_sim.app.http_request", request)
    facade = _GymEmbeddingFacade(SimpleNamespace(environment_server=server))

    embeddings = await facade.agenerate_text_embeddings(["first", "second"])

    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    request.assert_awaited_once_with(
        method="POST",
        url="https://embedding.example/v1/embeddings",
        json={"input": ["first", "second"], "model": "embedding"},
        headers={"Authorization": "Bearer embedding-key"},
    )


def _request() -> UserSimEpisodeRequest:
    return UserSimEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=0),
        task=MaterializedTask(
            task_id=TaskId(taskset="nemo_user_sim:example", task_id="task"),
            task_input=UserSimTaskInput(
                task_id="task",
                resolved_row=_resolved_row(),
                role_request_params={},
            ),
        ),
    )


@pytest.mark.asyncio
async def test_lost_resources_seed_response_still_attempts_close() -> None:
    server, client = _lifecycle_server()
    client.responses = [ClientConnectionError("lost seed response"), _Response({})]

    response = await server.run_request(_request())

    assert response.failure is not None
    assert response.failure.stage == "seed"
    assert response.failure.terminal is False
    assert [path for _, path, _ in client.calls] == ["/seed_session", "/close_session"]
    assert client.calls[-1][2]["cookies"] == {}


@pytest.mark.asyncio
async def test_lost_agent_seed_response_still_attempts_agent_and_resources_close() -> None:
    server, client = _lifecycle_server()
    client.responses = [
        _Response(
            {
                "resources_session_id": "placeholder",
                "sandbox_access": None,
            },
            cookie="resources-cookie",
        ),
        ClientConnectionError("lost agent seed response"),
        _Response({}),
        _Response({}),
    ]

    response = await server.run_request(_request())

    assert response.failure is not None
    assert response.failure.stage == "agent"
    assert response.failure.terminal is False
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/v1/agent_sessions/close",
        "/close_session",
    ]
    assert client.calls[2][2]["cookies"] == {}


def _seeded_participant_responses() -> list[_Response]:
    return [
        _Response(
            {
                "resources_session_id": "placeholder",
                "sandbox_access": None,
            },
            cookie="resources-cookie",
        ),
        _Response({}, cookie="user-cookie"),
        _Response({}, cookie="assistant-cookie"),
    ]


@pytest.mark.asyncio
async def test_non_retryable_assistant_response_failure_is_verified_as_reward_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, client = _lifecycle_server()
    client.responses = [
        *_seeded_participant_responses(),
        _Response({}),
        _Response({}),
        _Response(
            {
                "reward": 0.0,
                "mask_sample": False,
                "reward_components": {"assistant_quality": 0.0},
                "scenario_completed": False,
            }
        ),
        _Response({}),
    ]

    async def fail_assistant(*_args: Any) -> dict[str, Any]:
        error = ValueError("invalid assistant response")
        raise _ParticipantActivationError("assistant_model", error) from error

    monkeypatch.setattr(server, "_run_usersim", fail_assistant)

    response = await server.run_request(_request())

    assert response.failure is None
    assert response.result is not None
    assert response.result.reward == 0.0
    assert response.result.mask_sample is False
    assert response.result.usersim_result.simulation_outcome["failure_attribution"] == "assistant_model"
    assert "/verify" in [path for _, path, _ in client.calls]

    model = ModelServerRef(type="responses_api_models", name="support")
    resources_server = UserSimResourcesServer(
        config=UserSimResourcesServerConfig(
            name="resources",
            host="resources",
            port=8006,
            entrypoint="app.py",
            probe_scorer_model=model,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    resources_server.session_id_to_seed = {}
    resources_server.closed_resources_session_ids = {}
    resources_request = SimpleNamespace(session={SESSION_ID_KEY: "real-resources-session"})
    seed_body = next(call[2]["json"] for call in client.calls if call[1] == "/seed_session")
    verify_body = next(call[2]["json"] for call in client.calls if call[1] == "/verify")

    await resources_server.seed_session(resources_request, seed_body)
    verification = await resources_server.verify(resources_request, verify_body)

    assert verification.reward == 0.0
    assert verification.mask_sample is False
    assert verification.usersim_result is not None
    assert verification.usersim_result.trajectory_id == _resolved_row()["trajectory_id"]


@pytest.mark.asyncio
async def test_retryable_assistant_transport_failure_skips_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, client = _lifecycle_server()
    client.responses = [
        *_seeded_participant_responses(),
        _Response({}),
        _Response({}),
        _Response({}),
    ]

    async def fail_assistant(*_args: Any) -> dict[str, Any]:
        error = ClientConnectionError("assistant disconnected")
        raise _ParticipantActivationError("assistant_model", error) from error

    monkeypatch.setattr(server, "_run_usersim", fail_assistant)

    response = await server.run_request(_request())

    assert response.failure is not None
    assert response.failure.stage == "agent"
    assert response.failure.terminal is False
    assert "/verify" not in [path for _, path, _ in client.calls]


@pytest.mark.parametrize("status", [404, 408, 409, 425, 429, 500, 503])
def test_wrapped_participant_http_infrastructure_failures_are_retryable(status: int) -> None:
    original = ClientResponseError(MagicMock(), (), status=status)

    assert _is_retryable_dependency_error(_ParticipantActivationError("assistant_model", original)) is True
    assert _is_retryable_dependency_error(_ParticipantActivationError("user_model", original)) is True


@pytest.mark.parametrize("status", [404, 409])
def test_non_participant_http_contract_failures_are_not_retryable(status: int) -> None:
    assert _is_retryable_dependency_error(ClientResponseError(MagicMock(), (), status=status)) is False


def test_wrapped_assistant_invalid_response_is_not_retryable() -> None:
    invalid = _ParticipantActivationError("assistant_model", ValueError("invalid response"))
    bad_request = _ParticipantActivationError(
        "assistant_model",
        ClientResponseError(MagicMock(), (), status=400),
    )

    assert _is_retryable_dependency_error(invalid) is False
    assert _is_scorable_assistant_activation_error(invalid) is True
    assert _is_scorable_assistant_activation_error(bad_request) is True
    for status in (404, 409):
        infrastructure_error = _ParticipantActivationError(
            "assistant_model",
            ClientResponseError(MagicMock(), (), status=status),
        )
        assert _is_retryable_dependency_error(infrastructure_error) is True
        assert _is_scorable_assistant_activation_error(infrastructure_error) is False
    assert (
        _is_scorable_assistant_activation_error(
            _ParticipantActivationError("assistant_model", RuntimeError("bridge bug"))
        )
        is False
    )
    assert (
        _is_retryable_dependency_error(_ParticipantActivationError("assistant_model", TimeoutError("timed out")))
        is True
    )
