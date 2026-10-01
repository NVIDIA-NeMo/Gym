# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from environment_servers.usersim.app import (
    UserSimEnvironmentServer,
    UserSimEnvironmentServerConfig,
    _apply_activation_parameters,
    _GymModelFacade,
    _to_responses_input_items,
)
from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from resources_servers.usersim.episode_contracts import UserSimSeedResponse


def _server() -> UserSimEnvironmentServer:
    return UserSimEnvironmentServer(
        config=UserSimEnvironmentServerConfig(
            name="usersim_environment",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            cleanup_timeout_seconds=10,
            user_agent=AgentServerRef(type="responses_api_agents", name="usersim_user"),
            assistant_agent=AgentServerRef(type="responses_api_agents", name="usersim_assistant"),
            judge_model=ModelServerRef(type="responses_api_models", name="support_model"),
            summary_model=ModelServerRef(type="responses_api_models", name="support_model"),
            tool_simulation_model=ModelServerRef(type="responses_api_models", name="support_model"),
            resources_server=ResourcesServerRef(type="resources_servers", name="usersim_resources"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def test_aliases_route_participants_and_support_roles() -> None:
    config = _server().config

    assert config.target_for_alias("user_model").name == "usersim_user"
    assert config.target_for_alias("assistant_model").name == "usersim_assistant"
    assert config.target_for_alias("judge_model").name == "support_model"
    assert config.target_for_alias("summary_model").name == "support_model"
    assert config.target_for_alias("api_response_model").name == "support_model"


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


def test_activation_parameters_preserve_native_tools_and_sampling_controls() -> None:
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
    external.EpisodeLifecycleComplete = EpisodeLifecycleComplete
    external.HostRoleModel = HostRoleModel
    external.ProbeEpisodeRuntime = FakeRuntime
    monkeypatch.setitem(sys.modules, "usersim.engine.external", external)
    monkeypatch.setattr("environment_servers.usersim.app._configured_model_name", lambda _server, alias: alias)
    server = _server()
    bridge = SimpleNamespace(
        environment_server=server,
        invoke_activation=AsyncMock(
            side_effect=[
                ActivationResult(
                    activation_id="user-0",
                    response={"role": "assistant", "content": "Check the weather."},
                ),
                ActivationResult(
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
                ),
            ]
        ),
        record_state=MagicMock(),
    )
    seed = UserSimSeedResponse(
        resources_session_id="resources-session-0",
        sandbox_access=None,
        resolved_row={"trajectory_id": "trajectory-0"},
    )

    result = await server._run_usersim(bridge, seed)

    assert result["conversation_status"] is True
    assert [call.args[0].role for call in bridge.invoke_activation.await_args_list] == ["user", "assistant"]
    assert isinstance(FakeRuntime.instance.models["api_response_model"], _GymModelFacade)
    assert len(FakeRuntime.instance.results) == 2
    assert bridge.record_state.call_count == 2
    assert FakeRuntime.instance.closed is True
