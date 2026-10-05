# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Black-box parity between Gym's environment driver and the pinned native runtime."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


pytest.importorskip("usersim.engine.external")

from usersim.engine.external import ActivationResult, EpisodeLifecycleComplete, HostRoleModel, ProbeEpisodeRuntime

from environment_servers.usersim.app import UserSimEnvironmentServer, UserSimEnvironmentServerConfig
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


def _example_argument(name: str, schema: dict[str, Any]) -> Any:
    if name == "date_of_birth":
        return "1984-01-01"
    if name == "full_name":
        return "Sarah Johnson"
    if name in {"city", "location"}:
        return "Austin, TX"
    return {
        "array": [],
        "boolean": False,
        "integer": 1,
        "number": 1.0,
        "object": {},
        "string": "test",
    }.get(schema.get("type"), "test")


async def _activation_result(activation: Any) -> ActivationResult:
    content = ""
    tool_calls = None
    if activation.role == "assistant":
        if activation.tools:
            function = activation.tools[0].get("function", activation.tools[0])
            properties = function.get("parameters", {}).get("properties", {})
            arguments = {
                name: _example_argument(name, schema)
                for name, schema in properties.items()
                if name in function.get("parameters", {}).get("required", [])
            }
            tool_calls = [
                {
                    "id": f"call-{activation.activation_id}",
                    "type": "function",
                    "function": {"name": function["name"], "arguments": json.dumps(arguments)},
                }
            ]
        else:
            content = "I can help with that request safely."
    elif activation.role == "user":
        content = "Thanks, that answers my question."
    elif activation.role == "judge":
        content = "<explanation>The response addressed the request.</explanation>\n<rating>success</rating>"
    elif activation.role == "summary":
        content = "yes"
    response = {"role": "assistant", "content": content}
    if tool_calls is not None:
        response["tool_calls"] = tool_calls
    return ActivationResult(activation_id=activation.activation_id, response=response)


class _ToolSimulationFacade:
    model_name = "test-support-model"

    async def acompletion(self, _messages: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            message=SimpleNamespace(
                content='{"status":"ok","result":"success"}',
                reasoning_content=None,
                tool_calls=None,
            ),
            usage=None,
        )


async def _drive_native_runtime(row: dict[str, Any]) -> dict[str, Any]:
    models = {
        alias: HostRoleModel(model_name=alias)
        for alias in ("user_model", "assistant_model", "judge_model", "summary_model")
    }
    models["api_response_model"] = _ToolSimulationFacade()
    runtime = ProbeEpisodeRuntime.from_resolved_row(row, models=models)
    try:
        event = await runtime.advance()
        while not isinstance(event, EpisodeLifecycleComplete):
            event = await runtime.advance(await _activation_result(event))
        return event.result
    finally:
        await runtime.close()


def _without_timing(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    outcome = json.loads(normalized["simulation_outcome"])
    outcome.pop("wall_clock_s", None)
    outcome.pop("wall_clock_s_by_alias", None)
    normalized["simulation_outcome"] = outcome
    return normalized


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_type", ["tool_calling", "safety_agentic", "financial_services"])
async def test_gym_driver_matches_native_runtime_for_key_probes(
    probe_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        json.loads(line)["task_input"]["resolved_row"]
        for line in (Path(__file__).parents[2] / "environments/usersim/data/example.jsonl").read_text().splitlines()
    ]
    row = next(candidate for candidate in rows if candidate["probe_type"] == probe_type)
    native_result = await _drive_native_runtime(row)
    server = _server()
    monkeypatch.setattr("environment_servers.usersim.app._configured_model_name", lambda _server, alias: alias)

    class Bridge:
        environment_server = server

        async def invoke_activation(self, activation: Any) -> ActivationResult:
            return await _activation_result(activation)

        async def invoke(self, alias: str, _messages: Any, *, parameters: Any) -> SimpleNamespace:
            assert alias == "api_response_model"
            return await _ToolSimulationFacade().acompletion([], **parameters)

        def record_state(self, _state: dict[str, Any]) -> None:
            pass

    seed = UserSimSeedResponse(
        resources_session_id="resources-session-0",
        sandbox_access=None,
        resolved_row=row,
    )
    gym_result = await server._run_usersim(Bridge(), seed)

    assert _without_timing(gym_result) == _without_timing(native_result)
    assert json.loads(gym_result["simulation_outcome"])["status"] != "failed"
