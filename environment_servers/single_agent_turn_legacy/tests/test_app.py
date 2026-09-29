# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path

import orjson
import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ConfigDict

import nemo_gym.server_utils
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import BaseServerConfig, ServerClient, SimpleServer
from nemo_gym.single_agent_turn_types import SingleAgentTurnRequest, SingleAgentTurnTaskInput


class _Cookie:
    value = "cookie-value"


def test_legacy_module_exports_app_for_multi_worker_import(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_app = object()
    monkeypatch.setattr(nemo_gym.server_utils, "is_nemo_gym_fastapi_entrypoint", lambda _: True)
    monkeypatch.setattr(SimpleServer, "run_webserver", classmethod(lambda _: worker_app))

    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "app.py"),
        run_name="single_agent_turn_legacy.worker_test",
    )

    assert namespace["app"] is worker_app


class _Response:
    ok = True
    cookies = {"session": _Cookie()}

    def __init__(self, body: dict) -> None:
        self.body = orjson.dumps(body)

    async def read(self) -> bytes:
        return self.body


def _agent_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class _Client(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict]]
    responses: list[_Response]

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        self.calls.append((server_name, url_path, kwargs))
        response = self.responses.pop(0)
        payload = orjson.loads(response.body)
        body = kwargs.get("json")
        if url_path == "/seed_session" and "resources_session_id" in body:
            payload["resources_session_id"] = body["resources_session_id"]
        elif url_path == "/v1/agent_sessions":
            payload["agent_session_id"] = body["agent_session_id"]
        elif url_path == "/v1/agent_sessions/close":
            payload["agent_session_id"] = body["agent_session_id"]
        elif url_path == "/close_session":
            payload["resources_session_id"] = body["resources_session_id"]
        return _Response(payload)

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


def _environment_server() -> tuple[SingleAgentTurnEnvironmentServer, _Client]:
    global_config = OmegaConf.create(
        {
            "resources": {"resources_servers": {"test": {"host": "resources", "port": 8000, "entrypoint": "app.py"}}},
            "agent": {"responses_api_agents": {"test": {"host": "agent", "port": 8001, "entrypoint": "app.py"}}},
        }
    )
    response = _agent_response()
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        calls=[],
        responses=[
            _Response({"resources_session_id": "resources-session"}),
            _Response({"agent_session_id": "agent-session"}),
            _Response(response.model_dump(mode="json")),
            _Response(
                {
                    "agent_session_id": "agent-session",
                    "resources_cookies": {"session": "updated-cookie"},
                }
            ),
            _Response(
                {
                    "responses_create_params": {"input": "task"},
                    "response": response.model_dump(mode="json"),
                    "reward": 1.0,
                    "benchmark_field": "preserved",
                }
            ),
            _Response({"resources_session_id": "resources-session"}),
        ],
    )
    config = SingleAgentTurnEnvironmentServerConfig(
        name="environment",
        host="environment",
        port=8002,
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
        default_episode_timeout_seconds=10,
        cleanup_timeout_seconds=10,
    )
    return SingleAgentTurnEnvironmentServer(config=config, server_client=client), client


async def test_legacy_compatibility_is_a_separate_environment_deployment() -> None:
    environment_server, client = _environment_server()
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=environment_server.config, server_client=client)
    result = await adapter.run_legacy(
        {
            "_ng_task_index": 3,
            "_ng_rollout_index": 2,
            "_ng_attempt_index": 1,
            "instance_id": "task",
            "benchmark_field": "input",
            "responses_create_params": {"input": "task"},
        }
    )

    assert result["reward"] == 1.0
    assert result["benchmark_field"] == "preserved"
    assert result["agent_ref"] == {"name": "agent"}
    assert "verification" not in result
    assert "ng_agent_observations" not in result


def test_legacy_adapter_forwards_aggregate_metrics_to_resources() -> None:
    environment_server, client = _environment_server()
    client.responses = [_Response({"agent_metrics": {"mean/reward": 0.5}})]
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=environment_server.config, server_client=client)

    response = TestClient(adapter.setup_webserver()).post(
        "/aggregate_metrics",
        json={"verify_responses": [{"_ng_task_index": 0, "reward": 0.5}]},
    )

    assert response.status_code == 200
    assert response.json()["agent_metrics"] == {"mean/reward": 0.5}
    assert [(server, path) for server, path, _ in client.calls] == [("resources", "/aggregate_metrics")]


async def test_legacy_and_native_envelopes_project_the_same_result() -> None:
    legacy_environment, legacy_client = _environment_server()
    native_environment, native_client = _environment_server()
    legacy_adapter = SingleAgentTurnLegacyEnvironmentServer(
        config=legacy_environment.config,
        server_client=legacy_client,
    )
    native_adapter = SingleAgentTurnLegacyEnvironmentServer(
        config=native_environment.config,
        server_client=native_client,
    )
    flat_row = {
        "_ng_task_index": 3,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 1,
        "instance_id": "task",
        "benchmark_field": "input",
        "responses_create_params": {"input": "task"},
    }
    native_request = SingleAgentTurnRequest(
        episode_id=EpisodeId(rollout_id="3-2", attempt=1),
        task=MaterializedTask(
            task_id=TaskId(taskset="resources", task_id="task"),
            task_input=SingleAgentTurnTaskInput(
                responses_create_params={"input": "task"},
                task_data={"instance_id": "task", "benchmark_field": "input"},
            ),
        ),
    )

    legacy_result = await legacy_adapter.run_legacy(flat_row)
    native_result = await native_adapter.run_legacy(native_request.model_dump(mode="json"))

    assert native_result == legacy_result
