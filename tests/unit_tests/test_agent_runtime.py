# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import BaseServerConfig, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.agent_runtime import RuntimeRunRequest, SandboxedAgentHost
from nemo_gym.sandbox.agent_runtime_config import AgentRuntimeConfig
from nemo_gym.server_utils import ServerClient


class FileAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    marker: str


class FileAgent(SimpleResponsesAPIAgent):
    config: FileAgentConfig

    def model_post_init(self, context):
        Path(self.config.marker).write_text(str(os.getpid()))

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()):
        assert request.cookies["resource_session"] == "task-cookie"
        Path("answer.txt").write_text(body.input)
        return NeMoGymResponse(
            id="response",
            created_at=0,
            model="file-agent",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )

    async def run(self, request, body):
        raise AssertionError("Inner /run must never be called")


class Response:
    ok = True
    cookies = {"resource_session": "task-cookie"}

    def __init__(self, payload):
        self.payload = payload

    async def read(self):
        return json.dumps(self.payload).encode()


def make_host(tmp_path, **runtime):
    config = FileAgentConfig(
        host="localhost",
        port=8001,
        entrypoint="app.py",
        name="agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        marker=str(tmp_path / "initialized"),
        runtime={
            "type": "sandbox",
            "sandbox_source": "runtime",
            "provider": "sandbox",
            "python": sys.executable,
            "dependencies": {"enabled": False},
            "env": {"PYTHONPATH": str(Path.cwd())},
            "spec": {"workdir": str(tmp_path)},
            "timeout_s": 30,
            **runtime,
        },
    )
    client = ServerClient(
        head_server_config=BaseServerConfig(host="localhost", port=1),
        global_config_dict=OmegaConf.create(
            {
                "sandbox": {"local": {}},
                "resources": {"resources_servers": {"test": {"host": "localhost", "port": 8002, "answer": "SECRET"}}},
            }
        ),
    )
    return FileAgent.create_server(config, client)


def test_factory_places_initialization_inside_runtime(tmp_path):
    host = make_host(tmp_path)
    assert isinstance(host, SandboxedAgentHost)
    assert isinstance(host.config, FileAgentConfig)
    assert not (tmp_path / "initialized").exists()
    local_config = host.config.model_copy(update={"runtime": AgentRuntimeConfig()})
    assert isinstance(FileAgent.create_server(local_config, host.server_client), FileAgent)
    assert (tmp_path / "initialized").read_text() == str(os.getpid())


async def test_unchanged_harness_executes_in_separate_process_before_verification(tmp_path):
    host = make_host(tmp_path)
    calls = []

    async def post(server_name, url_path, json, cookies):
        calls.append(url_path)
        if url_path == "/seed_session":
            return Response({})
        assert url_path == "/verify"
        assert cookies["resource_session"] == "task-cookie"
        assert json["task_id"] == "task-1"
        assert (tmp_path / "answer.txt").read_text() == "42"
        assert (tmp_path / "initialized").read_text() != str(os.getpid())
        return Response(json | {"reward": 1.0, "custom_metric": 3})

    with patch.object(ServerClient, "post", AsyncMock(side_effect=post)):
        result = await host.run(
            MagicMock(cookies={}),
            RuntimeRunRequest.model_validate({"responses_create_params": {"input": "42"}, "task_id": "task-1"}),
        )
    assert calls == ["/seed_session", "/verify"]
    assert result.reward == 1.0
    assert result.custom_metric == 3


def test_worker_receives_routes_not_verifier_secrets_or_task_answers(tmp_path):
    host = make_host(tmp_path, server_urls={"resources": "https://reachable.example/gym"})
    body = RuntimeRunRequest.model_validate({"responses_create_params": {"input": "solve"}, "answer": "GOLDEN"})
    payload = host._worker_payload(body, {})
    encoded = json.dumps(payload)
    assert "SECRET" not in encoded
    assert "GOLDEN" not in encoded
    assert "reachable.example" in encoded
    assert payload["config"]["runtime"] == {"type": "local"}


@pytest.mark.parametrize("capture", [False, True])
def test_worker_preserves_rollout_and_training_capture_routes(tmp_path, capture):
    host = make_host(tmp_path)
    host.server_client.global_config_dict.observability_enabled = True
    host.server_client.global_config_dict.token_id_capture = {"enabled": capture, "all_agents": True}
    body = RuntimeRunRequest.model_validate(
        {
            "responses_create_params": {"input": "solve"},
            "_ng_task_index": 2,
            "_ng_rollout_index": 3,
        }
    )
    payload = host._worker_payload(body, {})
    assert payload["path"] == "/ng-rollout/2-3/" + ("training-token-capture/" if capture else "") + "v1/responses"
    assert payload["global_config"]["token_id_capture"]["enabled"] is capture
    assert payload["config"]["token_id_capture"] is capture


@pytest.mark.parametrize("failure", ["connect", "execute", "verify", "cancel"])
async def test_borrowed_workspace_cleanup_on_failure(tmp_path, failure):
    host = make_host(tmp_path, sandbox_source="environment", spec={})
    sandbox = MagicMock(stop=AsyncMock())
    provider = MagicMock(aclose=AsyncMock())
    descriptor = {"sandbox_id": "task", "workdir": "/testbed", "opaque": {"lease": 1}}
    calls = []

    async def post(server_name, url_path, json, cookies):
        calls.append(url_path)
        if url_path == "/seed_session":
            return Response({"workspace": {"provider": "sandbox", "descriptor": descriptor}})
        assert cookies["resource_session"] == "task-cookie"
        if url_path == "/verify":
            raise ValueError("verify failed")
        assert url_path == "/cleanup_session"
        return Response({})

    error = asyncio.CancelledError() if failure == "cancel" else ValueError("execute failed")
    response = NeMoGymResponse(
        id="response",
        created_at=0,
        model="file-agent",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    with (
        patch.object(ServerClient, "post", AsyncMock(side_effect=post)),
        patch("nemo_gym.sandbox.agent_runtime.create_provider", return_value=provider),
        patch(
            "nemo_gym.sandbox.agent_runtime.AsyncSandbox.connect",
            AsyncMock(
                return_value=sandbox,
                side_effect=ValueError("connect failed") if failure == "connect" else None,
            ),
        ) as connect,
        patch.object(
            SandboxedAgentHost,
            "_execute",
            AsyncMock(
                return_value={"response": response.model_dump(mode="json")},
                side_effect=error if failure in {"execute", "cancel"} else None,
            ),
        ),
    ):
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else ValueError):
            await host.run(MagicMock(cookies={}), RuntimeRunRequest(responses_create_params={"input": "solve"}))
        connect.assert_awaited_once_with(descriptor, provider=provider)
    assert calls[-1] == "/cleanup_session"
    sandbox.stop.assert_not_awaited()
    provider.aclose.assert_awaited_once()


@pytest.mark.parametrize(
    "config",
    [
        {"type": "typo"},
        {"concurrency": 0},
        {"agent_config": {"runtime": {"type": "sandbox"}}},
        {"sandbox_source": "environment", "spec": {"image": "task"}},
        {"sandbox_source": "typo"},
        {"workspace": "environment"},
    ],
)
def test_invalid_runtime_config(config):
    with pytest.raises(ValidationError):
        AgentRuntimeConfig.model_validate(config)
