# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.langgraph_agent.orchestrator_agent import OrchestratorAgent, OrchestratorAgentConfig
from responses_api_agents.langgraph_agent.parallel_thinking_agent import (
    ParallelThinkingAgent,
    ParallelThinkingAgentConfig,
)
from responses_api_agents.langgraph_agent.reflection_agent import ReflectionAgent, ReflectionAgentConfig
from responses_api_agents.langgraph_agent.rewoo_agent import ReWOOAgent, ReWOOAgentConfig


FLAVORS = [
    (ReflectionAgent, ReflectionAgentConfig, "reflection", {"max_reflections": 2}),
    (OrchestratorAgent, OrchestratorAgentConfig, "orchestrator", {"max_subtasks": 3}),
    (
        ParallelThinkingAgent,
        ParallelThinkingAgentConfig,
        "parallel_thinking",
        {"num_parallel_paths": 2, "perspective_prompts": ["Use arithmetic", "Use multiplication"]},
    ),
    (ReWOOAgent, ReWOOAgentConfig, "rewoo", {}),
]
MOCK_RESPONSE = {
    "id": "resp_test123",
    "created_at": 1770000000.0,
    "model": "test-model",
    "object": "response",
    "output": [
        {
            "id": "msg_test123",
            "content": [{"annotations": [], "text": "<answer>42</answer>", "type": "output_text"}],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


def _usage(input_tokens: int, output_tokens: int, reasoning_tokens: int, cached_tokens: int = 0) -> dict:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def _mock_model_response(payload: dict | None = None):
    payload = deepcopy(MOCK_RESPONSE if payload is None else payload)
    response = AsyncMock()
    response.json.return_value = payload
    response.read.return_value = json.dumps(payload).encode()
    response.cookies = {}
    response.ok = True
    return response


def _agent(agent_class=ReflectionAgent, config_class=ReflectionAgentConfig, options=None, **overrides):
    server_client = MagicMock(spec=ServerClient, global_config_dict={})
    server_client._build_server_base_url.return_value = "http://model-server:8080"
    return agent_class(
        config=config_class(
            name="graph",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            num_workers=1,
            model_server={"type": "responses_api_models", "name": "model"},
            **dict(options or {}, **overrides),
        ),
        server_client=server_client,
    )


def _seed(*, borrowed: bool = True) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="graph-session",
        episode_id=EpisodeId(rollout_id="graph-rollout", attempt=1),
        task_id=TaskId(taskset="test", task_id="multiply"),
        sandbox_access=(
            {
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "runtime",
                    "descriptor": {"sandbox_id": "task-sandbox"},
                },
                "workdir": "/workspace/task",
            }
            if borrowed
            else None
        ),
    )


class MemorySandbox:
    """File-transfer/exec boundary with the real agent and SandboxSession above it."""

    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}
        self.commands: list[tuple[str, str | None, float | None]] = []
        self.response = NeMoGymResponse.model_validate(
            deepcopy(MOCK_RESPONSE) | {"usage": _usage(17, 25, 9, 3)}
        ).model_dump(mode="json")
        self.output = {
            "response": self.response,
            "runtime": {"hostname": "task-sandbox", "pid": 123, "python": "/sandbox/venv/bin/python"},
        }
        self.cleanup_confirmed = True
        self.timed_out = False
        self.release_error: Exception | None = None
        self.stops = 0
        self.disconnects = 0
        self.started = []
        self.events: list[str] = []

    async def start(self, spec) -> None:
        self.started.append(spec)

    async def upload(self, local_path, remote_path) -> None:
        self.uploads[str(remote_path)] = Path(local_path).read_bytes()

    async def download(self, remote_path, local_path) -> None:
        if str(remote_path).endswith("/cleanup.json"):
            value = {
                "cleanup_confirmed": self.cleanup_confirmed,
                "return_code": 0,
                "error": None,
                "timed_out": self.timed_out,
            }
            self.events.append("cleanup-receipt")
        elif str(remote_path).endswith("/output.json"):
            value = self.output
            self.events.append("collect-output")
        else:
            raise FileNotFoundError(remote_path)
        Path(local_path).write_text(json.dumps(value), encoding="utf-8")

    async def exec(self, command, *, cwd=None, env=None, timeout_s=180, user=None) -> SandboxExecResult:
        self.commands.append((command, cwd, timeout_s))
        return SandboxExecResult(return_code=0, stdout="", stderr="")

    async def disconnect(self) -> None:
        self.events.append("disconnect")
        self.disconnects += 1
        if self.release_error is not None:
            raise self.release_error

    async def stop(self) -> None:
        self.events.append("stop")
        self.stops += 1
        if self.release_error is not None:
            raise self.release_error


@pytest.fixture
def sandbox_boundary(monkeypatch):
    from responses_api_agents.langgraph_agent import app as module
    from responses_api_agents.langgraph_agent.sandbox import LangGraphSandboxSession

    sandbox = MemorySandbox()
    factory = MagicMock(return_value=sandbox)
    factory.connect = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(module, "AsyncSandbox", factory)
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(module, "resolve_provider_config", lambda *_args: {"docker": {}})
    monkeypatch.setattr(module, "resolve_provider_metadata", lambda *_args: {})
    monkeypatch.setattr(module, "create_provider", lambda _config: MagicMock())
    monkeypatch.setattr("nemo_gym.agents.base_responses_api_agent.get_first_server_config_dict", lambda *_args: {})
    monkeypatch.setattr(LangGraphSandboxSession, "install_runtime", AsyncMock())
    return sandbox, factory


@pytest.mark.parametrize("agent_class,config_class,workflow,options", FLAVORS)
def test_entrypoints_do_not_build_graphs_on_the_host(agent_class, config_class, workflow, options):
    agent = _agent(agent_class, config_class, options)
    assert agent.workflow == workflow
    assert agent.calls_model_from_sandbox is True
    assert not hasattr(agent, "graph")
    assert not hasattr(agent, "build_graph")
    agent.server_client.post.assert_not_called()


@pytest.mark.parametrize("agent_class,config_class,workflow,options", FLAVORS)
def test_sessionless_responses_use_owned_sandbox_and_preserve_usage(
    sandbox_boundary, agent_class, config_class, workflow, options
):
    sandbox, factory = sandbox_boundary
    agent = _agent(agent_class, config_class, options, sandbox_provider="runtime")
    body = {"input": "What is 6 * 7?", "temperature": 0.2, "metadata": {"purpose": "test"}}
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/responses", json=body, headers={"cookie": "resource_secret=do-not-copy"})
    assert response.status_code == 200, response.text
    assert response.json()["output"] == sandbox.response["output"]
    assert response.json()["usage"] == sandbox.response["usage"]
    assert response.json()["metadata"]["harness_execution"] == "sandbox"
    payload = json.loads(next(value for path, value in sandbox.uploads.items() if path.endswith("/input.json")))
    assert payload["workflow"] == workflow
    # Only caller-set fields cross, so unset sampling fields cannot override the Model Server's defaults.
    assert payload["request"] == body
    assert payload["config"] == options
    assert payload["model_url"] == "http://model-server:8080/v1/responses"
    assert "do-not-copy" not in json.dumps(payload)
    assert "model_server" not in payload["config"]
    assert "sandbox_provider" not in payload["config"]
    factory.connect.assert_not_awaited()
    assert len(sandbox.started) == sandbox.stops == 1
    assert sandbox.disconnects == 0
    assert sandbox.events.index("collect-output") < sandbox.events.index("stop")
    agent.server_client.post.assert_not_called()


def test_sessionless_responses_need_a_provider_before_model_calls():
    agent = _agent()
    with TestClient(agent.setup_webserver()) as client:
        result = client.post("/v1/responses", json={"input": "question"})
    assert result.status_code == 422, result.text
    assert "sandbox" in result.json()["detail"].lower()
    agent.server_client.post.assert_not_called()


def test_unbound_legacy_run_is_rejected_before_resources():
    agent = _agent()
    with TestClient(agent.setup_webserver()) as client:
        result = client.post("/run", json={"responses_create_params": {"input": "question"}})
    assert result.status_code == 422
    assert "resources_server" in result.json()["detail"]
    agent.server_client.post.assert_not_called()
