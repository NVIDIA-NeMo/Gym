# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions in a borrowed sandbox, run on the local sandbox provider in a temporary directory."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import ServerClient
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from responses_api_agents.stirrup_agent import app, sandbox
from responses_api_agents.stirrup_agent.app import StirrupAgentWrapper, StirrupAgentWrapperConfig
from responses_api_agents.stirrup_agent.tests.test_sandbox_runner import (
    _TOOLS,
    _call,
    _ModelServer,
    _Resources,
    _resources_app,
)


_EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
_TOOL_ACCESS = DirectHTTPToolAccess(
    name="gdpval.direct_http", required=True, base_url="http://resources:8000", cookies={"session": "seeded"}
)


def _sandbox_access(workdir: str = "/root") -> SandboxAccess:
    return SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"id": "box"}), workdir=workdir
    )


@pytest.fixture
def agent() -> StirrupAgentWrapper:
    config = StirrupAgentWrapperConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="stirrup_agent",
        task="gdpval",
        resources_server=ResourcesServerRef(type="resources_servers", name="gdpval_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        finish_tool_names=["finish", "abandon_task_finish"],
    )
    return StirrupAgentWrapper(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def client(agent) -> TestClient:
    return TestClient(agent.setup_webserver())


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """The borrowed sandbox's working directory; the host's Stirrup install stands in for the sandbox runtime."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})

    # The local provider cannot be reattached, so connecting starts one in the working directory.
    async def connect(descriptor, *, provider):
        box = AsyncSandbox(provider)
        await box.start(SandboxSpec(workdir=str(root)))
        box._connected = True
        return box

    monkeypatch.setattr(AsyncSandbox, "connect", connect)
    monkeypatch.setattr(sandbox, "SANDBOX_PYTHON", sys.executable)
    monkeypatch.setattr(sandbox, "_PACKAGE_DIR", str(tmp_path / "harness/responses_api_agents/stirrup_agent"))
    return root


def _seed(client: TestClient, **grants):
    body = AgentSeedSessionRequest(
        agent_session_id="agent-session", episode_id=_EPISODE, task_id=TaskId(taskset="gdpval", task_id="t"), **grants
    )
    return client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))


def _close(client: TestClient):
    body = AgentCloseSessionRequest(agent_session_id="agent-session", episode_id=_EPISODE)
    return client.post("/v1/agent_sessions/close", json=body.model_dump(mode="json"))


def test_an_episode_runs_stirrup_in_the_borrowed_sandbox(agent, client, root, monkeypatch):
    calls: list = []
    resources = _Resources(_resources_app(calls))
    model = _ModelServer(
        [
            _call("code_exec", json.dumps({"cmd": "echo draft > report.txt"})),
            _call("web_search", json.dumps({"query": "gdp"})),
            _call("finish", json.dumps({"reason": "done", "paths": ["report.txt"]})),
        ]
    )
    monkeypatch.setattr(StirrupAgentWrapper, "resolve_model_base_url", lambda *_: model.base_url)
    tool_access = _TOOL_ACCESS.model_copy(update={"base_url": resources.base_url})
    try:
        seeded = _seed(client, sandbox_access=_sandbox_access(str(root)), tool_accesses=[tool_access])
        response = client.post(
            f"/ng-rollout/{_EPISODE.capture_key}/v1/responses",
            json={"input": [{"role": "user", "content": "Write the report."}], "model": "policy", "tools": _TOOLS},
        )
        closed = _close(client)
    finally:
        model.close()
        resources.close()

    assert seeded.status_code == 200
    assert response.status_code == 200, response.text
    output = response.json()["output"]
    assert [item["content"] for item in output if item.get("role") == "user"] == ["Write the report."]
    assert [item["name"] for item in output if item["type"] == "function_call"] == [
        "code_exec",
        "web_search",
        "finish",
    ]
    assert (root / "report.txt").read_text() == "draft\n"
    assert [name for name, *_ in calls] == ["web_search", "finish"]
    assert closed.status_code == 200
    assert closed.json()["resources_cookies"] == {"session": "after-search"}
    assert sorted(p.name for p in root.iterdir()) == ["report.txt"]


def test_a_session_without_a_sandbox_is_rejected(client):
    with pytest.raises(ValueError, match="requires sandbox access"):
        _seed(client, tool_accesses=[_TOOL_ACCESS])


def test_a_session_without_direct_tools_is_rejected(client):
    with pytest.raises(ValueError, match="exactly one direct HTTP tool access"):
        _seed(client, sandbox_access=_sandbox_access())


def test_a_session_requiring_mcp_tools_is_rejected(client):
    mcp = MCPToolAccess(
        name="gdpval.mcp", required=True, connection=MCPStreamableHTTPConnection(url="http://resources:8000/mcp")
    )

    with pytest.raises(ValueError, match="does not support required MCP"):
        _seed(client, sandbox_access=_sandbox_access(), tool_accesses=[_TOOL_ACCESS, mcp])
