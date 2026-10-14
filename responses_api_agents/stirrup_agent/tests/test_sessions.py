# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions in a borrowed sandbox, run on the local sandbox provider in a temporary directory."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, agent_tools
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
    usage = response.json()["usage"]
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (30, 15, 45)
    bundle = closed.json()["agent_observations"]
    assert bundle["gaps"] == []
    invocation, *tools = bundle["records"]
    assert invocation["status"] == "completed"
    assert [ref["response_id"] for ref in invocation["model_calls"]] == ["chatcmpl-1", "chatcmpl-2", "chatcmpl-3"]
    assert {ref["model_ref"]["name"] for ref in invocation["model_calls"]} == {"policy_model"}
    assert [item["call_id"] for item in invocation["conversation"] if item["type"] == "function_call_output"] == [
        tool["tool_call_id"] for tool in tools
    ]


def test_a_failed_episode_still_closes_with_its_evidence(agent, client, root, monkeypatch):
    resources = _Resources(_resources_app([]))
    model = _ModelServer([_call("fetch_web_page", json.dumps({"url": "x"}))])
    monkeypatch.setattr(StirrupAgentWrapper, "resolve_model_base_url", lambda *_: model.base_url)
    tool_access = _TOOL_ACCESS.model_copy(update={"base_url": resources.base_url})
    try:
        _seed(client, sandbox_access=_sandbox_access(str(root)), tool_accesses=[tool_access])
        with pytest.raises(RuntimeError, match="500"):
            client.post(
                f"/ng-rollout/{_EPISODE.capture_key}/v1/responses",
                json={"input": "Write the report.", "model": "policy", "tools": _TOOLS},
            )
        closed = _close(client)
    finally:
        model.close()
        resources.close()

    bundle = closed.json()["agent_observations"]
    (invocation,) = bundle["records"]
    assert invocation["status"] == "failed"
    assert [ref["response_id"] for ref in invocation["model_calls"]] == ["chatcmpl-1"]
    assert invocation["conversation"] == []
    assert bundle["gaps"] == []


@pytest.mark.parametrize(
    ("model_url", "reachable"), [("http://127.0.0.1:8000", False), ("http://10.0.0.5:8000", True)]
)
def test_a_remote_sandbox_needs_a_model_server_address_it_can_reach(client, monkeypatch, model_url, reachable):
    class Remote:
        name = "opensandbox"
        aclose = AsyncMock()

    class Connected(Exception):
        pass

    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}})
    monkeypatch.setattr(app, "create_provider", lambda _: Remote())
    monkeypatch.setattr(agent_tools, "get_server_url", lambda _: model_url)
    monkeypatch.setattr(AsyncSandbox, "connect", AsyncMock(side_effect=Connected))

    with pytest.raises(Connected if reachable else ValueError):
        _seed(client, sandbox_access=_sandbox_access(), tool_accesses=[_TOOL_ACCESS])


def test_a_session_without_a_sandbox_or_a_provider_is_rejected(client):
    with pytest.raises(ValueError, match="requires sandbox_access or a configured sandbox_provider"):
        _seed(client, tool_accesses=[_TOOL_ACCESS])


def test_a_session_with_two_direct_tool_accesses_is_rejected(client):
    second = _TOOL_ACCESS.model_copy(update={"name": "other.direct_http"})
    with pytest.raises(ValueError, match="at most one direct HTTP tool access"):
        _seed(client, sandbox_access=_sandbox_access(), tool_accesses=[_TOOL_ACCESS, second])


def test_an_episode_without_tools_needs_no_tool_access(agent, client, root, monkeypatch):
    agent.config.finish_tool_names = None
    model = _ModelServer(
        [
            _call("code_exec", json.dumps({"cmd": "echo done > report.txt"})),
            _call("finish", json.dumps({"reason": "done", "paths": ["report.txt"]})),
        ]
    )
    monkeypatch.setattr(StirrupAgentWrapper, "resolve_model_base_url", lambda *_: model.base_url)
    try:
        seeded = _seed(client, sandbox_access=_sandbox_access(str(root)))
        response = client.post(
            f"/ng-rollout/{_EPISODE.capture_key}/v1/responses", json={"input": "Write the report.", "model": "policy"}
        )
        _close(client)
    finally:
        model.close()

    assert seeded.status_code == 200
    assert response.status_code == 200, response.text
    assert [item["name"] for item in response.json()["output"] if item["type"] == "function_call"] == [
        "code_exec",
        "finish",
    ]
    assert (root / "report.txt").read_text() == "done\n"


def test_without_a_lent_sandbox_or_tools_the_agent_starts_its_own_sandbox(agent, root, monkeypatch):
    agent.config.sandbox_provider = "sandbox"
    agent.config.sandbox_config = {"workdir": str(root)}
    agent.config.finish_tool_names = None
    client = TestClient(agent.setup_webserver())
    model = _ModelServer(
        [
            _call("code_exec", json.dumps({"cmd": "echo done > report.txt"})),
            _call("finish", json.dumps({"reason": "done", "paths": ["report.txt"]})),
        ]
    )
    monkeypatch.setattr(StirrupAgentWrapper, "resolve_model_base_url", lambda *_: model.base_url)
    try:
        seeded = _seed(client)
        response = client.post(
            f"/ng-rollout/{_EPISODE.capture_key}/v1/responses", json={"input": "Write the report.", "model": "policy"}
        )
        closed = _close(client)
    finally:
        model.close()

    assert seeded.status_code == 200
    assert response.status_code == 200, response.text
    assert [item["name"] for item in response.json()["output"] if item["type"] == "function_call"] == [
        "code_exec",
        "finish",
    ]
    (invocation,) = closed.json()["agent_observations"]["records"][:1]
    assert invocation["status"] == "completed"


def test_request_tools_without_a_tool_access_are_rejected(client, root):
    _seed(client, sandbox_access=_sandbox_access(str(root)))

    with pytest.raises(ValueError, match="grants no tool access"):
        client.post(
            f"/ng-rollout/{_EPISODE.capture_key}/v1/responses",
            json={"input": "Write the report.", "model": "policy", "tools": _TOOLS},
        )


def test_a_session_requiring_mcp_tools_is_rejected(client):
    mcp = MCPToolAccess(
        name="gdpval.mcp", required=True, connection=MCPStreamableHTTPConnection(url="http://resources:8000/mcp")
    )

    with pytest.raises(ValueError, match="does not support required MCP"):
        _seed(client, sandbox_access=_sandbox_access(), tool_accesses=[_TOOL_ACCESS, mcp])


async def test_the_runtime_installs_with_a_uv_built_for_the_sandbox_architecture(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nemo_gym.agent_utils import uv_bootstrap

    host_machine = uv_bootstrap.platform.machine().lower()
    sandbox_machine = "x86_64" if host_machine in ("aarch64", "arm64") else "aarch64"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(uv_bootstrap.shutil, "which", lambda name: "/host/uv")
    monkeypatch.setattr(uv_bootstrap, "_host_uv_version", lambda uv: "0.9.0")
    downloads = []

    def download(version, target, path):
        downloads.append(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("uv")

    monkeypatch.setattr(uv_bootstrap, "_download_uv", download)
    probe = SimpleNamespace(return_code=0, stdout=f"Linux\n{sandbox_machine}\ngnu\n", stderr="")
    installed = SimpleNamespace(return_code=0, stdout="", stderr="")
    fake_sandbox = MagicMock(exec=AsyncMock(side_effect=[probe, installed]), upload=AsyncMock())
    state = SimpleNamespace(
        session=SimpleNamespace(sandbox=fake_sandbox, workdir="/root"),
        _runtime_installed=AsyncMock(return_value=True),
    )

    await sandbox.StirrupSessionState._install_runtime(state, timeout=60)

    (uploaded, _), _ = fake_sandbox.upload.await_args
    assert downloads and sandbox_machine in downloads[0]
    assert uploaded != "/host/uv" and Path(uploaded).read_text() == "uv"
