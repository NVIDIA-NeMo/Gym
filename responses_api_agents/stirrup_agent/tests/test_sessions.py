# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions in a borrowed sandbox, run on the local sandbox provider in a temporary directory."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nemo_gym.agent_utils import uv_bootstrap
from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxAccess,
    SandboxExecResult,
    SandboxSpec,
    agent_tools,
)
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from responses_api_agents.stirrup_agent import app, sandbox
from responses_api_agents.stirrup_agent.app import StirrupAgentWrapper, StirrupAgentWrapperConfig
from responses_api_agents.stirrup_agent.tests.fakes import _TOOLS, _call, _ModelServer, _Resources, _resources_app


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


def test_an_agent_that_runs_out_of_wall_time_answers_with_its_partial_run_as_incomplete(
    agent, client, root, monkeypatch
):
    resources = _Resources(_resources_app([]))
    model = _ModelServer([_call("code_exec", json.dumps({"cmd": "echo draft > report.txt"})), None])
    monkeypatch.setattr(StirrupAgentWrapper, "resolve_model_base_url", lambda *_: model.base_url)
    agent.config.sandbox_runner_timeout_seconds = 4
    tool_access = _TOOL_ACCESS.model_copy(update={"base_url": resources.base_url})
    try:
        _seed(client, sandbox_access=_sandbox_access(str(root)), tool_accesses=[tool_access])
        response = client.post(
            f"/ng-rollout/{_EPISODE.capture_key}/v1/responses",
            json={"input": [{"role": "user", "content": "Write the report."}], "model": "policy", "tools": _TOOLS},
        )
        closed = _close(client)
    finally:
        model.close()
        resources.close()

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["status"], body["metadata"]["stop_reason"]) == ("incomplete", "wall_time")
    assert [item["type"] for item in body["output"]].count("function_call") == 1
    assert (root / "report.txt").read_text() == "draft\n"
    invocation = next(r for r in closed.json()["agent_observations"]["records"] if "model_calls" in r)
    assert invocation["status"] == "incomplete"


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
    response = _seed(client, tool_accesses=[_TOOL_ACCESS])

    assert response.status_code == 422
    assert "requires sandbox_access or a configured sandbox_provider" in response.json()["detail"]


def test_a_session_with_two_direct_tool_accesses_is_rejected(client):
    second = _TOOL_ACCESS.model_copy(update={"name": "other.direct_http"})
    response = _seed(client, sandbox_access=_sandbox_access(), tool_accesses=[_TOOL_ACCESS, second])

    assert response.status_code == 422
    assert "at most one direct HTTP tool access" in response.json()["detail"]


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

    response = client.post(
        f"/ng-rollout/{_EPISODE.capture_key}/v1/responses",
        json={"input": "Write the report.", "model": "policy", "tools": _TOOLS},
    )

    assert response.status_code == 422
    assert "grants no tool access" in response.json()["detail"]


@pytest.mark.parametrize(
    ("request_body", "detail"),
    [
        (
            {"input": "Write the report.", "model": "policy", "tools": [t for t in _TOOLS if t["name"] != "finish"]},
            "Finish tools ['finish'] are not among the request's tools",
        ),
        (
            {
                "input": [
                    {"role": "user", "content": "Write the report."},
                    {"role": "assistant", "content": "Working on it."},
                ],
                "model": "policy",
                "tools": _TOOLS,
            },
            "Multi-turn input with assistant messages is not supported",
        ),
        (
            {"input": [{"role": "system", "content": "Be brief."}], "model": "policy", "tools": _TOOLS},
            "Stirrup requires a user message",
        ),
    ],
    ids=["a finish tool is missing", "multi-turn input", "no user message"],
)
def test_a_request_that_can_never_run_is_a_422_before_the_runner_starts(
    client, root, monkeypatch, request_body, detail
):
    started = []

    async def execute(*_args, **_kwargs):
        started.append(True)

    monkeypatch.setattr(sandbox.StirrupSessionState, "execute", execute)
    _seed(client, sandbox_access=_sandbox_access(str(root)), tool_accesses=[_TOOL_ACCESS])

    response = client.post(f"/ng-rollout/{_EPISODE.capture_key}/v1/responses", json=request_body)

    assert response.status_code == 422
    assert detail in response.json()["detail"]
    assert started == [], "nothing runs in the sandbox for a request that cannot succeed"


def test_a_session_requiring_mcp_tools_is_rejected(client):
    mcp = MCPToolAccess(
        name="gdpval.mcp", required=True, connection=MCPStreamableHTTPConnection(url="http://resources:8000/mcp")
    )

    response = _seed(client, sandbox_access=_sandbox_access(), tool_accesses=[_TOOL_ACCESS, mcp])

    assert response.status_code == 422
    assert "Stirrup does not support required MCP tool access: gdpval.mcp" in response.json()["detail"]


async def test_the_runtime_installs_with_a_uv_built_for_the_sandbox_architecture(tmp_path, monkeypatch):
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
    (install_command,), install_kwargs = fake_sandbox.exec.await_args
    assert f"uv venv {sandbox.RUNTIME_DIR}/venv --python {sandbox.PYTHON_VERSION}" in install_command
    assert f"uv pip install --python {sandbox.SANDBOX_PYTHON} {sandbox.STIRRUP_REQUIREMENT}" in install_command
    assert install_kwargs["timeout_s"] == 60


def _respond(client: TestClient, body: dict | None = None, rollout: str = _EPISODE.capture_key):
    body = body or {"input": "Write the report.", "model": "policy"}
    return client.post(f"/ng-rollout/{rollout}/v1/responses", json=body)


def _finishing_model() -> _ModelServer:
    return _ModelServer(
        [
            _call("code_exec", json.dumps({"cmd": "echo done > report.txt"})),
            _call("finish", json.dumps({"reason": "done", "paths": ["report.txt"]})),
        ]
    )


def test_responses_outside_an_agent_session_are_refused(client):
    response = client.post("/v1/responses", json={"input": "x", "model": "policy"})

    assert response.status_code == 409
    assert "runs only in agent sessions" in response.json()["detail"]


def test_responses_on_another_rollouts_route_are_refused(client, root):
    _seed(client, sandbox_access=_sandbox_access(str(root)))

    response = _respond(client, rollout="another-rollout")

    assert response.status_code == 409
    assert "does not match the rollout route" in response.json()["detail"]


def test_a_non_required_mcp_access_is_accepted(client, root):
    mcp = MCPToolAccess(
        name="gdpval.mcp", required=False, connection=MCPStreamableHTTPConnection(url="http://resources:8000/mcp")
    )

    response = _seed(client, sandbox_access=_sandbox_access(str(root)), tool_accesses=[_TOOL_ACCESS, mcp])

    assert response.status_code == 200


async def test_a_sandbox_connection_other_than_direct_is_rejected(agent):
    sandbox_access = SandboxAccess.model_construct(connection=SimpleNamespace(kind="server"), workdir="/root")
    body = AgentSeedSessionRequest.model_construct(sandbox_access=sandbox_access)

    with pytest.raises(HTTPException) as error:
        await agent._seed_agent_session_state(body)

    assert (error.value.status_code, error.value.detail) == (422, "Stirrup supports only direct sandbox connections")


def test_an_activation_is_answered_once_and_a_retry_gets_the_same_response(agent, client, root, monkeypatch):
    agent.config.finish_tool_names = None
    model = _finishing_model()
    monkeypatch.setattr(StirrupAgentWrapper, "resolve_model_base_url", lambda *_: model.base_url)
    try:
        _seed(client, sandbox_access=_sandbox_access(str(root)))
        first = _respond(client)
        retry = _respond(client)
        different = _respond(client, {"input": "Write something else.", "model": "policy"})
        _close(client)
    finally:
        model.close()

    assert first.status_code == 200, first.text
    assert retry.json() == first.json()
    assert len(model.requests) == 2
    assert different.status_code == 409
    assert "one activation" in different.json()["detail"]


def test_a_run_that_never_finishes_within_its_turns_answers_incomplete(agent, client, root, monkeypatch):
    agent.config.finish_tool_names = None
    agent.config.agent_max_turns = 2
    model = _ModelServer([_call("code_exec", json.dumps({"cmd": "true"}))] * 2)
    monkeypatch.setattr(StirrupAgentWrapper, "resolve_model_base_url", lambda *_: model.base_url)
    try:
        _seed(client, sandbox_access=_sandbox_access(str(root)))
        response = _respond(client)
        closed = _close(client)
    finally:
        model.close()

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "incomplete"
    assert "stop_reason" not in response.json()["metadata"]
    assert closed.json()["agent_observations"]["records"][0]["status"] == "incomplete"


@pytest.mark.parametrize("output", ["", "{not json"])
def test_a_runner_that_leaves_no_valid_output_fails_the_activation(agent, client, root, tmp_path, monkeypatch, output):
    agent.config.finish_tool_names = None
    fake_python = tmp_path / "python"
    # Only the runner is replaced; the supervisor that launches it runs on the real interpreter.
    fake_python.write_text(
        "#!/bin/sh\n"
        '[ "$1" = -c ] && exit 0\n'
        f"case \"$1\" in *sandbox_runner.py) printf '%s' '{output}' > \"$3\"; exit 0;; esac\n"
        f'exec {sys.executable} "$@"\n'
    )
    fake_python.chmod(0o755)
    monkeypatch.setattr(sandbox, "SANDBOX_PYTHON", str(fake_python))
    monkeypatch.setattr(StirrupAgentWrapper, "resolve_model_base_url", lambda *_: "http://policy.invalid/v1")
    _seed(client, sandbox_access=_sandbox_access(str(root)))

    with pytest.raises(RuntimeError, match="Stirrup sandbox runner exited without output"):
        _respond(client)


class _FakeSandbox:
    """A borrowed sandbox that records its commands and answers each with ``answer(command)``."""

    def __init__(self, answer):
        self.answer = answer
        self.commands: list[str] = []
        self.disconnected = False

    async def exec(self, command, *, cwd=None, timeout_s=None):
        self.commands.append(command)
        return self.answer(command)

    async def upload(self, local_path, remote_path):
        pass

    async def disconnect(self):
        self.disconnected = True


def _ok(stdout="", stderr="", return_code=0):
    return SandboxExecResult(stdout=stdout, stderr=stderr, return_code=return_code)


@pytest.fixture
def borrow(monkeypatch):
    """Lend ``_FakeSandbox(answer)`` to the next seed; the provider it is lent through records being closed."""
    provider = SimpleNamespace(name="local", closed=False)

    async def aclose():
        provider.closed = True

    provider.aclose = aclose
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    monkeypatch.setattr(app, "create_provider", lambda _: provider)

    def lend(answer):
        box = _FakeSandbox(answer)

        async def connect(descriptor, *, provider):
            return box

        monkeypatch.setattr(AsyncSandbox, "connect", connect)
        return box, provider

    return lend


def test_a_session_whose_sandbox_cannot_be_prepared_is_released_and_fails_the_seed(client, borrow):
    box, _ = borrow(
        lambda command: _ok(stderr="read-only file system", return_code=1) if "mkdir" in command else _ok()
    )

    with pytest.raises(RuntimeError, match="read-only file system"):
        _seed(client, sandbox_access=_sandbox_access())

    assert box.disconnected


def test_a_failed_runtime_install_fails_the_seed_and_releases_the_sandbox(client, borrow, monkeypatch):
    monkeypatch.setattr(uv_bootstrap.shutil, "which", lambda name: "/host/uv")

    def answer(command):
        if "mkdir" in command:
            return _ok()
        if "import stirrup" in command:
            return _ok(return_code=1)
        if "uv venv" in command:
            return _ok(stderr="no network", return_code=2)
        return _ok()

    box, _ = borrow(answer)

    with pytest.raises(RuntimeError, match="no network"):
        _seed(client, sandbox_access=_sandbox_access())

    assert any("pip install" in command for command in box.commands)
    assert box.disconnected


def test_a_seed_whose_cleanup_also_fails_keeps_the_session_for_a_close_retry(client, borrow):
    cleaning_up = {"fails": True}

    def answer(command):
        if "mkdir" in command:
            return _ok(stderr="read-only file system", return_code=1)
        if ".closed" in command and cleaning_up["fails"]:
            return _ok(return_code=1)
        return _ok()

    box, _ = borrow(answer)

    with pytest.raises(RuntimeError, match="read-only file system"):
        _seed(client, sandbox_access=_sandbox_access())
    assert not box.disconnected

    cleaning_up["fails"] = False
    closed = _close(client)

    assert closed.status_code == 200
    assert box.disconnected


def test_a_sandbox_that_cannot_be_reached_closes_the_provider(client, borrow, monkeypatch):
    _, provider = borrow(lambda command: _ok())

    async def unreachable(descriptor, *, provider):
        raise ConnectionError("sandbox is gone")

    monkeypatch.setattr(AsyncSandbox, "connect", unreachable)

    with pytest.raises(ConnectionError):
        _seed(client, sandbox_access=_sandbox_access())

    assert provider.closed
