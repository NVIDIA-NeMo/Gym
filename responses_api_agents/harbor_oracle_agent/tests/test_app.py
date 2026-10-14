# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.harbor_oracle_agent import app as module
from responses_api_agents.harbor_oracle_agent.app import OracleAgent, OracleAgentConfig


@dataclass
class FakeSandbox:
    """The sandbox the environment server seeded, with the solution already at ``/solution`` when ``solution`` is set.

    ``result`` is what solve.sh produced.
    """

    result: SandboxExecResult = SandboxExecResult(stdout="Done!\n", stderr="", return_code=0)
    solution: bool = True
    execs: list[dict] = field(default_factory=list)
    uploads: list[str] = field(default_factory=list)
    disconnected: bool = False

    async def exec(
        self, command, *, cwd=None, env=None, timeout_s=None, user=None, preserve_background_services=False
    ):
        self.execs.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_s": timeout_s,
                "user": user,
                "preserve_background_services": preserve_background_services,
            }
        )
        if command == "test -f /solution/solve.sh":
            return SandboxExecResult(stdout="", stderr="", return_code=0 if self.solution else 1)
        if "solve.sh" in command:
            return self.result
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload(self, local_path, remote_path):
        self.uploads.append(remote_path)

    async def disconnect(self):
        self.disconnected = True


def make_agent(monkeypatch: MonkeyPatch, *, sandbox: FakeSandbox | None = None):
    # The oracle needs the run config for one thing: the sandbox provider block its connection names.
    global_config = {"sandbox": {"opensandbox": {"connection": {"domain": "example.invalid"}}}}
    monkeypatch.setattr(module, "get_global_config_dict", lambda: global_config)
    sandbox = sandbox or FakeSandbox()
    provider = MagicMock()
    monkeypatch.setattr(module, "create_provider", lambda config: provider)

    async def connect(descriptor, *, provider, owns_provider=True):
        connect.descriptors.append(descriptor)
        return sandbox

    connect.descriptors = []
    monkeypatch.setattr(module.AsyncSandbox, "connect", connect)
    config = OracleAgentConfig(
        host="0.0.0.0",
        port=8081,
        entrypoint="",
        name="harbor_oracle_agent",
    )
    agent = OracleAgent(config=config, server_client=MagicMock(spec=ServerClient))
    return agent, sandbox, connect


def seed_body(
    *,
    session="ag-1",
    rollout="r1",
    with_sandbox=True,
    timeout: float | None = 120.0,
    user: str | None = "runner",
    env: dict[str, str] | None = None,
):
    """What the harbor environment server posts: the sandbox it seeded (the solution already in it), the task
    user, the `[solution].env` it resolved on its host, and the task's `[agent].timeout_sec`."""
    body = {
        "agent_session_id": session,
        "episode_id": {"rollout_id": rollout, "attempt": 0},
        "task_id": {"taskset": "ds", "task_id": "hello"},
        "tool_accesses": [],
    }
    if with_sandbox:
        body["sandbox_access"] = {
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "sb"}},
            "workdir": "/app",
            "user": user,
            "env": {"GREETING": "hi"} if env is None else env,
        }
    if timeout is not None:
        body["agent_timeout_sec"] = timeout
    return body


RESPONSES_BODY = {"input": [{"role": "user", "content": "Create hello.txt"}]}


def solve_run(sandbox: FakeSandbox) -> dict:
    return next(call for call in sandbox.execs if "bash /solution/solve.sh" in call["command"])


def test_oracle_runs_the_solution_the_server_placed_in_the_borrowed_sandbox(monkeypatch):
    agent, sandbox, connect = make_agent(monkeypatch)
    client = TestClient(agent.setup_webserver())

    seed = client.post("/v1/agent_sessions", json=seed_body())
    assert seed.status_code == 200, seed.text
    assert connect.descriptors == [{"sandbox_id": "sb"}]

    response = client.post("/v1/responses", json=RESPONSES_BODY)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["metadata"]["oracle"] == "solved"
    assert payload["metadata"]["oracle_return_code"] == "0"
    assert payload["output"][0]["content"][0]["text"] == "Done!\n"
    # The oracle uploads nothing: the environment server put /solution in place before the session opened.
    assert sandbox.uploads == []
    # It looks for the script as the task user, where Harbor's oracle copies the solution.
    probe = sandbox.execs[0]
    assert (probe["command"], probe["user"]) == ("test -f /solution/solve.sh", "runner")
    run = solve_run(sandbox)
    assert run["command"] == "bash /solution/solve.sh"
    # The budget is the seed's agent_timeout_sec.
    assert run["timeout_s"] == 120.0
    assert (run["cwd"], run["user"]) == ("/app", "runner")
    # The `[solution].env` the server resolved travels in the sandbox access, under Harbor's oracle default.
    assert run["env"] == {"DEBIAN_FRONTEND": "noninteractive", "GREETING": "hi"}
    assert run["preserve_background_services"] is True

    close = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
    )
    assert close.status_code == 200 and sandbox.disconnected


def test_oracle_reads_no_task_folder():
    """The task, its digest and its solution are the environment server's; the oracle has no path to them."""
    assert "environment_server" not in OracleAgentConfig.model_fields
    assert set(OracleAgentConfig.model_fields) - set(module.BaseResponsesAPIAgentConfig.model_fields) == {
        "max_solve_timeout_s",
        "output_tail_chars",
    }
    assert not any(name in dir(module) for name in ("load_task", "HarborTask", "upload_dir", "Path"))


def test_task_without_solution_is_unvalidated_not_failed(monkeypatch):
    """The server uploaded nothing for a task without `solution/`, so the script is absent from the sandbox."""
    sandbox = FakeSandbox(solution=False)
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"] == {"oracle": "unvalidated"}
    assert "solve.sh" in payload["output"][0]["content"][0]["text"]
    # Only the probe ran: nothing was executed or uploaded.
    assert [call["command"] for call in sandbox.execs] == ["test -f /solution/solve.sh"]
    assert sandbox.uploads == []


def test_sandbox_default_user_runs_the_solution_when_the_task_names_none(monkeypatch):
    agent, sandbox, _ = make_agent(monkeypatch)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body(user=None)).status_code == 200
    assert client.post("/v1/responses", json=RESPONSES_BODY).json()["metadata"]["oracle"] == "solved"
    assert sandbox.execs[0]["user"] is None and solve_run(sandbox)["user"] is None


def test_no_solution_env_means_only_the_oracle_default(monkeypatch):
    agent, sandbox, _ = make_agent(monkeypatch)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body(env={})).status_code == 200
    assert client.post("/v1/responses", json=RESPONSES_BODY).status_code == 200
    assert solve_run(sandbox)["env"] == {"DEBIAN_FRONTEND": "noninteractive"}


def test_failing_solution_is_reported(monkeypatch):
    sandbox = FakeSandbox(result=SandboxExecResult(stdout="", stderr="boom", return_code=2))
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"]["oracle"] == "failed"
    assert payload["metadata"]["oracle_return_code"] == "2"
    assert "boom" in payload["output"][0]["content"][0]["text"]


def test_seed_requires_sandbox_access(monkeypatch):
    agent, _, _ = make_agent(monkeypatch)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body(with_sandbox=False)).status_code == 422


def test_seed_requires_the_task_agent_timeout(monkeypatch):
    """The solve budget is the seed's agent_timeout_sec; without it the oracle has no task to read one from."""
    agent, _, _ = make_agent(monkeypatch)
    client = TestClient(agent.setup_webserver())
    response = client.post("/v1/agent_sessions", json=seed_body(timeout=None))
    assert response.status_code == 422
    assert "agent_timeout_sec" in response.text


def test_responses_requires_seeded_session(monkeypatch):
    agent, _, _ = make_agent(monkeypatch)
    assert TestClient(agent.setup_webserver()).post("/v1/responses", json=RESPONSES_BODY).status_code == 404


def test_legacy_run_route_is_not_supported(monkeypatch):
    agent, _, _ = make_agent(monkeypatch)
    response = TestClient(agent.setup_webserver()).post("/run", json={"responses_create_params": {"input": []}})
    assert response.status_code == 501


@pytest.mark.parametrize("cap,expected", [(None, 7200.0), (8000.0, 7200.0), (60.0, 60.0)])
def test_solve_uses_the_seeds_timeout_unless_explicitly_capped(monkeypatch, cap, expected):
    agent, sandbox, _ = make_agent(monkeypatch)
    if cap is not None:
        agent.config.max_solve_timeout_s = cap
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body(timeout=7200.0)).status_code == 200
    assert client.post("/v1/responses", json=RESPONSES_BODY).status_code == 200
    assert solve_run(sandbox)["timeout_s"] == expected


def test_transient_disconnect_is_retried_by_base_session_lifecycle(monkeypatch):
    agent, sandbox, _ = make_agent(monkeypatch)
    attempts = []

    async def disconnect():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("transient disconnect")
        sandbox.disconnected = True

    monkeypatch.setattr(sandbox, "disconnect", disconnect)
    client = TestClient(agent.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    body = {"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}}
    assert client.post("/v1/agent_sessions/close", json=body).status_code == 500
    assert client.post("/v1/agent_sessions/close", json=body).status_code == 200
    assert len(attempts) == 2 and sandbox.disconnected


def test_provider_error_in_the_probe_is_not_scored(monkeypatch):
    """A lost sandbox answering the solve.sh probe propagates (a retryable failed episode), never `unvalidated`."""

    class Lost(FakeSandbox):
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None, **kwargs):
            await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user, **kwargs)
            return SandboxExecResult(stdout="", stderr="container gone", return_code=1, error_type="sandbox")

    sandbox = Lost()
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    response = client.post("/v1/responses", json=RESPONSES_BODY)
    assert response.status_code == 500, response.text
    assert [call["command"] for call in sandbox.execs] == ["test -f /solution/solve.sh"]
