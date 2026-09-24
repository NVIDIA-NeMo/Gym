# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from nemo_gym.tasks.harbor import load_task
from responses_api_agents.oracle_agent import app as module
from responses_api_agents.oracle_agent.app import OracleAgent, OracleAgentConfig


TASK_TOML = """
schema_version = "1.4"

[agent]
timeout_sec = 120.0
user = "runner"

[solution.env]
GREETING = "hi"

[environment]
cpus = 1
"""


def write_task(root: Path, *, solution: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(TASK_TOML)
    (root / "instruction.md").write_text("Create hello.txt\n")
    (root / "environment").mkdir(exist_ok=True)
    (root / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app")
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test.sh").write_text("#!/bin/bash\n")
    if solution:
        (root / "solution").mkdir(exist_ok=True)
        (root / "solution" / "solve.sh").write_text("#!/bin/bash\necho hi > hello.txt\n")
    return root


@dataclass
class FakeSandbox:
    result: SandboxExecResult = SandboxExecResult(stdout="Done!\n", stderr="", return_code=0)
    execs: list[dict] = field(default_factory=list)
    uploads: list[str] = field(default_factory=list)
    disconnected: bool = False

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.execs.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user})
        if "solve.sh" in command:
            return self.result
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload(self, local_path, remote_path):
        self.uploads.append(remote_path)

    async def disconnect(self):
        self.disconnected = True


def make_agent(tmp_path: Path, monkeypatch: MonkeyPatch, *, solution: bool = True, sandbox: FakeSandbox | None = None):
    folder = tmp_path / "datasets" / "ds"
    task = load_task(write_task(folder / "hello", solution=solution))
    global_config = {
        "sandbox": {"opensandbox": {"connection": {"domain": "example.invalid"}}},
        "harbor_rs": {
            "resources_servers": {
                "harbor": {"tasksets": {"ds": {"folder": str(folder), "tasks": {"hello": task.digest}}}}
            }
        },
    }
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
        name="oracle_agent",
        resources_server={"type": "resources_servers", "name": "harbor_rs"},
    )
    agent = OracleAgent(config=config, server_client=MagicMock(spec=ServerClient))
    return agent, task, sandbox, connect


def seed_body(*, session="ag-1", rollout="r1", with_sandbox=True):
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
        }
    return body


RESPONSES_BODY = {"input": [{"role": "user", "content": "Create hello.txt"}]}


def test_oracle_runs_solution_in_borrowed_sandbox(tmp_path, monkeypatch):
    agent, task, sandbox, connect = make_agent(tmp_path, monkeypatch)
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
    assert any(remote.endswith(".tar.gz") for remote in sandbox.uploads)
    run = next(call for call in sandbox.execs if "solve.sh" in call["command"])
    assert run == {
        "command": "bash /solution/solve.sh",
        "cwd": "/app",
        "env": {"GREETING": "hi"},
        "timeout_s": 120.0,
        "user": "runner",
    }

    close = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
    )
    assert close.status_code == 200 and sandbox.disconnected


def test_task_without_solution_is_unvalidated_not_failed(tmp_path, monkeypatch):
    agent, _, sandbox, _ = make_agent(tmp_path, monkeypatch, solution=False)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"] == {"oracle": "unvalidated"}
    assert sandbox.execs == [] and sandbox.uploads == []


def test_failing_solution_is_reported(tmp_path, monkeypatch):
    sandbox = FakeSandbox(result=SandboxExecResult(stdout="", stderr="boom", return_code=2))
    agent, _, _, _ = make_agent(tmp_path, monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"]["oracle"] == "failed"
    assert payload["metadata"]["oracle_return_code"] == "2"
    assert "boom" in payload["output"][0]["content"][0]["text"]


def test_changed_folder_is_rejected(tmp_path, monkeypatch):
    agent, task, _, _ = make_agent(tmp_path, monkeypatch)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    (task.path / "solution" / "solve.sh").write_text("#!/bin/bash\necho changed\n")

    assert client.post("/v1/responses", json=RESPONSES_BODY).status_code == 409


def test_seed_requires_sandbox_access(tmp_path, monkeypatch):
    agent, _, _, _ = make_agent(tmp_path, monkeypatch)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body(with_sandbox=False)).status_code == 422


def test_responses_requires_seeded_session(tmp_path, monkeypatch):
    agent, _, _, _ = make_agent(tmp_path, monkeypatch)
    assert TestClient(agent.setup_webserver()).post("/v1/responses", json=RESPONSES_BODY).status_code == 404


def test_session_identity_and_close_rules(tmp_path, monkeypatch):
    agent, _, sandbox, connect = make_agent(tmp_path, monkeypatch)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    # Re-seeding is idempotent; another episode cannot take the id.
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    assert len(connect.descriptors) == 1
    assert client.post("/v1/agent_sessions", json=seed_body(rollout="other")).status_code == 409
    # Closing with the wrong episode is refused and keeps the session.
    wrong = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "other", "attempt": 0}},
    )
    assert wrong.status_code == 409 and not sandbox.disconnected
    right = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
    )
    assert right.status_code == 200 and sandbox.disconnected
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 409


def test_legacy_run_route_is_not_supported(tmp_path, monkeypatch):
    agent, _, _, _ = make_agent(tmp_path, monkeypatch)
    response = TestClient(agent.setup_webserver()).post("/run", json={"responses_create_params": {"input": []}})
    assert response.status_code == 501


@pytest.mark.parametrize("taskset,task_id,status", [("nope", "hello", 404), ("ds", "missing", 404)])
def test_unknown_task_lookup(tmp_path, monkeypatch, taskset, task_id, status):
    agent, _, _, _ = make_agent(tmp_path, monkeypatch)
    client = TestClient(agent.setup_webserver())
    body = seed_body()
    body["task_id"] = {"taskset": taskset, "task_id": task_id}
    assert client.post("/v1/agent_sessions", json=body).status_code == 200
    assert client.post("/v1/responses", json=RESPONSES_BODY).status_code == status
