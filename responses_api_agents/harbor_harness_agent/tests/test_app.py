# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.base_responses_api_agent import AgentCloseSessionResponse, AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import ServerClient
from responses_api_agents.harbor_harness_agent import app as agent_app
from responses_api_agents.harbor_harness_agent.app import HarborHarnessAgent, HarborHarnessAgentConfig
from responses_api_agents.harbor_harness_agent.tests.fake_harbor_agent import FakeHarborAgent


FAKE_AGENT = "responses_api_agents.harbor_harness_agent.tests.fake_harbor_agent:FakeHarborAgent"


class BorrowedSandbox:
    """A connected sandbox: commands succeed and directory downloads yield an empty archive."""

    instances: list["BorrowedSandbox"] = []

    def __init__(self, descriptor: dict) -> None:
        self.descriptor = descriptor
        self.commands: list[tuple[str, str | None]] = []
        self.disconnected = False
        self.stopped = False
        BorrowedSandbox.instances.append(self)

    @classmethod
    async def connect(cls, descriptor: dict, provider: object) -> "BorrowedSandbox":
        return cls(descriptor)

    async def exec(self, command: str, **kwargs):
        self.commands.append((command, kwargs.get("user")))
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    async def download(self, remote_path: str, local_path: Path) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz"):
            pass
        Path(local_path).write_bytes(buffer.getvalue())

    async def disconnect(self) -> None:
        self.disconnected = True

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture(autouse=True)
def borrowed_sandbox(monkeypatch: MonkeyPatch) -> type[BorrowedSandbox]:
    BorrowedSandbox.instances = []
    FakeHarborAgent.instances = []
    monkeypatch.setattr(agent_app, "AsyncSandbox", BorrowedSandbox)
    monkeypatch.setattr(agent_app, "create_provider", lambda config: object())
    monkeypatch.setattr(agent_app, "resolve_provider_config", lambda ref, global_config: {"docker": {}})
    monkeypatch.setattr(agent_app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(
        HarborHarnessAgent,
        "resolve_model_base_url",
        lambda self, name, rollout_id=None: f"http://model/ng-rollout/{rollout_id}/v1",
    )
    return BorrowedSandbox


def make_agent(tmp_path: Path, **kwargs) -> HarborHarnessAgent:
    config = HarborHarnessAgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="harbor_harness_agent",
        harbor_agent={"import_path": FAKE_AGENT, "model_name": "openai/policy", "kwargs": kwargs.pop("kwargs", {})},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        logs_dir=tmp_path / "logs",
        **kwargs,
    )
    return HarborHarnessAgent(config=config, server_client=MagicMock(spec=ServerClient))


def seed_body(*, sandbox_access: bool = True) -> dict:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session-1",
        episode_id=EpisodeId(rollout_id="3-1"),
        task_id=TaskId(taskset="harbor_tasks", task_id="hello-world"),
        sandbox_access=SandboxAccess(
            connection=DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"sandbox_id": "c1"}),
            workdir="/app",
        )
        if sandbox_access
        else None,
    ).model_dump(mode="json")


def activation(metadata: dict | None = None) -> dict:
    body = {"input": [{"role": "user", "content": "Create hello.txt."}], "temperature": 0.7}
    if metadata:
        body["metadata"] = metadata
    return body


def close_body() -> dict:
    return {"agent_session_id": "agent-session-1", "episode_id": {"rollout_id": "3-1", "attempt": 0}}


def test_session_runs_the_harbor_agent_in_the_borrowed_sandbox(tmp_path: Path) -> None:
    client = TestClient(make_agent(tmp_path).setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    agent = FakeHarborAgent.instances[0]
    # Setup ran at seed, and the agent got the rollout-prefixed model URL for token capture.
    assert agent.setup_calls == 1
    assert agent.api_base == "http://model/ng-rollout/3-1/v1"

    response = client.post(
        "/ng-rollout/3-1/v1/responses",
        json=activation({"harbor_agent_user": "agent", "harbor_agent_timeout_sec": "30"}),
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "completed"
    assert [item["type"] for item in result["output"]] == ["message", "message"]
    assert result["output"][1]["content"][0]["text"] == "done"
    assert result["usage"]["input_tokens"] == 10 and result["usage"]["output_tokens"] == 5
    assert result["temperature"] == 0.7
    assert agent.runs == [("Create hello.txt.", "agent")]

    sandbox = BorrowedSandbox.instances[0]
    assert ("bash -c 'echo run'", "agent") in sandbox.commands
    closed = AgentCloseSessionResponse.model_validate(
        client.post("/v1/agent_sessions/close", json=close_body()).json()
    )
    assert closed.agent_observations.source == "harbor:fake-harbor-agent"
    # Resources owns the sandbox: the agent disconnects and never stops it.
    assert sandbox.disconnected and not sandbox.stopped
    assert (tmp_path / "logs" / "3-1" / "agent" / "trajectory.json").is_file()


def test_timed_out_agent_returns_an_incomplete_response_so_verification_still_runs(tmp_path: Path) -> None:
    client = TestClient(make_agent(tmp_path, kwargs={"behavior": "hang"}).setup_webserver())
    client.post("/v1/agent_sessions", json=seed_body())
    response = client.post("/ng-rollout/3-1/v1/responses", json=activation({"harbor_agent_timeout_sec": "0.2"}))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "incomplete"
    assert result["metadata"]["harbor_agent_exit"].startswith("AgentTimeoutError")
    closed = client.post("/v1/agent_sessions/close", json=close_body()).json()
    codes = {gap["code"] for gap in closed["agent_observations"]["gaps"]}
    assert {"agent_timeout", "agent_transcript_unavailable"} <= codes


def test_config_timeout_caps_and_scales_the_task_timeout(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, agent_timeout_multiplier=2.0)
    agent.config.harbor_agent.max_timeout_sec = 100
    assert agent._agent_timeout_seconds("600") == 200
    assert agent._agent_timeout_seconds(None) is None
    agent.config.harbor_agent.override_timeout_sec = 30
    assert agent._agent_timeout_seconds("600") == 60


def test_installed_agents_receive_the_model_url_through_environment_variables(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, model_base_url_kwarg=None, model_base_url_env=["OPENAI_BASE_URL"])
    config = agent._agent_config("3-1")
    assert "api_base" not in config.kwargs
    assert config.env["OPENAI_BASE_URL"] == "http://model/ng-rollout/3-1/v1"


def test_activation_must_match_the_seeded_rollout(tmp_path: Path) -> None:
    client = TestClient(make_agent(tmp_path).setup_webserver(), raise_server_exceptions=False)
    client.post("/v1/agent_sessions", json=seed_body())
    assert client.post("/ng-rollout/9-9/v1/responses", json=activation()).status_code == 409
    assert FakeHarborAgent.instances[0].runs == []


def test_a_second_different_activation_is_rejected(tmp_path: Path) -> None:
    client = TestClient(make_agent(tmp_path).setup_webserver(), raise_server_exceptions=False)
    client.post("/v1/agent_sessions", json=seed_body())
    assert client.post("/ng-rollout/3-1/v1/responses", json=activation()).status_code == 200
    # A lost reply is retried with the same body and gets the same response without rerunning the agent.
    assert client.post("/ng-rollout/3-1/v1/responses", json=activation()).status_code == 200
    assert len(FakeHarborAgent.instances[0].runs) == 1
    other = activation() | {"input": [{"role": "user", "content": "Something else."}]}
    assert client.post("/ng-rollout/3-1/v1/responses", json=other).status_code == 409


def test_seed_requires_sandbox_access(tmp_path: Path) -> None:
    client = TestClient(make_agent(tmp_path).setup_webserver(), raise_server_exceptions=False)
    assert client.post("/v1/agent_sessions", json=seed_body(sandbox_access=False)).status_code == 422
    assert BorrowedSandbox.instances == []


def test_responses_outside_a_session_are_rejected(tmp_path: Path) -> None:
    client = TestClient(make_agent(tmp_path).setup_webserver(), raise_server_exceptions=False)
    assert client.post("/v1/responses", json=activation()).status_code == 409
