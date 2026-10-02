# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode sessions: the environment server seeds a borrowed sandbox and calls /v1/responses."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import ServerClient
from responses_api_agents.miniswe_sandboxed_agent import app as module
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessOutcome


TASK_CONTEXT = {"mcp_servers": [{"name": "fs", "url": "http://localhost:7000/mcp"}], "skills_dir": "/skills"}


class FakeHarness:
    """Records how the agent built it and returns a canned run."""

    instances: list["FakeHarness"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.budget = None
        self.closed = False
        FakeHarness.instances.append(self)

    async def setup(self) -> None:
        pass

    async def execute(self, budget):
        self.budget = budget
        extra = {"ng_agent_observations": {"source": "miniswe", "records": []}}
        return response_stub(), HarnessOutcome(reason="completed"), extra

    async def close(self) -> None:
        self.closed = True


def make_agent(tmp_path, monkeypatch, *, task_context=TASK_CONTEXT):
    FakeHarness.instances.clear()
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    agent = module.MiniSWESandboxedAgent(
        config=module.MiniSWESandboxedConfig(
            host="localhost",
            port=1,
            name="miniswe_sandboxed_agent",
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            artifacts_dir=tmp_path,
            agent_timeout_sec=120,
        ),
        server_client=server_client,
    )
    provider = SimpleNamespace(aclose=AsyncMock())

    async def exec_(command, **_kwargs):
        if command.startswith(f"cat {module.TASK_CONTEXT_FILE}"):
            return SimpleNamespace(return_code=0, stdout=json.dumps(task_context) if task_context else "")
        return SimpleNamespace(return_code=0, stdout="/app\n")

    sandbox = SimpleNamespace(exec=AsyncMock(side_effect=exec_), disconnect=AsyncMock())
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}})
    monkeypatch.setattr(module, "resolve_provider_config", lambda ref, config: {"opensandbox": {}})
    monkeypatch.setattr(module, "create_provider", MagicMock(return_value=provider))
    monkeypatch.setattr(module, "get_server_url", lambda name: "http://policy:1")
    monkeypatch.setattr(module, "MiniSWEHarness", FakeHarness)
    monkeypatch.setattr(type(agent), "base_url_for_run", lambda self, *, base_url, body: base_url)
    connect = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(module.AsyncSandbox, "connect", connect)
    return agent, TestClient(agent.setup_webserver()), sandbox, provider, connect


def seed_body(*, session="ag-1", rollout="r1", with_sandbox=True, workdir="/app"):
    body = {
        "agent_session_id": session,
        "episode_id": {"rollout_id": rollout, "attempt": 0},
        "task_id": {"taskset": "tb4", "task_id": "atrx-vep-crispr"},
        "tool_accesses": [],
    }
    if with_sandbox:
        body["sandbox_access"] = {
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "sb"}},
            "workdir": workdir,
        }
    return body


def response_stub(text="done"):
    return module.NeMoGymResponse(
        id="resp",
        created_at=0,
        model="policy_model",
        object="response",
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        tool_choice="auto",
        tools=[],
        parallel_tool_calls=False,
    )


def capture_path(rollout="r1"):
    return f"/ng-rollout/{EpisodeId(rollout_id=rollout, attempt=0).capture_key}/v1/responses"


def test_seed_connects_and_the_turn_runs_the_harness_on_that_sandbox(tmp_path, monkeypatch):
    agent, client, sandbox, provider, connect = make_agent(tmp_path, monkeypatch)

    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    connect.assert_awaited_once()
    assert connect.await_args.args[0] == {"sandbox_id": "sb"}

    response = client.post(capture_path(), json={"input": [{"role": "user", "content": "Write the report."}]})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["output"][0]["content"][0]["text"] == "done"
    assert payload["metadata"]["miniswe_termination"] == "completed"
    assert payload["metadata"]["miniswe_agent_started"] == "true"

    (harness,) = FakeHarness.instances
    assert harness.kwargs["sandbox"] is sandbox
    context = harness.kwargs["context"]
    assert (context.session_id, context.task_id, context.instruction, context.workdir) == (
        "ag-1",
        "atrx-vep-crispr",
        "Write the report.",
        "/app",
    )
    # The instruction is the turn's input; the tool grants come from the resources server's task.json.
    assert context.mcp_servers == TASK_CONTEXT["mcp_servers"] and context.skills_dir == "/skills"
    assert context.rollout_id == EpisodeId(rollout_id="r1", attempt=0).capture_key
    assert harness.kwargs["model_base_url"] == "http://policy:1/v1"
    assert harness.kwargs["directory"] == tmp_path / "ag-1"
    assert 0 < harness.budget <= 120
    # The harness is not closed after a started run; the sandbox outlives the agent until close.
    assert not harness.closed
    # `pwd` is never needed when the resources server hands over the working directory.
    assert all(not call.args[0].strip() == "pwd" for call in sandbox.exec.await_args_list)

    close = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
    )
    assert close.status_code == 200, close.text
    assert close.json()["agent_observations"]["source"] == "miniswe"
    sandbox.disconnect.assert_awaited_once()
    provider.aclose.assert_awaited_once()


def test_plain_responses_path_binds_the_only_session_and_disables_capture(tmp_path, monkeypatch):
    agent, client, _, _, _ = make_agent(tmp_path, monkeypatch, task_context=None)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    response = client.post("/v1/responses", json={"input": "Do it"})
    assert response.status_code == 200, response.text
    (harness,) = FakeHarness.instances
    assert harness.kwargs["context"].instruction == "Do it"
    assert harness.kwargs["context"].mcp_servers == [] and harness.kwargs["context"].skills_dir is None
    key = next(iter(agent._sessions))
    assert agent._sessions[key].capture_model_calls is False


def test_a_second_different_turn_on_the_same_session_is_refused(tmp_path, monkeypatch):
    _, client, _, _, _ = make_agent(tmp_path, monkeypatch)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    assert client.post(capture_path(), json={"input": "first"}).status_code == 200
    assert client.post(capture_path(), json={"input": "second"}).status_code == 409


def test_session_identity_rules(tmp_path, monkeypatch):
    _, client, sandbox, _, connect = make_agent(tmp_path, monkeypatch)
    assert client.post("/v1/responses", json={"input": "x"}).status_code == 409
    assert client.post("/v1/agent_sessions", json=seed_body(with_sandbox=False)).status_code == 422
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200  # idempotent
    assert connect.await_count == 1
    assert client.post("/v1/agent_sessions", json=seed_body(rollout="other")).status_code == 409
    wrong = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "other", "attempt": 0}},
    )
    assert wrong.status_code == 409 and not sandbox.disconnect.await_count
    right = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
    )
    assert right.status_code == 200
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 409


def test_run_without_a_resources_server_is_refused(tmp_path, monkeypatch):
    _, client, _, _, _ = make_agent(tmp_path, monkeypatch)
    response = client.post("/run", json={"responses_create_params": {"input": []}, "rollout_id": "r"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "value,expected",
    [
        ("plain", "plain"),
        (
            [{"role": "user", "content": "a"}, {"role": "user", "content": [{"type": "input_text", "text": "b"}]}],
            "a\n\nb",
        ),
    ],
)
def test_instruction_from_input(value, expected):
    assert module._instruction(value) == expected
