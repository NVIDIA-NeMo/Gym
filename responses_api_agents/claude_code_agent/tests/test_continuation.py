# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn continuation: later activations resume the Claude Code session the first one saved."""

import io
import json
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.agents import AGENT_ACTIVATION_HEADER, AgentSeedSessionRequest, SandboxSession
from nemo_gym.client import ServerClient
from nemo_gym.trajectory import AgentInvocation
from responses_api_agents.claude_code_agent import app
from responses_api_agents.claude_code_agent import sandbox as adapter


CLAUDE_SESSION = "0b5c3f0e-claude-session"


def _seed(*, continuation: bool = True) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session",
        episode_id={"rollout_id": "rollout"},
        task_id={"taskset": "fixture", "task_id": "task"},
        sandbox_access={
            "connection": {"provider_config_ref": "sandbox", "descriptor": {"id": "borrowed"}},
            "workdir": "/app",
        },
        continuation={"mode": "resume_harness_session"} if continuation else None,
    )


def _agent(**kwargs) -> app.ClaudeCodeAgent:
    return app.ClaudeCodeAgent(
        config=app.ClaudeCodeAgentConfig(
            name="claude", host="127.0.0.1", port=8080, entrypoint="app.py", claude_code_version="2.1.108", **kwargs
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={"sandbox": {"local": {}}}),
    )


def _transcript_record(uuid: str, role: str, text: str, timestamp: str) -> dict:
    message = {"role": role, "content": [{"type": "text", "text": text}]}
    if role == "assistant":
        message |= {"id": f"msg-{uuid}", "usage": {"input_tokens": 10, "output_tokens": 3}}
    return {"type": role, "uuid": uuid, "sessionId": CLAUDE_SESSION, "timestamp": timestamp, "message": message}


def _archive(records: list[dict]) -> bytes:
    data = "\n".join(json.dumps(record) for record in records).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(f"projects/-app/{CLAUDE_SESSION}.jsonl")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _events(answer: str) -> str:
    return "\n".join(
        json.dumps(event)
        for event in (
            {"type": "system", "subtype": "init", "session_id": CLAUDE_SESSION},
            {
                "type": "assistant",
                "session_id": CLAUDE_SESSION,
                "message": {"id": "msg", "role": "assistant", "content": [{"type": "text", "text": answer}]},
            },
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "num_turns": 1,
                "session_id": CLAUDE_SESSION,
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
        )
    )


# The rollout route the Environment Server sends each activation on.
RESPONSES = f"/ng-rollout/{_seed().episode_id.capture_key}/v1/responses"


class FakeClaude:
    """Stands in for the CLI: the saved transcript grows with each run, as a resumed session's does."""

    def __init__(self) -> None:
        self.launches: list[dict] = []
        self.transcript: list[dict] = []

    async def execute(self, state: adapter.ClaudeCodeSandboxSession, payload: dict, **_: object):
        command = payload["command"]
        prompt = command[command.index("--") + 1]
        self.launches.append({"command": command, "env": payload["env"], "session_dir": state.session.session_dir})
        turn = len(self.launches)
        answer = f"answer {turn} to {prompt}"
        self.transcript += [
            _transcript_record(f"user-{turn}", "user", prompt, f"2026-01-01T00:00:0{turn}Z"),
            _transcript_record(f"assistant-{turn}", "assistant", answer, f"2026-01-01T00:00:0{turn}Z"),
        ]
        return adapter.ClaudeCodeArtifacts(events=_events(answer), transcripts=_archive(self.transcript))


@pytest.fixture
def harness(monkeypatch):
    fake = FakeClaude()

    async def execute(self, payload, *, timeout, close_timeout):
        return await fake.execute(self, payload)

    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "execute", execute)
    remote = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None)),
        disconnect=AsyncMock(),
    )
    instance = _agent()

    async def seed_state(body):
        return adapter.ClaudeCodeSandboxSession(
            request=body,
            runtime="/runtime",
            session=SandboxSession(
                sandbox=remote, session_dir="/tmp/nemo-gym-claude-sessions/s", workdir="/app", harness="Claude Code"
            ),
        )

    monkeypatch.setattr(instance, "_seed_agent_session_state", seed_state)
    return SimpleNamespace(fake=fake, agent=instance)


def _activate(client: TestClient, index: int, text: str, **body: object):
    return client.post(RESPONSES, json={"input": text, **body}, headers={AGENT_ACTIVATION_HEADER: str(index)})


def _close(client: TestClient) -> dict:
    seed = _seed()
    response = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": seed.agent_session_id, "episode_id": seed.episode_id.model_dump(mode="json")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_capabilities_name_the_resume_mode():
    capabilities = _agent()._agent_continuation_capabilities()
    assert capabilities.mode == "resume_harness_session"
    assert {"trajectory", "timing"} <= set(capabilities.observations)


def test_later_activations_resume_the_saved_session(harness):
    with TestClient(harness.agent.setup_webserver()) as client:
        seeded = client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
        assert seeded.status_code == 200, seeded.text
        first = _activate(client, 1, "write notes.txt")
        second = _activate(client, 2, "append a line to notes.txt")
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        # A retried activation replays its reply instead of running Claude again.
        assert _activate(client, 2, "append a line to notes.txt").json() == second.json()
        closed = _close(client)

    launches = harness.fake.launches
    assert len(launches) == 2
    assert "--resume" not in launches[0]["command"]
    resume = launches[1]["command"].index("--resume")
    assert launches[1]["command"][resume + 1] == CLAUDE_SESSION
    # The saved session lives in one config directory; each launch has its own supervisor directory.
    assert launches[0]["env"]["CLAUDE_CONFIG_DIR"] == launches[1]["env"]["CLAUDE_CONFIG_DIR"]
    assert launches[0]["env"]["CLAUDE_CONFIG_DIR"] == "/tmp/nemo-gym-claude-sessions/s/config"
    assert launches[0]["session_dir"] != launches[1]["session_dir"]

    # Each reply carries only its own activation's output.
    texts = [item["content"][0]["text"] for item in second.json()["output"] if item["type"] == "message"]
    assert texts == ["answer 2 to append a line to notes.txt"]
    assert second.json()["metadata"]["activation_id"] == "2"
    assert second.json()["metadata"]["harness_session_id"] == CLAUDE_SESSION

    # Close returns one invocation per activation, each holding its own conversation, and the cumulative trajectory.
    roots = [
        AgentInvocation.model_validate(record)
        for record in closed["agent_observations"]["records"]
        if record["kind"] == "agent_invocation"
    ]
    assert [root.activation_id for root in roots] == [1, 2]
    assert {root.harness_session_id for root in roots} == {CLAUDE_SESSION}
    assert len({root.invocation_id for root in roots}) == 2
    second_conversation = json.dumps(roots[1].model_dump(mode="json")["conversation"])
    assert "append a line" in second_conversation
    assert "write notes.txt" not in second_conversation
    trajectory = closed["trajectory"]
    assert [turn["activation_id"] for turn in trajectory["turns"]] == [1, 2]
    assert [turn["question"] for turn in trajectory["turns"]] == ["write notes.txt", "append a line to notes.txt"]
    assert [invocation["invocation_id"] for invocation in trajectory["invocations"]] == [
        root.invocation_id for root in roots
    ]


def test_an_activation_cannot_replace_the_system_prompt(harness):
    with TestClient(harness.agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
        system = {"role": "system", "content": "Be brief."}
        first = _activate(client, 1, None, input=[system, {"role": "user", "content": "one"}])
        assert first.status_code == 200, first.text
        replaced = _activate(
            client, 2, None, input=[{"role": "system", "content": "Be verbose."}, {"role": "user", "content": "two"}]
        )
        assert replaced.status_code == 422
        # Without a system message, the next activation keeps the session's.
        kept = _activate(client, 2, "two")
        assert kept.status_code == 200, kept.text
        _close(client)
    second = harness.fake.launches[1]["command"]
    assert second[second.index("--append-system-prompt") + 1] == "Be brief."


def test_a_failed_preparation_still_reports_its_activation(harness, monkeypatch):
    async def fail(self, activation_id, *, timeout):
        raise RuntimeError("Cannot prepare Claude activation directory")

    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "prepare_activation", fail)
    with TestClient(harness.agent.setup_webserver(), raise_server_exceptions=False) as client:
        client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
        assert _activate(client, 1, "write notes.txt").status_code == 500
        closed = _close(client)
    assert harness.fake.launches == []
    roots = [record for record in closed["agent_observations"]["records"] if record["kind"] == "agent_invocation"]
    assert [(root["activation_id"], root["status"], root["error_type"]) for root in roots] == [
        (1, "failed", "sandbox_error")
    ]


def test_a_session_without_continuation_still_allows_one_activation(harness):
    with TestClient(harness.agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=_seed(continuation=False).model_dump(mode="json"))
        assert client.post(RESPONSES, json={"input": "one"}).status_code == 200
        assert client.post(RESPONSES, json={"input": "two"}).status_code == 409
