# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Auto Ontology Agent Server's session contract, with a fake sandbox."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.agents import AgentSeedSessionRequest, AgentSessionConflictError, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import (
    DirectSandboxConnection,
    SandboxAccess,
    SandboxExecResult,
    SandboxSpec,
)
from responses_api_agents.auto_ontology_agent import app as agent_app
from responses_api_agents.auto_ontology_agent.app import (
    SANDBOX_RUNNER,
    AutoOntologyAgent,
    AutoOntologyAgentConfig,
    AutoOntologySessionState,
    answer_message,
)
from responses_api_agents.auto_ontology_agent.episode_runner import MODEL_SERVER_API_KEY, episode_environment


MODEL_URL = "http://10.0.0.1:9000/ng-rollout/rollout/v1"
TASK = {
    "question": "How many schools?",
    "database": "california_schools",
    "evidence": "school = row",
}


def _agent(**kwargs: Any) -> AutoOntologyAgent:
    config = AutoOntologyAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="auto_ontology_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        sandbox_provider="auto_ontology_sandbox",
        sandbox_config={"image": "nemo-gym/bird-catalog:dev"},
        environment={"DEFAULT_MODELS_API_KEY": "embed-key"},
        **kwargs,
    )
    return AutoOntologyAgent(config=config, server_client=MagicMock(spec=ServerClient))


def _seed(**kwargs: Any) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="auto_ontology_react", task_id="bird-1"),
        **{"agent_context": {"values": TASK}, **kwargs},
    )


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """The sandbox the agent starts: its catalog starts, and nothing else runs in it."""
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    monkeypatch.setattr(AutoOntologyAgent, "resolve_model_base_url", lambda *args, **kwargs: MODEL_URL)
    monkeypatch.setattr(
        agent_app,
        "get_global_config_dict",
        lambda: {"auto_ontology_sandbox": {"docker": {}}},
    )
    monkeypatch.setattr(agent_app, "resolve_provider_config", lambda ref, config: config[ref])
    monkeypatch.setattr(agent_app, "create_provider", lambda config: AsyncMock())
    monkeypatch.setattr(agent_app, "AsyncSandbox", MagicMock(return_value=sandbox))
    return sandbox


@pytest.mark.parametrize("agent_context", [{"values": TASK}, None], ids=["with the task", "without a task"])
def test_a_seed_with_a_task_sandbox_is_refused(agent_context: Any) -> None:
    seed = _seed(
        agent_context=agent_context,
        sandbox_access=SandboxAccess(
            connection=DirectSandboxConnection(provider_config_ref="runtime", descriptor={"sandbox_id": "s"}),
            workdir="/app",
        ),
    )
    # The refusal the Environment Server compliance probe (rule A4) reads, whatever the context holds.
    assert "does not support sandbox access" in check_agent_seed_rejected(_agent().setup_webserver(), seed)


def test_a_seed_without_the_task_is_refused() -> None:
    seed = _seed(agent_context=None)
    assert "question: Field required" in check_agent_seed_rejected(_agent().setup_webserver(), seed)


def test_a_seed_with_an_unknown_task_key_is_refused() -> None:
    seed = _seed(agent_context={"values": {**TASK, "schema": "CREATE TABLE t"}})
    assert "schema" in check_agent_seed_rejected(_agent().setup_webserver(), seed)


async def test_seed_starts_the_image_and_its_catalog_and_close_stops_it(
    sandbox: AsyncMock,
) -> None:
    agent = _agent()

    state = await agent._seed_agent_session_state(_seed())

    assert state.task.database == "california_schools"
    [spec] = sandbox.start.await_args.args
    assert isinstance(spec, SandboxSpec)
    assert spec.image == "nemo-gym/bird-catalog:dev"
    assert "def run_episode" not in spec.files[SANDBOX_RUNNER] and "run_episode(" in spec.files[SANDBOX_RUNNER]
    # Long enough for setup, the episode and its close.
    assert spec.ttl_s == 300 + 1800 + 2 * 60
    [start] = sandbox.exec.await_args_list
    assert start.args == ("/opt/bird-catalog/start-catalog",)
    assert start.kwargs["timeout_s"] == 300
    assert 20000 <= int(start.kwargs["env"]["POSTGRES_PORT"]) < 60000
    assert state.catalog_port == int(start.kwargs["env"]["POSTGRES_PORT"])
    assert state.session.owns_sandbox

    closed = await agent._close_agent_session_state(state)

    sandbox.stop.assert_awaited_once()
    assert closed.cleanup_confirmed is True
    # No episode ran, so nothing was captured.
    assert "observation_capture_failed" in {gap.code for gap in closed.agent_observations.gaps}


async def test_a_catalog_that_does_not_start_fails_the_seed_and_stops_the_sandbox(
    sandbox: AsyncMock,
) -> None:
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="no PGDATA", return_code=1)

    with pytest.raises(RuntimeError, match="no PGDATA"):
        await _agent()._seed_agent_session_state(_seed())

    # Each attempt takes another port before the seed gives up.
    starts = [c for c in sandbox.exec.await_args_list if c.args == ("/opt/bird-catalog/start-catalog",)]
    assert len(starts) == 3
    sandbox.stop.assert_awaited_once()


async def test_a_taken_port_is_retried_on_another(sandbox: AsyncMock) -> None:
    sandbox.exec.side_effect = [
        SandboxExecResult(stdout="", stderr="could not bind: address in use", return_code=1),
        SandboxExecResult(stdout="", stderr="", return_code=0),
    ]

    state = await _agent()._seed_agent_session_state(_seed())

    ports = [int(c.kwargs["env"]["POSTGRES_PORT"]) for c in sandbox.exec.await_args_list]
    assert len(ports) == 2
    assert state.catalog_port == ports[1]


class _FakeSession:
    """A sandbox session that stages the activation, then collects ``result`` as the runner would write it."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.session_dir = "/tmp/auto-ontology-sessions/s1"
        self.sandbox = object()
        self.result = result
        self.uploads: dict[str, str] = {}
        self.argv: list[str] = []
        self.runs = 0

    async def execute(self, *, stage_activation, collect, timeout, close_timeout):  # noqa: ANN001
        self.runs += 1
        self.argv = (await stage_activation()).argv
        return await collect()

    async def read_output_log(self) -> str:
        return ""


@pytest.fixture
def staged(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    files: dict[str, str] = {}

    async def upload_text(sandbox: object, *, path: str, text: str) -> None:
        files[path] = text

    async def read_text(sandbox: object, *, path: str) -> str:
        return files[path]

    monkeypatch.setattr(agent_app, "upload_text", upload_text)
    monkeypatch.setattr(agent_app, "read_text", read_text)
    return files


@pytest.fixture
def activate(monkeypatch: pytest.MonkeyPatch, staged: dict[str, str]):  # noqa: ANN201
    """Seat ``session`` in the agent's one seeded session, as the seed would have."""

    def seat(agent: AutoOntologyAgent, session: _FakeSession) -> AutoOntologySessionState:
        seed = _seed()
        state = AutoOntologySessionState(
            request=seed,
            task=agent.parse_agent_context(seed),
            session=session,  # type: ignore[arg-type]
            catalog_port=34567,
        )
        monkeypatch.setattr(
            AutoOntologyAgent,
            "resolve_model_base_url",
            lambda *args, **kwargs: MODEL_URL,
        )
        monkeypatch.setattr(AutoOntologyAgent, "_require_agent_session", lambda self, session_id: state)
        monkeypatch.setattr(
            AutoOntologyAgent,
            "_agent_session_id_from_request",
            lambda self, request: "session",
        )
        staged[f"{session.session_dir}/result.json"] = json.dumps(session.result)
        return state

    return seat


def _request(rollout_id: str = "rollout") -> SimpleNamespace:
    return SimpleNamespace(path_params={"rollout_id": rollout_id})


async def test_the_activation_runs_one_episode_and_answers_with_its_query(activate, staged: dict[str, str]) -> None:  # noqa: ANN001
    agent = _agent()
    session = _FakeSession({"sql": "SELECT COUNT(*) FROM schools ", "reason": "submitted", "tool_calls": 5})
    activate(agent, session)
    body = NeMoGymResponseCreateParamsNonStreaming(input="ignored: the agent writes its own prompts")

    reply = await agent.responses(_request(), body)
    again = await agent.responses(_request(), body)

    assert session.runs == 1
    assert reply.output_text == "```sql\nSELECT COUNT(*) FROM schools\n```"
    assert again.output_text == reply.output_text
    assert reply.metadata["auto_ontology_end_reason"] == "submitted"
    task_path = f"{session.session_dir}/task.json"
    assert session.argv == [
        "/opt/auto-ontology/.venv/bin/python",
        SANDBOX_RUNNER,
        task_path,
        f"{session.session_dir}/result.json",
    ]
    task = json.loads(staged[task_path])
    assert task == {
        **TASK,
        "model": "openai/policy",
        "model_base_url": MODEL_URL,
        "env_file": "/opt/bird-catalog/.env",
        "environment": {
            "DEFAULT_MODELS_API_KEY": "embed-key",
            "POSTGRES_PORT": "34567",
        },
    }


async def test_a_different_second_activation_is_refused(activate) -> None:  # noqa: ANN001
    agent = _agent()
    activate(agent, _FakeSession({"sql": "SELECT 1", "reason": "submitted"}))
    await agent.responses(_request(), NeMoGymResponseCreateParamsNonStreaming(input="first"))

    with pytest.raises(AgentSessionConflictError) as refused:
        await agent.responses(_request(), NeMoGymResponseCreateParamsNonStreaming(input="second"))
    assert refused.value.status_code == 409


async def test_an_activation_on_another_rollout_is_refused(activate) -> None:  # noqa: ANN001
    agent = _agent()
    session = _FakeSession({"sql": "SELECT 1", "reason": "submitted"})
    activate(agent, session)

    with pytest.raises(AgentSessionConflictError) as refused:
        await agent.responses(_request("other"), NeMoGymResponseCreateParamsNonStreaming(input="x"))
    assert refused.value.status_code == 409
    assert session.runs == 0


async def test_close_cancels_a_running_episode(sandbox: AsyncMock) -> None:
    agent = _agent()
    state = await agent._seed_agent_session_state(_seed())
    state.activation = asyncio.create_task(asyncio.sleep(3600))

    await agent._close_agent_session_state(state)
    await asyncio.sleep(0)

    assert state.activation.cancelled()
    sandbox.stop.assert_awaited_once()


def test_an_episode_without_a_query_says_why() -> None:
    message = answer_message({"sql": None, "reason": "budget_exhausted"})
    assert message.content[0].text == "Auto Ontology submitted no query (budget_exhausted)."


def test_the_episode_runs_on_the_policy_whatever_the_settings_say(tmp_path) -> None:
    """The image's settings, then the agent's, reach Auto Ontology, but never replace its model with another."""
    env_file = tmp_path / ".env"
    env_file.write_text("CONNECTION_STRINGS=sqlite:////db.sqlite\nEMBED_MODEL=embed-a\nREASONING_MODEL=from-image\n")

    environment = episode_environment(
        {
            "model_base_url": MODEL_URL,
            "model": "openai/policy",
            "env_file": str(env_file),
            "environment": {"REASONING_MODEL": "azure/other", "EMBED_MODEL": "embed-b", "DEFAULT_MODELS_API_KEY": "k"},
        }
    )

    assert environment == {
        "CONNECTION_STRINGS": "sqlite:////db.sqlite",
        "EMBED_MODEL": "embed-b",
        "DEFAULT_MODELS_API_KEY": "k",
        "REASONING_ENDPOINT": MODEL_URL,
        "REASONING_MODEL": "openai/policy",
        "REASONING_API_KEY": MODEL_SERVER_API_KEY,
    }
