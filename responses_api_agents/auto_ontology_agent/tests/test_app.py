# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Auto Ontology Agent Server's session contract, with a fake sandbox."""

from __future__ import annotations

import asyncio
import http.client
import io
import json
import urllib.error
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from fastapi import HTTPException

from nemo_gym.agents import (
    AgentContinuationRequirements,
    AgentSeedSessionRequest,
    AgentSessionConflictError,
    AgentSessionSetupError,
    OutcomeAction,
    agent_outcome_policy,
    check_agent_seed_rejected,
)
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
from responses_api_agents.auto_ontology_agent import episode_runner
from responses_api_agents.auto_ontology_agent.app import (
    SANDBOX_RUNNER,
    AutoOntologyAgent,
    AutoOntologyAgentConfig,
    AutoOntologyEpisodeError,
    AutoOntologySessionState,
    answer_message,
)
from responses_api_agents.auto_ontology_agent.episode_runner import (
    MODEL_SERVER_API_KEY,
    episode_environment,
    model_failure,
)
from responses_api_agents.auto_ontology_agent.tests.adk_texts import (
    C_INTERACT_INSTRUCTION,
    PHASE_1_MESSAGE,
)


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


async def test_a_failed_setup_whose_cleanup_fails_keeps_the_sandbox_for_close(sandbox: AsyncMock) -> None:
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="no PGDATA", return_code=1)
    sandbox.stop.side_effect = [RuntimeError("stop timed out"), None]
    agent = _agent()

    with pytest.raises(AgentSessionSetupError) as failed:
        await agent._seed_agent_session_state(_seed())

    # The seed's error is the catalog's, and the state keeps the sandbox that did not stop.
    assert "no PGDATA" in str(failed.value.error)
    closed = await agent._close_agent_session_state(failed.value.state)
    assert sandbox.stop.await_count == 2
    assert closed.cleanup_confirmed is True


async def test_a_sandbox_that_start_could_not_stop_is_kept_for_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """start() created the sandbox, failed to deliver the runner, and its own stop timed out, as did the seed's."""
    provider = AsyncMock()
    provider.upload_file.side_effect = RuntimeError("upload failed")
    provider.close.side_effect = [TimeoutError("stop timed out"), TimeoutError("stop timed out"), None]
    monkeypatch.setattr(AutoOntologyAgent, "resolve_model_base_url", lambda *args, **kwargs: MODEL_URL)
    monkeypatch.setattr(agent_app, "get_global_config_dict", lambda: {"auto_ontology_sandbox": {"docker": {}}})
    monkeypatch.setattr(agent_app, "resolve_provider_config", lambda ref, config: config[ref])
    monkeypatch.setattr(agent_app, "create_provider", lambda config: provider)
    agent = _agent()

    with pytest.raises(AgentSessionSetupError) as failed:
        await agent._seed_agent_session_state(_seed())

    # The runner never arrived, so the catalog did not start. The provider's client stays open for the session's
    # close, which stops the sandbox and then releases it.
    provider.upload_file.assert_awaited_once()
    assert not [c for c in provider.exec.await_args_list if "/opt/bird-catalog/start-catalog" in c.args]
    provider.aclose.assert_not_awaited()
    closed = await agent._close_agent_session_state(failed.value.state)
    assert provider.close.await_count == 3
    provider.aclose.assert_awaited_once()
    assert closed.cleanup_confirmed is True


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


async def test_an_episode_that_raised_fails_the_activation_with_its_error(activate) -> None:  # noqa: ANN001
    agent = _agent()
    activate(agent, _FakeSession({"error": "KeyError: 'No source database 'alien''"}))
    body = NeMoGymResponseCreateParamsNonStreaming(input="ignored: the agent writes its own prompts")

    with pytest.raises(AutoOntologyEpisodeError, match="No source database"):
        await agent.responses(_request(), body)


async def test_a_different_second_activation_is_refused(activate) -> None:  # noqa: ANN001
    agent = _agent()
    activate(agent, _FakeSession({"sql": "SELECT 1", "reason": "submitted"}))
    await agent.responses(_request(), NeMoGymResponseCreateParamsNonStreaming(input="first"))

    with pytest.raises(AgentSessionConflictError) as refused:
        await agent.responses(_request(), NeMoGymResponseCreateParamsNonStreaming(input="second"))
    assert refused.value.status_code == 409


@pytest.mark.parametrize(
    "fields",
    [{"max_output_tokens": 1}, {"temperature": 0.2}, {"tool_choice": "none"}, {"instructions": "Be brief."}],
)
async def test_an_activation_with_a_control_the_episode_does_not_honor_is_refused(activate, fields: dict) -> None:  # noqa: ANN001
    agent = _agent()
    session = _FakeSession({"sql": "SELECT 1", "reason": "submitted"})
    activate(agent, session)

    with pytest.raises(HTTPException) as refused:
        await agent.responses(_request(), NeMoGymResponseCreateParamsNonStreaming(input="x", **fields))
    assert refused.value.status_code == 422
    assert list(fields)[0] in refused.value.detail
    assert session.runs == 0
    # The default request, with the configured model and the stream flags off, runs.
    body = NeMoGymResponseCreateParamsNonStreaming(input="x", model="openai/policy", stream=False, metadata={"k": "v"})
    await agent.responses(_request(), body)
    assert session.runs == 1


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


@pytest.mark.parametrize(
    ("failure", "action"),
    [
        ({"status_code": 503, "no_reply": False, "error": "InternalServerError: 503"}, OutcomeAction.RETRY),
        ({"status_code": 400, "no_reply": False, "error": "BadRequestError: 400"}, OutcomeAction.VERIFY),
        ({"status_code": None, "no_reply": True, "error": "APIConnectionError: refused"}, OutcomeAction.RETRY),
    ],
    ids=["unavailable", "bad request", "no reply"],
)
async def test_a_model_call_the_loop_gave_up_on_fails_the_activation_for_the_outcome_policy(
    activate,  # noqa: ANN001
    failure: dict[str, Any],
    action: OutcomeAction,
) -> None:
    agent = _agent()
    activate(agent, _FakeSession({"sql": None, "reason": "model_error", "model_failure": failure}))

    reply = await agent.responses(_request(), NeMoGymResponseCreateParamsNonStreaming(input="question"))

    assert reply.status == "failed"
    assert reply.error.message == failure["error"]
    assert agent_outcome_policy(reply) == action


async def test_an_episode_that_submitted_nothing_is_graded(activate) -> None:  # noqa: ANN001
    agent = _agent()
    activate(agent, _FakeSession({"sql": None, "reason": "budget_exhausted", "model_failure": None}))

    reply = await agent.responses(_request(), NeMoGymResponseCreateParamsNonStreaming(input="question"))

    assert reply.status != "failed"
    assert agent_outcome_policy(reply) == OutcomeAction.VERIFY


def test_the_runner_reports_the_model_server_s_status_or_no_reply() -> None:
    request = httpx.Request("POST", MODEL_URL)
    unavailable = openai.InternalServerError("unavailable", response=httpx.Response(503, request=request), body=None)
    assert model_failure(unavailable)["status_code"] == 503
    assert model_failure(unavailable)["no_reply"] is False
    for no_reply in (openai.APIConnectionError(request=request), openai.APITimeoutError(request=request)):
        assert model_failure(no_reply) == {
            "status_code": None,
            "no_reply": True,
            "error": f"{type(no_reply).__name__}: {no_reply}",
        }


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
        # The interactive pipeline also calls Auto Ontology's non-reasoning model: it runs on the policy too.
        "NON_REASONING_ENDPOINT": MODEL_URL,
        "NON_REASONING_MODEL": "openai/policy",
        "NON_REASONING_API_KEY": MODEL_SERVER_API_KEY,
    }


# ----- c-Interact ----- #

C_INTERACT_GRANT = {
    "kind": "direct_http",
    "name": "bird_interact",
    "required": True,
    "base_url": "http://127.0.0.1:7000/episodes/e1/tools",
    "headers": {"Authorization": "Bearer episode-token"},
}
INSTRUCTION = C_INTERACT_INSTRUCTION.format(
    db_name="gym_members",
    db_schema="CREATE TABLE members (id integer);",
    external_kg=json.dumps([{"id": 1, "knowledge": "Active", "description": "Visited this year."}], indent=2),
    max_turn=5,
)
PHASE_1 = PHASE_1_MESSAGE.format(amb_user_query="Show the active members.", max_turn=5)
ASK_USER = {
    "type": "function",
    "name": "ask_user",
    "description": "Ask the user a clarification question about their query.",
    "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
    "strict": True,
}
SUBMIT_SQL = {**ASK_USER, "name": "submit_sql", "description": "Submit your final SQL query for evaluation."}


def _c_interact_seed(**kwargs: Any) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="bird_interact:c_interact", task_id="gym_members_1"),
        continuation=AgentContinuationRequirements(),
        **{"tool_accesses": [C_INTERACT_GRANT], **kwargs},
    )


def _first_turn() -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": PHASE_1}],
        tools=[ASK_USER, SUBMIT_SQL],
    )


def _turn(message: str) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": message}])


def test_a_c_interact_seed_with_a_task_context_is_refused() -> None:
    seed = _c_interact_seed(agent_context={"values": TASK})
    assert "reads its task from its requests" in check_agent_seed_rejected(_agent().setup_webserver(), seed)


def test_a_c_interact_seed_without_a_direct_http_grant_is_refused() -> None:
    seed = _c_interact_seed(tool_accesses=[])
    assert "needs one direct HTTP grant" in check_agent_seed_rejected(_agent().setup_webserver(), seed)


def test_a_c_interact_seed_with_a_required_mcp_grant_is_refused() -> None:
    mcp = {
        "kind": "mcp",
        "name": "bird_interact_mcp",
        "required": True,
        "connection": {"transport": "streamable_http", "url": "http://127.0.0.1:7000/mcp"},
    }
    seed = _c_interact_seed(tool_accesses=[C_INTERACT_GRANT, mcp])
    assert "direct HTTP only" in check_agent_seed_rejected(_agent().setup_webserver(), seed)


async def test_a_c_interact_seed_keeps_the_grant_and_lives_for_four_turns(sandbox: AsyncMock) -> None:
    state = await _agent()._seed_agent_session_state(_c_interact_seed())

    assert state.task is None
    assert str(state.tool_access.base_url) == C_INTERACT_GRANT["base_url"]
    [spec] = sandbox.start.await_args.args
    assert spec.ttl_s == 300 + 4 * 1800 + 2 * 60


class _TurnSession(_FakeSession):
    """A sandbox session that runs each turn in its own directory, collecting the next of ``results``."""

    def __init__(self, results: list[dict[str, Any]], staged: dict[str, str]) -> None:
        super().__init__({})
        self.root = "/tmp/auto-ontology-sessions/s1"
        self.results = results
        self.staged = staged
        self.turn_dirs: list[str] = []

    @property
    def root_dir(self) -> str:
        return self.root

    async def begin_activation(self, *, session_dir: str, timeout: float) -> None:
        self.session_dir = session_dir
        self.turn_dirs.append(session_dir)
        self.staged[f"{session_dir}/result.json"] = json.dumps(self.results[len(self.turn_dirs) - 1])

    cleanup_confirmed = True

    async def close(self, *, timeout: float) -> None:
        pass


def _c_interact_state(agent: AutoOntologyAgent, session: _TurnSession) -> AutoOntologySessionState:
    seed = _c_interact_seed()
    return AutoOntologySessionState(
        request=seed,
        task=None,
        session=session,  # type: ignore[arg-type]
        catalog_port=34567,
        tool_access=agent.effective_tool_accesses(seed)[0],
    )


async def test_c_interact_turns_ask_the_user_submit_through_the_server_and_remember(
    monkeypatch: pytest.MonkeyPatch, staged: dict[str, str]
) -> None:
    monkeypatch.setattr(AutoOntologyAgent, "resolve_model_base_url", lambda *args, **kwargs: MODEL_URL)
    agent = _agent()
    session = _TurnSession(
        [
            {
                "sql": "SELECT id FROM members WHERE visits > 0",
                "reason": "submitted",
                "tool_calls": 4,
                "environment_tool_calls": [
                    # Refused by the server, so not asked: it neither answers nor spends a question.
                    {
                        "name": "ask_user",
                        "arguments": {"question": ""},
                        "error": "The tool call failed with HTTP 422: question is empty",
                        "refused": True,
                    },
                    {
                        "name": "ask_user",
                        "arguments": {"question": "What makes a member active?"},
                        "reply": "A visit this year.\n\n[SYSTEM NOTE: Clarification turns remaining: 4/5]",
                    },
                ],
                "submit_reply": "SQL failed Phase 1. Your SQL is not correct.\nBudget remaining: 0 bird-coins",
            },
            {"sql": None, "reason": "budget_exhausted", "tool_calls": 30, "environment_tool_calls": []},
        ],
        staged,
    )
    state = _c_interact_state(agent, session)

    first = await agent._activate_agent_session_state(state, _first_turn(), activation_id=1)

    task = json.loads(staged[f"{session.root}/turn-1/task.json"])
    assert task["question"] == "Show the active members."
    assert task["database"] == "gym_members"
    assert task["evidence"] == "Knowledge the user shared:\n- Active: Visited this year."
    assert "You may ask 5 more questions" in task["custom_prompts"]
    endpoint = {"headers": {"Authorization": "Bearer episode-token"}, "cookies": {}}
    assert task["tools"] == [
        {
            "definition": {
                "type": "function",
                "function": {k: ASK_USER[k] for k in ("name", "description", "parameters")},
            },
            "url": f"{C_INTERACT_GRANT['base_url']}/ask_user",
            **endpoint,
        }
    ]
    assert task["submit"] == {"url": f"{C_INTERACT_GRANT['base_url']}/submit_sql", **endpoint}
    assert first.output_text.startswith("```sql\nSELECT id FROM members WHERE visits > 0\n```\n\nSQL failed Phase 1.")

    second = await agent._activate_agent_session_state(
        state,
        _turn("Your SQL is not correct. You have one more chance. Please fix and call submit_sql."),
        activation_id=2,
    )

    assert session.turn_dirs == [f"{session.root}/turn-1", f"{session.root}/turn-2"]
    task = json.loads(staged[f"{session.root}/turn-2/task.json"])
    assert task["question"] == "Show the active members."
    assert "Q: What makes a member active?\nA: A visit this year.\n\n" in task["evidence"]
    assert "Your last query for this request:\nSELECT id FROM members WHERE visits > 0\n" in task["evidence"]
    assert "You may ask 4 more questions" in task["custom_prompts"]
    assert second.output_text == "Auto Ontology submitted no query (budget_exhausted)."

    closed = await agent._close_agent_session_state(state)

    # The server does not see the model calls: the trajectory says its turns are unavailable, rather than none taken.
    assert [gap.code for gap in closed.trajectory.gaps] == ["turns_unavailable"]
    invocations = closed.trajectory.invocations
    assert [(i.invocation_id, i.activation_id, i.status, i.stop_reason) for i in invocations] == [
        ("root", 1, "completed", "submitted"),
        ("activation-2", 2, "completed", "budget_exhausted"),
    ]
    assert invocations[0].conversation[0].content == PHASE_1
    assert all(invocation.duration_ms is not None for invocation in invocations)


async def test_a_c_interact_turn_whose_model_call_failed_fails_for_the_outcome_policy(
    monkeypatch: pytest.MonkeyPatch, staged: dict[str, str]
) -> None:
    monkeypatch.setattr(AutoOntologyAgent, "resolve_model_base_url", lambda *args, **kwargs: MODEL_URL)
    agent = _agent()
    failure = {"status_code": 429, "no_reply": False, "error": "RateLimitError: 429"}
    result = {"sql": None, "reason": "model_error", "environment_tool_calls": [], "model_failure": failure}
    state = _c_interact_state(agent, _TurnSession([result], staged))

    reply = await agent._activate_agent_session_state(state, _first_turn(), activation_id=1)

    assert reply.status == "failed"
    assert agent_outcome_policy(reply) == OutcomeAction.RETRY
    closed = await agent._close_agent_session_state(state)
    [invocation] = closed.trajectory.invocations
    assert (invocation.status, invocation.stop_reason) == ("failed", "model_error")


def test_a_first_turn_without_the_instruction_is_refused() -> None:
    agent = _agent()
    state = _c_interact_state(agent, _TurnSession([], {}))

    with pytest.raises(HTTPException) as refused:
        agent._validate_agent_activation(state, _turn(PHASE_1))
    assert refused.value.status_code == 422


def test_a_first_turn_without_ask_user_is_refused() -> None:
    agent = _agent()
    state = _c_interact_state(agent, _TurnSession([], {}))
    body = _first_turn().model_copy(update={"tools": [SUBMIT_SQL]})

    with pytest.raises(HTTPException, match="declares no ask_user"):
        agent._validate_agent_activation(state, body)


def test_settings_can_name_the_episodes_catalog_port(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CONNECTION_STRINGS=postgresql://root@localhost:${POSTGRES_PORT}/gym_members\n")

    environment = episode_environment(
        {
            "model_base_url": MODEL_URL,
            "model": "openai/policy",
            "env_file": str(env_file),
            "environment": {"POSTGRES_PORT": "34567"},
        }
    )

    assert environment["CONNECTION_STRINGS"] == "postgresql://root@localhost:34567/gym_members"


@pytest.mark.parametrize(
    ("body", "text"), [(b'"A visit this year."', "A visit this year."), (b'{"a": 1}', '{"a": 1}')]
)
def test_a_tool_reply_is_a_json_strings_text_or_the_json(
    monkeypatch: pytest.MonkeyPatch, body: bytes, text: str
) -> None:
    sent = {}

    class Reply:
        def __enter__(self) -> "Reply":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return body

    def urlopen(request, timeout):  # noqa: ANN001, ANN202
        sent.update(url=request.full_url, data=request.data, headers=dict(request.header_items()))
        return Reply()

    monkeypatch.setattr(episode_runner.urllib.request, "urlopen", urlopen)

    reply = episode_runner.call_tool(
        {"url": "http://env/ask_user", "headers": {"Authorization": "Bearer t"}, "cookies": {"s": "1"}},
        {"question": "What is active?"},
    )

    assert reply == text
    assert sent["url"] == "http://env/ask_user"
    assert json.loads(sent["data"]) == {"question": "What is active?"}
    assert sent["headers"]["Authorization"] == "Bearer t"
    assert sent["headers"]["Cookie"] == "s=1"


@pytest.mark.parametrize(
    ("error", "refused"),
    [
        (urllib.error.HTTPError("http://env/ask_user", 422, "Unprocessable", {}, io.BytesIO(b"no question")), True),
        (
            urllib.error.HTTPError("http://env/submit_sql", 500, "Internal Server Error", {}, io.BytesIO(b"boom")),
            False,
        ),
        (urllib.error.URLError("connection refused"), False),
        (TimeoutError("timed out"), False),
    ],
)
def test_a_tool_call_the_server_does_not_answer_raises(
    monkeypatch: pytest.MonkeyPatch, error: Exception, refused: bool
) -> None:
    def urlopen(request, timeout):  # noqa: ANN001, ANN202
        raise error

    monkeypatch.setattr(episode_runner.urllib.request, "urlopen", urlopen)

    with pytest.raises(episode_runner.ToolCallError) as failed:
        episode_runner.call_tool({"url": "http://env/submit_sql"}, {"sql": "SELECT 1"})
    # A 4xx refused the arguments; anything else is a call the server never answered.
    assert failed.value.refused is refused


@pytest.fixture
def extra_tool_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto Ontology's ExtraTool, which the agent's own environment does not install."""
    import sys
    from dataclasses import dataclass

    @dataclass
    class ExtraTool:
        definition: dict
        run: Any

    module = SimpleNamespace(ExtraTool=ExtraTool)
    monkeypatch.setitem(sys.modules, "auto_ontology.retrieval.text_to_sql.react.loop", module)


def _ask_tool(monkeypatch: pytest.MonkeyPatch, error: Exception) -> tuple[Any, list[dict[str, Any]]]:
    def urlopen(request, timeout):  # noqa: ANN001, ANN202
        raise error

    monkeypatch.setattr(episode_runner.urllib.request, "urlopen", urlopen)
    calls: list[dict[str, Any]] = []
    tool = {"definition": {"type": "function", "function": {"name": "ask_user"}}, "url": "http://env/ask_user"}
    [extra] = episode_runner.extra_tools({"tools": [tool]}, calls)
    return extra.run, calls


def test_a_refused_question_replies_with_the_refusal(monkeypatch: pytest.MonkeyPatch, extra_tool_module: None) -> None:
    ask, calls = _ask_tool(
        monkeypatch, urllib.error.HTTPError("http://env/ask_user", 422, "Unprocessable", {}, io.BytesIO(b"empty"))
    )

    assert ask(question="") == "The tool call failed with HTTP 422: empty"
    assert calls == [
        {
            "name": "ask_user",
            "arguments": {"question": ""},
            "error": "The tool call failed with HTTP 422: empty",
            "refused": True,
        }
    ]


def test_an_unanswered_question_fails_the_episode_after_the_loop(
    monkeypatch: pytest.MonkeyPatch, extra_tool_module: None
) -> None:
    ask, calls = _ask_tool(monkeypatch, urllib.error.URLError("connection refused"))

    # Auto Ontology's loop catches what a tool raises; run() raises the recorded failure once the loop returns.
    with pytest.raises(episode_runner.ToolCallError):
        ask(question="What is active?")
    assert calls[0]["refused"] is False and "connection refused" in calls[0]["error"]


@pytest.mark.parametrize(
    "error", [http.client.IncompleteRead(b'"A vis', 20), ConnectionResetError("reset by peer")], ids=type
)
def test_a_reply_cut_short_fails_the_episode_after_the_loop(
    monkeypatch: pytest.MonkeyPatch, extra_tool_module: None, error: Exception
) -> None:
    """The server answered 200 and may have counted the question, but its reply did not arrive whole."""

    class Reply:
        def __enter__(self) -> "Reply":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            raise error

    monkeypatch.setattr(episode_runner.urllib.request, "urlopen", lambda request, timeout: Reply())
    calls: list[dict[str, Any]] = []
    tool = {"definition": {"type": "function", "function": {"name": "ask_user"}}, "url": "http://env/ask_user"}
    [ask] = episode_runner.extra_tools({"tools": [tool]}, calls)

    with pytest.raises(episode_runner.ToolCallError):
        ask.run(question="What is active?")
    # Recorded as a call that got no reply, which run() raises once the loop returns, so the question is not repeated.
    assert calls == [
        {
            "name": "ask_user",
            "arguments": {"question": "What is active?"},
            "error": f"The tool call failed: {type(error).__name__}: {error}",
            "refused": False,
        }
    ]


def test_an_episode_that_raises_writes_its_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    def run(task: dict[str, Any]) -> dict[str, Any]:
        raise KeyError(f"No source database {task['database']!r}")

    monkeypatch.setattr(episode_runner, "run", run)
    task, result = tmp_path / "task.json", tmp_path / "result.json"
    task.write_text(json.dumps({"database": "alien"}))

    assert episode_runner.main([str(task), str(result)]) == 1
    assert json.loads(result.read_text()) == {"error": "KeyError: \"No source database 'alien'\""}
