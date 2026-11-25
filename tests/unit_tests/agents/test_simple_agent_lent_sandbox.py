# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple Agent runs its loop where its seed says: in the sandbox the seed lends, borrowed with ``borrow()``, or in
its own process when lent nothing.

The loop installed in the sandbox is a recording stand-in for ``responses_api_agents.simple_agent.sandbox``, so these
tests cover the agent's side of a lent sandbox: which seeds borrow, what the loop is staged with, and that a borrowed
loop is closed and never falls back to the host. The sandbox runtime has its own tests beside the server.
"""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlsplit

import pytest

from nemo_gym.agents.base_responses_api_agent import (
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
)
from nemo_gym.agents.testing.agent_session_conformance import check_agent_seed_rejected
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import ModelServerRef
from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.core.responses_api_types import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from responses_api_agents.simple_agent.app import (
    _SANDBOX_LOOP_CONFIG,
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentSessionState,
)


APP = "responses_api_agents.simple_agent.app"
ACCESS = SandboxAccess(
    connection=DirectSandboxConnection(provider_config_ref="attempt_sandboxes", descriptor={"id": "sb-1"}),
    workdir="/workspace",
)
MODEL_PATH = "/ng-rollout/rollout-a1/v1/responses"


def _config() -> SimpleAgentConfig:
    return SimpleAgentConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="simple",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_model_retries=0,
    )


def _agent(agent_type: type[SimpleAgent] = SimpleAgent) -> SimpleAgent:
    return agent_type(config=_config(), server_client=MagicMock(spec=ServerClient))


def _seed(**fields: Any) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session",
        episode_id=EpisodeId(rollout_id="rollout", attempt=1),
        task_id=TaskId(taskset="example", task_id="0"),
        **fields,
    )


def _reply() -> dict[str, Any]:
    return NeMoGymResponse(
        id="resp-1",
        created_at=0,
        model="policy",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    ).model_dump(mode="json")


def _loop_arguments() -> dict[str, Any]:
    return {
        "model_url_path": MODEL_PATH,
        "resources_server_cookies": None,
        "model_server_cookies": None,
        "tool_access": None,
        "resources_prefix": "",
        "in_session": True,
        "execute_tools": True,
        "invocation_id": "root",
        "rollout_id": "rollout-a1",
        "collect_trajectory": True,
        "activation": None,
    }


class _Loop:
    """The loop installed in a borrowed sandbox: records what it is asked to do, and fails where it is told to."""

    def __init__(self, sandbox: Any, workdir: str, *, fail_install: Exception | None, fail_close: Exception | None):
        self.sandbox, self.workdir = sandbox, workdir
        self.fail_install, self.fail_close = fail_install, fail_close
        self.installed: list[float] = []
        self.payloads: list[dict[str, Any]] = []
        self.closed = 0

    async def install(self, *, timeout: float) -> None:
        self.installed.append(timeout)
        if self.fail_install is not None:
            raise self.fail_install

    async def run(self, payload: dict[str, Any], *, timeout: float, close_timeout: float) -> dict[str, Any]:
        self.payloads.append(payload)
        return {
            "response": _reply(),
            "trajectory": None,
            "model_server_cookies": {"model": "1"},
            "resources_server_cookies": None,
        }

    async def close(self, *, timeout: float) -> None:
        self.closed += 1
        if self.fail_close is not None:
            raise self.fail_close


@pytest.fixture
def lent(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A lent sandbox borrows as ``sandbox:<id>`` and gets a recording loop; the sandbox reaches the model server at
    10.0.0.5:8000."""
    state = SimpleNamespace(loops=[], borrowed=[], fail_install=None, fail_close=None)

    class SandboxLoop:
        @classmethod
        def borrowed(cls, sandbox: Any, *, workdir: str) -> _Loop:
            loop = _Loop(sandbox, workdir, fail_install=state.fail_install, fail_close=state.fail_close)
            state.loops.append(loop)
            return loop

    def model_server_endpoint(url: str) -> tuple[str, int]:
        parts = urlsplit(url)
        return parts.hostname, parts.port

    # The agent imports its sandbox runtime only for a seed that lends a sandbox; this stand-in takes its place.
    module = ModuleType("responses_api_agents.simple_agent.sandbox")
    module.SandboxLoop = SandboxLoop
    module.model_server_endpoint = model_server_endpoint
    monkeypatch.setitem(sys.modules, module.__name__, module)

    async def borrow(access: SandboxAccess) -> str:
        state.borrowed.append(access)
        return f"sandbox:{access.connection.descriptor['id']}"

    monkeypatch.setattr(f"{APP}.borrow", borrow)
    monkeypatch.setattr(f"{APP}.sandbox_server_url", lambda name, require_reachable: "http://10.0.0.5:8000")
    return state


async def test_a_seed_that_lends_a_sandbox_runs_the_loop_there_and_one_that_lends_none_in_process(
    lent: SimpleNamespace,
) -> None:
    agent = _agent()

    in_process = await agent._seed_agent_session_state(_seed())
    assert in_process.sandbox_loop is None
    assert lent.borrowed == []

    state = await agent._seed_agent_session_state(_seed(sandbox_access=ACCESS))
    [loop] = lent.loops
    assert lent.borrowed == [ACCESS]
    assert state.sandbox_loop is loop
    assert (loop.sandbox, loop.workdir) == ("sandbox:sb-1", "/workspace")
    assert loop.installed == [agent.config.sandbox_install_timeout_seconds]
    # The loop calls the model server at the address the sandbox reaches, never a loopback one.
    assert state.model_server_endpoint == ("10.0.0.5", 8000)


async def test_an_activation_of_a_lent_session_runs_in_its_sandbox_with_the_agents_settings(
    lent: SimpleNamespace,
) -> None:
    agent = _agent()
    state = await agent._seed_agent_session_state(_seed(sandbox_access=ACCESS))
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="Which option?", metadata={"baseline_response": "Reference answer", "task": "quiz"}
    )

    response, trajectory, model_cookies, resources_cookies = await agent._create_session_episode(
        state, body, **_loop_arguments()
    )

    assert (response.id, trajectory, model_cookies, resources_cookies) == ("resp-1", None, {"model": "1"}, None)
    [payload] = lent.loops[0].payloads
    # The settings the loop reads, with their defaults but no unset ones, and the model server as the sandbox
    # reaches it.
    assert set(payload["config"]) == _SANDBOX_LOOP_CONFIG - {"resources_server", "max_steps"}
    assert payload["model_server"] == {"host": "10.0.0.5", "port": 8000}
    # Judge-only metadata never reaches the policy's requests.
    assert payload["body"] == {"input": "Which option?", "metadata": {"task": "quiz"}}
    assert payload["loop"]["model_url_path"] == MODEL_PATH
    # No checkpoint waits on a loop in a borrowed sandbox: its session restarts instead.
    assert "activation" not in payload["loop"]
    assert agent._seed_must_restart(_seed(sandbox_access=ACCESS)) is True


@pytest.mark.parametrize(
    ("unreachable", "refusal"),
    [("model", "cannot reach its model server"), ("tools", "cannot reach its tool grant 'mcqa.direct_http'")],
)
async def test_a_seed_whose_servers_the_sandbox_cannot_reach_is_refused_before_borrowing(
    lent: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, unreachable: str, refusal: str
) -> None:
    def server_url(name: str, require_reachable: bool) -> str:
        if unreachable == "model":
            raise ValueError("loopback address 127.0.0.1")
        return "http://10.0.0.5:8000"

    monkeypatch.setattr(f"{APP}.sandbox_server_url", server_url)
    # A loopback tool grant: the tool calls would leave from the sandbox, which cannot reach it.
    grant = {"name": "mcqa.direct_http", "kind": "direct_http", "required": True, "base_url": "http://127.0.0.1:9"}

    with pytest.raises(AgentSeedRejectedError, match=refusal):
        await _agent()._seed_agent_session_state(_seed(sandbox_access=ACCESS, tool_accesses=[grant]))
    assert lent.borrowed == []


async def test_a_loop_that_cannot_install_releases_the_sandbox_and_fails_the_seed(lent: SimpleNamespace) -> None:
    lent.fail_install = RuntimeError("the runtime could not import the agent loop")

    with pytest.raises(RuntimeError, match="could not import the agent loop"):
        await _agent()._seed_agent_session_state(_seed(sandbox_access=ACCESS))
    assert lent.loops[0].closed == 1

    # When the release fails too, the session is kept for a close to release it again.
    lent.fail_close = RuntimeError("release failed")
    with pytest.raises(AgentSessionSetupError) as raised:
        await _agent()._seed_agent_session_state(_seed(sandbox_access=ACCESS))
    assert raised.value.state.sandbox_loop is lent.loops[1]
    assert lent.loops[1].closed == 1


async def test_closing_or_retiring_a_lent_session_stops_its_loop_and_releases_the_sandbox(
    lent: SimpleNamespace,
) -> None:
    agent = _agent()
    closed = await agent._seed_agent_session_state(_seed(sandbox_access=ACCESS))

    reply = await agent._close_agent_session_state(closed)

    assert reply.agent_session_id == "agent-session"
    assert lent.loops[0].closed == 1

    retired = await agent._seed_agent_session_state(_seed(sandbox_access=ACCESS))
    agent._install_restored_session("agent-session", retired)
    await agent.retire_agent_session("agent-session")
    assert lent.loops[1].closed == 1
    assert "agent-session" not in agent._session_records


async def test_a_session_lent_a_sandbox_never_runs_its_loop_in_process() -> None:
    agent = _agent()
    # As a restore would rebuild it: lent a sandbox, but with no loop in it.
    state = SimpleAgentSessionState(request=_seed(sandbox_access=ACCESS), tool_access=None, resources_cookies=None)

    with pytest.raises(RuntimeError, match="only in that sandbox"):
        await agent._create_session_episode(
            state, NeMoGymResponseCreateParamsNonStreaming(input="hi"), **_loop_arguments()
        )


def test_a_subclass_that_changes_the_agent_loop_refuses_a_lent_sandbox() -> None:
    class _OwnLoop(SimpleAgent):
        async def _create_episode(self, body: Any, **kwargs: Any) -> Any:
            raise AssertionError("not reached")

    detail = check_agent_seed_rejected(_agent(_OwnLoop).setup_webserver(), _seed(sandbox_access=ACCESS))

    assert "changes the agent loop" in detail
