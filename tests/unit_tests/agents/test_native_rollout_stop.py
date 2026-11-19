# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native adapters expose their existing supervisor to the shared stop lifecycle."""

import asyncio
from importlib import import_module
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.agents import AgentSessionState, SandboxSession
from nemo_gym.agents.base_responses_api_agent import RolloutStopRequest
from nemo_gym.core import NeMoGymResponse
from tests.unit_tests.agents.test_agent_sessions import _seed_request


ADAPTERS = [
    ("codex", "CodexAgent", "CodexSandboxSession"),
    ("claude_code", "ClaudeCodeAgent", "ClaudeCodeSandboxSession"),
    ("pi", "PiAgent", "PiSandboxSession"),
    ("hermes", "HermesAgent", "HermesSandboxSession"),
    ("openclaw", "OpenClawAgent", "OpenClawSandboxSession"),
    ("opencode", "OpenCodeAgent", "OpenCodeSandboxSession"),
    ("nooa", "NOOAAgent", "NOOASessionState"),
]


@pytest.mark.parametrize("name,agent_name,state_name", ADAPTERS)
@pytest.mark.parametrize("during_seed", [False, True])
async def test_native_adapter_stops_its_owned_supervisor(monkeypatch, name, agent_name, state_name, during_seed):
    try:
        module = import_module(f"responses_api_agents.{name}_agent.app")
    except ModuleNotFoundError as error:
        if error.name in {"model_tools", "nooa", "nooa_bench"}:
            pytest.skip(f"Optional native runtime unavailable: {error.name}")
        raise
    agent_type = getattr(module, agent_name)
    state_type = getattr(module, state_name)
    seed = _seed_request()
    sandbox = MagicMock(uploads_independent_of_execs=True, upload=AsyncMock())
    session = SandboxSession(sandbox=sandbox, session_dir="/session", workdir="/app", harness=name)
    if name == "nooa":
        # Runtime setup is deliberately absent; only the real runner's existing session is needed.
        runner = object.__new__(module.SandboxNOOARunner)
        runner.session = session
        state = state_type(request=seed, runner=runner)
    else:
        kwargs = {"request": seed, "session": session}
        if "runtime" in state_type.__dataclass_fields__:
            kwargs["runtime"] = "/runtime"
        state = state_type(**kwargs)
    installed = asyncio.Event()
    if not during_seed:
        installed.set()
        session.launch_started = True

    async def seed_state(self, body):
        await installed.wait()
        return state

    monkeypatch.setattr(agent_type, "_seed_agent_session_state", seed_state)
    # Avoid native startup: this exercises the real shared seed/stop methods and adapter hook.
    agent = agent_type.model_construct(config=MagicMock(num_workers=1, concurrency=1), server_client=MagicMock())
    assert agent._rollout_sandbox_session(AgentSessionState(request=seed)) is None
    task = asyncio.create_task(agent.seed_agent_session(MagicMock(session={}), seed))
    await asyncio.sleep(0)
    with pytest.raises(HTTPException) as unknown:
        await agent.stop_rollout("another-rollout")
    assert unknown.value.status_code == 404
    first = await agent.stop_rollout("rollout-a2", RolloutStopRequest(reason="budget"))
    second = await agent.stop_rollout("rollout-a2", RolloutStopRequest(reason="other"))
    assert first.stopped and not second.stopped
    installed.set()
    await task
    assert session.expire_reason == "budget"
    if during_seed:
        sandbox.upload.assert_not_awaited()
    else:
        sandbox.upload.assert_awaited_once()
        assert sandbox.upload.await_args.args[1] == "/session/expire.request"
    # A completed supervisor receipt, not the stop acknowledgement, makes this a timeout.
    assert not session.expired
    session.cleanup = {"timed_out": True}
    assert session.expired
    response = NeMoGymResponse(
        id="served",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
        status="incomplete",
        metadata={"existing": "retained"},
    )
    reported = agent.report_rollout_stop(response, session)
    assert reported.metadata == {"existing": "retained", "ng_rollout_stop": '{"reason":"budget"}'}
    assert reported.model_dump(exclude={"metadata"}) == response.model_dump(exclude={"metadata"})
