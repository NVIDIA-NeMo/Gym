# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.miniswe_sandboxed_agent import episode
from responses_api_agents.miniswe_sandboxed_agent.app import MiniSWESandboxedConfig, empty_response
from responses_api_agents.miniswe_sandboxed_agent.episode import MiniSWEEpisodeAgent
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessOutcome


@pytest.fixture
def native(tmp_path, monkeypatch):
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"borrowed": {"local": {}}}
    agent = MiniSWEEpisodeAgent(
        config=MiniSWESandboxedConfig(
            host="localhost",
            port=1,
            name="agent",
            entrypoint="episode.py",
            model_server={"type": "responses_api_models", "name": "model"},
            artifacts_dir=tmp_path,
            shutdown_timeout_sec=0.1,
        ),
        server_client=client,
    )
    request = SimpleNamespace(
        session={SESSION_ID_KEY: "owner"}, cookies={"session": "cookie"}, path_params={"rollout_id": "rollout-a2"}
    )
    seed = AgentSeedSessionRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task_id=TaskId(taskset="unrelated-benchmark", task_id="problem-17"),
        sandbox_access={
            "connection": {"provider_config_ref": "borrowed", "descriptor": {"sandbox_id": "test"}},
            "workdir": "/problem",
        },
    )
    harness = MagicMock()
    harness.setup, harness.execute = AsyncMock(), AsyncMock()
    harness.close, harness.dispose = AsyncMock(), AsyncMock()
    harness.result = None

    def make_harness(**kwargs):
        harness.context = kwargs["context"]
        harness.query = kwargs["query"]
        return harness

    monkeypatch.setattr(episode, "MiniSWEHarness", make_harness)
    provider = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(episode, "create_provider", lambda _: provider)
    monkeypatch.setattr(episode.AsyncSandbox, "connect", AsyncMock())
    return SimpleNamespace(agent=agent, request=request, seed=seed, harness=harness, provider=provider)


def result(params, reason="nonzero_exit"):
    return empty_response(params, "model"), HarnessOutcome(reason=reason, detail="LimitsExceeded"), {}


async def seed_and_close(f):
    seeded = await f.agent.seed_agent_session(f.request, f.seed)
    return AgentCloseSessionRequest(agent_session_id=seeded.agent_session_id, episode_id=f.seed.episode_id)


async def test_setup_precedes_publication_and_activation_is_once(native):
    f = native

    async def setup():
        assert "miniswe_agent_session_id" not in f.request.session

    f.harness.setup.side_effect = setup
    close = await seed_and_close(f)
    assert (await f.agent.seed_agent_session(f.request, f.seed)).agent_session_id == close.agent_session_id
    f.harness.setup.assert_awaited_once()
    params = NeMoGymResponseCreateParamsNonStreaming(
        input="Fix the public project", instructions="Be concise", max_output_tokens=500, temperature=0.2
    )
    f.harness.execute.return_value = result(params)
    response = await f.agent.responses(f.request, params)
    assert f.harness.context.instruction == params.input
    assert f.harness.context.workdir == "/problem"
    assert f.harness.context.task_id == "problem-17"
    assert f.harness.params == params
    assert response.status == "incomplete"
    with pytest.raises(HTTPException, match="one activation"):
        await f.agent.responses(f.request, params)
    assert await f.agent.close_agent_session(f.request, close) == await f.agent.close_agent_session(f.request, close)
    f.harness.close.assert_awaited_once()
    f.harness.dispose.assert_awaited_once()
    f.provider.aclose.assert_awaited_once()
    f.agent.server_client.post.assert_not_called()
    with pytest.raises(HTTPException, match="closed"):
        await f.agent.responses(f.request, params)


async def test_seed_failure_does_not_publish_session(native):
    f = native
    f.harness.setup.side_effect = RuntimeError("image cannot install runtime")
    with pytest.raises(HTTPException, match="install runtime"):
        await f.agent.seed_agent_session(f.request, f.seed)
    assert not f.agent._agent_sessions
    assert "miniswe_agent_session_id" not in f.request.session
    f.harness.dispose.assert_awaited_once()
    f.provider.aclose.assert_awaited_once()


async def test_native_agent_rejects_identity_changes_and_foreign_cookie(native):
    f = native
    close = await seed_and_close(f)
    changed = f.seed.model_copy(deep=True)
    changed.task_id = TaskId(taskset="other", task_id="task")
    with pytest.raises(HTTPException, match="bound"):
        await f.agent.seed_agent_session(f.request, changed)
    with pytest.raises(HTTPException, match="Close identity"):
        await f.agent.close_agent_session(
            f.request, close.model_copy(update={"episode_id": EpisodeId(rollout_id="other")})
        )
    f.request.path_params["rollout_id"] = "rollout"
    with pytest.raises(HTTPException, match="Rollout route"):
        await f.agent.responses(f.request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    f.request.session[SESSION_ID_KEY] = "foreign-owner"
    with pytest.raises(HTTPException, match="Unknown"):
        await f.agent.close_agent_session(f.request, close)


async def test_retried_close_does_not_interrupt_worker_cleanup(native):
    f = native
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    params = NeMoGymResponseCreateParamsNonStreaming(input="task")

    async def execute(budget):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            return result(params, "cancelled")

    f.harness.execute.side_effect = execute
    close = await seed_and_close(f)
    pending = asyncio.create_task(f.agent.responses(f.request, params))
    await started.wait()
    first_close = asyncio.create_task(f.agent.close_agent_session(f.request, close))
    await cleaning.wait()
    retry_close = asyncio.create_task(f.agent.close_agent_session(f.request, close))
    await asyncio.sleep(0)
    release.set()
    assert (await pending).metadata["termination_reason"] == "cancelled"
    assert await first_close == await retry_close
    f.harness.close.assert_awaited_once()


@pytest.mark.parametrize("failure", ["close", "dispose", "disconnect"])
async def test_failed_cleanup_retains_state_for_retry(native, failure):
    f = native
    close = await seed_and_close(f)
    operation = f.provider.aclose if failure == "disconnect" else getattr(f.harness, failure)
    operation.side_effect = RuntimeError("cleanup unconfirmed")
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await f.agent.close_agent_session(f.request, close)
    state = f.agent._agent_sessions[close.agent_session_id]
    assert state.close_result is None
    assert state.closing
    operation.side_effect = None
    await f.agent.close_agent_session(f.request, close)
    assert state.close_result is not None
    state.closed_at -= f.agent.config.closed_session_retention_sec + 1
    with pytest.raises(HTTPException, match="Unknown"):
        await f.agent.close_agent_session(f.request, close)


async def test_unsupported_options_rejected_before_activation(native):
    f = native
    await seed_and_close(f)
    for kwargs in ({"previous_response_id": "previous"}, {"tool_choice": "none"}, {"tools": []}):
        with pytest.raises(HTTPException, match="Unsupported"):
            await f.agent.responses(f.request, NeMoGymResponseCreateParamsNonStreaming(input="task", **kwargs))
    f.harness.execute.assert_not_awaited()
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "developer", "content": "Keep tests"}, {"role": "user", "content": "Fix bug"}]
    )
    assert episode.task_instruction(params) == "developer: Keep tests\n\nuser: Fix bug"
