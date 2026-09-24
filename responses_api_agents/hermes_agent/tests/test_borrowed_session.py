# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig, HermesAgentSessionState


async def test_runner_exit_rechecks_output_before_reporting_failure(monkeypatch):
    agent = HermesAgent(
        config=HermesAgentConfig(
            host="localhost",
            port=1,
            name="hermes",
            entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "resources"},
            model_server={"type": "responses_api_models", "name": "model"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    sandbox = AsyncMock()
    runner = AsyncMock()
    runner.wait_exit.return_value = 0
    sandbox.pty.create.return_value = runner
    states = iter(["running", "output"])

    async def probe(*args, **kwargs):
        await asyncio.sleep(0)
        return SimpleNamespace(return_code=0, stdout=next(states), stderr="")

    sandbox.exec.side_effect = probe
    state = HermesAgentSessionState(
        request=AgentSeedSessionRequest(
            episode_id={"rollout_id": "rollout"}, task_id={"taskset": "test", "task_id": "task"}
        ),
        sandbox=sandbox,
        workdir="/app",
        session_dir="/tmp/hermes-test",
    )
    monkeypatch.setattr(HermesAgent, "_upload_json", AsyncMock())
    monkeypatch.setattr(
        HermesAgent,
        "_download_json",
        AsyncMock(
            return_value={
                "result": {"messages": [{"role": "assistant", "content": "Done"}], "completed": True},
                "runtime": {"hostname": "task", "pid": 123, "python": "/runtime/python"},
                "observations": None,
            }
        ),
    )
    episode = await agent._run_sandbox_episode(
        request=SimpleNamespace(),
        body=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "task"}]),
        agent_session_id="session",
        state=state,
    )
    assert episode.response.status == "completed"
    assert "else echo exited" in sandbox.exec.await_args_list[1].args[0]
    runner.close.assert_awaited_once()
