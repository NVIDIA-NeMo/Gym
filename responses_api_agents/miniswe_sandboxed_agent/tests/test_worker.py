# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from minisweagent.agents.default import DefaultAgent

from responses_api_agents.miniswe_sandboxed_agent.harness import GymModel, SandboxEnvironment, WorkerBridge


async def test_default_agent_submits_in_sandbox_environment(tmp_path):
    bridge = WorkerBridge()
    commands = []

    async def query(messages):
        return {"role": "assistant", "content": "Submit", "extra": {"actions": [{"command": "submit"}]}}

    async def execute(command):
        commands.append(command["command"])
        return {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nfinished", "returncode": 0}

    agent = DefaultAgent(
        GymModel(bridge, query),
        SandboxEnvironment(bridge, execute, {"system": "Linux"}),
        system_template="System",
        instance_template="{{task}}",
        cost_limit=0,
        output_path=tmp_path / "trajectory.json",
    )
    result = await asyncio.to_thread(agent.run, "Generic task without SWE-bench fields")
    bridge.close()
    assert result["exit_status"] == "Submitted"
    assert result["submission"] == "finished"
    assert commands == ["submit"]
    assert (tmp_path / "trajectory.json").is_file()
    assert agent.messages[1]["content"] == "Generic task without SWE-bench fields"


async def test_cancellation_stops_worker_before_verification():
    bridge = WorkerBridge()
    entered = asyncio.Event()
    exited = asyncio.Event()

    async def query():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    worker = asyncio.create_task(asyncio.to_thread(bridge.call, query))
    await entered.wait()
    bridge.close()
    await asyncio.gather(worker, return_exceptions=True)
    await exited.wait()
    with pytest.raises(RuntimeError, match="closed"):
        await asyncio.to_thread(bridge.call, query)


@pytest.mark.parametrize("stop", ["timeout", "cancel", "model_failure", "step_limit"])
async def test_harness_stops_real_worker_and_retains_partial_trajectory(tmp_path, stop):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.sandbox import SandboxExecResult
    from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, MiniSWEConfig, MiniSWEHarness

    entered, exited = asyncio.Event(), asyncio.Event()
    calls = 0

    async def query(params):
        nonlocal calls
        calls += 1
        if calls == 1:
            return NeMoGymResponse(
                id="first",
                created_at=0,
                object="response",
                model="model",
                parallel_tool_calls=False,
                tools=[],
                tool_choice="auto",
                output=[
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "bash",
                        "arguments": '{"command": "echo inspected"}',
                        "status": "completed",
                    }
                ],
            )
        entered.set()
        try:
            if stop == "model_failure":
                raise ConnectionError("model unavailable")
            await asyncio.Event().wait()
        finally:
            exited.set()

    sandbox = SimpleNamespace(exec=AsyncMock(return_value=SandboxExecResult("inspected", "", 0)))
    harness = MiniSWEHarness(
        sandbox=sandbox,
        context=HarnessContext(session_id="task", instruction="inspect"),
        config=MiniSWEConfig(step_limit=1 if stop == "step_limit" else 0),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path,
    )
    harness.system_info = {"system": "Linux", "release": "6", "version": "test", "machine": "x86_64"}
    run = asyncio.create_task(harness.execute(0.2 if stop == "timeout" else 5))
    if stop == "cancel":
        await entered.wait()
        run.cancel()
    response, outcome, extra = await run
    assert (
        outcome.reason
        == {
            "cancel": "cancelled",
            "timeout": "timeout",
            "model_failure": "infrastructure_error",
            "step_limit": "nonzero_exit",
        }[stop]
    )
    assert len(response.output) == 2
    if stop != "step_limit":
        assert exited.is_set()
    assert extra["mini_swe_trajectory"]["messages"]
    assert (tmp_path / "trajectory.json").exists()
    sandbox.exec.assert_awaited_once()
