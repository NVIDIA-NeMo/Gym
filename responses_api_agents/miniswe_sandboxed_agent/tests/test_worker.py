# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from minisweagent.agents.default import DefaultAgent

from responses_api_agents.miniswe_sandboxed_agent.app import BorrowedEnvironment, GymModel, WorkerBridge


async def test_default_agent_submits_in_borrowed_environment(tmp_path):
    bridge = WorkerBridge()
    commands = []

    async def query(messages):
        return {"role": "assistant", "content": "Submit", "extra": {"actions": [{"command": "submit"}]}}

    async def execute(command):
        commands.append(command)
        return {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nfinished", "returncode": 0}

    agent = DefaultAgent(
        GymModel(bridge, query),
        BorrowedEnvironment(bridge, execute),
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
