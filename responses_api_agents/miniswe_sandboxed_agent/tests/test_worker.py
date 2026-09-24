# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from pathlib import Path

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, MiniSWEConfig


@pytest.mark.parametrize("stop", ["timeout", "cancel", "model_failure", "step_limit", "tool_cancel"])
async def test_runner_stops_before_verification_and_retains_partial_trajectory(tmp_path, runner_factory, stop):
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
                        "arguments": json.dumps(
                            {"command": "echo $$ > tool.pid; sleep 60" if stop == "tool_cancel" else "echo inspected"}
                        ),
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

    harness = await runner_factory(
        context=HarnessContext(session_id="worker-test", instruction="inspect"),
        config=MiniSWEConfig(step_limit=1 if stop == "step_limit" else 0),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path / "artifacts",
        observability_enabled=True,
    )
    run = asyncio.create_task(harness.execute(10 if stop == "timeout" else 15))
    if stop == "cancel":
        await asyncio.wait_for(entered.wait(), 10)
        run.cancel()
    elif stop == "tool_cancel":
        async with asyncio.timeout(10):
            while not Path(harness.context.workdir, "tool.pid").exists():
                await asyncio.sleep(0.025)
        run.cancel()
    response, outcome, extra = await run
    assert (
        outcome.reason
        == {
            "cancel": "cancelled",
            "timeout": "timeout",
            "model_failure": "infrastructure_error",
            "step_limit": "nonzero_exit",
            "tool_cancel": "cancelled",
        }[stop]
    ), outcome
    assert len(response.output) == (1 if stop == "tool_cancel" else 2)
    if stop not in ("step_limit", "tool_cancel"):
        assert exited.is_set()
    assert extra["mini_swe_trajectory"]["messages"]
    assert (harness.directory / "trajectory.json").exists()
    await asyncio.wait_for(harness.sandbox.runners[0].wait(), 2)
    if stop == "tool_cancel":
        pid = int(Path(harness.context.workdir, "tool.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert extra["ng_agent_observations"]["records"][1]["status"] == "cancelled"


async def test_cleanup_transport_failure_keeps_captured_response(tmp_path, runner_factory, monkeypatch):
    async def query(params):
        return NeMoGymResponse(
            id="submitted",
            created_at=0,
            object="response",
            model="model",
            parallel_tool_calls=False,
            tools=[],
            tool_choice="auto",
            output=[
                {
                    "type": "function_call",
                    "call_id": "submit",
                    "name": "bash",
                    "arguments": '{"command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}',
                }
            ],
        )

    harness = await runner_factory(
        context=HarnessContext(session_id="cleanup-test", instruction="submit"),
        config=MiniSWEConfig(),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path / "artifacts",
    )
    close = harness.close

    async def fail_close():
        await close()
        raise ConnectionError("lost cleanup response")

    monkeypatch.setattr(harness, "close", fail_close)
    response, outcome, extra = await harness.execute(15)
    assert outcome.reason == "infrastructure_error"
    assert "lost cleanup response" in outcome.detail
    assert len(response.output) == 2
    assert extra["mini_swe_trajectory"]["info"]["exit_status"] == "Submitted"
