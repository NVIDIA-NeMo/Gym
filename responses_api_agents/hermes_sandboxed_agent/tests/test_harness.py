# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.hermes_sandboxed_agent.harness import HarnessContext, HermesConfig, HermesHarness


@pytest.mark.parametrize("stop", ["completed", "timeout", "cancelled", "turn_limit", "partial", "tool_error"])
async def test_real_hermes_loop_preserves_sandbox_results_and_stops_before_return(tmp_path, monkeypatch, stop):
    """Hermes retains native turns and paired tool results, including when model I/O is interrupted."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    requests, commands = [], []
    pending = asyncio.Event()
    closed = asyncio.Event()

    async def query(params):
        requests.append(params)
        if len(requests) == 2 and stop in {"timeout", "cancelled"}:
            pending.set()
            try:
                await asyncio.Future()
            finally:
                # Yield during cleanup to detect returning before asynchronous I/O has stopped.
                await asyncio.sleep(0.05)
                closed.set()
        items = (
            [
                {"type": "reasoning", "id": "reason", "summary": [{"type": "summary_text", "text": "inspect"}]},
                {
                    "type": "function_call",
                    "call_id": "command",
                    "name": "terminal",
                    "arguments": json.dumps({"command": "printf sandbox-result"}),
                },
            ]
            if len(requests) == 1
            else [
                {
                    "type": "message",
                    "id": "final",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Done", "annotations": []}],
                }
            ]
        )
        return NeMoGymResponse(
            id=f"response-{len(requests)}",
            created_at=0,
            model="model",
            object="response",
            status="incomplete" if stop == "partial" and len(requests) == 2 else "completed",
            output=items,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
            usage={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        )

    async def execute(command, **kwargs):
        commands.append((command, kwargs))
        if stop == "tool_error" and len(commands) == 2:
            raise RuntimeError("Sandbox disconnected")
        return SandboxExecResult("sandbox-result", "", 0)

    harness = HermesHarness(
        sandbox=SimpleNamespace(exec=execute),
        context=HarnessContext(session_id="test", instruction="Inspect the sandbox", workdir="/task"),
        config=HermesConfig(name="hermes", max_turns=1 if stop == "turn_limit" else 3),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path,
    )
    await harness.setup()
    task = asyncio.create_task(harness.execute(1 if stop == "timeout" else 20))
    if stop == "cancelled":
        await asyncio.wait_for(pending.wait(), 10)
        task.cancel()
    response, outcome, extra = await asyncio.wait_for(task, 15)
    assert outcome.reason == {
        "turn_limit": "nonzero_exit",
        "partial": "nonzero_exit",
        "tool_error": "infrastructure_error",
    }.get(stop, stop)
    assert commands[-1][1]["cwd"] == "/task"
    assert commands[-1][0].endswith("printf sandbox-result'")
    call = next(item for item in response.output if item.type == "function_call")
    result = next(item for item in response.output if item.type == "function_call_output")
    assert call.call_id == result.call_id == "command"
    if stop == "tool_error":
        assert json.loads(result.output)["error"] == "RuntimeError: Sandbox disconnected"
    else:
        assert json.loads(result.output)["output"] == "sandbox-result"
    assert response.output[0].type == "reasoning"
    assert response.usage.total_tokens == (24 if stop in {"completed", "partial"} else 12)
    native = json.loads((tmp_path / "trajectory.json").read_text())["messages"]
    assert any(message["role"] == "tool" and message["tool_call_id"] == "command" for message in native)
    if stop in {"timeout", "cancelled"}:
        assert closed.is_set()

    trajectory = extra["ng_trajectory"]
    invocation = trajectory["invocations"][0]
    expected_calls = 2 if stop in {"completed", "partial"} else 1
    assert len(invocation["model_calls"]) == len(trajectory["turns"]) == expected_calls
    assert (
        next(item for item in invocation["conversation"] if item["type"] == "function_call_output")["call_id"]
        == "command"
    )
    tool = trajectory["tool_calls"][0]
    assert tool["tool_call_id"] == "command"
    assert tool["output"] == result.output
    assert tool["status"] == ("failed" if stop == "tool_error" else "completed")
