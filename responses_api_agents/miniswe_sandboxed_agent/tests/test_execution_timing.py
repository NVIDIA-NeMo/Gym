# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from time import sleep, time

import pytest

from nemo_gym.execution_timing import ExecutionTiming
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_collection import _build_trajectory_record
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, MiniSWEConfig
from responses_api_agents.miniswe_sandboxed_agent.runtime import timed_environment as runtime


def test_unequal_overlapping_native_commands_exclude_dispatch_and_result_delays(tmp_path):
    environment = runtime.TimedLocalEnvironment(cwd=str(tmp_path))
    dispatch = time()
    sleep(0.1)
    with ThreadPoolExecutor(2) as pool:
        short = pool.submit(environment.execute, {"command": "sleep 0.2; echo short", "tool_call_id": "short"})
        long = pool.submit(environment.execute, {"command": "sleep 0.6; echo long", "tool_call_id": "long"})
        short, long = short.result(), long.result()
    sleep(0.1)
    received = time()
    a, b = [ExecutionTiming.model_validate(x["extra"]["execution_timing"]) for x in (short, long)]
    assert a.clock_domain == b.clock_domain
    assert max(a.started_at, b.started_at) < min(a.completed_at, b.completed_at)
    assert 150 < a.duration_ms < b.duration_ms - 200
    assert dispatch + 0.09 < min(a.started_at, b.started_at)
    assert max(a.completed_at, b.completed_at) < received - 0.09
    assert short["output"].strip() == "short" and long["output"].strip() == "long"
    assert a.execution_id == "short" and b.execution_id == "long"


@pytest.mark.parametrize("command,status", [("exit 7", 7), ("sleep 3", -1)])
def test_native_failure_and_timeout_retain_runtime_interval(tmp_path, command, status):
    result = runtime.TimedLocalEnvironment(cwd=str(tmp_path), timeout=1).execute(
        {"command": command, "tool_call_id": "failed"}
    )
    assert result["returncode"] == status
    timing = ExecutionTiming.model_validate(result["extra"]["execution_timing"])
    assert timing.completed_at >= timing.started_at
    assert timing.duration_ms >= (1000 if status == -1 else 0)
    if status == -1:
        assert result["extra"]["exception_type"] == "TimeoutExpired"


async def test_rollout_persists_runtime_clocks_and_exact_call_result_join(tmp_path, runner_factory):
    commands = ["sleep 0.1; echo first", "exit 7", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
    requests = []

    async def query(params):
        index = len(requests)
        requests.append(params)
        await asyncio.sleep(0.1)  # Model/dispatch delay must not become command duration.
        return NeMoGymResponse(
            id=f"resp_timing_{index}",
            created_at=int(time()),
            object="response",
            model="test",
            parallel_tool_calls=False,
            tools=[],
            tool_choice="auto",
            output=[
                {
                    "type": "function_call",
                    "id": f"fc_{index}",
                    "call_id": f"call_{index}",
                    "name": "bash",
                    "arguments": json.dumps({"command": commands[index]}),
                    "status": "completed",
                }
            ],
        )

    harness = await runner_factory(
        context=HarnessContext(session_id="timing", instruction="run commands"),
        config=MiniSWEConfig(),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="test",
        directory=tmp_path / "artifacts",
        observability_enabled=True,
    )
    response, termination, extra = await harness.execute(30)
    assert termination.reason == "completed"
    # This is the same projection used when the collector persists rollout JSONL.
    trajectory = _build_trajectory_record(
        {"trajectory_identity": {"task_id": "task", "rollout_id": "rollout"}},
        {"response": response.model_dump(mode="json"), **extra},
    )
    persisted = tmp_path / "rollout.jsonl"
    persisted.write_text(json.dumps({"ng_trajectory": trajectory.model_dump(mode="json")}) + "\n")
    record = json.loads(persisted.read_text())["ng_trajectory"]
    tools = record["tool_calls"]
    assert [t["tool_call_id"] for t in tools] == ["call_0", "call_1", "call_2"]
    assert [t["status"] for t in tools] == ["completed", "failed", "incomplete"]
    assert tools[0]["execution_timing"]["execution_id"] == "call_0"
    assert tools[0]["execution_timing"]["duration_ms"] >= 100
    assert tools[0]["execution_timing"]["clock_domain"] == tools[1]["execution_timing"]["clock_domain"]
    assert tools[2]["execution_timing"] is None
    assert "first" in tools[0]["output"]
    conversation = record["invocations"][0]["conversation"]
    calls = {x["call_id"]: x for x in conversation if x["type"] == "function_call"}
    results = {x["call_id"]: x for x in conversation if x["type"] == "function_call_output"}
    assert calls["call_0"]["name"] == "bash"
    assert json.loads(calls["call_0"]["arguments"])["command"] == commands[0]
    assert "first" in results["call_0"]["output"]
