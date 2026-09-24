# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import platform
from pathlib import Path

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.miniswe_sandboxed_agent import harness as module


@pytest.mark.parametrize("observability_enabled", [False, True])
@pytest.mark.parametrize("response_id", ["present", ""])
async def test_real_runner_preserves_model_history_and_tool_observations(
    tmp_path, runner_factory, observability_enabled, response_id
):
    requests = []
    full_output = "start" + "x" * 6000 + "MIDDLE_MUST_BE_ELIDED" + "y" * 6000 + "end"
    commands = ["echo first", "cat large.txt; sleep 10", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]

    async def query(params):
        index = len(requests)
        requests.append(params)
        return NeMoGymResponse.model_validate(
            {
                "id": f"resp_test_{index}" if response_id else "",
                "created_at": 0,
                "object": "response",
                "model": "test",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {"type": "reasoning", "id": f"rs_{index}", "summary": []},
                    {
                        "type": "function_call",
                        "id": f"fc_{index}",
                        "call_id": f"call_{index}",
                        "name": "bash",
                        "arguments": json.dumps({"command": commands[index]}),
                        "status": "completed",
                    },
                ],
                "usage": {
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 3,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 13,
                },
            }
        )

    harness = await runner_factory(
        context=module.HarnessContext(
            session_id="task", instruction="Official task instruction", skills_dir="/skills"
        ),
        config=module.MiniSWEConfig(step_timeout_sec=1),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[], tool_choice="required"),
        query=query,
        model_name="model",
        directory=tmp_path / "artifacts",
        observability_enabled=observability_enabled,
    )
    Path(harness.context.workdir, "large.txt").write_text(full_output)
    response, termination, extra = await harness.execute(30)
    assert termination.reason == "completed", termination
    assert extra["harness_version"] == "2.4.6"
    assert extra["runtime"]["pid"] != os.getpid()
    cleanup = json.loads((harness.directory / "cleanup.json").read_text())
    assert extra["runtime"]["pid"] == cleanup["worker_pid"]
    assert cleanup["supervisor_pid"] == harness.sandbox.runners[0].pid
    assert cleanup["remaining_pids"] == []
    assert len(response.output) == 9
    assert response.usage.total_tokens == 39
    assert response.tool_choice == "required"
    assert (harness.directory / "trajectory.json").is_file()
    for request in requests:
        NeMoGymResponseCreateParamsNonStreaming.model_validate(request)
        assert request["tools"] == [{"type": "function", **module.BASH_TOOL["function"], "strict": False}]
    assert requests[0]["input"][0]["content"] == module.MINI_CONFIG["agent"]["system_template"].rstrip("\n")
    prompt = requests[0]["input"][1]["content"]
    assert "Please solve this issue: Official task instruction" in prompt
    assert "Task skills are in /skills" in prompt
    assert platform.system() in prompt
    outcomes = [item for item in response.output if item.type == "function_call_output"]
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in outcomes[-1].output
    observation = json.loads(outcomes[1].output)
    assert observation["returncode"] == -1
    assert observation["output_head"] == full_output[:5000]
    assert observation["output_tail"] == full_output[-5000:]
    assert observation["exception_info"] == "Command timed out after 1 seconds."
    native = extra["mini_swe_trajectory"]
    assert (
        next(m for m in native["messages"] if m.get("tool_call_id") == "call_1")["extra"]["raw_output"] == full_output
    )
    assert [item["id"] for item in requests[-1]["input"] if item.get("type") == "reasoning"] == ["rs_0", "rs_1"]
    # The host dispatches the runner and relay I/O, never the model's shell commands.
    assert not any(command in dispatched for command in commands for dispatched in harness.sandbox.commands)
    if observability_enabled:
        records = extra["ng_agent_observations"]["records"]
        assert records[0]["status"] == "completed"
        assert [r["status"] for r in records[1:]] == ["completed", "timeout", "completed"]
        assert all(r["duration_ms"] >= 0 for r in records[1:])
        assert [t["step_count"] for t in extra["ng_trajectory"]["turns"]] == [1, 2, 3]
        assert len(extra["ng_trajectory"]["gaps"]) == (0 if response_id else 3)
    else:
        assert "ng_agent_observations" not in extra and "ng_trajectory" not in extra
