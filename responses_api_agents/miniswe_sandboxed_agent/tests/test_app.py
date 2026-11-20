# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import platform
from pathlib import Path

import pytest
import yaml
from minisweagent.config import builtin_config_dir
from minisweagent.models.utils.actions_toolcall_response import BASH_TOOL_RESPONSE_API

from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.trajectory import project_bundle_v1
from responses_api_agents.miniswe_sandboxed_agent import harness as module


MINI_CONFIG = yaml.safe_load((builtin_config_dir / "mini.yaml").read_text())


@pytest.mark.parametrize("observability_enabled", [False, True])
@pytest.mark.parametrize("response_id", ["present", ""])
async def test_real_runner_preserves_model_history_and_tool_observations(
    tmp_path, runner_factory, observability_enabled, response_id
):
    requests = []
    full_output = "start" + "x" * 6000 + "MIDDLE_MUST_BE_ELIDED" + "y" * 6000 + "end"
    # Leave time for the shell and cat to start on loaded CI hosts so the
    # timeout exercises output truncation, not an empty startup observation.
    step_timeout = 10
    commands = ["echo first", "cat large.txt; sleep 60", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]

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
        config=module.MiniSWEConfig(step_timeout_sec=step_timeout),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[], tool_choice="required"),
        query=query,
        model_name="model",
        directory=tmp_path / "artifacts",
        observability_enabled=observability_enabled,
    )
    Path(harness.context.workdir, "large.txt").write_text(full_output)
    # The runner budget includes startup, separate from the command timeout under test.
    response, termination, extra = await harness.execute(120)
    assert termination.reason == "completed", termination
    assert extra["harness_version"] == "2.4.6"
    assert len(response.output) == 8
    assert response.usage.total_tokens == 39
    assert response.tool_choice == "required"
    assert (harness.directory / "trajectory.json").is_file()
    for request in requests:
        NeMoGymResponseCreateParamsNonStreaming.model_validate(request)
        assert request["tools"] == [{**BASH_TOOL_RESPONSE_API, "strict": False}]
    assert requests[0]["input"][0]["content"] == MINI_CONFIG["agent"]["system_template"].rstrip("\n")
    prompt = requests[0]["input"][1]["content"]
    assert "Please solve this issue: Official task instruction" in prompt
    assert "Task skills are in /skills" in prompt
    assert platform.system() in prompt
    outcomes = [item for item in response.output if item.type == "function_call_output"]
    assert len(outcomes) == 2  # Native submission exits before saving its tool observation.
    observation = json.loads(outcomes[1].output)
    assert observation["returncode"] == -1
    assert observation["output_head"] == full_output[:5000]
    # Some shells append a killed-process notice to stderr after the timeout.
    assert "end" in observation["output_tail"] and len(observation["output_tail"]) == 5000
    assert f"timed out after {step_timeout} seconds" in observation["exception_info"]
    native = extra["mini_swe_trajectory"]
    assert full_output in next(m for m in native["messages"] if m.get("call_id") == "call_1")["extra"]["raw_output"]
    assert [item["id"] for item in requests[-1]["input"] if item.get("type") == "reasoning"] == ["rs_0", "rs_1"]
    # The host dispatches one runner command, never the model's shell commands.
    assert not any(command in dispatched for command in commands for dispatched in harness.sandbox.commands)
    if observability_enabled:
        bundle = extra["ng_agent_observations"]
        records = bundle["records"]
        assert records[0]["status"] == "completed"
        tools = [r for r in records if r["kind"] == "tool_call"]
        # The submission has no tool output, but its successful command was recorded where it executed.
        assert [(t["tool_call_id"], t["status"]) for t in tools] == [
            ("call_0", "completed"),
            ("call_1", "timeout"),
            ("call_2", "completed"),
        ]
        for index, tool in enumerate(tools):
            (timing,) = tool["timings"]
            assert (timing["boundary"], timing["timing_source"]) == ("executor", "miniswe.environment")
            assert timing["started_at"] <= timing["completed_at"] and timing["duration_ms"] >= 0
            # The call joins the response that requested it.
            reference = tool.get("model_call_ref")
            assert (reference or {}).get("response_id") == (f"resp_test_{index}" if response_id else None)
        # The v1 view labels the executor boundary as such, with the recorded times unchanged.
        for tool, projected in zip(
            tools, (r for r in project_bundle_v1(bundle)["records"] if r["kind"] == "tool_call")
        ):
            assert projected["timing_source"] == "executor"
            for field in ("started_at", "completed_at", "duration_ms"):
                assert projected[field] == tool["timings"][0][field]
        # The saved trajectory is not a timing source.
        saved = json.loads((harness.directory / "trajectory.json").read_text())
        assert not any("started_at" in message.get("extra", {}) for message in saved["messages"])
        assert [t["step_count"] for t in extra["ng_trajectory"]["turns"]] == [1, 2, 2]
        assert [gap["code"] for gap in extra["ng_trajectory"]["gaps"]] == (
            [] if response_id else ["model_call_reference_unavailable"] * 3
        )
    else:
        assert "ng_agent_observations" not in extra and "ng_trajectory" not in extra
