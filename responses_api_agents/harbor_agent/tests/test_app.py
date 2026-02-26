# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import tempfile
from asyncio import Semaphore
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.harbor_agent.app import (
    HarborAgent,
    HarborAgentConfig,
    HarborRunRequest,
)
from responses_api_agents.harbor_agent.utils import HarborAgentUtils


# ---------------------------------------------------------------------------
# Trajectory / step builders
# ---------------------------------------------------------------------------

_DEFAULT_AGENT_META = {"name": "terminus-2", "version": "2.0.0", "model_name": "hosted_vllm/test_model"}


def _make_step_user(step_id: int, message: str) -> Dict[str, Any]:
    return {"step_id": step_id, "source": "user", "message": message}


def _make_step_agent(
    step_id: int,
    message: str,
    *,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    observation_content: str = "",
    reasoning_content: Optional[str] = None,
    prompt_token_ids: Optional[List[int]] = None,
    completion_token_ids: Optional[List[int]] = None,
    logprobs: Optional[List[float]] = None,
    prompt_tokens: int = 500,
    completion_tokens: int = 100,
) -> Dict[str, Any]:
    step: Dict[str, Any] = {
        "step_id": step_id,
        "source": "agent",
        "model_name": "hosted_vllm/test_model",
        "message": message,
    }
    if reasoning_content is not None:
        step["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        step["tool_calls"] = tool_calls
    step["observation"] = {"results": [{"content": observation_content}]}
    metrics: Dict[str, Any] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if prompt_token_ids is not None:
        metrics["prompt_token_ids"] = prompt_token_ids
    if completion_token_ids is not None:
        metrics["completion_token_ids"] = completion_token_ids
    if logprobs is not None:
        metrics["logprobs"] = logprobs
    step["metrics"] = metrics
    return step


def _make_trajectory(
    steps: List[Dict[str, Any]],
    session_id: str = "test-session",
    total_prompt: int = 1200,
    total_completion: int = 180,
) -> Dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.5",
        "session_id": session_id,
        "agent": _DEFAULT_AGENT_META,
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cached_tokens": 0,
        },
    }


def _bash_tool_call(call_id: str, keystrokes: str, duration: float = 0.1) -> Dict[str, Any]:
    return {
        "tool_call_id": call_id,
        "function_name": "bash_command",
        "arguments": {"keystrokes": keystrokes, "duration": duration},
    }


def _raw_msg(analysis: str, plan: str, commands: list, task_complete: bool = False) -> str:
    return json.dumps({"analysis": analysis, "plan": plan, "commands": commands, "task_complete": task_complete})


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

DEFAULT_TRIAL_RESULT = {
    "task_name": "test_task_123",
    "agent_result": {"n_input_tokens": 100, "n_output_tokens": 50, "rollout_details": []},
    "verifier_result": {"rewards": {"reward": 1.0}},
}

_USER_STEP = _make_step_user(1, "You are an AI assistant. Solve this task:\nFix the bug in foo.py.")

DEFAULT_TRAJECTORY = _make_trajectory(
    steps=[
        _USER_STEP,
        _make_step_agent(
            2,
            "Analysis: I will look at foo.py.\nPlan: Read the file and fix the bug.",
            reasoning_content="Hidden reasoning step 1.",
            tool_calls=[_bash_tool_call("call_0_1", "cat foo.py\n")],
            observation_content="def foo():\n    return 1 + '2'\n",
            prompt_token_ids=[100, 101, 102],
            completion_token_ids=[200, 201, 202],
            logprobs=[-0.01, -0.02, -0.03],
        ),
        _make_step_agent(
            3,
            "Analysis: Found the bug. Fixing it now.\nPlan: Change '2' to 2.",
            reasoning_content="Hidden reasoning step 2.",
            tool_calls=[_bash_tool_call("call_1_1", "sed -i 's/+ '2'/+ 2/' foo.py\n")],
            prompt_tokens=700,
            completion_tokens=80,
            prompt_token_ids=[103, 104, 105],
            completion_token_ids=[203, 204, 205],
            logprobs=[-0.04, -0.05],
        ),
    ],
)

TRAJECTORY_RAW_CONTENT = _make_trajectory(
    steps=[
        _USER_STEP,
        _make_step_agent(
            2,
            _raw_msg(
                "I will look at foo.py.",
                "Read the file and fix the bug.",
                [{"keystrokes": "cat foo.py\n", "duration": 0.1}],
            ),
            observation_content="def foo():\n    return 1 + '2'\n",
            prompt_token_ids=[100, 101, 102],
            completion_token_ids=[200, 201, 202],
            logprobs=[-0.01, -0.02, -0.03],
        ),
        _make_step_agent(
            3,
            _raw_msg(
                "Found the bug. Fixing it now.",
                "Change '2' to 2.",
                [{"keystrokes": "sed -i 's/+ '2'/+ 2/' foo.py\n", "duration": 0.1}],
                task_complete=True,
            ),
            prompt_tokens=700,
            completion_tokens=80,
            prompt_token_ids=[103, 104, 105],
            completion_token_ids=[203, 204, 205],
            logprobs=[-0.04, -0.05],
        ),
    ],
)

TRAJECTORY_RAW_CONTENT_MULTI_CMD = _make_trajectory(
    steps=[
        _make_step_user(1, "Create hello.txt with Hello, world!"),
        _make_step_agent(
            2,
            _raw_msg(
                "I need to create the file.",
                "Write and verify the file.",
                [
                    {"keystrokes": "echo 'Hello, world!' > hello.txt\n", "duration": 0.1},
                    {"keystrokes": "cat hello.txt\n", "duration": 0.1},
                ],
            ),
            observation_content="Hello, world!\n",
            prompt_tokens=300,
            completion_tokens=60,
        ),
    ],
    total_prompt=300,
    total_completion=60,
)

TRAJECTORY_NO_TOKEN_DETAILS = _make_trajectory(
    steps=[
        _USER_STEP,
        _make_step_agent(
            2,
            "Analysis: I will look at foo.py.\nPlan: Read the file and fix the bug.",
            tool_calls=[_bash_tool_call("call_0_1", "cat foo.py\n")],
            observation_content="def foo():\n    return 1 + '2'\n",
        ),
        _make_step_agent(
            3,
            "Analysis: Found the bug. Fixing it now.\nPlan: Change '2' to 2.",
            tool_calls=[_bash_tool_call("call_1_1", "sed -i 's/+ '2'/+ 2/' foo.py\n")],
            prompt_tokens=700,
            completion_tokens=80,
        ),
    ],
)


# ---------------------------------------------------------------------------
# App helpers
# ---------------------------------------------------------------------------


def _make_server(**config_overrides) -> HarborAgent:
    """Create Harbor agent server with test defaults."""
    defaults: Dict[str, Any] = dict(
        name="harbor_agent",
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        concurrency=1,
        model_server={"type": "responses_api_models", "name": "test_model_server"},
        harbor_agent_name="terminus-2",
        harbor_datasets={"scientific": {"local_dataset_path": "/tmp/test_dataset"}},
        harbor_environment_type="docker",
        harbor_jobs_dir="/tmp/harbor_jobs",
    )
    defaults.update(config_overrides)
    config = HarborAgentConfig(**defaults)
    return HarborAgent.model_construct(
        config=config,
        server_client=MagicMock(),
        sem=Semaphore(config.concurrency),
    )


def _make_run_request(instance_id="scientific::test_task_123", **kwargs) -> HarborRunRequest:
    params: Dict[str, Any] = dict(temperature=1.0, top_p=1.0, input=[])
    params.update(kwargs)
    return HarborRunRequest(
        instance_id=instance_id,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(**params),
    )


_GLOBAL_CONFIG = {
    "policy_model_name": "test_model",
    "test_model_server": {"responses_api_models": {"vllm_model": {"host": "policy-host", "port": 9000}}},
}


@contextmanager
def _harbor_run_mocks(
    trial_result: Optional[Dict[str, Any]] = None,
    trajectory: Optional[Dict[str, Any]] = None,
    side_effect: Optional[Exception] = None,
):
    """Patch external deps and wire up mocks for HarborAgent.run()."""
    with (
        patch("responses_api_agents.harbor_agent.app.get_global_config_dict") as mock_gc,
        patch("responses_api_agents.harbor_agent.app.runner_ray_remote") as mock_ray,
        patch("asyncio.to_thread") as mock_to_thread,
        patch.object(HarborAgent, "_build_job_config", return_value={"job_name": "mock_job"}),
    ):
        mock_gc.return_value = _GLOBAL_CONFIG
        mock_ray.remote.return_value = MagicMock()

        if side_effect:
            mock_to_thread.side_effect = side_effect
        else:
            trial_dir = tempfile.mkdtemp(prefix="harbor_trial_")
            (Path(trial_dir) / "result.json").write_text(json.dumps(trial_result or DEFAULT_TRIAL_RESULT))
            if trajectory is not None:
                agent_dir = Path(trial_dir) / "agent"
                agent_dir.mkdir(parents=True, exist_ok=True)
                (agent_dir / "trajectory.json").write_text(json.dumps(trajectory))
            mock_to_thread.return_value = trial_dir

        yield


# ===========================================================================
#  App tests
# ===========================================================================


class TestApp:
    async def test_run_with_token_details(self):
        server = _make_server()
        with _harbor_run_mocks(trajectory=DEFAULT_TRAJECTORY):
            response = await server.run(_make_run_request())

        assert response.reward == 1.0
        assert len(response.response.output) == 6

        msg0, msg3 = response.response.output[0], response.response.output[3]
        assert msg0.prompt_token_ids == [100, 101, 102]
        assert msg0.generation_token_ids == [200, 201, 202]
        assert msg0.generation_log_probs == [-0.01, -0.02, -0.03]
        assert msg3.prompt_token_ids == [103, 104, 105]
        assert msg3.generation_token_ids == [203, 204, 205]

        assert response.response.parallel_tool_calls is False
        assert response.response.id.startswith("resp_")
        assert len(response.responses_create_params.input) == 1
        assert "Fix the bug" in response.responses_create_params.input[0].content

    async def test_run_without_token_details(self):
        server = _make_server()
        trial_result = {
            **DEFAULT_TRIAL_RESULT,
            "agent_result": {"n_input_tokens": 1200, "n_output_tokens": 180, "rollout_details": []},
        }
        with _harbor_run_mocks(trial_result=trial_result, trajectory=TRAJECTORY_NO_TOKEN_DETAILS):
            response = await server.run(_make_run_request())

        out = [o.model_dump() for o in response.response.output[:3]]
        assert [o["type"] for o in out] == ["message", "function_call", "function_call_output"]
        assert "prompt_token_ids" not in out[0]
        assert "I will look at foo.py" in out[0]["content"][0]["text"]
        assert response.response.usage.total_tokens == 1380

    async def test_run_failed_execution(self):
        server = _make_server()
        with _harbor_run_mocks(side_effect=Exception("Harbor job failed")):
            response = await server.run(
                _make_run_request(instance_id="scientific::fail_task", temperature=0.3, top_p=0.95)
            )

        assert response.reward == 0.0
        assert len(response.response.output) == 0
        assert response.responses_create_params.temperature == 0.3
        assert response.responses_create_params.input == []

    @pytest.mark.parametrize(
        "model_name, expected",
        [
            ("/lustre/models/nano-v3-sft-hf", "nano-v3-sft-hf"),
            ("Qwen/Qwen3-8B", "Qwen3-8B"),
            ("my-model", "my-model"),
        ],
    )
    def test_extract_model_name(self, model_name, expected) -> None:
        assert HarborAgent._extract_model_name(model_name) == expected

    def test_path_sanitization(self) -> None:
        server = _make_server()
        ts = datetime(2026, 2, 10, 12, 34, 56, tzinfo=timezone.utc)

        assert (
            server._get_results_output_dir("deepseek-ai/DeepSeek-V3.2", "scientific", ts).parts[-1] == "DeepSeek-V3.2"
        )
        assert server._get_jobs_output_dir("deepseek-ai/DeepSeek-V3.2", "scientific", ts).parts[-1] == "DeepSeek-V3.2"
        assert server._get_results_output_dir("my-plain-model", "scientific", ts).parts[-1] == "my-plain-model"

    @pytest.mark.parametrize(
        "instance_id, expected_alias, expected_task",
        [
            ("scientific::task_001", "scientific", "task_001"),
            ("terminal_bench::tb2_math_7", "terminal_bench", "tb2_math_7"),
        ],
    )
    def test_parse_instance_id(self, instance_id: str, expected_alias: str, expected_task: str) -> None:
        alias, task = HarborAgent._parse_instance_id(instance_id)
        assert alias == expected_alias
        assert task == expected_task

    @pytest.mark.parametrize("instance_id", ["", "scientific", "::task", "scientific::"])
    def test_parse_instance_id_rejects_invalid_values(self, instance_id: str) -> None:
        with pytest.raises(ValueError, match="instance_id must be in the form"):
            HarborAgent._parse_instance_id(instance_id)


# ===========================================================================
#  Utils tests
# ===========================================================================


class TestExtractInputFromTrajectory:
    def test_extracts_user_messages(self) -> None:
        msgs = HarborAgentUtils.extract_input_from_trajectory(DEFAULT_TRAJECTORY)
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert "Fix the bug in foo.py" in msgs[0].content

    @pytest.mark.parametrize("trajectory", [None, {"steps": []}])
    def test_returns_empty(self, trajectory) -> None:
        assert HarborAgentUtils.extract_input_from_trajectory(trajectory) == []

    def test_stops_at_first_agent_step(self) -> None:
        trajectory = {
            "steps": [
                {"step_id": 1, "source": "user", "message": "System prompt"},
                {"step_id": 2, "source": "user", "message": "Task description"},
                {"step_id": 3, "source": "agent", "message": "OK"},
                {"step_id": 4, "source": "user", "message": "Follow-up"},
            ]
        }
        msgs = HarborAgentUtils.extract_input_from_trajectory(trajectory)
        assert len(msgs) == 2
        assert msgs[1].content == "Task description"


class TestTrialResultToResponses:
    def test_training_fields(self) -> None:
        items = HarborAgentUtils.trial_result_to_responses(DEFAULT_TRIAL_RESULT, DEFAULT_TRAJECTORY)
        assert len(items) == 6
        assert items[0]["prompt_token_ids"] == [100, 101, 102]
        assert items[0]["generation_token_ids"] == [200, 201, 202]
        assert items[3]["generation_token_ids"] == [203, 204, 205]
        assert "<think>Hidden reasoning step 1.</think>" in items[0]["content"][0]["text"]
        assert "<think>Hidden reasoning step 2.</think>" in items[3]["content"][0]["text"]

    def test_returns_empty_without_trajectory(self) -> None:
        assert HarborAgentUtils.trial_result_to_responses(DEFAULT_TRIAL_RESULT, None) == []

    def test_omits_training_fields_without_token_details(self) -> None:
        items = HarborAgentUtils.trial_result_to_responses(DEFAULT_TRIAL_RESULT, TRAJECTORY_NO_TOKEN_DETAILS)
        assert len(items) == 6
        assert "prompt_token_ids" not in items[0]
        assert "I will look at foo.py" in items[0]["content"][0]["text"]


class TestExtractUsage:
    @pytest.mark.parametrize(
        "trial_result, trajectory, expected_total",
        [
            (DEFAULT_TRIAL_RESULT, DEFAULT_TRAJECTORY, 1380),
            (DEFAULT_TRIAL_RESULT, None, 150),
            ({"agent_result": None}, None, 0),
        ],
    )
    def test_extract_usage(self, trial_result, trajectory, expected_total) -> None:
        assert HarborAgentUtils.extract_usage(trial_result, trajectory)["total_tokens"] == expected_total


class TestExtractReward:
    @pytest.mark.parametrize(
        "verifier_result, expected",
        [
            ({"rewards": {"reward": 1.0}}, 1.0),
            ({"rewards": {"reward": 0.0}}, 0.0),
            (None, 0.0),
            ({}, 0.0),
            ({"rewards": {"accuracy": 0.75}}, 0.75),
        ],
    )
    def test_extract_reward(self, verifier_result, expected) -> None:
        assert HarborAgentUtils.extract_reward(verifier_result) == expected


# ===========================================================================
#  Raw content parsing tests
# ===========================================================================


class TestExtractJsonObject:
    def test_valid_json(self) -> None:
        assert HarborAgentUtils._extract_json_object('{"a": 1}') == {"a": 1}

    def test_json_with_surrounding_text(self) -> None:
        result = HarborAgentUtils._extract_json_object('Here:\n{"a": 1, "b": [{"c": 2}]}\nDone.')
        assert result == {"a": 1, "b": [{"c": 2}]}

    @pytest.mark.parametrize("text", ["not json", "", "{broken", "[1, 2, 3]"])
    def test_returns_none_for_invalid(self, text) -> None:
        assert HarborAgentUtils._extract_json_object(text) is None


class TestParseRawContentToolCalls:
    def test_single_command(self) -> None:
        calls = HarborAgentUtils._parse_raw_content_tool_calls(
            _raw_msg("test", "test", [{"keystrokes": "cat foo.py\n", "duration": 0.1}]),
            0,
        )
        assert len(calls) == 1
        assert calls[0]["tool_call_id"] == "call_0_1"
        assert calls[0]["function_name"] == "bash_command"
        assert calls[0]["arguments"]["keystrokes"] == "cat foo.py\n"

    def test_multiple_commands(self) -> None:
        calls = HarborAgentUtils._parse_raw_content_tool_calls(
            _raw_msg("test", "test", [{"keystrokes": "echo hi\n"}, {"keystrokes": "cat f\n"}]),
            2,
        )
        assert [c["tool_call_id"] for c in calls] == ["call_2_1", "call_2_2"]

    def test_task_complete(self) -> None:
        calls = HarborAgentUtils._parse_raw_content_tool_calls(
            _raw_msg("Done", "done", [{"keystrokes": "echo done\n"}], task_complete=True),
            3,
        )
        assert len(calls) == 1
        assert calls[0]["function_name"] == "bash_command"

    def test_task_complete_string_true(self) -> None:
        msg = json.dumps({"analysis": "Done", "plan": "done", "commands": [], "task_complete": "true"})
        calls = HarborAgentUtils._parse_raw_content_tool_calls(msg, 0)
        assert len(calls) == 0

    def test_missing_duration_defaults(self) -> None:
        calls = HarborAgentUtils._parse_raw_content_tool_calls(
            _raw_msg("test", "test", [{"keystrokes": "ls\n"}]),
            0,
        )
        assert calls[0]["arguments"]["duration"] == 1.0

    def test_empty_commands(self) -> None:
        assert HarborAgentUtils._parse_raw_content_tool_calls(_raw_msg("w", "w", []), 0) == []

    @pytest.mark.parametrize("text", ["not json", ""])
    def test_invalid_message(self, text) -> None:
        assert HarborAgentUtils._parse_raw_content_tool_calls(text, 0) == []

    def test_skips_invalid_commands(self) -> None:
        msg = json.dumps(
            {
                "analysis": "t",
                "plan": "t",
                "commands": [
                    "not a dict",
                    {"no_keystrokes": True},
                    {"keystrokes": "ls\n", "duration": 0.1},
                ],
            }
        )
        calls = HarborAgentUtils._parse_raw_content_tool_calls(msg, 0)
        assert len(calls) == 1
        assert calls[0]["tool_call_id"] == "call_0_3"


class TestTrajectoryToResponsesRawContent:
    def test_parses_function_calls_from_raw_message(self) -> None:
        items = HarborAgentUtils.trajectory_to_responses(TRAJECTORY_RAW_CONTENT)
        # Step 1: message + fc + fco = 3; Step 2: message + fc + fco = 3
        assert len(items) == 6

    def test_raw_message_preserved(self) -> None:
        items = HarborAgentUtils.trajectory_to_responses(TRAJECTORY_RAW_CONTENT)
        assert '"analysis"' in items[0]["content"][0]["text"]

    def test_function_call_ids(self) -> None:
        items = HarborAgentUtils.trajectory_to_responses(TRAJECTORY_RAW_CONTENT)
        assert items[1]["call_id"] == "call_0_1"
        assert items[1]["name"] == "bash_command"
        assert items[4]["call_id"] == "call_1_1"
        assert items[5]["type"] == "function_call_output"
        assert items[5]["call_id"] == "call_1_1"
        assert all(i.get("name") != "mark_task_complete" for i in items if i.get("type") == "function_call")

    def test_observation_linked(self) -> None:
        items = HarborAgentUtils.trajectory_to_responses(TRAJECTORY_RAW_CONTENT)
        assert items[2]["type"] == "function_call_output"
        assert items[2]["call_id"] == "call_0_1"
        assert "def foo():" in items[2]["output"]

    def test_training_fields_preserved(self) -> None:
        items = HarborAgentUtils.trajectory_to_responses(TRAJECTORY_RAW_CONTENT)
        assert items[0]["prompt_token_ids"] == [100, 101, 102]
        assert items[0]["generation_token_ids"] == [200, 201, 202]

    def test_multi_command_step(self) -> None:
        items = HarborAgentUtils.trajectory_to_responses(TRAJECTORY_RAW_CONTENT_MULTI_CMD)
        assert len(items) == 4
        assert [i["type"] for i in items] == ["message", "function_call", "function_call", "function_call_output"]
        assert items[1]["call_id"] == "call_0_1"
        assert items[2]["call_id"] == "call_0_2"

    def test_existing_tool_calls_not_overridden(self) -> None:
        """When tool_calls are present (raw_content=false), raw parsing is skipped."""
        items = HarborAgentUtils.trajectory_to_responses(DEFAULT_TRAJECTORY)
        assert len(items) == 6
        assert items[1]["call_id"] == "call_0_1"


# ===========================================================================
#  Merge reasoning tests
# ===========================================================================


class TestMergeMessageAndReasoning:
    def test_prepends_reasoning_in_think_tags(self) -> None:
        assert HarborAgentUtils._merge_message_and_reasoning("answer", "thinking") == "<think>thinking</think>answer"

    def test_returns_message_when_no_reasoning(self) -> None:
        assert HarborAgentUtils._merge_message_and_reasoning("answer", None) == "answer"
        assert HarborAgentUtils._merge_message_and_reasoning("answer", "") == "answer"
