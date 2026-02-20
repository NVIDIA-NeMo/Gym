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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
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
# Test data
# ---------------------------------------------------------------------------

DEFAULT_TRIAL_RESULT = {
    "task_name": "test_task_123",
    "agent_result": {
        "n_input_tokens": 100,
        "n_output_tokens": 50,
        "rollout_details": [
            {
                "prompt_token_ids": [[1, 2, 3], [4, 5, 6]],
                "completion_token_ids": [[10, 11, 12], [13, 14, 15]],
                "logprobs": [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]],
            }
        ],
    },
    "verifier_result": {"rewards": {"reward": 1.0}},
}

DEFAULT_TRAJECTORY = {
    "schema_version": "ATIF-v1.5",
    "session_id": "test-session-123",
    "agent": {"name": "terminus-2", "version": "2.0.0", "model_name": "hosted_vllm/test_model"},
    "steps": [
        {
            "step_id": 1,
            "source": "user",
            "message": "You are an AI assistant. Solve this task:\nFix the bug in foo.py.",
        },
        {
            "step_id": 2,
            "source": "agent",
            "model_name": "hosted_vllm/test_model",
            "message": "Analysis: I will look at foo.py.\nPlan: Read the file and fix the bug.",
            "reasoning_content": "Hidden reasoning step 1.",
            "tool_calls": [
                {
                    "tool_call_id": "call_0_1",
                    "function_name": "bash_command",
                    "arguments": {"keystrokes": "cat foo.py\n", "duration": 0.1},
                }
            ],
            "observation": {"results": [{"content": "def foo():\n    return 1 + '2'\n"}]},
            "metrics": {
                "prompt_tokens": 500,
                "completion_tokens": 100,
                "prompt_token_ids": [100, 101, 102],
                "completion_token_ids": [200, 201, 202],
                "logprobs": [-0.01, -0.02, -0.03],
            },
        },
        {
            "step_id": 3,
            "source": "agent",
            "model_name": "hosted_vllm/test_model",
            "message": "Analysis: Found the bug. Fixing it now.\nPlan: Change '2' to 2.",
            "reasoning_content": "Hidden reasoning step 2.",
            "tool_calls": [
                {
                    "tool_call_id": "call_1_1",
                    "function_name": "bash_command",
                    "arguments": {"keystrokes": "sed -i 's/+ '2'/+ 2/' foo.py\n", "duration": 0.1},
                }
            ],
            "observation": {"results": [{"content": ""}]},
            "metrics": {
                "prompt_tokens": 700,
                "completion_tokens": 80,
                "prompt_token_ids": [103, 104, 105],
                "completion_token_ids": [203, 204, 205],
                "logprobs": [-0.04, -0.05],
            },
        },
    ],
    "final_metrics": {"total_prompt_tokens": 1200, "total_completion_tokens": 180, "total_cached_tokens": 0},
}


# Trajectory without token-level details (no prompt_token_ids, completion_token_ids, logprobs).
# Used to verify output messages are plain NeMoGymResponseOutputMessage (no training fields).
# Trajectory produced with raw_content=true: step.message contains the full
# raw LLM JSON, and there are NO tool_calls.  Function calls must be parsed
# from the message on the Gym side.
TRAJECTORY_RAW_CONTENT = {
    "schema_version": "ATIF-v1.5",
    "session_id": "test-session-raw",
    "agent": {"name": "terminus-2", "version": "2.0.0", "model_name": "hosted_vllm/test_model"},
    "steps": [
        {
            "step_id": 1,
            "source": "user",
            "message": "You are an AI assistant. Solve this task:\nFix the bug in foo.py.",
        },
        {
            "step_id": 2,
            "source": "agent",
            "model_name": "hosted_vllm/test_model",
            "message": json.dumps({
                "analysis": "I will look at foo.py.",
                "plan": "Read the file and fix the bug.",
                "commands": [
                    {"keystrokes": "cat foo.py\n", "duration": 0.1},
                ],
                "task_complete": False,
            }),
            "observation": {"results": [{"content": "def foo():\n    return 1 + '2'\n"}]},
            "metrics": {
                "prompt_tokens": 500,
                "completion_tokens": 100,
                "prompt_token_ids": [100, 101, 102],
                "completion_token_ids": [200, 201, 202],
                "logprobs": [-0.01, -0.02, -0.03],
            },
        },
        {
            "step_id": 3,
            "source": "agent",
            "model_name": "hosted_vllm/test_model",
            "message": json.dumps({
                "analysis": "Found the bug. Fixing it now.",
                "plan": "Change '2' to 2.",
                "commands": [
                    {"keystrokes": "sed -i 's/+ '2'/+ 2/' foo.py\n", "duration": 0.1},
                ],
                "task_complete": True,
            }),
            "observation": {"results": [{"content": ""}]},
            "metrics": {
                "prompt_tokens": 700,
                "completion_tokens": 80,
                "prompt_token_ids": [103, 104, 105],
                "completion_token_ids": [203, 204, 205],
                "logprobs": [-0.04, -0.05],
            },
        },
    ],
    "final_metrics": {"total_prompt_tokens": 1200, "total_completion_tokens": 180, "total_cached_tokens": 0},
}


# Trajectory with raw_content=true and multiple commands per step.
TRAJECTORY_RAW_CONTENT_MULTI_CMD = {
    "schema_version": "ATIF-v1.5",
    "session_id": "test-session-raw-multi",
    "agent": {"name": "terminus-2", "version": "2.0.0", "model_name": "hosted_vllm/test_model"},
    "steps": [
        {
            "step_id": 1,
            "source": "user",
            "message": "Create hello.txt with Hello, world!",
        },
        {
            "step_id": 2,
            "source": "agent",
            "model_name": "hosted_vllm/test_model",
            "message": json.dumps({
                "analysis": "I need to create the file.",
                "plan": "Write and verify the file.",
                "commands": [
                    {"keystrokes": "echo 'Hello, world!' > hello.txt\n", "duration": 0.1},
                    {"keystrokes": "cat hello.txt\n", "duration": 0.1},
                ],
                "task_complete": False,
            }),
            "observation": {"results": [{"content": "Hello, world!\n"}]},
            "metrics": {"prompt_tokens": 300, "completion_tokens": 60},
        },
    ],
    "final_metrics": {"total_prompt_tokens": 300, "total_completion_tokens": 60, "total_cached_tokens": 0},
}


TRAJECTORY_NO_TOKEN_DETAILS = {
    "schema_version": "ATIF-v1.5",
    "session_id": "test-session-456",
    "agent": {"name": "terminus-2", "version": "2.0.0", "model_name": "hosted_vllm/test_model"},
    "steps": [
        {
            "step_id": 1,
            "source": "user",
            "message": "You are an AI assistant. Solve this task:\nFix the bug in foo.py.",
        },
        {
            "step_id": 2,
            "source": "agent",
            "model_name": "hosted_vllm/test_model",
            "message": "Analysis: I will look at foo.py.\nPlan: Read the file and fix the bug.",
            "tool_calls": [
                {
                    "tool_call_id": "call_0_1",
                    "function_name": "bash_command",
                    "arguments": {"keystrokes": "cat foo.py\n", "duration": 0.1},
                }
            ],
            "observation": {"results": [{"content": "def foo():\n    return 1 + '2'\n"}]},
            "metrics": {"prompt_tokens": 500, "completion_tokens": 100},
        },
        {
            "step_id": 3,
            "source": "agent",
            "model_name": "hosted_vllm/test_model",
            "message": "Analysis: Found the bug. Fixing it now.\nPlan: Change '2' to 2.",
            "tool_calls": [
                {
                    "tool_call_id": "call_1_1",
                    "function_name": "bash_command",
                    "arguments": {"keystrokes": "sed -i 's/+ '2'/+ 2/' foo.py\n", "duration": 0.1},
                }
            ],
            "observation": {"results": [{"content": ""}]},
            "metrics": {"prompt_tokens": 700, "completion_tokens": 80},
        },
    ],
    "final_metrics": {"total_prompt_tokens": 1200, "total_completion_tokens": 180, "total_cached_tokens": 0},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Normalize output items that may be dicts or Pydantic models."""
    return obj if isinstance(obj, dict) else obj.model_dump()


def create_test_config(**overrides) -> HarborAgentConfig:
    """Build an ``HarborAgentConfig`` with sensible test defaults.

    Pass keyword overrides for any field you want to change, e.g.
    ``create_test_config(harbor_agent_kwargs={"temperature": 0.5})``.
    """
    defaults: Dict[str, Any] = dict(
        name="harbor_agent",
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        concurrency=1,
        model_server={"type": "responses_api_models", "name": "test_model_server"},
        harbor_agent_name="terminus-2",
        harbor_local_dataset_path="/tmp/test_dataset",
        harbor_environment_type="docker",
        harbor_jobs_dir="/tmp/harbor_jobs",
    )
    defaults.update(overrides)
    return HarborAgentConfig(**defaults)


def setup_harbor_run_mock(
    mock_to_thread,
    mock_runner_ray_remote,
    mock_get_global_config,
    trial_result: Optional[Dict[str, Any]] = None,
    trajectory: Optional[Dict[str, Any]] = None,
):
    """Wire up all mocks for a successful ``run()`` call.

    Sets up global config mock, writes result/trajectory files to a temp
    directory, and routes the Ray mock to return it.
    """
    # Global config
    mock_get_global_config.return_value = {
        "policy_model_name": "test_model",
        "test_model_server": {
            "responses_api_models": {
                "vllm_model": {
                    "host": "policy-host",
                    "port": 9000,
                }
            }
        },
    }

    # Trial directory with result.json (+ optional trajectory.json)
    if trial_result is None:
        trial_result = DEFAULT_TRIAL_RESULT
    trial_dir = tempfile.mkdtemp(prefix="harbor_trial_")
    (Path(trial_dir) / "result.json").write_text(json.dumps(trial_result))
    if trajectory is not None:
        agent_dir = Path(trial_dir) / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "trajectory.json").write_text(json.dumps(trajectory))

    # Ray
    mock_runner_ray_remote.remote.return_value = MagicMock()
    mock_to_thread.return_value = trial_dir


def create_run_request(instance_id="test_task_123", **kwargs) -> HarborRunRequest:
    params: Dict[str, Any] = dict(temperature=1.0, top_p=1.0, input=[])
    params.update(kwargs)
    return HarborRunRequest(
        instance_id=instance_id,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(**params),
    )


def _make_server(**config_overrides) -> HarborAgent:
    """Create Harbor agent server with test defaults.

    Uses ``model_construct`` to bypass Pydantic validation of the
    ``server_client`` field (which expects a real ``ServerClient`` instance).
    """
    config = create_test_config(**config_overrides)
    server = HarborAgent.model_construct(
        config=config,
        server_client=MagicMock(),
        sem=Semaphore(config.concurrency),
    )
    return server


# ===========================================================================
#  Core app tests
# ===========================================================================


class TestApp:
    @patch("responses_api_agents.harbor_agent.app.get_global_config_dict")
    @patch("responses_api_agents.harbor_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_with_trajectory_token_details(self, mock_to_thread, mock_ray, mock_gc):
        server = _make_server()
        setup_harbor_run_mock(mock_to_thread, mock_ray, mock_gc, trajectory=DEFAULT_TRAJECTORY)

        response = await server.run(create_run_request())

        assert response.reward == 1.0
        assert len(response.response.output) == 6

        msg0 = response.response.output[0]
        msg3 = response.response.output[3]
        # Token details come from trajectory step metrics
        assert msg0.prompt_token_ids == [100, 101, 102]
        assert msg0.generation_token_ids == [200, 201, 202]
        assert msg0.generation_log_probs == [-0.01, -0.02, -0.03]
        assert msg3.prompt_token_ids == [103, 104, 105]
        assert msg3.generation_token_ids == [203, 204, 205]
        assert msg3.generation_log_probs == [-0.04, -0.05]

        # Contract requested in this thread.
        assert response.response.parallel_tool_calls is False
        assert response.response.id.startswith("resp_")
        assert len(response.responses_create_params.input) == 1
        assert "Fix the bug" in response.responses_create_params.input[0].content

    @patch("responses_api_agents.harbor_agent.app.get_global_config_dict")
    @patch("responses_api_agents.harbor_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_without_token_details_omits_training_fields(self, mock_to_thread, mock_ray, mock_gc):
        server = _make_server()
        trial_result = {
            **DEFAULT_TRIAL_RESULT,
            "agent_result": {"n_input_tokens": 1200, "n_output_tokens": 180, "rollout_details": []},
        }
        setup_harbor_run_mock(
            mock_to_thread, mock_ray, mock_gc,
            trial_result=trial_result,
            trajectory=TRAJECTORY_NO_TOKEN_DETAILS,
        )

        response = await server.run(create_run_request())

        output0 = _as_dict(response.response.output[0])
        output1 = _as_dict(response.response.output[1])
        output2 = _as_dict(response.response.output[2])
        assert output0["type"] == "message"
        assert output1["type"] == "function_call"
        assert output2["type"] == "function_call_output"
        assert "prompt_token_ids" not in output0
        assert "generation_token_ids" not in output0
        assert "generation_log_probs" not in output0
        assert "I will look at foo.py" in output0["content"][0]["text"]
        assert "Fix the bug" in response.responses_create_params.input[0].content
        assert response.response.usage.input_tokens == 1200
        assert response.response.usage.output_tokens == 180
        assert response.response.usage.total_tokens == 1380

    @patch("responses_api_agents.harbor_agent.app.get_global_config_dict")
    @patch("responses_api_agents.harbor_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_failed_execution(self, mock_to_thread, mock_ray, mock_gc):
        server = _make_server()
        mock_gc.return_value = {
            "policy_model_name": "test_model",
            "test_model_server": {
                "responses_api_models": {
                    "vllm_model": {
                        "host": "host",
                        "port": 9000,
                    }
                }
            },
        }
        mock_ray.remote.return_value = MagicMock()
        mock_to_thread.side_effect = Exception("Harbor job failed")

        response = await server.run(create_run_request(instance_id="fail_task", temperature=0.3, top_p=0.95))

        assert response.reward == 0.0
        assert len(response.response.output) == 0
        assert response.responses_create_params.temperature == 0.3
        assert response.responses_create_params.input == []

    def test_build_job_config_raises_without_dataset(self) -> None:
        server = _make_server(harbor_dataset_name=None, harbor_local_dataset_path=None)
        with pytest.raises(ValueError, match="requires a dataset"):
            server._build_job_config(
                "test_task",
                "hosted_vllm/test_model",
                "http://localhost:8000/v1",
                job_name="test_task__run",
                jobs_dir=Path("/tmp/harbor_jobs"),
            )

    def test_results_and_job_paths_sanitize_model_and_job_name(self) -> None:
        server = _make_server(harbor_dataset_name="terminal-bench", harbor_dataset_version="2.0")
        ts = datetime(2026, 2, 10, 12, 34, 56, tzinfo=timezone.utc)

        results_dir = server._get_results_output_dir("deepseek-ai/DeepSeek-V3.2", ts)
        jobs_dir = server._get_jobs_output_dir("deepseek-ai/DeepSeek-V3.2", ts)
        job_name = server._build_job_name("20260210_123456_1a2b")

        assert "deepseek-ai__DeepSeek-V3.2" == results_dir.parts[-1]
        assert "deepseek-ai__DeepSeek-V3.2" == jobs_dir.parts[-1]
        assert not job_name.startswith("ng_")


# ===========================================================================
#  Core utils tests
# ===========================================================================


class TestExtractInputFromTrajectory:
    def test_extracts_user_messages(self) -> None:
        msgs = HarborAgentUtils.extract_input_from_trajectory(DEFAULT_TRAJECTORY)
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert "Fix the bug in foo.py" in msgs[0].content

    def test_returns_empty_for_none(self) -> None:
        assert HarborAgentUtils.extract_input_from_trajectory(None) == []

    def test_returns_empty_for_no_steps(self) -> None:
        assert HarborAgentUtils.extract_input_from_trajectory({"steps": []}) == []

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
        assert msgs[0].content == "System prompt"
        assert msgs[1].content == "Task description"


class TestTrialResultToResponses:
    def test_reads_training_fields_from_trajectory_metrics(self) -> None:
        """Token IDs and logprobs come from trajectory step metrics."""
        items = HarborAgentUtils.trial_result_to_responses(DEFAULT_TRIAL_RESULT, DEFAULT_TRAJECTORY)
        assert len(items) == 6
        assert items[0]["prompt_token_ids"] == [100, 101, 102]
        assert items[0]["generation_token_ids"] == [200, 201, 202]
        assert items[0]["generation_log_probs"] == [-0.01, -0.02, -0.03]
        assert items[3]["prompt_token_ids"] == [103, 104, 105]
        assert items[3]["generation_token_ids"] == [203, 204, 205]
        assert items[3]["generation_log_probs"] == [-0.04, -0.05]
        assert "I will look at foo.py" in items[0]["content"][0]["text"]
        assert "<think>Hidden reasoning step 1.</think>" in items[0]["content"][0]["text"]
        assert "<think>Hidden reasoning step 2.</think>" in items[3]["content"][0]["text"]

    def test_returns_empty_without_trajectory(self) -> None:
        """Without a trajectory, output is empty regardless of trial_result."""
        items = HarborAgentUtils.trial_result_to_responses(DEFAULT_TRIAL_RESULT, None)
        assert items == []

    def test_omits_training_fields_without_token_details(self) -> None:
        """When trajectory metrics lack token IDs/logprobs, no training fields appear."""
        items = HarborAgentUtils.trial_result_to_responses(DEFAULT_TRIAL_RESULT, TRAJECTORY_NO_TOKEN_DETAILS)
        assert len(items) == 6
        assert "prompt_token_ids" not in items[0]
        assert "generation_token_ids" not in items[0]
        assert "generation_log_probs" not in items[0]
        assert "I will look at foo.py" in items[0]["content"][0]["text"]

    def test_falls_back_to_empty_output(self) -> None:
        result = {**DEFAULT_TRIAL_RESULT, "agent_result": {"rollout_details": [], "n_input_tokens": 100, "n_output_tokens": 50}}
        items = HarborAgentUtils.trial_result_to_responses(result, None)
        assert items == []

class TestExtractUsage:
    @pytest.mark.parametrize(
        "trial_result, trajectory, expected_total",
        [
            (DEFAULT_TRIAL_RESULT, DEFAULT_TRAJECTORY, 1380),
            (DEFAULT_TRIAL_RESULT, None, 150),
            ({"agent_result": None}, None, 0),
        ],
    )
    def test_extract_usage_paths(self, trial_result, trajectory, expected_total) -> None:
        usage = HarborAgentUtils.extract_usage(trial_result, trajectory)
        assert usage["total_tokens"] == expected_total


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
    def test_direct_json(self) -> None:
        msg = '{"analysis": "ok", "plan": "go", "commands": []}'
        result = HarborAgentUtils._extract_json_object(msg)
        assert result == {"analysis": "ok", "plan": "go", "commands": []}

    def test_json_with_surrounding_text(self) -> None:
        msg = 'Here is my response:\n{"analysis": "ok", "plan": "go", "commands": []}\nDone.'
        result = HarborAgentUtils._extract_json_object(msg)
        assert result is not None
        assert result["analysis"] == "ok"

    def test_nested_json(self) -> None:
        msg = json.dumps({
            "analysis": "test",
            "plan": "test",
            "commands": [{"keystrokes": "ls\n", "duration": 0.1}],
        })
        result = HarborAgentUtils._extract_json_object(msg)
        assert result is not None
        assert len(result["commands"]) == 1

    def test_returns_none_for_invalid(self) -> None:
        assert HarborAgentUtils._extract_json_object("not json at all") is None
        assert HarborAgentUtils._extract_json_object("") is None
        assert HarborAgentUtils._extract_json_object("{broken") is None

    def test_returns_none_for_non_dict(self) -> None:
        assert HarborAgentUtils._extract_json_object("[1, 2, 3]") is None


class TestParseRawContentToolCalls:
    def test_single_command(self) -> None:
        msg = json.dumps({
            "analysis": "Looking at foo.py",
            "plan": "Read the file",
            "commands": [{"keystrokes": "cat foo.py\n", "duration": 0.1}],
        })
        calls = HarborAgentUtils._parse_raw_content_tool_calls(msg, 0)
        assert len(calls) == 1
        assert calls[0]["tool_call_id"] == "call_0_1"
        assert calls[0]["function_name"] == "bash_command"
        assert calls[0]["arguments"]["keystrokes"] == "cat foo.py\n"
        assert calls[0]["arguments"]["duration"] == 0.1

    def test_multiple_commands(self) -> None:
        msg = json.dumps({
            "analysis": "Setup",
            "plan": "Create and verify",
            "commands": [
                {"keystrokes": "echo 'hello' > file.txt\n", "duration": 0.1},
                {"keystrokes": "cat file.txt\n", "duration": 0.1},
            ],
        })
        calls = HarborAgentUtils._parse_raw_content_tool_calls(msg, 2)
        assert len(calls) == 2
        assert calls[0]["tool_call_id"] == "call_2_1"
        assert calls[1]["tool_call_id"] == "call_2_2"

    def test_task_complete(self) -> None:
        msg = json.dumps({
            "analysis": "Done",
            "plan": "Mark complete",
            "commands": [{"keystrokes": "echo done\n", "duration": 0.1}],
            "task_complete": True,
        })
        calls = HarborAgentUtils._parse_raw_content_tool_calls(msg, 3)
        assert len(calls) == 2
        assert calls[0]["function_name"] == "bash_command"
        assert calls[1]["tool_call_id"] == "call_3_task_complete"
        assert calls[1]["function_name"] == "mark_task_complete"

    def test_task_complete_string_true(self) -> None:
        msg = json.dumps({
            "analysis": "Done",
            "plan": "Mark complete",
            "commands": [],
            "task_complete": "true",
        })
        calls = HarborAgentUtils._parse_raw_content_tool_calls(msg, 0)
        assert len(calls) == 1
        assert calls[0]["function_name"] == "mark_task_complete"

    def test_missing_duration_defaults(self) -> None:
        msg = json.dumps({
            "analysis": "test",
            "plan": "test",
            "commands": [{"keystrokes": "ls\n"}],
        })
        calls = HarborAgentUtils._parse_raw_content_tool_calls(msg, 0)
        assert calls[0]["arguments"]["duration"] == 1.0

    def test_empty_commands(self) -> None:
        msg = json.dumps({
            "analysis": "Waiting",
            "plan": "Wait for output",
            "commands": [],
        })
        calls = HarborAgentUtils._parse_raw_content_tool_calls(msg, 0)
        assert calls == []

    def test_invalid_message(self) -> None:
        assert HarborAgentUtils._parse_raw_content_tool_calls("not json", 0) == []
        assert HarborAgentUtils._parse_raw_content_tool_calls("", 0) == []

    def test_skips_invalid_commands(self) -> None:
        msg = json.dumps({
            "analysis": "test",
            "plan": "test",
            "commands": [
                "not a dict",
                {"no_keystrokes": True},
                {"keystrokes": "ls\n", "duration": 0.1},
            ],
        })
        calls = HarborAgentUtils._parse_raw_content_tool_calls(msg, 0)
        assert len(calls) == 1
        assert calls[0]["tool_call_id"] == "call_0_3"


class TestTrajectoryToResponsesRawContent:
    """Tests for trajectory_to_responses with raw_content=true trajectories."""

    def test_parses_function_calls_from_raw_message(self) -> None:
        items = HarborAgentUtils.trajectory_to_responses(TRAJECTORY_RAW_CONTENT)
        # 2 agent steps, each with: message + function_call + function_call_output
        # Step 2 also has task_complete → extra function_call
        # Step 1: message + bash_command fc + fco = 3
        # Step 2: message + bash_command fc + task_complete fc + fco = 4
        assert len(items) == 7

    def test_raw_message_preserved_in_output_text(self) -> None:
        items = HarborAgentUtils.trajectory_to_responses(TRAJECTORY_RAW_CONTENT)
        msg0_text = items[0]["content"][0]["text"]
        # The raw JSON should be in the output text
        assert '"analysis"' in msg0_text
        assert "I will look at foo.py" in msg0_text

    def test_function_call_ids_use_agent_step_index(self) -> None:
        items = HarborAgentUtils.trajectory_to_responses(TRAJECTORY_RAW_CONTENT)
        # First agent step (index 0): bash_command
        fc0 = items[1]
        assert fc0["type"] == "function_call"
        assert fc0["call_id"] == "call_0_1"
        assert fc0["name"] == "bash_command"
        args0 = json.loads(fc0["arguments"])
        assert args0["keystrokes"] == "cat foo.py\n"

        # Second agent step (index 1): bash_command + task_complete
        fc1 = items[4]
        assert fc1["type"] == "function_call"
        assert fc1["call_id"] == "call_1_1"
        fc_complete = items[5]
        assert fc_complete["call_id"] == "call_1_task_complete"
        assert fc_complete["name"] == "mark_task_complete"

    def test_observation_linked_to_function_call(self) -> None:
        items = HarborAgentUtils.trajectory_to_responses(TRAJECTORY_RAW_CONTENT)
        # First step: fco should link to call_0_1
        fco0 = items[2]
        assert fco0["type"] == "function_call_output"
        assert fco0["call_id"] == "call_0_1"
        assert "def foo():" in fco0["output"]

    def test_training_fields_preserved(self) -> None:
        items = HarborAgentUtils.trajectory_to_responses(TRAJECTORY_RAW_CONTENT)
        msg0 = items[0]
        assert msg0["prompt_token_ids"] == [100, 101, 102]
        assert msg0["generation_token_ids"] == [200, 201, 202]
        assert msg0["generation_log_probs"] == [-0.01, -0.02, -0.03]

    def test_multi_command_step(self) -> None:
        items = HarborAgentUtils.trajectory_to_responses(TRAJECTORY_RAW_CONTENT_MULTI_CMD)
        # 1 agent step: message + 2 function_calls + 1 function_call_output = 4
        assert len(items) == 4
        assert items[0]["type"] == "message"
        assert items[1]["type"] == "function_call"
        assert items[1]["call_id"] == "call_0_1"
        assert items[2]["type"] == "function_call"
        assert items[2]["call_id"] == "call_0_2"
        assert items[3]["type"] == "function_call_output"
        assert items[3]["call_id"] == "call_0_1"

    def test_existing_tool_calls_not_overridden(self) -> None:
        """When tool_calls are present (raw_content=false), parsing is skipped."""
        items = HarborAgentUtils.trajectory_to_responses(DEFAULT_TRAJECTORY)
        # Should work exactly as before — 6 items
        assert len(items) == 6
        fc0 = items[1]
        assert fc0["call_id"] == "call_0_1"
