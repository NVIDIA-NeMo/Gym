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
            "tool_calls": [
                {
                    "tool_call_id": "call_0_1",
                    "function_name": "bash_command",
                    "arguments": {"keystrokes": "cat foo.py\n", "duration": 0.1},
                }
            ],
            "observation": {"results": [{"content": "def foo():\n    return 1 + '2'\n"}]},
            "metrics": {"prompt_tokens": 500, "completion_tokens": 100, "logprobs": [-0.01, -0.02, -0.03]},
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
            "metrics": {"prompt_tokens": 700, "completion_tokens": 80, "logprobs": [-0.04, -0.05]},
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
        harbor_model_prefix="hosted_vllm",
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
    """Create Harbor agent server with test defaults."""
    return HarborAgent(config=create_test_config(**config_overrides), server_client=MagicMock())


# ===========================================================================
#  Core app tests
# ===========================================================================


class TestApp:
    @patch("responses_api_agents.harbor_agent.app.get_global_config_dict")
    @patch("responses_api_agents.harbor_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_with_rollout_details_overlays_training_fields(self, mock_to_thread, mock_ray, mock_gc):
        server = _make_server()
        setup_harbor_run_mock(mock_to_thread, mock_ray, mock_gc, trajectory=DEFAULT_TRAJECTORY)

        response = await server.run(create_run_request())

        assert response.reward == 1.0
        assert len(response.response.output) == 6

        msg0 = response.response.output[0]
        msg3 = response.response.output[3]
        assert msg0.prompt_token_ids == [1, 2, 3]
        assert msg0.generation_token_ids == [10, 11, 12]
        assert msg0.generation_log_probs == [-0.1, -0.2, -0.3]
        assert msg3.prompt_token_ids == [4, 5, 6]
        assert msg3.generation_token_ids == [13, 14, 15]
        assert msg3.generation_log_probs == [-0.4, -0.5, -0.6]

        # Contract requested in this thread.
        assert response.response.parallel_tool_calls is False
        assert response.response.id.startswith("resp_")
        assert len(response.responses_create_params.input) == 1
        assert "Fix the bug" in response.responses_create_params.input[0].content

    @patch("responses_api_agents.harbor_agent.app.get_global_config_dict")
    @patch("responses_api_agents.harbor_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_without_rollout_details_omits_training_fields(self, mock_to_thread, mock_ray, mock_gc):
        server = _make_server()
        trial_result = {
            **DEFAULT_TRIAL_RESULT,
            "agent_result": {"n_input_tokens": 1200, "n_output_tokens": 180, "rollout_details": []},
        }
        setup_harbor_run_mock(mock_to_thread, mock_ray, mock_gc, trial_result=trial_result, trajectory=DEFAULT_TRAJECTORY)

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

    def test_resolve_model_name_requires_prefix(self) -> None:
        with pytest.raises(ValueError, match="harbor_model_prefix is required"):
            _make_server(harbor_model_prefix=None)._resolve_model_name("test_model")

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
    def test_prefers_rollout_details(self) -> None:
        items = HarborAgentUtils.trial_result_to_responses(DEFAULT_TRIAL_RESULT, DEFAULT_TRAJECTORY)
        assert len(items) == 6
        assert items[0]["prompt_token_ids"] == [1, 2, 3]
        assert items[3]["prompt_token_ids"] == [4, 5, 6]
        assert "I will look at foo.py" in items[0]["content"][0]["text"]

    def test_rollout_only_without_trajectory(self) -> None:
        items = HarborAgentUtils.trial_result_to_responses(DEFAULT_TRIAL_RESULT, None)
        assert len(items) == 2
        assert items[0]["prompt_token_ids"] == [1, 2, 3]
        assert items[1]["prompt_token_ids"] == [4, 5, 6]
        assert items[0]["content"][0]["text"] == ""

    def test_falls_back_to_trajectory(self) -> None:
        result = {**DEFAULT_TRIAL_RESULT, "agent_result": {"rollout_details": [], "n_input_tokens": 100, "n_output_tokens": 50}}
        items = HarborAgentUtils.trial_result_to_responses(result, DEFAULT_TRAJECTORY)
        assert len(items) == 6
        assert "generation_log_probs" not in items[0]

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
