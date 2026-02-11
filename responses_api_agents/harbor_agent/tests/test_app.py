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
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
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
    "trial_name": "test_task_123__abc1234",
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


def _get(obj, key):
    """Access a field on either a dict or a Pydantic model.

    Pydantic's discriminated-union parsing sometimes leaves output items as
    raw dicts instead of resolving to the concrete model type.
    """
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


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
    mock_load_from_global_config,
    trial_result: Optional[Dict[str, Any]] = None,
    trajectory: Optional[Dict[str, Any]] = None,
):
    """Wire up all mocks for a successful ``run()`` call.

    Sets up the ServerClient, writes result/trajectory files to a temp
    directory, and routes the Ray mock to return it.
    """
    # ServerClient
    sc = MagicMock()
    sc.global_config_dict = {"policy_model_name": "test_model", "policy_base_url": "http://policy-host:9000/v1"}
    mock_load_from_global_config.return_value = sc

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
    """Shorthand: create an ``HarborAgent`` with a mock ``ServerClient``."""
    return HarborAgent(config=create_test_config(**config_overrides), server_client=MagicMock(spec=ServerClient))


# ===========================================================================
#  TestApp — agent lifecycle, run(), and _build_job_config
# ===========================================================================


class TestApp:
    def test_sanity(self) -> None:
        _make_server()

    # ---- run() --------------------------------------------------------- #

    @patch("responses_api_agents.harbor_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.harbor_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_rollout_details_take_priority(self, mock_to_thread, mock_ray, mock_sc):
        """When rollout_details + trajectory exist, keep full trajectory and enrich assistant turns."""
        server = _make_server()
        setup_harbor_run_mock(mock_to_thread, mock_ray, mock_sc, trajectory=DEFAULT_TRAJECTORY)

        response = await server.run(create_run_request())

        assert response.reward == 1.0
        # Keep rich trajectory output: 2 agent steps x (message + function_call + function_call_output) = 6
        assert len(response.response.output) == 6
        out0 = response.response.output[0]
        assert out0.prompt_token_ids == [1, 2, 3]
        assert out0.generation_token_ids == [10, 11, 12]
        assert out0.generation_log_probs == [-0.1, -0.2, -0.3]
        assert "I will look at foo.py" in out0.content[0].text
        # Second assistant turn also enriched from rollout_details
        out3 = response.response.output[3]
        assert out3.prompt_token_ids == [4, 5, 6]
        assert out3.generation_token_ids == [13, 14, 15]
        assert out3.generation_log_probs == [-0.4, -0.5, -0.6]
        # Input still populated from trajectory
        assert len(response.responses_create_params.input) == 1
        assert "Fix the bug" in response.responses_create_params.input[0].content

    @patch("responses_api_agents.harbor_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.harbor_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_falls_back_to_trajectory(self, mock_to_thread, mock_ray, mock_sc):
        """Empty rollout_details -> ATIF trajectory used for output."""
        server = _make_server()
        trial_result = {
            **DEFAULT_TRIAL_RESULT,
            "agent_result": {"n_input_tokens": 1200, "n_output_tokens": 180, "rollout_details": []},
        }
        setup_harbor_run_mock(mock_to_thread, mock_ray, mock_sc, trial_result=trial_result, trajectory=DEFAULT_TRAJECTORY)

        response = await server.run(create_run_request())

        assert response.reward == 1.0
        output = response.response.output
        # 2 agent steps x (message + function_call + function_call_output) = 6
        assert len(output) == 6

        # Assistant message with logprobs
        assert _get(output[0], "type") == "message"
        assert "I will look at foo.py" in _get(_get(output[0], "content")[0], "text")
        assert _get(output[0], "generation_log_probs") == [-0.01, -0.02, -0.03]
        assert _get(output[0], "prompt_token_ids") == []
        assert _get(output[0], "generation_token_ids") == []

        # Function call + output
        assert _get(output[1], "type") == "function_call"
        assert _get(output[1], "name") == "bash_command"
        assert _get(output[2], "type") == "function_call_output"
        assert "def foo" in _get(output[2], "output")

        # Second agent step
        assert _get(output[3], "generation_log_probs") == [-0.04, -0.05]

        # Input from trajectory
        assert "Fix the bug" in response.responses_create_params.input[0].content

        # Usage from final_metrics
        assert response.response.usage.input_tokens == 1200
        assert response.response.usage.output_tokens == 180
        assert response.response.usage.total_tokens == 1380

    @patch("responses_api_agents.harbor_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.harbor_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_failed_execution(self, mock_to_thread, mock_ray, mock_sc):
        """Harbor job exception -> reward=0, empty output."""
        server = _make_server()
        sc = MagicMock()
        sc.global_config_dict = {"policy_model_name": "test_model", "policy_base_url": "http://host:9000/v1"}
        mock_sc.return_value = sc
        mock_ray.remote.return_value = MagicMock()
        mock_to_thread.side_effect = Exception("Harbor job failed")

        response = await server.run(create_run_request(instance_id="fail_task", temperature=0.3, top_p=0.95))

        assert response.reward == 0.0
        assert len(response.response.output) == 0
        assert response.responses_create_params.temperature == 0.3
        assert response.responses_create_params.input == []

    @patch("responses_api_agents.harbor_agent.app.ServerClient.load_from_global_config")
    @patch("responses_api_agents.harbor_agent.app.runner_ray_remote")
    @patch("asyncio.to_thread")
    async def test_run_uses_trial_name_as_response_id(self, mock_to_thread, mock_ray, mock_sc):
        """response.id should map to Harbor trial_name when available."""
        server = _make_server()
        setup_harbor_run_mock(mock_to_thread, mock_ray, mock_sc, trajectory=DEFAULT_TRAJECTORY)

        response = await server.run(create_run_request())
        assert response.response.id == DEFAULT_TRIAL_RESULT["trial_name"]

    # ---- responses() --------------------------------------------------- #

    async def test_responses_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            await _make_server().responses(NeMoGymResponseCreateParamsNonStreaming(temperature=0.7, top_p=0.9, input=[]))

    # ---- _build_job_config --------------------------------------------- #

    def test_build_job_config_agent_settings(self) -> None:
        server = _make_server(
            harbor_agent_kwargs={
                "collect_rollout_details": True,
                "model_info": {"max_input_tokens": 65536, "max_output_tokens": 8192, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
            }
        )
        jc = server._build_job_config("test_task", "hosted_vllm/test_model", "http://localhost:8000/v1")
        agent = jc["agents"][0]
        assert agent["kwargs"]["collect_rollout_details"] is True
        assert agent["kwargs"]["api_base"] == "http://localhost:8000/v1"
        assert agent["kwargs"]["model_info"]["max_input_tokens"] == 65536
        assert agent["override_timeout_sec"] is None
        assert jc["job_name"].startswith("ng_")
        assert jc["job_name"].endswith("_test_task")

    def test_build_job_config_raises_without_dataset(self) -> None:
        server = _make_server(harbor_dataset_name=None, harbor_local_dataset_path=None)
        with pytest.raises(ValueError, match="requires a dataset"):
            server._build_job_config("test_task", "hosted_vllm/test_model", "http://localhost:8000/v1")

    def test_build_job_config_custom_agent_import_path(self) -> None:
        server = _make_server(harbor_agent_import_path="my_package.agents.MyCustomAgent")
        agent = server._build_job_config("test_task", "hosted_vllm/test_model", "http://localhost:8000/v1")["agents"][0]
        assert agent["name"] is None
        assert agent["import_path"] == "my_package.agents.MyCustomAgent"

    def test_build_job_config_custom_environment_import_path(self) -> None:
        server = _make_server(harbor_environment_import_path="my_package.envs.MyCustomEnv")
        env = server._build_job_config("test_task", "hosted_vllm/test_model", "http://localhost:8000/v1")["environment"]
        assert env["type"] is None
        assert env["import_path"] == "my_package.envs.MyCustomEnv"

    def test_build_job_config_extra_agent_kwargs(self) -> None:
        server = _make_server(harbor_agent_kwargs={"temperature": 0.5, "max_turns": 100})
        agent = server._build_job_config("test_task", "hosted_vllm/test_model", "http://localhost:8000/v1")["agents"][0]
        assert agent["kwargs"]["temperature"] == 0.5
        assert agent["kwargs"]["max_turns"] == 100
        assert agent["kwargs"]["api_base"] == "http://localhost:8000/v1"

    def test_build_job_config_extra_environment_kwargs(self) -> None:
        server = _make_server(harbor_environment_kwargs={"override_cpus": 4})
        env_kw = server._build_job_config("test_task", "hosted_vllm/test_model", "http://localhost:8000/v1")["environment"]["kwargs"]
        assert env_kw["override_cpus"] == 4

    def test_resolve_model_name_requires_prefix(self) -> None:
        with pytest.raises(ValueError, match="harbor_model_prefix is required"):
            _make_server(harbor_model_prefix=None)._resolve_model_name("test_model")

    def test_endpoints_registered(self) -> None:
        client = TestClient(_make_server().setup_webserver(), raise_server_exceptions=False)
        assert client.post("/v1/responses", json={"temperature": 0.7, "top_p": 0.9, "input": []}).status_code == 500
        assert client.post("/run", json={}).status_code != 404


# ===========================================================================
#  HarborAgentUtils unit tests
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


class TestTrajectoryToResponses:
    @pytest.fixture()
    def items(self):
        return HarborAgentUtils.trajectory_to_responses(DEFAULT_TRAJECTORY)

    def test_item_count(self, items) -> None:
        # 2 agent steps x (message + tool_call + tool_output) = 6
        assert len(items) == 6

    def test_message_has_logprobs(self, items) -> None:
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "assistant"
        assert items[0]["generation_log_probs"] == [-0.01, -0.02, -0.03]

    def test_function_call(self, items) -> None:
        assert items[1]["type"] == "function_call"
        assert items[1]["name"] == "bash_command"
        assert items[1]["call_id"] == "call_0_1"
        assert items[1]["generation_log_probs"] == []

    def test_function_call_output(self, items) -> None:
        assert items[2]["type"] == "function_call_output"
        assert items[2]["call_id"] == "call_0_1"
        assert "def foo" in items[2]["output"]

    def test_empty_trajectory(self) -> None:
        assert HarborAgentUtils.trajectory_to_responses({"steps": []}) == []


class TestTrialResultToResponses:
    def test_prefers_rollout_details(self) -> None:
        items = HarborAgentUtils.trial_result_to_responses(DEFAULT_TRIAL_RESULT, DEFAULT_TRAJECTORY)
        # Keep rich trajectory structure even when rollout_details are available
        assert len(items) == 6
        assert items[0]["prompt_token_ids"] == [1, 2, 3]
        assert items[3]["prompt_token_ids"] == [4, 5, 6]
        assert "I will look at foo.py" in items[0]["content"][0]["text"]

    def test_rollout_only_without_trajectory(self) -> None:
        items = HarborAgentUtils.trial_result_to_responses(DEFAULT_TRIAL_RESULT, None)
        assert len(items) == 2
        assert items[0]["prompt_token_ids"] == [1, 2, 3]
        assert items[1]["prompt_token_ids"] == [4, 5, 6]

    def test_falls_back_to_trajectory(self) -> None:
        result = {**DEFAULT_TRIAL_RESULT, "agent_result": {"rollout_details": [], "n_input_tokens": 100, "n_output_tokens": 50}}
        items = HarborAgentUtils.trial_result_to_responses(result, DEFAULT_TRAJECTORY)
        assert len(items) == 6
        assert items[0]["generation_log_probs"] == [-0.01, -0.02, -0.03]

    def test_falls_back_to_empty_output(self) -> None:
        result = {**DEFAULT_TRIAL_RESULT, "agent_result": {"rollout_details": [], "n_input_tokens": 100, "n_output_tokens": 50}}
        items = HarborAgentUtils.trial_result_to_responses(result, None)
        assert items == []


class TestExtractUsage:
    def test_from_trajectory(self) -> None:
        usage = HarborAgentUtils.extract_usage(DEFAULT_TRIAL_RESULT, DEFAULT_TRAJECTORY)
        assert usage["input_tokens"] == 1200
        assert usage["output_tokens"] == 180
        assert usage["total_tokens"] == 1380

    def test_from_trial_result_fallback(self) -> None:
        usage = HarborAgentUtils.extract_usage(DEFAULT_TRIAL_RESULT, None)
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_empty(self) -> None:
        usage = HarborAgentUtils.extract_usage({"agent_result": None}, None)
        assert usage["total_tokens"] == 0


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
