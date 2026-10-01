# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.models.trajectories import Trajectory

from responses_api_agents.harbor_agent_general.app import HarborAgent, HarborVerifyResponse


def _trajectory(**step_updates) -> Trajectory:
    step = {
        "step_id": 1,
        "source": "agent",
        "message": "answer",
    }
    step.update(step_updates)
    return Trajectory.model_validate(
        {
            "agent": {"name": "test-agent", "version": "1"},
            "steps": [step],
        }
    )


def test_prompt_token_ids_do_not_emit_training_metadata() -> None:
    warnings = []
    output = HarborAgent.convert_atif_to_gym_responses(_trajectory(metrics={"prompt_token_ids": [1, 2]}), warnings)

    assert "prompt_token_ids" not in output[0]
    assert "generation_token_ids" not in output[0]
    assert "generation_log_probs" not in output[0]
    assert "completion_token_ids are missing or empty" in warnings[0]


def test_mismatched_completion_metadata_is_omitted_with_warning() -> None:
    warnings = []
    trajectory = _trajectory(metrics={"prompt_token_ids": [0], "completion_token_ids": [1, 2], "logprobs": [-0.1]})

    output = HarborAgent.convert_atif_to_gym_responses(trajectory, warnings)

    assert "generation_token_ids" not in output[0]
    assert "completion_token_ids and logprobs have different lengths" in warnings[0]


def test_completion_token_ids_emit_aligned_training_metadata() -> None:
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            metrics={
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3, 4],
                "logprobs": [-0.1, -0.2],
            }
        )
    )

    assert output[0]["prompt_token_ids"] == [1, 2]
    assert output[0]["generation_token_ids"] == [3, 4]
    assert output[0]["generation_log_probs"] == [-0.1, -0.2]


def test_multimodal_message_is_serialized_with_warning() -> None:
    warnings = []
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            message=[
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"media_type": "image/png", "path": "/tmp/image.png"}},
            ]
        ),
        warnings,
    )

    assert json.loads(output[0]["content"][0]["text"]) == [
        {"type": "text", "text": "look"},
        {"type": "image", "source": {"media_type": "image/png", "path": "/tmp/image.png"}},
    ]
    assert "multimodal message serialized as JSON" in warnings[0]


def test_multimodal_tool_output_uses_gym_content_parts() -> None:
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            tool_calls=[{"tool_call_id": "call-1", "function_name": "inspect", "arguments": {}}],
            observation={
                "results": [
                    {
                        "source_call_id": "call-1",
                        "content": [
                            {"type": "text", "text": "result"},
                            {
                                "type": "image",
                                "source": {"media_type": "image/png", "path": "https://example.com/image.png"},
                            },
                        ],
                    }
                ]
            },
        )
    )

    assert output[2]["output"] == [
        {"text": "result", "type": "input_text"},
        {"detail": "auto", "image_url": "https://example.com/image.png", "type": "input_image"},
    ]


def test_local_tool_image_and_missing_call_id_are_documented() -> None:
    warnings = []
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            observation={
                "results": [
                    {
                        "content": [
                            {
                                "type": "image",
                                "source": {"media_type": "image/png", "path": "/tmp/image.png"},
                            }
                        ]
                    }
                ]
            }
        ),
        warnings,
    )

    assert json.loads(output[1]["output"])[0]["source"]["path"] == "/tmp/image.png"
    assert output[1]["call_id"] == "atif-step-1-observation-0"
    assert any("local image paths" in warning for warning in warnings)
    assert any("synthetic call_id" in warning for warning in warnings)


def test_copied_context_training_metadata_is_omitted_with_warning() -> None:
    warnings = []
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            is_copied_context=True,
            metrics={
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3, 4],
                "logprobs": [-0.1, -0.2],
            },
        ),
        warnings,
    )

    assert "generation_token_ids" not in output[0]
    assert "step is copied context" in warnings[0]


def test_user_step_is_preserved_without_warning() -> None:
    warnings = []

    output = HarborAgent.convert_atif_to_gym_responses(_trajectory(source="user"), warnings)

    assert output == [{"content": "answer", "role": "user", "type": "message", "phase": None}]
    assert warnings == []


def test_system_multimodal_step_uses_gym_content_parts() -> None:
    warnings = []
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            source="system",
            message=[
                {"type": "text", "text": "inspect"},
                {
                    "type": "image",
                    "source": {"media_type": "image/png", "path": "data:image/png;base64,YQ=="},
                },
            ],
        ),
        warnings,
    )

    assert output[0]["role"] == "system"
    assert output[0]["content"] == [
        {"text": "inspect", "type": "input_text"},
        {"detail": "auto", "image_url": "data:image/png;base64,YQ==", "type": "input_image"},
    ]
    assert warnings == []


def test_verify_response_preserves_conversion_diagnostics() -> None:
    response = HarborVerifyResponse.model_validate(
        {
            "responses_create_params": {"input": []},
            "response": {
                "id": "response-id",
                "created_at": 0,
                "model": "test-model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
            },
            "reward": 1.0,
            "reward_components": {"reward": 1.0},
            "atif_conversion": {"lossless": False, "warnings": ["documented limitation"]},
        }
    )

    assert response.model_dump()["atif_conversion"] == {
        "lossless": False,
        "warnings": ["documented limitation"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_location", ["trial", "step", None])
async def test_run_job_rejects_failed_steps_and_invalidates_resume(tmp_path, monkeypatch, failure_location):
    from harbor.models.job.config import JobConfig

    from responses_api_agents.harbor_agent_general import app

    trial_dir = tmp_path / "job" / "trial"
    trial_dir.mkdir(parents=True)
    result_path = trial_dir / "result.json"
    result_path.write_text("{}")
    failure = SimpleNamespace(exception_type="RuntimeError", exception_message="notebook unavailable")
    trial = SimpleNamespace(
        task_name="task",
        exception_info=failure if failure_location == "trial" else None,
        step_results=[SimpleNamespace(exception_info=failure if failure_location == "step" else None)],
    )
    monkeypatch.setattr(app.TrialResult, "model_validate_json", lambda _: trial)
    monkeypatch.setattr(app.Job, "create", AsyncMock(return_value=SimpleNamespace(run=AsyncMock())))
    config = JobConfig(job_name="job", jobs_dir=tmp_path)

    if failure_location is not None:
        with pytest.raises(RuntimeError, match="notebook unavailable"):
            await HarborAgent.run_job(config.model_dump(mode="json"), "task")
        assert not result_path.exists(), "Gym retries must not reuse a failed step's result"
    else:
        assert Path(await HarborAgent.run_job(config.model_dump(mode="json"), "task")) == trial_dir
        assert result_path.exists()


@pytest.mark.parametrize("multi_step", [False, True])
def test_success_preserves_structured_rewards_and_step_boundaries(
    tmp_path: Path, multi_step: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harbor.models.trial.paths import TrialPaths
    from harbor.models.trial.result import TrialResult

    from nemo_gym.reward_profile import compute_aggregate_metrics
    from responses_api_agents.harbor_agent_general.app import HarborAgentConfig, HarborRunRequest
    from responses_api_agents.harbor_agent_general.tests.test_prepare import make_task

    task = make_task(tmp_path, "task", multi_step=multi_step)
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    paths = TrialPaths(trial_dir)
    steps = ["first", "second"] if multi_step else [None]
    for index, name in enumerate(steps):
        agent_dir = paths.step_agent_dir(name) if name else paths.agent_dir
        agent_dir.mkdir(parents=True)
        trajectory = _trajectory(message=f"answer {index}")
        (agent_dir / "trajectory.json").write_text(trajectory.model_dump_json())
    # Trial-level scores deliberately differ from the final step's scores.
    trial = TrialResult.model_validate(
        {
            "task_name": "task",
            "trial_name": "trial",
            "trial_uri": str(trial_dir),
            "task_id": {"path": str(task)},
            "task_checksum": "test-checksum",
            "config": {"task": {"path": str(task)}},
            "agent_info": {"name": "opencode", "version": "test"},
            "verifier_result": {"rewards": {"correctness": 0.5, "format": 1.0}},
            "step_results": [
                {"step_name": name, "verifier_result": {"rewards": {"correctness": index, "format": 1.0}}}
                for index, name in enumerate(steps)
            ]
            if multi_step
            else None,
        }
    )
    paths.result_path.write_text(trial.model_dump_json())
    config = HarborAgentConfig(
        name="harbor_agent_general",
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        harbor_jobs_dir=tmp_path / "jobs",
        harbor_reward_key="correctness",
        harbor_agent={"name": "opencode", "model_name": "test-model"},
    )
    monkeypatch.setattr("responses_api_agents.harbor_agent_general.app.get_global_config_dict", lambda: {})
    agent = HarborAgent.model_construct(config=config)
    body = HarborRunRequest.model_validate(
        {"task_name": "task", "responses_create_params": {"input": []}, "_ng_task_index": 0, "_ng_rollout_index": 0}
    )
    result = agent.success_response(body, trial_dir).model_dump(mode="json")
    assert result["reward"] == 0.5
    assert result["reward_components"] == {"correctness": 0.5, "format": 1.0}
    assert result["harbor_reward/correctness"] == 0.5
    assert result["response"]["output"][0]["content"][0]["text"] == f"answer {len(steps) - 1}"
    assert result["responses_create_params"]["input"][0]["content"] == ("Solve second" if multi_step else "Solve task")
    assert len(result["harbor_steps"]) == len(steps)
    if multi_step:
        assert result["harbor_steps"][0]["output"][0]["content"][0]["text"] == "answer 0"
        assert result["harbor_steps"][1]["rewards"]["correctness"] == 1
        assert not result["atif_conversion"]["lossless"]
        assert any("Multi-step" in warning for warning in result["atif_conversion"]["warnings"])
    metrics = compute_aggregate_metrics([result])
    assert metrics.agent_metrics["mean/harbor_reward/correctness"] == 0.5
    config.harbor_reward_key = "missing"
    with pytest.raises(ValueError, match="Set harbor_reward_key explicitly"):
        agent.success_response(body, trial_dir)
    failed = agent.failure_response(body, RuntimeError("failure"))
    assert failed.reward_components == {}
