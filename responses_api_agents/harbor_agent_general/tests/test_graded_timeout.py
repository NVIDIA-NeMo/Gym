# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from harbor.models.trial.result import ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from pydantic import ValidationError

from responses_api_agents.harbor_agent_general import app
from responses_api_agents.harbor_agent_general.app import HarborAgent, HarborAgentConfig, HarborRunRequest


@pytest.mark.parametrize("error", ["AgentTimeoutError", "NonZeroAgentExitCodeError"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_agent_error_keeps_the_official_verifier_reward(error, reward):
    trial = TrialResult.model_construct(
        exception_info=ExceptionInfo.from_exception(type(error, (Exception,), {})("agent stopped")),
        verifier_result=VerifierResult(rewards={"reward": reward}),
    )
    assert HarborAgent.has_graded_agent_exception(trial)


@pytest.mark.parametrize(
    "error,verifier",
    [
        ("AgentTimeoutError", None),
        ("AgentTimeoutError", VerifierResult(rewards={})),
        ("VerifierTimeoutError", VerifierResult(rewards={"reward": 1.0})),
        ("SandboxError", None),
        (None, VerifierResult(rewards={"reward": 1.0})),
    ],
)
def test_ungraded_or_infrastructure_errors_are_not_accepted(error, verifier):
    trial = TrialResult.model_construct(
        exception_info=ExceptionInfo.from_exception(type(error, (Exception,), {})()) if error else None,
        verifier_result=verifier,
    )
    assert not HarborAgent.has_graded_agent_exception(trial)


@pytest.fixture
def saved_trial(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Solve the task.")
    trial_dir = tmp_path / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    trial = TrialResult.model_validate(
        {
            "task_name": "task",
            "trial_name": "trial",
            "trial_uri": trial_dir.as_uri(),
            "task_id": {"path": task_dir},
            "task_checksum": "test-checksum",
            "config": {"task": {"path": task_dir}},
            "agent_info": {"name": "opencode", "version": "1.17.11"},
            "agent_result": {"n_input_tokens": 10, "n_output_tokens": 20},
        }
    )
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    agent = HarborAgent.model_construct(
        config=HarborAgentConfig.model_construct(
            harbor_jobs_dir=tmp_path,
            harbor_dataset=app.DatasetConfig(path=tmp_path),
            harbor_agent=app.AgentConfig(name="opencode", model_name="test-model"),
        ),
        server_client=SimpleNamespace(),
    )
    body = HarborRunRequest.model_validate(
        {
            "task_name": "task",
            "_ng_task_index": 0,
            "_ng_rollout_index": 0,
            "responses_create_params": {"input": []},
        }
    )
    return agent, body, trial, trial_dir


@pytest.mark.parametrize("error", ["AgentTimeoutError", "NonZeroAgentExitCodeError"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_graded_error_without_atif_returns_reward_and_discloses_missing_output(saved_trial, error, reward):
    agent, body, trial, trial_dir = saved_trial
    trial.exception_info = ExceptionInfo.from_exception(type(error, (Exception,), {})("agent stopped"))
    trial.verifier_result = VerifierResult(rewards={"reward": reward})
    (trial_dir / "result.json").write_text(trial.model_dump_json())

    response = agent.success_response(body, trial_dir)

    assert response.reward == reward
    assert response.response.status == "completed"
    assert response.response.output == []
    assert response.response.usage.total_tokens == 30
    assert response.harbor_exception == {"type": error, "message": "agent stopped"}
    assert response.atif_conversion["lossless"] is False
    assert "ATIF trajectory missing" in response.atif_conversion["warnings"][0]
    assert response.atif_conversion["trajectories"] == []
    assert response.ng_agent_observations.source == "opencode"
    assert response.ng_agent_observations.gaps  # DB observations are still attempted independently of ATIF.


@pytest.mark.parametrize(
    "error,rewards", [(None, {"reward": 1.0}), ("AgentTimeoutError", {}), ("SandboxError", {"reward": 0.0})]
)
def test_missing_atif_still_fails_for_success_or_ungraded_or_infrastructure_error(saved_trial, error, rewards):
    agent, body, trial, trial_dir = saved_trial
    trial.exception_info = ExceptionInfo.from_exception(type(error, (Exception,), {})()) if error else None
    trial.verifier_result = VerifierResult(rewards=rewards)
    (trial_dir / "result.json").write_text(trial.model_dump_json())
    with pytest.raises(FileNotFoundError):
        agent.success_response(body, trial_dir)


@pytest.mark.parametrize("malformed", [False, True])
def test_existing_atif_is_converted_or_validation_error_propagates(saved_trial, malformed):
    agent, body, trial, trial_dir = saved_trial
    trial.exception_info = ExceptionInfo.from_exception(type("AgentTimeoutError", (Exception,), {})("agent stopped"))
    trial.verifier_result = VerifierResult(rewards={"reward": 1.0})
    (trial_dir / "result.json").write_text(trial.model_dump_json())
    (trial_dir / "agent/trajectory.json").write_text(
        "invalid json"
        if malformed
        else app.Trajectory.model_validate(
            {
                "agent": {"name": "opencode", "version": "1.17.11"},
                "steps": [{"step_id": 1, "source": "agent", "message": "partial work"}],
            }
        ).model_dump_json()
    )
    if malformed:
        with pytest.raises(ValidationError):
            agent.success_response(body, trial_dir)
    else:
        response = agent.success_response(body, trial_dir)
        assert response.reward == 1.0
        assert response.response.output[0].content[0].text == "partial work"
        assert response.atif_conversion["lossless"] is True
        assert len(response.atif_conversion["trajectories"]) == 1
