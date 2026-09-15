# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.processors import (
    AgentTurn,
    EpisodeFailure,
    EpisodeId,
    EpisodeRequest,
    EpisodeResponse,
    EpisodeVerification,
    TaskIdentity,
)
from nemo_gym.rollout_collection import _episode_request_from_row, _project_episode_response


def _response(response_id: str = "response-1") -> dict:
    return {
        "id": response_id,
        "created_at": 1,
        "model": "model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _request() -> EpisodeRequest:
    return EpisodeRequest(
        episode_id=EpisodeId(rollout_id="4-2"),
        task=TaskIdentity(task_source="nemo_sim", task_id="4"),
        responses_create_params={"input": []},
        task_data={"nemo_sim_sampling": {"locale": "en_US", "seed": 42}},
    )


def test_episode_result_requires_exactly_one_outcome() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="exactly one"):
        EpisodeResponse(episode_id=request.episode_id, task=request.task)
    with pytest.raises(ValidationError, match="exactly one"):
        EpisodeResponse(
            episode_id=request.episode_id,
            task=request.task,
            response=_response(),
            verification=EpisodeVerification(reward=1),
            failure=EpisodeFailure(kind="internal", message="failed", retryable=False),
        )


def test_verification_rejects_non_finite_rewards() -> None:
    with pytest.raises(ValidationError):
        EpisodeVerification(reward=float("nan"))


def test_agent_turn_keeps_exact_request_and_response() -> None:
    turn = AgentTurn(
        sequence=0,
        participant="user",
        request={"input": [{"role": "user", "content": "hello"}]},
        response=_response(),
        state_after={"preference": "vegetarian"},
        termination_reason="user_goal_satisfied",
    )
    assert turn.participant == "user"
    assert turn.request.input[0].content == "hello"
    assert turn.state_after == {"preference": "vegetarian"}
    assert turn.termination_reason == "user_goal_satisfied"


def test_collector_materializes_native_request_and_projects_success() -> None:
    row = {
        "_ng_task_index": 4,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 1,
        "processor_ref": {"type": "processors", "name": "nemo_sim"},
        "responses_create_params": {"input": []},
        "nemo_sim_sampling": {"locale": "en_US", "seed": 42},
    }
    request = _episode_request_from_row(row)
    assert request.episode_id == EpisodeId(rollout_id="4-2", attempt=1)
    assert request.task_data == {"nemo_sim_sampling": {"locale": "en_US", "seed": 42}}

    native = EpisodeResponse(
        episode_id=request.episode_id,
        task=request.task,
        response=_response(),
        verification=EpisodeVerification(reward=1, verifier_data={"scenario_completed": True}),
    )
    projected = _project_episode_response(request, native.model_dump(mode="json"))
    assert projected["reward"] == 1
    assert projected["scenario_completed"] is True


def test_projection_rejects_verifier_field_collisions() -> None:
    request = _request()
    native = EpisodeResponse(
        episode_id=request.episode_id,
        task=request.task,
        response=_response(),
        verification=EpisodeVerification(reward=1, verifier_data={"nemo_sim_sampling": {}}),
    )
    with pytest.raises(ValueError, match="collides"):
        _project_episode_response(request, native.model_dump(mode="json"))
