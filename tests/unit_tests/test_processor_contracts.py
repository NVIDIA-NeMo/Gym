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


def _turn(sequence: int = 0, agent_id: str = "assistant") -> AgentTurn:
    return AgentTurn(
        sequence=sequence,
        agent_id=agent_id,
        request={"input": []},
        response=_response(f"response-{sequence}"),
    )


def test_episode_result_requires_exactly_one_outcome() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="exactly one"):
        EpisodeResponse(episode_id=request.episode_id, task=request.task)
    with pytest.raises(ValidationError, match="exactly one"):
        EpisodeResponse(
            episode_id=request.episode_id,
            task=request.task,
            agent_turns=[_turn()],
            output_turn_sequence=0,
            verification=EpisodeVerification(reward=1),
            failure=EpisodeFailure(kind="internal", message="failed", retryable=False),
        )


def test_verified_episode_requires_a_valid_output_turn() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="requires an output turn"):
        EpisodeResponse(
            episode_id=request.episode_id,
            task=request.task,
            agent_turns=[_turn()],
            verification=EpisodeVerification(reward=1),
        )
    with pytest.raises(ValidationError, match="must reference an agent turn"):
        EpisodeResponse(
            episode_id=request.episode_id,
            task=request.task,
            agent_turns=[_turn()],
            output_turn_sequence=1,
            verification=EpisodeVerification(reward=1),
        )


def test_episode_turns_must_be_contiguous_and_ordered() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="contiguous and ordered"):
        EpisodeResponse(
            episode_id=request.episode_id,
            task=request.task,
            agent_turns=[_turn(sequence=1)],
            output_turn_sequence=0,
            verification=EpisodeVerification(reward=1),
        )


def test_verification_rejects_non_finite_rewards() -> None:
    with pytest.raises(ValidationError):
        EpisodeVerification(reward=float("nan"))


def test_agent_turn_keeps_exact_request_and_response() -> None:
    turn = AgentTurn(
        sequence=0,
        agent_id="user",
        request={"input": [{"role": "user", "content": "hello"}]},
        response=_response(),
        state_after={"preference": "vegetarian"},
        termination_reason="user_goal_satisfied",
    )
    assert turn.agent_id == "user"
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
        agent_turns=[_turn()],
        output_turn_sequence=0,
        verification=EpisodeVerification(reward=1, verifier_data={"scenario_completed": True}),
    )
    projected = _project_episode_response(request, native.model_dump(mode="json"))
    assert projected["reward"] == 1
    assert projected["scenario_completed"] is True
    assert projected["response"]["id"] == "response-0"


def test_projection_preserves_all_turns_and_derives_compatibility_response() -> None:
    request = _request()
    native = EpisodeResponse(
        episode_id=request.episode_id,
        task=request.task,
        agent_turns=[_turn(agent_id="user"), _turn(sequence=1)],
        output_turn_sequence=1,
        verification=EpisodeVerification(reward=1),
    )

    projected = _project_episode_response(request, native.model_dump(mode="json"))

    assert [turn["agent_id"] for turn in projected["agent_turns"]] == ["user", "assistant"]
    assert projected["output_turn_sequence"] == 1
    assert projected["response"]["id"] == "response-1"


def test_projection_rejects_verifier_field_collisions() -> None:
    request = _request()
    native = EpisodeResponse(
        episode_id=request.episode_id,
        task=request.task,
        agent_turns=[_turn()],
        output_turn_sequence=0,
        verification=EpisodeVerification(reward=1, verifier_data={"nemo_sim_sampling": {}}),
    )
    with pytest.raises(ValueError, match="collides"):
        _project_episode_response(request, native.model_dump(mode="json"))
