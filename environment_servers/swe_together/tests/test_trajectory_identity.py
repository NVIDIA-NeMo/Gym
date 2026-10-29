# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from environment_servers.swe_together.schemas import SWETRequest
from nemo_gym.agents import AgentCloseSessionResponse
from nemo_gym.config import ATTEMPT_INDEX_KEY_NAME, ROLLOUT_ID_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.rollouts.rollout_collection import _episode_record, _episode_request_body
from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record
from nemo_gym.trajectory import TrajectoryRecord


@pytest.mark.parametrize("reverse_task_keys", [False, True])
@pytest.mark.parametrize(
    ("indices", "rollout_id"),
    [
        ({ROLLOUT_INDEX_KEY_NAME: 0}, "7-0"),
        ({ROLLOUT_INDEX_KEY_NAME: 1}, "7-1"),
        ({ROLLOUT_INDEX_KEY_NAME: 1, ATTEMPT_INDEX_KEY_NAME: 2}, "7-1-a2"),
        (
            {ROLLOUT_INDEX_KEY_NAME: 1, ROLLOUT_ID_KEY_NAME: "custom-physical", ATTEMPT_INDEX_KEY_NAME: 2},
            "custom-physical-a2",
        ),
    ],
)
@pytest.mark.parametrize("failed", [False, True])
def test_forwarded_trajectory_matches_collector_without_changing_harness_evidence(
    reverse_task_keys: bool, indices: dict[str, int | str], rollout_id: str, failed: bool
) -> None:
    task_id = {"taskset": "swe_together:full109", "task_id": "task-logical"}
    if reverse_task_keys:
        task_id = dict(reversed(list(task_id.items())))
    row = {
        "task_id": task_id,
        "task_input": {
            "task_data": {"task_id": "task-logical", "image": "image", "image_digest": "sha256:" + "1" * 64}
        },
        TASK_INDEX_KEY_NAME: 7,
        **indices,
    }
    request = SWETRequest.model_validate(_episode_request_body(row))
    trajectory = TrajectoryRecord.model_validate(
        {
            "task_id": "task-logical",
            "rollout_id": rollout_id,
            "model_calls": [{"model_call_id": "call-1", "request": {"input": "original prompt"}}],
            "turns": [
                {
                    "invocation_id": "root",
                    "activation_id": index,
                    "task_id": "task-logical",
                    "rollout_id": rollout_id,
                    "turn_no": index,
                    "timestamp": float(index),
                    "step_count": index - 1,
                    "question": f"question {index}",
                    "answer": f"candidate decision {index}",
                    "reasoning_content": f"reason {index}",
                    "resolved": index == 2,
                    "model_calls": [{"model_call_id": "call-1"}],
                }
                for index in [1, 2]
            ],
        }
    )
    close = AgentCloseSessionResponse(agent_session_id="candidate", cleanup_confirmed=True, trajectory=trajectory)
    original = deepcopy(close.model_dump(mode="json"))
    forwarded = request.trajectory_for_collection(close.trajectory)
    assert forwarded is not None
    response = {"episode_id": request.episode_id.model_dump(mode="json"), "task_id": task_id}
    if failed:
        response["failure"] = {
            "failure_reason": "candidate failed after execution",
            "terminal": True,
            "trajectory": forwarded.model_dump(mode="json"),
        }
    else:
        response["result"] = {"task_id": "task-logical", "ng_trajectory": forwarded.model_dump(mode="json")}

    record = _episode_record(response)
    projected = _build_trajectory_record(row, record).model_dump(mode="json")

    assert projected["task_id"] == str(task_id)
    assert projected["rollout_id"] == rollout_id
    assert "producer_trajectory_identity_mismatch" not in {gap["code"] for gap in projected["gaps"]}
    assert projected["turns"] == forwarded.model_dump(mode="json")["turns"]
    assert projected["model_calls"] == original["trajectory"]["model_calls"]
    for before, after in zip(original["trajectory"]["turns"], projected["turns"], strict=True):
        assert {key: value for key, value in before.items() if key != "task_id"} == {
            key: value for key, value in after.items() if key != "task_id"
        }
    assert close.model_dump(mode="json") == original
    assert request.task.task_id.task_id == "task-logical"
    assert request.model_dump()["task"]["task_id"] == {"taskset": "swe_together:full109", "task_id": "task-logical"}
    assert "_collector_task_id" not in request.model_dump()
    if not failed:
        assert record["task_id"] == "task-logical"


def test_programmatic_request_preserves_its_identity_when_revalidated() -> None:
    request = SWETRequest(
        episode_id=EpisodeId(rollout_id="one"),
        task={
            "task_id": TaskId(taskset="swet", task_id="task"),
            "task_input": {"task_data": {"task_id": "task", "image": "image", "image_digest": "sha256:" + "1" * 64}},
        },
    )
    assert request.trajectory_for_collection(None) is None
    trajectory = TrajectoryRecord(task_id="task", rollout_id="one")
    revalidated = SWETRequest.model_validate(request)
    forwarded = revalidated.trajectory_for_collection(trajectory)
    assert forwarded is not None
    assert forwarded.task_id == "{'taskset': 'swet', 'task_id': 'task'}"
    assert trajectory.task_id == "task"


def test_identity_adaptation_retains_wrong_task_evidence() -> None:
    row = {
        "task_id": {"taskset": "swet", "task_id": "task"},
        "task_input": {"task_data": {"task_id": "task", "image": "image", "image_digest": "sha256:" + "1" * 64}},
        TASK_INDEX_KEY_NAME: 0,
        ROLLOUT_INDEX_KEY_NAME: 0,
    }
    request = SWETRequest.model_validate(_episode_request_body(row))
    trajectory = TrajectoryRecord.model_validate(
        {
            "task_id": "wrong-task",
            "rollout_id": "0-0",
            "turns": [
                {
                    "invocation_id": "root",
                    "task_id": "wrong-task",
                    "rollout_id": "0-0",
                    "turn_no": 1,
                    "timestamp": 1.0,
                    "step_count": 0,
                    "answer": "evidence from the wrong task",
                }
            ],
        }
    )
    forwarded = request.trajectory_for_collection(trajectory)
    assert forwarded is not None
    assert forwarded.turns[0].task_id == "wrong-task"
    assert forwarded is trajectory
    projected = _build_trajectory_record(row, {"ng_trajectory": forwarded.model_dump(mode="json")})
    assert any(
        gap.code == "producer_trajectory_identity_mismatch" and gap.detail == "task_id" for gap in projected.gaps
    )
