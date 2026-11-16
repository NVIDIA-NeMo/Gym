# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.rollouts.rollout_collection import _episode_record
from resources_servers.gdpval.app import GDPValVerifyRequest, GDPValVerifyResponse


def _response(**indices):
    request = GDPValVerifyRequest.model_validate(
        {
            "task_id": "document-task",
            "responses_create_params": {"input": []},
            "response": {
                "id": "resp",
                "created_at": 0,
                "model": "model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
            **indices,
        }
    )
    return GDPValVerifyResponse(**request.model_dump(), reward=0.75)


def test_unset_indices_do_not_invalidate_a_typed_episode_result():
    # FastAPI serializes response aliases; the collector owns these index keys.
    payload = _response().model_dump(mode="json", by_alias=True)
    record = _episode_record(
        {
            "episode_id": {"rollout_id": "run", "attempt": 0},
            "task_id": {"taskset": "gdpval", "task_id": "document-task"},
            "result": payload,
            "failure": None,
        }
    )
    assert record["reward"] == 0.75
    for key in ("_ng_task_index", "_ng_rollout_index", "_ng_attempt_index"):
        assert key not in payload


@pytest.mark.parametrize("index", [0, 7])
def test_explicit_legacy_indices_still_round_trip(index):
    indices = {key: index for key in ("_ng_task_index", "_ng_rollout_index", "_ng_attempt_index")}
    payload = _response(**indices).model_dump(mode="json", by_alias=True)
    assert {key: payload[key] for key in indices} == indices
