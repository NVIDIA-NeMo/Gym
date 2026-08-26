# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
from pydantic import ValidationError

from resources_servers.gym_v.schemas import (
    GymVSeedSessionRequest,
    GymVStepRequest,
    GymVTaskRow,
)


def _row(**overrides):
    row = {
        "env_id": "Games/FrozenLake-v0",
        "env_kwargs": {"size": 4, "num_holes": 3, "tile_size": 32},
        "seed": 1234,
        "task_id": "frozenlake_4x4_h3_seed1234_stage1",
        "horizon_cap": 30,
        "task_metadata": {"difficulty_stage": 1, "category": "Games"},
        "responses_create_params": {
            "model": "policy_model",
            "input": [],
            "temperature": 0.7,
            "max_output_tokens": 1024,
            "tools": [],
        },
    }
    row.update(overrides)
    return row


def test_task_row_round_trips_frozenlake_worked_example() -> None:
    parsed = GymVTaskRow.model_validate(_row())

    dumped = parsed.model_dump(mode="json")
    reparsed = GymVTaskRow.model_validate(dumped)

    assert reparsed.env_id == "Games/FrozenLake-v0"
    assert reparsed.responses_create_params.input == []
    assert reparsed.responses_create_params.tools == []


def test_task_row_ignores_extra_fields() -> None:
    # Legacy JSONL rows may still carry act_grammar_regex. The new schema
    # doesn't declare it, but Pydantic's default is to ignore unknown fields
    # (BaseModel doesn't forbid extras unless configured to).
    row = _row(act_grammar_regex=r"^\[(up|down|left|right)\]$")
    parsed = GymVTaskRow.model_validate(row)

    assert not hasattr(parsed, "act_grammar_regex")


def test_seed_session_request_requires_task_selector() -> None:
    with pytest.raises(ValidationError):
        GymVSeedSessionRequest.model_validate({})


def test_seed_session_request_accepts_task_idx() -> None:
    parsed = GymVSeedSessionRequest.model_validate({"task_idx": 3})
    assert parsed.task_idx == 3
    assert parsed.task_row is None


def test_step_request_requires_action_string() -> None:
    with pytest.raises(ValidationError):
        GymVStepRequest.model_validate({"env_id": "some-uuid"})


def test_step_request_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        GymVStepRequest.model_validate({"env_id": "some-uuid", "action_string": "[up]", "tool_calls": []})
