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

import pytest
from harbor.models.trajectories import Trajectory

from responses_api_agents.harbor_agent_general.app import HarborAgent


def _trajectory(metrics: dict) -> Trajectory:
    return Trajectory.model_validate(
        {
            "agent": {"name": "test-agent", "version": "1"},
            "steps": [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "answer",
                    "metrics": metrics,
                }
            ],
        }
    )


def test_prompt_token_ids_do_not_emit_training_metadata() -> None:
    output = HarborAgent.convert_atif_to_gym_responses(_trajectory({"prompt_token_ids": [1, 2]}))

    assert "prompt_token_ids" not in output[0]
    assert "generation_token_ids" not in output[0]
    assert "generation_log_probs" not in output[0]


def test_completion_token_ids_require_matching_logprobs() -> None:
    trajectory = _trajectory({"completion_token_ids": [1, 2], "logprobs": [-0.1]})

    with pytest.raises(ValueError, match="completion_token_ids and logprobs must have matching lengths"):
        HarborAgent.convert_atif_to_gym_responses(trajectory)


def test_completion_token_ids_emit_aligned_training_metadata() -> None:
    output = HarborAgent.convert_atif_to_gym_responses(
        _trajectory(
            {
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3, 4],
                "logprobs": [-0.1, -0.2],
            }
        )
    )

    assert output[0]["prompt_token_ids"] == [1, 2]
    assert output[0]["generation_token_ids"] == [3, 4]
    assert output[0]["generation_log_probs"] == [-0.1, -0.2]
