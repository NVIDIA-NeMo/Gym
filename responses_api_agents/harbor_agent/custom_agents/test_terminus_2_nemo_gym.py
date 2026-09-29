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
from types import SimpleNamespace
from unittest.mock import MagicMock

from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM
from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import Terminus2NemoGym


def test_attach_routed_experts_matches_by_rollout_details_not_call_order():
    llm = NemoGymLLM(model_name="test-model", api_base="http://localhost:8000/v1")
    unmatched_route = [[[1]]]
    matched_route = [[[2]]]
    llm._store_routed_experts_for_rollout_details([1], [10], [-0.1], unmatched_route)
    llm._store_routed_experts_for_rollout_details([2], [20], [-0.2], matched_route)

    metrics = SimpleNamespace(
        prompt_token_ids=[2],
        completion_token_ids=[20],
        logprobs=[-0.2],
        extra=None,
    )

    agent = Terminus2NemoGym.__new__(Terminus2NemoGym)
    agent._llm = llm
    agent._trajectory_steps = [
        SimpleNamespace(source="agent", metrics=metrics),
    ]
    agent._dump_trajectory = MagicMock()

    Terminus2NemoGym._attach_routed_experts_to_trajectory(agent)

    assert metrics.extra == {"routed_experts": matched_route}
    agent._dump_trajectory.assert_called_once()


def test_attach_routed_experts_skips_steps_without_rollout_details():
    llm = NemoGymLLM(model_name="test-model", api_base="http://localhost:8000/v1")
    llm._store_routed_experts_for_rollout_details([1], [10], [-0.1], [[[1]]])

    metrics = SimpleNamespace(extra={})

    agent = Terminus2NemoGym.__new__(Terminus2NemoGym)
    agent._llm = llm
    agent._trajectory_steps = [
        SimpleNamespace(source="agent", metrics=None),
        SimpleNamespace(source="agent", metrics=metrics),
    ]
    agent._dump_trajectory = MagicMock()

    Terminus2NemoGym._attach_routed_experts_to_trajectory(agent)

    assert metrics.extra == {}
    agent._dump_trajectory.assert_not_called()
