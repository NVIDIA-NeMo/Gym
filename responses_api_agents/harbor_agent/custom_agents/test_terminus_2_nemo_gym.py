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


def test_attach_routed_experts_advances_history_for_agent_steps_without_metrics():
    llm = NemoGymLLM(model_name="test-model", api_base="http://localhost:8000/v1")
    route_1 = [[[1]]]
    route_2 = [[[2]]]
    route_3 = [[[3]]]
    llm._routed_experts_history = [route_1, route_2, route_3]

    first_metrics = SimpleNamespace(extra=None)
    third_metrics = SimpleNamespace(extra={})

    agent = Terminus2NemoGym.__new__(Terminus2NemoGym)
    agent._llm = llm
    agent._trajectory_steps = [
        SimpleNamespace(source="agent", metrics=first_metrics),
        SimpleNamespace(source="agent", metrics=None),
        SimpleNamespace(source="agent", metrics=third_metrics),
    ]
    agent._dump_trajectory = MagicMock()

    Terminus2NemoGym._attach_routed_experts_to_trajectory(agent)

    assert first_metrics.extra == {"routed_experts": route_1}
    assert third_metrics.extra == {"routed_experts": route_3}
    agent._dump_trajectory.assert_called_once()
