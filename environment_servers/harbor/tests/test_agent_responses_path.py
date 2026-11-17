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
"""The path the Harbor environment server activates its agent on."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from environment_servers.harbor.app import HarborEnvironmentServer
from nemo_gym.core import EpisodeId


def _agent_responses_path(global_config: dict, *, attempt: int = 0) -> str:
    server = SimpleNamespace(
        server_client=SimpleNamespace(global_config_dict=OmegaConf.create(global_config)),
        config=SimpleNamespace(agent_server=SimpleNamespace(name="agent")),
        model_sampling_segment=HarborEnvironmentServer.model_sampling_segment,
    )
    request = SimpleNamespace(
        episode_id=EpisodeId(rollout_id="r1", attempt=attempt), skip_token_capture=False, model_sampling=None
    )
    return HarborEnvironmentServer._agent_responses_path(server, request)


def _config(*, enabled: bool, all_agents: bool = False, agent_opt_in: bool = False) -> dict:
    return {
        "token_id_capture": {"enabled": enabled, "all_agents": all_agents},
        "agent": {"responses_api_agents": {"simple_agent": {"token_id_capture": agent_opt_in}}},
    }


@pytest.mark.parametrize(
    "global_config, path",
    [
        ({"agent": {"responses_api_agents": {"simple_agent": {}}}}, "/ng-rollout/r1/v1/responses"),
        (_config(enabled=False, all_agents=True, agent_opt_in=True), "/ng-rollout/r1/v1/responses"),
        (_config(enabled=True), "/ng-rollout/r1/v1/responses"),
        (_config(enabled=True, agent_opt_in=True), "/ng-rollout/r1/training-token-capture/v1/responses"),
        (_config(enabled=True, all_agents=True), "/ng-rollout/r1/training-token-capture/v1/responses"),
    ],
)
def test_the_agent_is_activated_on_the_rollout_path_with_the_training_tag_when_capture_is_on(global_config, path):
    assert _agent_responses_path(global_config) == path


def test_a_later_attempt_is_activated_on_its_own_capture_key():
    assert (
        _agent_responses_path(_config(enabled=True, all_agents=True), attempt=2)
        == "/ng-rollout/r1-a2/training-token-capture/v1/responses"
    )
