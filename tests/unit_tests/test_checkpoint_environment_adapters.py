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
"""Initial environment adapters preserve logical state across attempts."""

import random
from unittest.mock import MagicMock

import pytest

from nemo_gym._checkpoint import ResourceSnapshot
from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.blackjack.app import BlackjackEnv
from resources_servers.example_session_state_mgmt.app import (
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
)
from resources_servers.workplace_assistant.app import (
    _TOOLKITS,
    WorkbenchResourcesServer,
    WorkbenchResourcesServerConfig,
    get_tools,
)


def _server(server_type, config_type):
    config = config_type(host="", port=0, entrypoint="", name="resources")
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return server_type(config=config, server_client=client)


@pytest.mark.asyncio
async def test_counter_restores_under_replacement_attempt() -> None:
    server = _server(StatefulCounterResourcesServer, StatefulCounterResourcesServerConfig)
    server.execution_to_session[("rollout-a", 0)] = "session-a"
    server.session_id_to_counter["session-a"] = 17
    state = await server.export_checkpoint_state("rollout-a", 0)

    await server.restore_checkpoint_states(
        [ResourceSnapshot(rollout_id="rollout-a", attempt_index=1, state_revision=3, state=state)]
    )
    restored_session = server.execution_to_session[("rollout-a", 1)]
    assert server.session_id_to_counter[restored_session] == 17


@pytest.mark.asyncio
async def test_workplace_restores_every_dataframe_before_activation() -> None:
    server = _server(WorkbenchResourcesServer, WorkbenchResourcesServerConfig)
    server.execution_to_session[("rollout-a", 0)] = "session-a"
    server.session_id_to_tool_env["session-a"] = get_tools(_TOOLKITS)
    state = await server.export_checkpoint_state("rollout-a", 0)

    await server.restore_checkpoint_states(
        [ResourceSnapshot(rollout_id="rollout-a", attempt_index=1, state_revision=2, state=state)]
    )
    restored = server.session_id_to_tool_env[server.execution_to_session[("rollout-a", 1)]]
    for name, frames in state["containers"].items():
        for attribute, payload in frames.items():
            assert getattr(restored["containers"][name], attribute).to_json(orient="split") == payload


@pytest.mark.asyncio
async def test_blackjack_restores_rng_position() -> None:
    server = _server(BlackjackEnv, BaseResourcesServerConfig)
    rng = random.Random(1234)
    server.execution_to_session[("rollout-a", 0)] = "session-a"
    server.session_state["session-a"] = {"player": ["10", "2"], "dealer": ["9", "7"], "rng": rng}
    state = await server.export_checkpoint_state("rollout-a", 0)
    expected_next = rng.choice(["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"])

    await server.restore_checkpoint_states(
        [ResourceSnapshot(rollout_id="rollout-a", attempt_index=1, state_revision=1, state=state)]
    )
    restored = server.session_state[server.execution_to_session[("rollout-a", 1)]]
    assert restored["rng"].choice(["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]) == expected_next
