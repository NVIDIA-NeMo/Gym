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
from resources_servers.example_multi_turn_gymnasium.app import ExampleMultiTurnEnv
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
async def test_counter_retire_removes_mapping_state_and_keeps_tombstone() -> None:
    server = _server(StatefulCounterResourcesServer, StatefulCounterResourcesServerConfig)
    server.execution_to_session[("rollout-a", 0)] = "session-a"
    server.session_id_to_counter["session-a"] = 17
    participant = server.checkpoint_participant()
    participant.bind("rollout-a", 0)

    await participant.retire_execution("rollout-a", 0)

    assert ("rollout-a", 0) not in server.execution_to_session
    assert "session-a" not in server.session_id_to_counter
    assert participant.is_tombstoned("rollout-a", 0)
    assert participant.status()["lock_entries"] == 0


@pytest.mark.asyncio
async def test_workplace_restores_every_dataframe_before_activation() -> None:
    server = _server(WorkbenchResourcesServer, WorkbenchResourcesServerConfig)
    server.execution_to_session[("rollout-a", 0)] = "session-a"
    server.session_id_to_tool_env["session-a"] = get_tools(_TOOLKITS)
    state = await server.export_checkpoint_state("rollout-a", 0)
    snapshot = ResourceSnapshot(rollout_id="rollout-a", attempt_index=1, state_revision=2, state=state)
    snapshot = ResourceSnapshot.model_validate_json(snapshot.model_dump_json())

    await server.restore_checkpoint_states([snapshot])
    restored = server.session_id_to_tool_env[server.execution_to_session[("rollout-a", 1)]]
    for name, frames in state["containers"].items():
        for attribute, payload in frames.items():
            original = server.session_id_to_tool_env["session-a"]["containers"][name]
            restored_frame = getattr(restored["containers"][name], attribute)
            original_frame = getattr(original, attribute)
            assert restored_frame.equals(original_frame)
            assert type(restored_frame.index) is type(original_frame.index)


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


def test_gymnasium_checkpoint_continuation_is_per_subclass_opt_in() -> None:
    assert _server(BlackjackEnv, BaseResourcesServerConfig).checkpoint_state_enabled()
    assert not _server(ExampleMultiTurnEnv, BaseResourcesServerConfig).checkpoint_state_enabled()


def test_checkpoint_route_classification_defaults_unknown_posts_to_mutation() -> None:
    server = _server(StatefulCounterResourcesServer, StatefulCounterResourcesServerConfig)
    assert server.checkpoint_route_kind("/get_counter_value", "POST") == "read"
    assert server.checkpoint_route_kind("/new_stateful_tool", "POST") == "mutation"
    assert server.checkpoint_route_kind("/aggregate_metrics", "POST") is None
    assert server.checkpoint_route_kind("/ng-control/v1/custom", "POST") is None
    assert server.checkpoint_route_kind("/mcp", "POST") is None
