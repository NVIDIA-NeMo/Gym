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
import asyncio
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.nullable_tool_args.app import (
    VERIFIER_FIXTURE,
    NullableToolArgsConfig,
    NullableToolArgsResourcesServer,
)


def _server() -> NullableToolArgsResourcesServer:
    config = NullableToolArgsConfig(host="0.0.0.0", port=8080, entrypoint="", name="nullable_tool_args")
    return NullableToolArgsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
        )
    )


def test_any_tool_name_is_acknowledged() -> None:
    client = TestClient(_server().setup_webserver())
    assert client.post("/create_contact", json={"middle_name": None}).json() == {"status": "ok"}


def test_catch_all_tool_route_keeps_the_session_contract() -> None:
    check_resources_session_contract(
        _server().setup_webserver(),
        ResourcesSeedSessionRequest(
            resources_session_id="contract-session",
            episode_id=EpisodeId(rollout_id="rollout", attempt=0),
            task_id=TaskId(taskset="example", task_id="0"),
            task_data={},
        ),
        keeps_state=False,
    )


def test_null_rate_is_reported_per_position() -> None:
    results = [
        {"reward": 1.0, "verifier_metadata": {"position": "first"}},
        {"reward": 0.0, "verifier_metadata": {"position": "first"}},
        {"reward": 1.0, "verifier_metadata": {"position": "mid"}},
    ]
    assert _server().compute_metrics([results]) == {"first/null_rate": 0.5, "mid/null_rate": 1.0}
