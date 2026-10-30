# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import (
    BaseVerifyRequest,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.testing.session_conformance import check_resources_session_contract
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.example_single_tool_call.app import (
    VERIFIER_FIXTURE,
    SimpleWeatherResourcesServer,
    SimpleWeatherResourcesServerConfig,
)


class TestApp:
    @staticmethod
    def _server() -> SimpleWeatherResourcesServer:
        return SimpleWeatherResourcesServer(
            config=SimpleWeatherResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="weather",
            ),
            server_client=MagicMock(spec=ServerClient),
        )

    def test_sanity(self) -> None:
        config = SimpleWeatherResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        SimpleWeatherResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_verifier_fixture(self) -> None:
        asyncio.run(
            exercise_verifier_fixture(
                VERIFIER_FIXTURE,
                reward_range=(0.0, 1.0),
                higher_is_better=True,
                determinism="unknown",
            )
        )

    async def test_environment_server_session_lifecycle_and_verification(self) -> None:
        server = self._server()
        episode_id = EpisodeId(rollout_id="rollout", attempt=0)
        task_id = TaskId(taskset="example", task_id="0")
        seed = await server.seed_session(
            ResourcesSeedSessionRequest(
                resources_session_id="resources-session",
                episode_id=episode_id,
                task_id=task_id,
                task_data={},
            )
        )
        repeated = await server.seed_session(
            ResourcesSeedSessionRequest(
                resources_session_id="resources-session",
                episode_id=episode_id,
                task_id=task_id,
                task_data={},
            )
        )
        assert repeated == seed

        response = NeMoGymResponse(
            id="response",
            created_at=0,
            model="model",
            object="response",
            output=[
                {
                    "id": "call",
                    "call_id": "call",
                    "name": "get_weather",
                    "arguments": '{"city":"San Francisco"}',
                    "type": "function_call",
                    "status": "completed",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        # single_agent_turn sends the same flat body an Agent's /run sends.
        verification = await server.verify(
            BaseVerifyRequest(responses_create_params={"input": "weather?"}, response=response)
        )
        assert verification.reward == 1.0

        close = await server.close_resources_session(
            ResourcesCloseSessionRequest(
                resources_session_id=seed.resources_session_id,
                episode_id=episode_id,
            )
        )
        assert close.resources_session_id == seed.resources_session_id

        repeated_close = await server.close_resources_session(
            ResourcesCloseSessionRequest(
                resources_session_id=seed.resources_session_id,
                episode_id=episode_id,
            )
        )
        assert repeated_close == close
        with pytest.raises(HTTPException, match="already closed") as refused:
            await server.seed_session(
                ResourcesSeedSessionRequest(
                    resources_session_id="resources-session",
                    episode_id=episode_id,
                    task_id=task_id,
                    task_data={},
                )
            )
        assert refused.value.status_code == 409

    def test_follows_the_session_contract(self) -> None:
        server = self._server()
        check_resources_session_contract(
            server.setup_webserver(),
            ResourcesSeedSessionRequest(
                resources_session_id="contract-session",
                episode_id=EpisodeId(rollout_id="rollout", attempt=0),
                task_id=TaskId(taskset="example", task_id="0"),
                task_data={},
            ),
            keeps_state=True,
            holds_state=lambda resources_session_id: resources_session_id in server._session_episodes,
        )

    def test_close_route_releases_the_session(self) -> None:
        server = self._server()
        client = TestClient(server.setup_webserver())
        episode_id = EpisodeId(rollout_id="rollout", attempt=0)
        seed = client.post(
            "/seed_session",
            json=ResourcesSeedSessionRequest(
                resources_session_id="resources-session",
                episode_id=episode_id,
                task_id=TaskId(taskset="example", task_id="0"),
                task_data={},
            ).model_dump(mode="json"),
        )

        assert seed.status_code == 200
        resources_session_id = seed.json()["resources_session_id"]
        close = client.post(
            "/close_session",
            json=ResourcesCloseSessionRequest(
                resources_session_id=resources_session_id,
                episode_id=episode_id,
            ).model_dump(mode="json"),
        )
        assert close.status_code == 200
        # The route reached this server's close, not the base default, so the session was released.
        assert resources_session_id not in server._session_episodes
        assert server._fence.closed_episode(resources_session_id) == episode_id

    def test_close_of_a_closed_session_naming_another_episode_is_refused(self) -> None:
        server = self._server()
        client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
        episode_id = EpisodeId(rollout_id="rollout", attempt=0)

        def close(episode: EpisodeId) -> int:
            body = ResourcesCloseSessionRequest(resources_session_id="resources-session", episode_id=episode)
            return client.post("/close_session", json=body.model_dump(mode="json")).status_code

        assert close(episode_id) == 200
        assert close(EpisodeId(rollout_id="other", attempt=0)) == 409
        # The refused close leaves the record of the episode that closed the session.
        assert server._fence.closed_episode("resources-session") == episode_id
