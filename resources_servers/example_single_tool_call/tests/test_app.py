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
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.single_agent_turn_types import (
    SingleAgentTurnResourcesVerifyRequest,
    SingleAgentTurnVerificationInput,
)
from nemo_gym.verifier_fixture import exercise_verifier_fixture
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

    async def test_native_session_lifecycle_and_verification(self) -> None:
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
        verification = await server.verify(
            SingleAgentTurnResourcesVerifyRequest(
                episode_id=episode_id,
                task_id=task_id,
                verification_input=SingleAgentTurnVerificationInput(
                    responses_create_params={"input": "weather?"},
                    response=response,
                ),
            )
        )
        assert verification.reward == 1.0

        close = await server.close_session(
            ResourcesCloseSessionRequest(
                resources_session_id=seed.resources_session_id,
                episode_id=episode_id,
            )
        )
        assert close.resources_session_id == seed.resources_session_id

        repeated_close = await server.close_session(
            ResourcesCloseSessionRequest(
                resources_session_id=seed.resources_session_id,
                episode_id=episode_id,
            )
        )
        assert repeated_close == close
        with pytest.raises(ValueError, match="already closed"):
            await server.seed_session(
                ResourcesSeedSessionRequest(
                    resources_session_id="resources-session",
                    episode_id=episode_id,
                    task_id=task_id,
                    task_data={},
                )
            )

    def test_native_session_routes_bind_the_native_contract(self) -> None:
        client = TestClient(self._server().setup_webserver())
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
