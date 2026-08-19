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

from nemo_gym.base_resources_server import (
    BaseCloseSessionRequest,
    BaseCloseSessionResponse,
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


def _resources_server() -> SimpleResourcesServer:
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")

    class TestSimpleResourcesServer(SimpleResourcesServer):
        async def verify(self, body):
            pass

    return TestSimpleResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestBaseVerifyResponse:
    def test_failure_reason_defaults_none_and_round_trips(self) -> None:
        response = BaseVerifyResponse(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            response=NeMoGymResponse.model_construct(id="resp-1", output=[]),
            reward=0.0,
        )
        assert response.failure_reason is None
        assert response.model_dump()["failure_reason"] is None

        rescued = response.model_copy(update={"failure_reason": "judge response unparseable after 3 attempts"})
        assert rescued.model_dump()["failure_reason"] == "judge response unparseable after 3 attempts"
        assert rescued.reward == 0.0


class TestBaseMultiRewardVerifyResponse:
    def test_reward_components_round_trip(self) -> None:
        response = BaseMultiRewardVerifyResponse(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            response=NeMoGymResponse.model_construct(id="resp-1", output=[]),
            reward=2.0,
            reward_components={"correctness": 1.0, "format": 1.0},
        )
        dumped = response.model_dump()
        assert dumped["reward_components"] == {"correctness": 1.0, "format": 1.0}
        assert dumped["reward"] == 2.0


class TestBaseResourcesServer:
    def test_sanity(self) -> None:
        _resources_server().setup_webserver()

    def test_reverify_mode(self) -> None:
        assert asyncio.run(_resources_server().get_reverify_mode()) == ReverifyMode.UNKNOWN


class TestSessionLifecycle:
    """Teardown must reach the environment even when the rollout never got to verify."""

    def test_close_session_is_a_no_op_by_default(self) -> None:
        result = asyncio.run(_resources_server().close_session(BaseCloseSessionRequest()))
        assert isinstance(result, BaseCloseSessionResponse)

    def test_route_is_registered(self) -> None:
        routes = {route.path for route in _resources_server().setup_webserver().routes}
        assert "/close_session" in routes
        assert "/seed_session" in routes

    def test_sweeper_is_off_unless_a_ttl_is_configured(self) -> None:
        assert BaseResourcesServerConfig(host="", port=0, entrypoint="", name="").session_ttl_s is None

    def test_idle_sessions_are_reclaimed_and_active_ones_are_not(self) -> None:
        closed: list[str] = []

        class _CountingServer(SimpleResourcesServer):
            async def verify(self, body):
                pass

            async def close_session(self, body: BaseCloseSessionRequest) -> BaseCloseSessionResponse:
                closed.append("called")
                return BaseCloseSessionResponse()

        server = _CountingServer(
            config=BaseResourcesServerConfig(host="", port=0, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )
        server.config.session_ttl_s = 0.05
        server.config.session_sweep_interval_s = 0.01

        async def _run() -> None:
            server.touch_session("idle")
            task = asyncio.create_task(server._sweep_idle_sessions())
            await asyncio.sleep(0.12)
            server.touch_session("active")
            await asyncio.sleep(0.02)
            task.cancel()

        asyncio.run(_run())
        assert closed, "an idle session should have been reclaimed"
        assert "idle" not in server._session_last_seen
        assert "active" in server._session_last_seen

    def test_forget_session_stops_tracking(self) -> None:
        server = _resources_server()
        server.touch_session("s1")
        server.forget_session("s1")
        server.forget_session("s1")  # idempotent
        assert "s1" not in server._session_last_seen
