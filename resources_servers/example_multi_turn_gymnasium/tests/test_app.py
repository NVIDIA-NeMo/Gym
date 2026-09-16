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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.example_multi_turn_gymnasium.app import ExampleMultiTurnEnv
from resources_servers.gymnasium import EnvStepRequest


def _step_request() -> EnvStepRequest:
    return EnvStepRequest(
        responses_create_params={"input": []},
        response=NeMoGymResponse(
            id="response",
            created_at=0.0,
            model="model",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        ),
    )


class TestApp:
    def test_sanity(self) -> None:
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")
        ExampleMultiTurnEnv(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("terminated", "truncated"), [(True, False), (False, True)])
    async def test_terminal_step_releases_turn_state(self, terminated: bool, truncated: bool) -> None:
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")
        server = ExampleMultiTurnEnv(config=config, server_client=MagicMock(spec=ServerClient))
        server.session_turns["sid"] = 3
        server.session_state["sid"] = {"owned": True}

        with patch.object(type(server), "step", new=AsyncMock(return_value=(None, 0.0, terminated, truncated, {}))):
            response = await server._step_endpoint(_step_request(), SimpleNamespace(session={SESSION_ID_KEY: "sid"}))

        assert response.terminated is terminated
        assert response.truncated is truncated
        assert "sid" not in server.session_turns
        assert "sid" not in server.session_state

    @pytest.mark.asyncio
    async def test_cleanup_is_idempotent_under_concurrent_calls(self) -> None:
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")
        server = ExampleMultiTurnEnv(config=config, server_client=MagicMock(spec=ServerClient))
        server.session_turns["sid"] = 3

        await asyncio.gather(*(server.close_session("sid") for _ in range(3)))

        assert "sid" not in server.session_turns

    @pytest.mark.asyncio
    async def test_reset_replaces_state_through_cleanup_path(self) -> None:
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")
        server = ExampleMultiTurnEnv(config=config, server_client=MagicMock(spec=ServerClient))
        server.session_turns["sid"] = 3
        server.session_state["sid"] = {"owned": True}

        await server.reset({}, "sid")

        assert server.session_turns["sid"] == 0
        assert "sid" not in server.session_state
