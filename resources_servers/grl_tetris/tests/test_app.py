# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.grl_tetris.app import (
    GrlTetrisResourcesServer,
    GrlTetrisResourcesServerConfig,
    TetrisSessionState,
)
from resources_servers.gymnasium import EnvResetRequest, EnvStepRequest


_RESET_CREATE_PARAMS = NeMoGymResponseCreateParamsNonStreaming(input="placeholder")


def _reset_payload(**metadata) -> dict:
    return EnvResetRequest(responses_create_params=_RESET_CREATE_PARAMS, **metadata).model_dump(mode="json")


def _step_payload(action_text: str, **metadata) -> dict:
    return {
        "responses_create_params": _RESET_CREATE_PARAMS.model_dump(mode="json"),
        "response": {
            "id": "resp_test",
            "object": "response",
            "created_at": 0.0,
            "status": "completed",
            "output": [
                {
                    "id": "msg_test",
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                    "content": [{"annotations": [], "text": action_text, "type": "output_text"}],
                }
            ],
            "model": "gpt-4.1",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        },
        **metadata,
    }


class TestApp:
    def test_sanity(self) -> None:
        config = GrlTetrisResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        GrlTetrisResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_reset_and_step_flow(self) -> None:
        config = GrlTetrisResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = GrlTetrisResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        class FakeEnv:
            ACTION_LOOKUP = {0: "Left", 1: "Right", 2: "Down"}

            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def reset(self, seed=None):  # noqa: ARG002
                return "Initial observation"

            def step(self, action):
                assert action == 1  # Right
                return "Next observation", 0.5, False, {"success": False}

            def close(self):
                pass

        with patch("resources_servers.grl_tetris.app.TetrisEnv", return_value=FakeEnv()):
            app = server.setup_webserver()
            client = TestClient(app)

            response = client.post("/reset", json=_reset_payload(seed=7))
            assert response.status_code == 200
            assert "Initial observation" in response.json()["observation"]
            cookies = response.cookies

            response = client.post("/step", json=_step_payload("<action>Right</action>"), cookies=cookies)
            payload = response.json()
            assert response.status_code == 200
            assert payload["reward"] == 0.5
            assert payload["terminated"] is False
            assert payload["info"]["num_actions_applied"] == 1

    def test_batched_actions_stop_on_done(self) -> None:
        """Multiple <action> tags in one response are applied in order and break on done."""
        config = GrlTetrisResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = GrlTetrisResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        class FakeEnv:
            ACTION_LOOKUP = {0: "Left", 1: "Right", 2: "Down"}

            def __init__(self, *_args, **_kwargs) -> None:
                self.calls = 0

            def reset(self, seed=None):  # noqa: ARG002
                return "Initial"

            def step(self, action):
                self.calls += 1
                done = self.calls == 2  # second action ends the game
                return f"after step {self.calls}", 1.0, done, {"success": done}

            def close(self):
                pass

        fake_env = FakeEnv()
        with patch("resources_servers.grl_tetris.app.TetrisEnv", return_value=fake_env):
            app = server.setup_webserver()
            client = TestClient(app)

            response = client.post("/reset", json=_reset_payload(seed=11))
            cookies = response.cookies

            # Three action tags, but the env signals done after 2 — third skipped.
            text = "<action>Left</action><action>Right</action><action>Down</action>"
            response = client.post("/step", json=_step_payload(text), cookies=cookies)
            payload = response.json()
            assert response.status_code == 200
            assert payload["reward"] == 2.0
            assert payload["terminated"] is True
            assert payload["info"]["num_actions_applied"] == 2
            assert fake_env.calls == 2

    def test_step_no_action_tags_raises(self) -> None:
        config = GrlTetrisResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = GrlTetrisResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        class FakeEnv:
            ACTION_LOOKUP = {0: "Left", 1: "Right", 2: "Down"}

            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def reset(self, seed=None):  # noqa: ARG002
                return "Initial"

            def step(self, action):  # pragma: no cover - not reached
                return "x", 0.0, False, {}

            def close(self):
                pass

        with patch("resources_servers.grl_tetris.app.TetrisEnv", return_value=FakeEnv()):
            app = server.setup_webserver()
            client = TestClient(app)

            response = client.post("/reset", json=_reset_payload(seed=1))
            cookies = response.cookies

            response = client.post("/step", json=_step_payload("plain text, no tags"), cookies=cookies)
            assert response.status_code == 400
            assert "No <action>" in response.json()["detail"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("terminated", "truncated"), [(True, False), (False, True)])
    async def test_terminal_step_releases_session_state(self, terminated: bool, truncated: bool) -> None:
        config = GrlTetrisResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = GrlTetrisResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        env = MagicMock()
        server.session_id_to_state["sid"] = TetrisSessionState(env=env, observation="board")

        with patch.object(type(server), "step", new=AsyncMock(return_value=(None, 0.0, terminated, truncated, {}))):
            response = await server._step_endpoint(
                EnvStepRequest.model_validate(_step_payload("<action>Down</action>")),
                SimpleNamespace(session={SESSION_ID_KEY: "sid"}),
            )

        assert response.terminated is terminated
        assert response.truncated is truncated
        assert "sid" not in server.session_id_to_state
        env.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_cleanup_is_idempotent_under_concurrent_calls(self) -> None:
        config = GrlTetrisResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = GrlTetrisResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        env = MagicMock()
        server.session_id_to_state["sid"] = TetrisSessionState(env=env, observation="board")
        both_steps_started = asyncio.Event()
        started = 0

        async def finish_together(*_args, **_kwargs):
            nonlocal started
            started += 1
            if started == 2:
                both_steps_started.set()
            await both_steps_started.wait()
            return None, 0.0, True, False, {}

        request = EnvStepRequest.model_validate(_step_payload("<action>Down</action>"))
        http_request = SimpleNamespace(session={SESSION_ID_KEY: "sid"})
        with patch.object(type(server), "step", new=finish_together):
            responses = await asyncio.gather(
                server._step_endpoint(request, http_request),
                server._step_endpoint(request, http_request),
            )
        await server.close_session("sid")

        assert all(response.terminated for response in responses)
        assert "sid" not in server.session_id_to_state
        env.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_reset_releases_replaced_environment(self) -> None:
        config = GrlTetrisResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = GrlTetrisResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        old_env = MagicMock()
        new_env = MagicMock()
        new_env.reset.return_value = "new board"
        server.session_id_to_state["sid"] = TetrisSessionState(env=old_env, observation="old board")

        with patch("resources_servers.grl_tetris.app.TetrisEnv", return_value=new_env):
            await server.reset({}, "sid")

        old_env.close.assert_called_once_with()
        assert server.session_id_to_state["sid"].env is new_env
