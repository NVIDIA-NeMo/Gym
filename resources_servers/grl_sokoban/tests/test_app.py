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
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.base_gymnasium import EnvResetRequest
from resources_servers.grl_sokoban.app import (
    GrlSokobanResourcesServer,
    GrlSokobanResourcesServerConfig,
)


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
        config = GrlSokobanResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        GrlSokobanResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_reset_and_step_flow(self) -> None:
        config = GrlSokobanResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = GrlSokobanResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        class FakeEnv:
            ACTION_LOOKUP = {1: "Up"}

            def reset(self, seed=None):  # noqa: ARG002
                return "Initial observation"

            def step(self, action):
                assert action == 1
                return "Next observation", 1.0, True, {"success": True}

            def close(self):
                pass

        with patch("resources_servers.grl_sokoban.app.SokobanEnv", return_value=FakeEnv()):
            app = server.setup_webserver()
            client = TestClient(app)

            response = client.post("/reset", json=_reset_payload(seed=123))
            assert response.status_code == 200
            assert "Initial observation" in response.json()["observation"]

            cookies = response.cookies
            response = client.post("/step", json=_step_payload("<action>Up</action>"), cookies=cookies)
            payload = response.json()
            assert response.status_code == 200
            assert payload["reward"] == 1.0
            assert payload["terminated"] is True
            assert payload["info"]["success"] is True
            assert payload["info"]["action_label"] == "Up"
            assert payload["info"]["total_reward"] == 1.0

    def test_reset_merges_metadata_into_env_config(self) -> None:
        config = GrlSokobanResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = GrlSokobanResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        class FakeEnv:
            ACTION_LOOKUP = {1: "Up"}

            def __init__(self, config):
                self.config = config

            def reset(self, seed=None):  # noqa: ARG002
                return "Board"

            def close(self):
                pass

        with patch("resources_servers.grl_sokoban.app.SokobanEnv", side_effect=FakeEnv) as env_ctor:
            app = server.setup_webserver()
            client = TestClient(app)
            response = client.post("/reset", json=_reset_payload(seed=7, dim_room=[5, 5], num_boxes=2))
            assert response.status_code == 200
            env_config = env_ctor.call_args.args[0]
            assert env_config["dim_room"] == [5, 5]
            assert env_config["num_boxes"] == 2

    def test_step_accepts_action_words_without_tags(self) -> None:
        config = GrlSokobanResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = GrlSokobanResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        class FakeEnv:
            ACTION_LOOKUP = {1: "Up", 2: "Down"}

            def reset(self, seed=None):  # noqa: ARG002
                return "Init"

            def step(self, action):
                assert action == 2
                return "Obs", 0.5, False, {"success": False}

            def close(self):
                pass

        with patch("resources_servers.grl_sokoban.app.SokobanEnv", return_value=FakeEnv()):
            app = server.setup_webserver()
            client = TestClient(app)

            seed_resp = client.post("/reset", json=_reset_payload())
            cookies = seed_resp.cookies
            resp = client.post("/step", json=_step_payload("I choose down."), cookies=cookies)
            payload = resp.json()
            assert resp.status_code == 200
            assert payload["terminated"] is False
            assert payload["reward"] == 0.5
            assert payload["info"]["action_label"] == "Down"

    def test_step_invalid_action_raises(self) -> None:
        config = GrlSokobanResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = GrlSokobanResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        class FakeEnv:
            ACTION_LOOKUP = {1: "Up"}

            def reset(self, seed=None):  # noqa: ARG002
                return "Init"

            def close(self):
                pass

        with patch("resources_servers.grl_sokoban.app.SokobanEnv", return_value=FakeEnv()):
            app = server.setup_webserver()
            client = TestClient(app)

            seed_resp = client.post("/reset", json=_reset_payload())
            cookies = seed_resp.cookies
            resp = client.post("/step", json=_step_payload("Move diagonally"), cookies=cookies)
            assert resp.status_code == 400
            assert resp.json()["detail"].startswith("Unable to parse action")
