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

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.grl_sokoban.app import (
    GrlSokobanResourcesServer,
    GrlSokobanResourcesServerConfig,
)


_VERIFY_CREATE_PARAMS = NeMoGymResponseCreateParamsNonStreaming(
    input="placeholder",
)

_VERIFY_RESPONSE = NeMoGymResponse.model_construct(
    id="resp_test",
    object="response",
    created_at=0.0,
    status="completed",
    output=[],
    model="gpt-4.1",
    parallel_tool_calls=True,
    tool_choice="auto",
    tools=[],
)


def _verify_payload() -> dict:
    return {
        "responses_create_params": _VERIFY_CREATE_PARAMS.model_dump(mode="json"),
        "response": _VERIFY_RESPONSE.model_dump(mode="json"),
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

    def test_seed_and_step_flow(self) -> None:
        config = GrlSokobanResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        server = GrlSokobanResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )

        class FakeEnv:
            ACTION_LOOKUP = {1: "Up"}

            def __init__(self, *_args, **_kwargs) -> None:
                self._closed = False
                self.step_calls = 0

            def reset(self, seed=None):  # noqa: ARG002
                return "Initial observation"

            def step(self, action):
                self.step_calls += 1
                assert action == 1
                reward = 1.0
                done = self.step_calls >= 1
                info = {"success": done}
                return "Next observation", reward, done, info

            def close(self):
                self._closed = True

        fake_env = FakeEnv()
        with patch("resources_servers.grl_sokoban.app.SokobanEnv", return_value=fake_env):
            app = server.setup_webserver()
            client = TestClient(app)

            response = client.post("/seed_session", json={"seed": 123})
            assert response.status_code == 200
            assert response.json()["observation"] == "Initial observation"

            cookies = response.cookies
            response = client.post("/step", json={"actions": ["Up"]}, cookies=cookies)
            payload = response.json()
            assert response.status_code == 200
            assert payload["observation"] == "Next observation"
            assert payload["reward"] == 1.0
            assert payload["done"] is True
            assert payload["steps"][0]["action_label"] == "Up"
            assert fake_env.step_calls == 1

            response = client.post("/verify", json=_verify_payload(), cookies=cookies)
            assert response.status_code == 200
            payload = response.json()
            assert payload["success"] is True
            assert payload["reward"] == 1.0

    def test_step_action_mapping_stops_after_done(self) -> None:
        config = GrlSokobanResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        server = GrlSokobanResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )

        class FakeEnv:
            ACTION_LOOKUP = {1: "Up", 2: "Down"}

            def __init__(self, *_args, **_kwargs) -> None:
                self.calls = 0
                self.closed = False

            def reset(self, seed=None):  # noqa: ARG002
                return "Init"

            def step(self, action):
                self.calls += 1
                if self.calls == 1:
                    assert action == 1
                    return "Obs1", 0.5, True, {"success": True}
                raise AssertionError("Env.step should not be called after done")

            def close(self):
                self.closed = True

        fake_env = FakeEnv()
        with patch("resources_servers.grl_sokoban.app.SokobanEnv", return_value=fake_env):
            app = server.setup_webserver()
            client = TestClient(app)

            seed_resp = client.post("/seed_session", json={})
            cookies = seed_resp.cookies
            resp = client.post("/step", json={"actions": ["Up", "Down"]}, cookies=cookies)
            payload = resp.json()
            assert resp.status_code == 200
            assert payload["done"] is True
            assert payload["steps"][0]["action_label"] == "Up"
            assert len(payload["steps"]) == 1
            assert len(payload["history"]) == 1
            assert fake_env.calls == 1

    def test_step_invalid_action_raises(self) -> None:
        config = GrlSokobanResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        server = GrlSokobanResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )

        class FakeEnv:
            ACTION_LOOKUP = {1: "Up"}

            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def reset(self, seed=None):  # noqa: ARG002
                return "Init"

            def step(self, action):  # pragma: no cover - not reached
                raise AssertionError("Should not call step for invalid action")

            def close(self):
                pass

        with patch("resources_servers.grl_sokoban.app.SokobanEnv", return_value=FakeEnv()):
            app = server.setup_webserver()
            client = TestClient(app)

            seed_resp = client.post("/seed_session", json={})
            cookies = seed_resp.cookies
            resp = client.post("/step", json={"actions": ["Left"]}, cookies=cookies)
            assert resp.status_code == 400
            assert resp.json()["detail"].startswith("Unable to parse action")

    def test_verify_failure_zero_reward_and_cleanup(self) -> None:
        config = GrlSokobanResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        server = GrlSokobanResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )

        class FakeEnv:
            ACTION_LOOKUP = {1: "Up"}

            def __init__(self, *_args, **_kwargs) -> None:
                self.closed = False

            def reset(self, seed=None):  # noqa: ARG002
                return "Init"

            def step(self, action):
                return "Obs", 0.0, False, {"success": False}

            def close(self):
                self.closed = True

        fake_env = FakeEnv()
        with patch("resources_servers.grl_sokoban.app.SokobanEnv", return_value=fake_env):
            app = server.setup_webserver()
            client = TestClient(app)

            seed_resp = client.post("/seed_session", json={})
            cookies = seed_resp.cookies
            client.post("/step", json={"actions": [1]}, cookies=cookies)

            verify_resp = client.post(
                "/verify",
                json=_verify_payload(),
                cookies=cookies,
            )
            assert verify_resp.status_code == 200
            payload = verify_resp.json()
            assert payload["success"] is False
            assert payload["reward"] == 0.0
            assert fake_env.closed is True
            assert server.session_id_to_state == {}
