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
import contextlib
import sys
import types
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.gymnasium import EnvResetRequest
from resources_servers.tales.app import (
    TALESResourcesServer,
    TALESResourcesServerConfig,
)


_CREATE_PARAMS = NeMoGymResponseCreateParamsNonStreaming(input="placeholder")
_FRAMEWORKS = ("textworld", "textworld_express", "alfworld", "scienceworld", "jericho")


def _make_server(**config_kwargs) -> TALESResourcesServer:
    config = TALESResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **config_kwargs)
    return TALESResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _reset_payload(**metadata) -> dict:
    return EnvResetRequest(responses_create_params=_CREATE_PARAMS, **metadata).model_dump(mode="json")


def _step_payload(action_text: str, **metadata) -> dict:
    return {
        "responses_create_params": _CREATE_PARAMS.model_dump(mode="json"),
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


class FakeEnv:
    def __init__(self, steps=None, reset_info=None):
        self._steps = list(steps or [("obs1", 1.0, False, {}), ("obs2", 2.0, True, {})])
        self._reset_info = reset_info or {}
        self._i = 0

    def reset(self, seed=None):  # noqa: ARG002
        return "initial observation", dict(self._reset_info)

    def step(self, command):  # noqa: ARG002
        result = self._steps[min(self._i, len(self._steps) - 1)]
        self._i += 1
        return result

    def close(self):
        pass


def _fake_framework_module(n_train=3, n_test=1):
    return types.SimpleNamespace(
        train_environments=[("game", f"t{i}") for i in range(n_train)],
        environments=[("game", f"e{i}") for i in range(n_test)],
    )


@contextlib.contextmanager
def _patch_env(fake_env: FakeEnv, framework_module=None):
    framework_module = framework_module or _fake_framework_module()
    names = ["tales"] + [f"tales.{fw}" for fw in _FRAMEWORKS]
    saved = {n: sys.modules.get(n) for n in names}
    sys.modules["tales"] = types.ModuleType("tales")
    for fw in _FRAMEWORKS:
        sys.modules[f"tales.{fw}"] = framework_module
    try:
        with patch("resources_servers.tales.app.gym.make", return_value=fake_env):
            yield
    finally:
        for n, mod in saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


class TestApp:
    def test_sanity(self) -> None:
        _make_server()

    def test_reset_and_step_flow(self) -> None:
        server = _make_server()
        fake = FakeEnv(steps=[("you win", 1.0, True, {})])
        with _patch_env(fake):
            client = TestClient(server.setup_webserver())
            r = client.post("/reset", json=_reset_payload(framework="alfworld", task_no=0))
            assert r.status_code == 200
            assert r.json()["observation"] == "initial observation"
            cookies = r.cookies

            s = client.post("/step", json=_step_payload("take apple"), cookies=cookies)
            payload = s.json()
            assert s.status_code == 200
            assert payload["reward"] == 1.0
            assert payload["terminated"] is True
            assert payload["truncated"] is False
            assert payload["info"]["total_score"] == 1.0

    def test_textworld_uses_delta_reward(self) -> None:
        server = _make_server()
        fake = FakeEnv(steps=[("o", 3.0, False, {}), ("o", 3.0, False, {})])
        with _patch_env(fake):
            client = TestClient(server.setup_webserver())
            cookies = client.post("/reset", json=_reset_payload(framework="textworld")).cookies
            s1 = client.post("/step", json=_step_payload("go north"), cookies=cookies).json()
            s2 = client.post("/step", json=_step_payload("look"), cookies=cookies).json()
            assert s1["reward"] == 3.0
            assert s2["reward"] == 0.0
            assert s2["info"]["total_score"] == 3.0

    def test_truncates_at_max_episode_steps(self) -> None:
        server = _make_server(max_episode_steps=2)
        fake = FakeEnv(steps=[("o", 0.0, False, {})])
        with _patch_env(fake):
            client = TestClient(server.setup_webserver())
            cookies = client.post("/reset", json=_reset_payload(framework="jericho")).cookies
            s1 = client.post("/step", json=_step_payload("wait"), cookies=cookies).json()
            s2 = client.post("/step", json=_step_payload("wait"), cookies=cookies).json()
            assert s1["truncated"] is False
            assert s2["truncated"] is True
            assert s2["terminated"] is False

    def test_admissible_commands_gating(self) -> None:
        info = {"admissible_commands": ["look", "inventory"]}
        server = _make_server(expose_admissible_commands=False)
        with _patch_env(FakeEnv(reset_info=info)):
            client = TestClient(server.setup_webserver())
            r = client.post("/reset", json=_reset_payload(framework="alfworld")).json()
            assert "admissible_commands" not in r["info"]
        server = _make_server(expose_admissible_commands=True)
        with _patch_env(FakeEnv(reset_info=info)):
            client = TestClient(server.setup_webserver())
            r = client.post("/reset", json=_reset_payload(framework="alfworld")).json()
            assert r["info"]["admissible_commands"] == ["look", "inventory"]

    def test_invalid_task_no_returns_400(self) -> None:
        server = _make_server()
        with _patch_env(FakeEnv(), framework_module=_fake_framework_module(n_train=2)):
            client = TestClient(server.setup_webserver())
            r = client.post("/reset", json=_reset_payload(framework="alfworld", task_no=99))
            assert r.status_code == 400
            assert "Invalid task number" in r.json()["detail"]

    def test_step_before_reset_returns_400(self) -> None:
        server = _make_server()
        client = TestClient(server.setup_webserver())
        r = client.post("/step", json=_step_payload("look"))
        assert r.status_code == 400

    def test_concurrent_sessions_keep_independent_state(self) -> None:
        server = _make_server()
        app = server.setup_webserver()
        env_a = FakeEnv(steps=[("a", 5.0, False, {})])
        env_b = FakeEnv(steps=[("b", 2.0, False, {})])

        client_a = TestClient(app)
        client_b = TestClient(app)
        with _patch_env(env_a):
            cookies_a = client_a.post("/reset", json=_reset_payload(framework="textworld")).cookies
        with _patch_env(env_b):
            cookies_b = client_b.post("/reset", json=_reset_payload(framework="alfworld")).cookies

        a = client_a.post("/step", json=_step_payload("x"), cookies=cookies_a).json()
        b = client_b.post("/step", json=_step_payload("y"), cookies=cookies_b).json()
        assert a["reward"] == 5.0
        assert b["reward"] == 2.0
        assert a["info"]["total_score"] == 5.0
        assert b["info"]["total_score"] == 2.0
        assert len(server.session_id_to_state) == 2
