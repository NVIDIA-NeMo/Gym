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
        self.closed = False

    def reset(self, seed=None):  # noqa: ARG002
        return "initial observation", dict(self._reset_info)

    def step(self, command):  # noqa: ARG002
        result = self._steps[min(self._i, len(self._steps) - 1)]
        self._i += 1
        return result

    def close(self):
        self.closed = True


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

    def test_terminated_episode_closes_env_and_clears_session(self) -> None:
        server = _make_server()
        fake = FakeEnv(steps=[("you win", 1.0, True, {})])
        with _patch_env(fake):
            client = TestClient(server.setup_webserver())
            cookies = client.post("/reset", json=_reset_payload(framework="alfworld")).cookies
            client.post("/step", json=_step_payload("take apple"), cookies=cookies)
        assert fake.closed is True
        assert len(server.session_id_to_state) == 0

    def test_truncated_episode_closes_env_and_clears_session(self) -> None:
        server = _make_server(max_episode_steps=1)
        fake = FakeEnv(steps=[("o", 0.0, False, {})])
        with _patch_env(fake):
            client = TestClient(server.setup_webserver())
            cookies = client.post("/reset", json=_reset_payload(framework="jericho")).cookies
            s = client.post("/step", json=_step_payload("wait"), cookies=cookies).json()
        assert s["truncated"] is True
        assert fake.closed is True
        assert len(server.session_id_to_state) == 0

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

    def test_step_info_preserves_upstream_score_and_tracks_highscore(self) -> None:
        # Non-monotonic game: cumulative score peaks at 30 then drops to 20.
        # info["score"] must stay the upstream cumulative (not the per-step value),
        # and highscore/normalized_highscore must track the running max (paper metric).
        server = _make_server()
        steps = [
            ("o", 30.0, False, {"score": 30, "max_score": 100, "won": False}),
            ("o", -10.0, True, {"score": 20, "max_score": 100, "won": False}),
        ]
        with _patch_env(FakeEnv(steps=steps)):
            client = TestClient(server.setup_webserver())
            r = client.post("/reset", json=_reset_payload(framework="scienceworld"))
            assert r.json()["info"]["framework"] == "scienceworld"
            cookies = r.cookies
            s1 = client.post("/step", json=_step_payload("mix"), cookies=cookies).json()
            s2 = client.post("/step", json=_step_payload("pour"), cookies=cookies).json()
        assert s1["info"]["score"] == 30  # upstream cumulative, not clobbered
        assert s1["info"]["step_score"] == 30.0
        assert s1["info"]["game_score"] == 30.0
        assert s1["info"]["highscore"] == 30.0
        assert s1["info"]["normalized_highscore"] == 0.3
        assert s2["info"]["score"] == 20
        assert s2["info"]["highscore"] == 30.0  # running max survives the drop
        assert s2["info"]["normalized_highscore"] == 0.3
        assert s2["info"]["framework"] == "scienceworld"

    def test_step_info_falls_back_to_positional_score(self) -> None:
        # Upstream info without "score"/"max_score": game_score falls back to the
        # positional value and normalized_highscore is None (cannot normalize).
        server = _make_server()
        with _patch_env(FakeEnv(steps=[("o", 1.0, True, {})])):
            client = TestClient(server.setup_webserver())
            cookies = client.post("/reset", json=_reset_payload(framework="alfworld")).cookies
            s = client.post("/step", json=_step_payload("x"), cookies=cookies).json()
        assert s["info"]["game_score"] == 1.0
        assert s["info"]["highscore"] == 1.0
        assert s["info"]["normalized_highscore"] is None

    def test_compute_metrics_per_family_macro_average(self) -> None:
        server = _make_server()

        def rollout(framework, normalized, won):
            return {"info": {"framework": framework, "normalized_highscore": normalized, "won": won}}

        tasks = [
            [rollout("jericho", 0.10, False), rollout("jericho", 0.20, False)],  # task mean 15.0
            [rollout("jericho", 0.30, False)],  # task mean 30.0 -> jericho family mean 22.5
            [rollout("alfworld", 1.0, True), rollout("alfworld", 0.0, False)],  # family mean 50.0
            [{"info": {}}],  # missing fields -> skipped
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["tales/jericho/normalized_highscore"] == 22.5
        assert metrics["tales/alfworld/normalized_highscore"] == 50.0
        assert metrics["tales/alfworld/success_rate"] == 0.5
        assert metrics["tales/macro_avg/normalized_highscore"] == 36.25
        assert metrics["tales/num_families"] == 2
        assert metrics["tales/rollouts_missing_score_fields"] == 1

        key = server.get_key_metrics({"tales/macro_avg/normalized_highscore": 36.25, "mean/reward": -19.8})
        assert key == {"tales/macro_avg/normalized_highscore": 36.25}
