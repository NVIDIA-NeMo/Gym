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
import subprocess
import urllib.error
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import responses_api_models.switchyard_model.app as app_module
from nemo_gym.server_utils import ServerClient
from responses_api_models.switchyard_model.app import (
    NeMoGymAsyncOpenAI,
    SwitchyardModel,
    SwitchyardModelConfig,
    _RolloutSessionMiddleware,
)


# Captured before any test monkeypatches subprocess.Popen, so mocks keep a real spec.
_REAL_POPEN = subprocess.Popen


def _response_data() -> dict:
    return {
        "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
        "created_at": 1753983920.0,
        "model": "openai/gpt-5.2",
        "object": "response",
        "output": [
            {
                "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                "content": [{"annotations": [], "text": "Hello!", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _chat_data() -> dict:
    return {
        "id": "chatcmpl-BzRdCFjIEIp59xXLBNYjdPPrcpDaa",  # pragma: allowlist secret
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "Hello!", "role": "assistant"},
            }
        ],
        "created": 1753983922,
        "model": "openai/gpt-5.2",
        "object": "chat.completion",
    }


class TestConfig:
    def test_attach_mode_requires_base_url(self) -> None:
        with pytest.raises(ValueError, match="switchyard_base_url is required"):
            SwitchyardModelConfig(host="0.0.0.0", port=8081, entrypoint="", name="sy", switchyard_model="policy-model")

    def test_launch_mode_requires_routing_profiles(self) -> None:
        with pytest.raises(ValueError, match="routing_profiles is required"):
            SwitchyardModelConfig(
                host="0.0.0.0",
                port=8081,
                entrypoint="",
                name="sy",
                switchyard_model="policy-model",
                launch_proxy=True,
            )

    def test_launch_mode_accepts_routing_profiles(self) -> None:
        config = SwitchyardModelConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="sy",
            switchyard_model="policy-model",
            launch_proxy=True,
            routing_profiles="/tmp/routes.yaml",
        )
        assert config.routing_profiles == "/tmp/routes.yaml"


class TestRolloutSessionMiddleware:
    async def test_non_http_scope_is_forwarded_untouched(self) -> None:
        seen: dict = {}

        async def inner(scope, receive, send):
            seen["scope"] = scope

        middleware = _RolloutSessionMiddleware(inner)
        await middleware({"type": "lifespan"}, None, None)

        assert seen["scope"] == {"type": "lifespan"}


class TestApp:
    def _setup_server(self, **overrides) -> SwitchyardModel:
        config = SwitchyardModelConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="test_switchyard_model",
            switchyard_base_url="http://127.0.0.1:4000/v1",
            switchyard_api_key="dummy_key",  # pragma: allowlist secret
            switchyard_model="policy-model",
            **overrides,
        )
        return SwitchyardModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))

    def test_sanity(self) -> None:
        server = self._setup_server()
        assert server._client.base_url == "http://127.0.0.1:4000/v1"

    def test_max_concurrent_requests_builds_semaphore(self) -> None:
        server = self._setup_server(max_concurrent_requests=2)
        assert server._semaphore._value == 2

    def test_responses_forwards_route_and_session_id(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server()
        seen: dict = {}

        async def mock_create_response(self, **kwargs):
            seen["headers"] = self.default_headers
            seen["kwargs"] = kwargs
            return _response_data()

        monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_response", mock_create_response)
        client = TestClient(server.setup_webserver())

        response = client.post(
            "/ng-rollout/task0-r1-a0/v1/responses",
            json={"input": [{"role": "user", "content": "hi"}], "model": "ignored"},
        )

        assert response.status_code == 200
        # The route name always wins -- a caller-supplied model must not bypass routing.
        assert seen["kwargs"]["model"] == "policy-model"
        assert seen["headers"]["proxy_x_session_id"] == "task0-r1-a0"
        # The routed target is visible on the response.
        assert response.json()["model"] == "openai/gpt-5.2"

    def test_chat_completions_forwards_route_and_session_id(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server()
        seen: dict = {}

        async def mock_create_chat_completion(self, **kwargs):
            seen["headers"] = self.default_headers
            seen["kwargs"] = kwargs
            return _chat_data()

        monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", mock_create_chat_completion)
        client = TestClient(server.setup_webserver())

        response = client.post(
            "/ng-rollout/task0-r1/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 200
        assert seen["kwargs"]["model"] == "policy-model"
        assert seen["headers"]["proxy_x_session_id"] == "task0-r1"

    def test_uncorrelated_call_sends_no_session_id(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server(default_headers={"x-team": "gym"})
        seen: dict = {}

        async def mock_create_response(self, **kwargs):
            seen["headers"] = self.default_headers
            return _response_data()

        monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_response", mock_create_response)
        client = TestClient(server.setup_webserver())

        response = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hi"}]})

        assert response.status_code == 200
        assert "proxy_x_session_id" not in seen["headers"]
        assert seen["headers"]["x-team"] == "gym"

    def test_forward_session_id_disabled(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server(forward_session_id=False)
        seen: dict = {}

        async def mock_create_response(self, **kwargs):
            seen["headers"] = self.default_headers
            return _response_data()

        monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_response", mock_create_response)
        client = TestClient(server.setup_webserver())

        response = client.post(
            "/ng-rollout/task0-r1/v1/responses",
            json={"input": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 200
        assert "proxy_x_session_id" not in seen["headers"]

    def test_extra_body_is_merged(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server(extra_body={"max_output_tokens": 16})
        seen: dict = {}

        async def mock_create_response(self, **kwargs):
            seen.update(kwargs)
            return _response_data()

        monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_response", mock_create_response)
        client = TestClient(server.setup_webserver())

        response = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hi"}]})

        assert response.status_code == 200
        assert seen["max_output_tokens"] == 16


class TestProxyLifecycle:
    def _launch_config(self, **overrides) -> SwitchyardModelConfig:
        return SwitchyardModelConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="test_switchyard_model",
            switchyard_model="policy-model",
            launch_proxy=True,
            routing_profiles="/tmp/routes.yaml",
            proxy_port=4123,
            **overrides,
        )

    def _launch_server(self, monkeypatch: MonkeyPatch) -> SwitchyardModel:
        """Build a launch-mode server without actually spawning a proxy."""
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: MagicMock(spec=_REAL_POPEN))
        monkeypatch.setattr(SwitchyardModel, "wait_for_proxy", lambda self, root_url, proc: None)
        return SwitchyardModel(
            config=self._launch_config(),
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )

    def test_start_proxy_builds_command_and_waits(self, monkeypatch: MonkeyPatch) -> None:
        commands: list = []
        process = MagicMock(spec=_REAL_POPEN)
        process.poll.return_value = None

        def mock_popen(command, *args, **kwargs):
            commands.append(command)
            return process

        monkeypatch.setattr(subprocess, "Popen", mock_popen)
        monkeypatch.setattr(SwitchyardModel, "wait_for_proxy", lambda self, root_url, proc: None)

        server = SwitchyardModel(
            config=self._launch_config(),
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )

        assert server._client.base_url == "http://127.0.0.1:4123/v1"
        assert commands[0] == [
            "switchyard",
            "--routing-profiles",
            "/tmp/routes.yaml",
            "--",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "4123",
        ]

    def test_wait_for_proxy_raises_when_process_dies(self, monkeypatch: MonkeyPatch) -> None:
        server = self._launch_server(monkeypatch)
        # Drop the construction-time patches so the real wait_for_proxy runs below.
        monkeypatch.undo()

        dead = MagicMock(spec=_REAL_POPEN)
        dead.poll.return_value = 1
        dead.returncode = 1

        with pytest.raises(RuntimeError, match="exited during startup"):
            server.wait_for_proxy("http://127.0.0.1:4123", dead)

    def test_stop_proxy_terminates_then_kills(self, monkeypatch: MonkeyPatch) -> None:
        server = self._launch_server(monkeypatch)

        process = MagicMock(spec=_REAL_POPEN)
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired(cmd="switchyard", timeout=30)
        server._proxy_process = process

        server.stop_proxy()

        process.terminate.assert_called_once()
        process.kill.assert_called_once()

    def test_wait_for_proxy_returns_once_healthy(self, monkeypatch: MonkeyPatch) -> None:
        server = self._launch_server(monkeypatch)
        monkeypatch.undo()

        alive = MagicMock(spec=_REAL_POPEN)
        alive.poll.return_value = None

        attempts = {"count": 0}

        @contextmanager
        def mock_urlopen(url, timeout=None):
            attempts["count"] += 1
            # First probe refused, as it is while the proxy is still binding.
            if attempts["count"] == 1:
                raise urllib.error.URLError("connection refused")
            yield MagicMock(status=200)

        monkeypatch.setattr(app_module.urllib.request, "urlopen", mock_urlopen)
        monkeypatch.setattr(app_module.time, "sleep", lambda seconds: None)

        server.wait_for_proxy("http://127.0.0.1:4123", alive)

        assert attempts["count"] == 2

    def test_wait_for_proxy_times_out(self, monkeypatch: MonkeyPatch) -> None:
        server = self._launch_server(monkeypatch)
        monkeypatch.undo()
        server.config.proxy_startup_timeout_s = 0.0
        server._proxy_process = None

        alive = MagicMock(spec=_REAL_POPEN)
        alive.poll.return_value = None

        with pytest.raises(TimeoutError, match="did not become healthy"):
            server.wait_for_proxy("http://127.0.0.1:4123", alive)

    def test_stop_proxy_is_noop_when_already_exited(self, monkeypatch: MonkeyPatch) -> None:
        server = self._launch_server(monkeypatch)

        process = MagicMock(spec=_REAL_POPEN)
        process.poll.return_value = 0
        server._proxy_process = process

        server.stop_proxy()

        process.terminate.assert_not_called()
