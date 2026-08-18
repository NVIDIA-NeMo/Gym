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
import importlib.util
import socket
import sys
import types
import urllib.request
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
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


class _FakeNativeServer:
    """Stands in for switchyard_rust.server.Server: bound once constructed, closed explicitly.

    The real constructor loads the TOML deployment, binds loopback, and returns serving -- so the
    fake's contract is just to record what it was asked to host and whether it was closed.
    """

    def __init__(self, config: str, *, port: int = 0) -> None:
        self.config = config
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.close_calls = 0

    def close(self, *, timeout_secs: float = 2.0) -> None:
        self.close_calls += 1


def _install_fake_switchyard(monkeypatch: MonkeyPatch, server_cls: type) -> None:
    """Publish a stand-in switchyard_rust.server module without importing the real package.

    app.py imports the native server lazily inside start_proxy, so seeding both sys.modules
    entries is enough -- the import system returns them without touching any installed wheel.
    """
    module = types.ModuleType("switchyard_rust.server")
    module.Server = server_cls
    package = types.ModuleType("switchyard_rust")
    package.server = module
    monkeypatch.setitem(sys.modules, "switchyard_rust", package)
    monkeypatch.setitem(sys.modules, "switchyard_rust.server", module)


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
    def test_requires_a_deployment_or_a_base_url(self) -> None:
        with pytest.raises(ValueError, match="deployment"):
            SwitchyardModelConfig(host="0.0.0.0", port=8081, entrypoint="", name="sy", switchyard_model="policy-model")

    def test_deployment_alone_means_gym_hosts(self) -> None:
        config = SwitchyardModelConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="sy",
            switchyard_model="policy-model",
            deployment="/tmp/routes.toml",
        )
        assert config.launches_proxy is True

    def test_both_set_attaches_and_warns(self, caplog) -> None:
        config = SwitchyardModelConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="sy",
            switchyard_model="policy-model",
            deployment="/tmp/routes.toml",
            switchyard_base_url="http://127.0.0.1:4000/v1",
        )

        assert config.launches_proxy is False
        assert "both switchyard_base_url and deployment are set" in caplog.text

    def test_base_url_alone_means_attach(self) -> None:
        config = SwitchyardModelConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="sy",
            switchyard_model="policy-model",
            switchyard_base_url="http://127.0.0.1:4000/v1",
        )
        assert config.launches_proxy is False


class TestRolloutSessionMiddleware:
    async def _rollout_id_for(self, path: str) -> object:
        seen: dict = {}

        async def inner(scope, receive, send):
            seen["rollout_id"] = app_module._ROLLOUT_ID.get()

        await _RolloutSessionMiddleware(inner)({"type": "http", "path": path}, None, None)
        return seen["rollout_id"]

    async def test_well_formed_rollout_id_is_published(self) -> None:
        assert await self._rollout_id_for("/ng-rollout/task0-r1-a0/v1/responses") == "task0-r1-a0"

    async def test_rollout_id_outside_the_contract_charset_is_ignored(self) -> None:
        """The id becomes an upstream header value, so anything off-contract is dropped, not sent."""
        assert await self._rollout_id_for("/ng-rollout/bad\r\nx-injected: 1/v1/responses") is None

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
            deployment="/tmp/routes.toml",
            proxy_port=4123,
            **overrides,
        )

    def _build(self) -> SwitchyardModel:
        return SwitchyardModel(
            config=self._launch_config(),
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )

    def test_constructing_the_server_does_not_host_a_proxy(self, monkeypatch: MonkeyPatch) -> None:
        """The proxy belongs to serving, not to config validation."""

        class Explode:
            def __init__(self, *args, **kwargs):
                raise AssertionError("constructing SwitchyardModel must not host a proxy")

        _install_fake_switchyard(monkeypatch, Explode)

        self._build()

    def test_missing_dependency_explains_how_to_fix(self, monkeypatch: MonkeyPatch) -> None:
        server = self._build()
        # None in sys.modules makes the import fail the way an uninstalled package does.
        monkeypatch.setitem(sys.modules, "switchyard_rust", None)
        monkeypatch.setitem(sys.modules, "switchyard_rust.server", None)

        with pytest.raises(RuntimeError, match="nemo-switchyard"):
            server.setup_webserver()

    def test_start_proxy_hosts_the_deployment(self, monkeypatch: MonkeyPatch) -> None:
        _install_fake_switchyard(monkeypatch, _FakeNativeServer)

        server = self._build()
        server.setup_webserver()

        # The deployment and the configured port reach the native server; the client is built on
        # the address the server reports, not one assembled independently.
        assert server._proxy_server.config == "/tmp/routes.toml"
        assert server._proxy_server.port == 4123
        assert server._client.base_url == "http://127.0.0.1:4123/v1"

    def test_stop_proxy_closes_the_server(self, monkeypatch: MonkeyPatch) -> None:
        _install_fake_switchyard(monkeypatch, _FakeNativeServer)
        server = self._build()
        server.setup_webserver()
        hosted = server._proxy_server

        server.stop_proxy()

        assert hosted.close_calls == 1

    def test_stop_proxy_closes_only_once(self, monkeypatch: MonkeyPatch) -> None:
        """Shutdown converges from several paths (lifespan, explicit); close must not double-fire."""
        _install_fake_switchyard(monkeypatch, _FakeNativeServer)
        server = self._build()
        server.setup_webserver()
        hosted = server._proxy_server

        server.stop_proxy()
        server.stop_proxy()

        assert hosted.close_calls == 1

    def test_stop_proxy_is_noop_when_nothing_was_hosted(self) -> None:
        server = self._build()

        server.stop_proxy()  # attach mode and pre-startup both reach here with no proxy


class TestProxyShutdown:
    """The proxy runs in-process, so shutdown is about promptness, not survival.

    An in-process server cannot outlive its owner the way a subprocess could -- what these tests
    pin down is that the graceful paths close it explicitly, which stops the listener without
    waiting for interpreter teardown and flushes Switchyard's telemetry.
    """

    def _launch_server(self, monkeypatch: MonkeyPatch) -> SwitchyardModel:
        _install_fake_switchyard(monkeypatch, _FakeNativeServer)
        return SwitchyardModel(
            config=SwitchyardModelConfig(
                host="0.0.0.0",
                port=8081,
                entrypoint="",
                name="test_switchyard_model",
                switchyard_model="policy-model",
                deployment="/tmp/routes.toml",
                proxy_port=4123,
            ),
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )

    def test_graceful_shutdown_closes_the_proxy(self, monkeypatch: MonkeyPatch) -> None:
        server = self._launch_server(monkeypatch)

        app = server.setup_webserver()
        hosted = server._proxy_server

        # Entering and leaving the TestClient context runs the app's startup/shutdown events.
        with TestClient(app):
            assert hosted.close_calls == 0

        assert hosted.close_calls == 1

    def test_failed_startup_still_closes_the_proxy(self, monkeypatch: MonkeyPatch) -> None:
        """The proxy is up before the app finishes starting, so a failed startup must still close it."""
        server = self._launch_server(monkeypatch)
        server._build_client(server.start_proxy())
        hosted = server._proxy_server

        app = FastAPI()

        @asynccontextmanager
        async def failing_lifespan(_app):
            raise RuntimeError("startup failed")
            yield  # pragma: no cover - unreachable; present so this is a generator

        app.router.lifespan_context = failing_lifespan
        server.setup_proxy_shutdown(app)

        with pytest.raises(RuntimeError, match="startup failed"):
            with TestClient(app):
                pass  # pragma: no cover - startup raises before the body runs

        assert hosted.close_calls == 1


@pytest.mark.skipif(
    importlib.util.find_spec("switchyard_rust") is None,
    reason="nemo-switchyard is not installed",
)
class TestNativeServerIntegration:
    """Host a real native server from a real TOML deployment -- no mocks.

    The upstream target points at a closed port, which is fine: these tests exercise hosting,
    health, and shutdown, none of which call upstream.
    """

    _DEPLOYMENT = """
schema_version = 1

[llm_clients.upstream]
format = "openai_chat"
base_url = "http://127.0.0.1:9/v1"
api_key_env = "SWITCHYARD_TEST_API_KEY"

[targets.policy]
id = "upstream/model"
llm_client = "upstream"

[routes.policy-model]
id = "policy-model"
type = "passthrough"
target = "policy"
"""

    def test_hosts_serves_health_and_stops(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("SWITCHYARD_TEST_API_KEY", "dummy")  # pragma: allowlist secret
        deployment = tmp_path / "routes.toml"
        deployment.write_text(self._DEPLOYMENT)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        server = SwitchyardModel(
            config=SwitchyardModelConfig(
                host="0.0.0.0",
                port=8081,
                entrypoint="",
                name="test_switchyard_model",
                switchyard_model="policy-model",
                deployment=str(deployment),
                proxy_port=port,
            ),
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )

        base_url = server.start_proxy()
        try:
            assert base_url == f"http://127.0.0.1:{port}/v1"
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                assert response.status == 200
        finally:
            server.stop_proxy()

    def test_invalid_deployment_fails_at_startup(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        """A bad routing config is a startup error with the validator's message, not a timeout."""
        deployment = tmp_path / "routes.toml"
        deployment.write_text("schema_version = 1\n[routes.broken]\n")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        server = SwitchyardModel(
            config=SwitchyardModelConfig(
                host="0.0.0.0",
                port=8081,
                entrypoint="",
                name="test_switchyard_model",
                switchyard_model="policy-model",
                deployment=str(deployment),
                proxy_port=port,
            ),
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )

        with pytest.raises(RuntimeError):
            server.start_proxy()
