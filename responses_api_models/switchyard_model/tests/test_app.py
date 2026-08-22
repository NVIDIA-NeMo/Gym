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
import json
import socket
import sys
import threading
import types
import urllib.request
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    fake's contract is just to record what it was asked to host and whether it was closed. Every
    instance registers on the class-level list, which _install_fake_switchyard resets, so a test
    can reach a proxy that stop_proxy has already unlinked from the model server.
    """

    instances: list = []

    def __init__(self, config: str, *, port: int = 0) -> None:
        self.config = config
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.close_calls = 0
        type(self).instances.append(self)

    def close(self, *, timeout_secs: float = 2.0) -> None:
        self.close_calls += 1


def _install_fake_switchyard(monkeypatch: MonkeyPatch, server_cls: type) -> None:
    """Publish a stand-in switchyard_rust.server module without importing the real package.

    app.py imports the native server lazily inside start_proxy, so seeding both sys.modules
    entries is enough -- the import system returns them without touching any installed wheel.
    """
    if issubclass(server_cls, _FakeNativeServer):
        server_cls.instances = []
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
        assert seen["headers"]["x-switchyard-session-id"] == "task0-r1-a0"
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
        assert seen["headers"]["x-switchyard-session-id"] == "task0-r1"

    def test_extra_session_id_headers_all_carry_the_id(self, monkeypatch: MonkeyPatch) -> None:
        """Attach-mode proxies can key other subsystems on other names; every configured name is sent."""
        server = self._setup_server(session_id_headers=["x-switchyard-session-id", "proxy_x_session_id"])
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
        assert seen["headers"]["x-switchyard-session-id"] == "task0-r1"
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
        assert "x-switchyard-session-id" not in seen["headers"]
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
        assert "x-switchyard-session-id" not in seen["headers"]

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

    def test_building_the_app_does_not_host_a_proxy(self, monkeypatch: MonkeyPatch) -> None:
        """The proxy belongs to serving: neither construction nor app assembly may host one."""

        class Explode:
            def __init__(self, *args, **kwargs):
                raise AssertionError("only app startup may host a proxy")

        _install_fake_switchyard(monkeypatch, Explode)

        server = self._build()
        server.setup_webserver()

    def test_missing_dependency_explains_how_to_fix(self, monkeypatch: MonkeyPatch) -> None:
        server = self._build()
        # None in sys.modules makes the import fail the way an uninstalled package does.
        monkeypatch.setitem(sys.modules, "switchyard_rust", None)
        monkeypatch.setitem(sys.modules, "switchyard_rust.server", None)

        with pytest.raises(RuntimeError, match="nemo-switchyard"):
            server.start_proxy()

    def test_serving_hosts_the_deployment(self, monkeypatch: MonkeyPatch) -> None:
        _install_fake_switchyard(monkeypatch, _FakeNativeServer)
        server = self._build()
        app = server.setup_webserver()

        # Entering the TestClient context runs the app's startup, which hosts the proxy.
        with TestClient(app):
            hosted = server._proxy_server
            # The deployment and the configured port reach the native server; the client is
            # built on the address the server reports, not one assembled independently.
            assert hosted.config == "/tmp/routes.toml"
            assert hosted.port == 4123
            assert server._client.base_url == "http://127.0.0.1:4123/v1"
            assert hosted.close_calls == 0

        assert hosted.close_calls == 1

    def test_stop_proxy_closes_only_once(self, monkeypatch: MonkeyPatch) -> None:
        """Shutdown converges from several paths (lifespan, explicit); close must not double-fire."""
        _install_fake_switchyard(monkeypatch, _FakeNativeServer)
        server = self._build()
        server.start_proxy()
        (hosted,) = _FakeNativeServer.instances

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

    def test_failed_startup_still_closes_the_proxy(self, monkeypatch: MonkeyPatch) -> None:
        """The proxy is up before the app finishes starting, so a failed startup must still close it."""
        server = self._launch_server(monkeypatch)

        app = FastAPI()

        @asynccontextmanager
        async def failing_lifespan(_app):
            raise RuntimeError("startup failed")
            yield  # pragma: no cover - unreachable; present so this is a generator

        app.router.lifespan_context = failing_lifespan
        server.setup_proxy_lifespan(app)

        with pytest.raises(RuntimeError, match="startup failed"):
            with TestClient(app):
                pass  # pragma: no cover - startup raises before the body runs

        (hosted,) = _FakeNativeServer.instances
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


class _StubUpstream(BaseHTTPRequestHandler):
    """A loopback OpenAI-chat upstream that records what the proxy sends it.

    Serving chat completions only is deliberate: a Responses call through the chain then proves
    Switchyard's responses<->chat translation, which is the code path the 0.2.0 pin exists for.
    """

    requests: list  # (path, headers, body) tuples, appended per call; reset by the fixture

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, dict(self.headers), body))
        payload = json.dumps(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 1755400000,
                "model": body.get("model", "upstream/model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from upstream!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        """Keep test output clean; the recorded requests are the observable."""


@pytest.mark.skipif(
    importlib.util.find_spec("switchyard_rust") is None,
    reason="nemo-switchyard is not installed",
)
class TestFullChainIntegration:
    """Drive Gym's own app through a real hosted proxy to a local stub upstream -- no mocks.

    Chain under test: Gym FastAPI app -> NeMoGymAsyncOpenAI -> native Switchyard server
    (routing + wire-format translation in Rust) -> stub upstream. This is the integration the
    exact 0.2.0 pin protects: the TOML schema, the /v1 endpoints, the session header, and the
    chat->responses translation emitting the usage detail objects NeMoGymResponse requires.
    """

    @pytest.fixture(autouse=True)
    def _fresh_upstream_requests(self):
        _StubUpstream.requests = []

    # Class-scoped on purpose: Gym's aiohttp client is a process-wide singleton bound to the
    # first event loop that uses it, so all of these tests must share one TestClient loop.
    @pytest.fixture(scope="class")
    def upstream(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubUpstream)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield httpd.server_address[1]
        finally:
            httpd.shutdown()
            thread.join(timeout=5)

    @pytest.fixture(scope="class")
    def gym_app(self, upstream, tmp_path_factory):
        monkeypatch = MonkeyPatch()
        # A present config dict keeps Gym's client stack from booting Hydra inside the request.
        monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", "test_switchyard_model: {}\n")
        monkeypatch.setenv("SWITCHYARD_TEST_API_KEY", "stub-upstream-key")  # pragma: allowlist secret
        deployment = tmp_path_factory.mktemp("deployment") / "routes.toml"
        deployment.write_text(
            f"""
schema_version = 1

[llm_clients.upstream]
format = "openai_chat"
base_url = "http://127.0.0.1:{upstream}/v1"
api_key_env = "SWITCHYARD_TEST_API_KEY"

[targets.policy]
id = "upstream/model"
llm_client = "upstream"

[routes.policy-model]
id = "policy-model"
type = "passthrough"
target = "policy"
"""
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            proxy_port = probe.getsockname()[1]

        server = SwitchyardModel(
            config=SwitchyardModelConfig(
                host="0.0.0.0",
                port=8081,
                entrypoint="",
                name="test_switchyard_model",
                switchyard_model="policy-model",
                deployment=str(deployment),
                proxy_port=proxy_port,
            ),
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )
        app = server.setup_webserver()
        try:
            with TestClient(app) as client:
                yield client, proxy_port
        finally:
            monkeypatch.undo()

    def test_responses_round_trip_translates_usage(self, gym_app, capfd) -> None:
        client, _ = gym_app

        response = client.post(
            "/ng-rollout/task0-r1-a0/v1/responses",
            json={"input": [{"role": "user", "content": "hi"}], "model": "caller-supplied"},
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["output"][0]["content"][0]["text"] == "Hello from upstream!"
        # The routed target, not the route name, is what served the call.
        assert data["model"] == "upstream/model"
        # The usage detail objects are the reason the pin starts at 0.2.0: published 0.1.0
        # omitted them and every response failed NeMoGymResponse validation.
        assert data["usage"]["input_tokens"] == 7
        assert data["usage"]["output_tokens"] == 3
        assert "cached_tokens" in data["usage"]["input_tokens_details"]
        assert "reasoning_tokens" in data["usage"]["output_tokens_details"]

        # The rollout id reached Switchyard's routing metadata: its request log (Rust tracing on
        # stderr, captured by capfd) records the session id it parsed from Gym's header.
        assert 'session_id="task0-r1-a0"' in capfd.readouterr().err

        # The upstream saw one translated chat call for the target model, not the route name,
        # carrying the deployment's credential.
        (path, headers, body) = _StubUpstream.requests[0]
        headers_lower = {name.lower(): value for name, value in headers.items()}
        assert path == "/v1/chat/completions"
        assert body["model"] == "upstream/model"
        assert headers_lower["authorization"] == "Bearer stub-upstream-key"  # pragma: allowlist secret
        # Switchyard 0.2.0 parses x-switchyard-session-id into routing metadata but does not
        # strip it before forwarding, so the upstream also sees the opaque rollout id. Asserted
        # as-is so a change in either direction on upgrade is caught, not discovered in the field.
        assert headers_lower["x-switchyard-session-id"] == "task0-r1-a0"

    def test_chat_completions_round_trip(self, gym_app) -> None:
        client, _ = gym_app

        response = client.post(
            "/ng-rollout/task0-r1/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello from upstream!"
        assert data["model"] == "upstream/model"
        assert data["usage"]["prompt_tokens"] == 7

    def test_proxy_counts_the_traffic(self, gym_app) -> None:
        """/v1/stats is the surface routing-aware evals read; a routed call must show up there."""
        client, proxy_port = gym_app

        response = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 200, response.text

        with urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/v1/stats", timeout=5) as stats_response:
            stats = json.loads(stats_response.read())
        assert json.dumps(stats).count("upstream/model"), stats
