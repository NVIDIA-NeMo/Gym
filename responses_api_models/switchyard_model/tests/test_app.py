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
import hashlib
import importlib.util
import json
import socket
import sys
import threading
import types
import urllib.request
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import aiohttp
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import nemo_gym.server_utils as server_utils
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


def _install_fake_stats_endpoint(
    monkeypatch: MonkeyPatch, payload: dict | None = None, error: Exception | None = None
) -> list:
    """Stand in for nemo_gym.server_utils.request on the stats path; return the recorded calls.

    Unit tests must not touch Gym's real aiohttp singleton: it parses the process's CLI args the
    first time it is built, which under pytest is a SystemExit, and it binds to whichever event
    loop builds it. The fake keeps the calls observable -- (method, url, kwargs) tuples -- so a
    test can assert what would have gone on the wire.
    """
    calls: list = []
    payload = payload if payload is not None else {"total_requests": 1, "models": {}}

    class _FakeStatsResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        async def json(self):
            return payload

    async def fake_request(method: str, url: str, **kwargs):
        calls.append((method, url, kwargs))
        if error is not None:
            raise error
        return _FakeStatsResponse()

    monkeypatch.setattr(app_module, "request", fake_request)
    return calls


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

    def test_hosting_rejects_multiple_workers(self) -> None:
        """Each worker process would host its own proxy -- split affinity, split stats."""
        with pytest.raises(ValueError, match="num_workers"):
            SwitchyardModelConfig(
                host="0.0.0.0",
                port=8081,
                entrypoint="",
                name="sy",
                switchyard_model="policy-model",
                deployment="/tmp/routes.toml",
                num_workers=2,
            )

    def test_attaching_allows_multiple_workers(self) -> None:
        """All workers share the one external proxy, so parallelism is coherent here."""
        config = SwitchyardModelConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="sy",
            switchyard_model="policy-model",
            switchyard_base_url="http://127.0.0.1:4000/v1",
            num_workers=2,
        )
        assert config.launches_proxy is False

    def test_max_concurrent_requests_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            SwitchyardModelConfig(
                host="0.0.0.0",
                port=8081,
                entrypoint="",
                name="sy",
                switchyard_model="policy-model",
                switchyard_base_url="http://127.0.0.1:4000/v1",
                max_concurrent_requests=0,
            )


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


class TestConditionRecord:
    """The routing-condition record is what makes routed runs comparable after the fact."""

    def _launch_server(self, monkeypatch: MonkeyPatch, tmp_path, **overrides) -> SwitchyardModel:
        _install_fake_switchyard(monkeypatch, _FakeNativeServer)
        deployment = tmp_path / "routes.toml"
        deployment.write_text(
            "schema_version = 1\n"
            "[llm_clients.up]\n"
            'api_key_env = "K"\n'
            'api_key = "sk-inline-secret"\n'  # pragma: allowlist secret
            "[llm_clients.up.extra_headers]\n"
            'Authorization = "Bearer header-secret"\n'
            'X-Team = "routing"\n'
            "[targets.t]\n"
            'mirrors = [{ extra_headers = { Authorization = "Bearer list-secret" } }]\n'
        )
        config = {
            "host": "0.0.0.0",
            "port": 8081,
            "entrypoint": "",
            "name": "test_switchyard_model",
            "switchyard_model": "policy-model",
            "deployment": str(deployment),
            "proxy_port": 4123,
            "condition_dir": str(tmp_path / "condition"),
            **overrides,
        }
        return SwitchyardModel(
            config=SwitchyardModelConfig(**config),
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )

    def test_hosted_manifest_records_the_condition(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        _install_fake_stats_endpoint(monkeypatch)
        server = self._launch_server(monkeypatch, tmp_path)

        with TestClient(server.setup_webserver()):
            manifest = json.loads((tmp_path / "condition" / "switchyard-condition.json").read_text())

        assert manifest["route"] == "policy-model"
        assert manifest["mode"] == "hosted"
        assert manifest["proxy_root_url"] == "http://127.0.0.1:4123"
        deployment_bytes = (tmp_path / "routes.toml").read_bytes()
        assert manifest["deployment_sha256"] == hashlib.sha256(deployment_bytes).hexdigest()
        # The archive keeps env-var references but never an inline credential -- neither an
        # api_key literal nor any extra_headers value, which providers receive verbatim.
        assert 'api_key_env = "K"' in manifest["deployment_toml"]
        assert "sk-inline-secret" not in manifest["deployment_toml"]
        assert "Bearer header-secret" not in manifest["deployment_toml"]
        assert "routing" not in manifest["deployment_toml"]
        assert "Bearer list-secret" not in manifest["deployment_toml"]

    def test_unparseable_deployment_is_hashed_but_not_archived(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        """A file that does not parse cannot be searched for secrets, so only its hash is kept."""
        _install_fake_stats_endpoint(monkeypatch)
        server = self._launch_server(monkeypatch, tmp_path)
        deployment = tmp_path / "routes.toml"
        deployment.write_text("this is [not TOML")

        with TestClient(server.setup_webserver()):
            manifest = json.loads((tmp_path / "condition" / "switchyard-condition.json").read_text())

        assert manifest["deployment_sha256"] == hashlib.sha256(deployment.read_bytes()).hexdigest()
        assert manifest["deployment_toml"] is None

    def test_attached_manifest_records_the_proxy(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        _install_fake_stats_endpoint(monkeypatch)
        server = self._launch_server(
            monkeypatch,
            tmp_path,
            deployment=None,
            switchyard_base_url="http://127.0.0.1:4000/v1",
        )

        with TestClient(server.setup_webserver()):
            manifest = json.loads((tmp_path / "condition" / "switchyard-condition.json").read_text())

        assert manifest["mode"] == "attached"
        assert manifest["proxy_root_url"] == "http://127.0.0.1:4000"
        assert manifest["deployment_sha256"] is None
        assert manifest["deployment_toml"] is None
        # The local wheel never served a request in attach mode, so its version says nothing
        # about the proxy; identity is the caller's to supply.
        assert manifest["nemo_switchyard_version"] is None
        assert manifest["proxy_provenance"] is None

    def test_attached_manifest_carries_caller_provenance(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        _install_fake_stats_endpoint(monkeypatch)
        provenance = {"switchyard_commit": "1fc9ab8", "deployment_sha256": "abc123"}
        server = self._launch_server(
            monkeypatch,
            tmp_path,
            deployment=None,
            switchyard_base_url="http://127.0.0.1:4000/v1",
            proxy_provenance=provenance,
        )

        with TestClient(server.setup_webserver()):
            manifest = json.loads((tmp_path / "condition" / "switchyard-condition.json").read_text())

        assert manifest["proxy_provenance"] == provenance

    def test_no_condition_dir_writes_nothing(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        _install_fake_stats_endpoint(monkeypatch)
        server = self._launch_server(monkeypatch, tmp_path, condition_dir=None)

        with TestClient(server.setup_webserver()):
            pass

        assert not (tmp_path / "condition").exists()

    def test_stats_snapshot_authenticates_as_the_configured_caller(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        """A proxy that requires auth serves model calls and would 401 an anonymous stats read."""
        calls = _install_fake_stats_endpoint(monkeypatch, payload={"total_requests": 7})
        server = self._launch_server(
            monkeypatch,
            tmp_path,
            switchyard_api_key="stats-key",  # pragma: allowlist secret
            default_headers={"x-org": "nv"},
        )

        with TestClient(server.setup_webserver()):
            pass

        (call,) = calls
        method, url, kwargs = call
        assert (method, url) == ("GET", "http://127.0.0.1:4123/v1/stats")
        assert kwargs["headers"] == {"x-org": "nv", "Authorization": "Bearer stats-key"}
        snapshot = json.loads((tmp_path / "condition" / "switchyard-stats.json").read_text())
        assert snapshot["mode"] == "hosted"
        assert snapshot["scope"] == "this run (proxy hosted for exactly this run)"
        assert snapshot["stats"] == {"total_requests": 7}

    def test_attached_stats_snapshot_is_labeled_an_aggregate(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        """A shared proxy counts every run it has served; the file must say so, not imply run scope."""
        _install_fake_stats_endpoint(monkeypatch, payload={"total_requests": 900})
        server = self._launch_server(
            monkeypatch,
            tmp_path,
            deployment=None,
            switchyard_base_url="http://127.0.0.1:4000/v1",
        )

        with TestClient(server.setup_webserver()):
            pass

        snapshot = json.loads((tmp_path / "condition" / "switchyard-stats.json").read_text())
        assert snapshot["mode"] == "attached"
        assert snapshot["scope"] == "proxy-lifetime aggregate"
        assert snapshot["stats"] == {"total_requests": 900}

    def test_unreachable_stats_endpoint_does_not_fail_shutdown(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        """A proxy that answers nothing must cost a warning at shutdown, not a raise."""
        _install_fake_stats_endpoint(monkeypatch, error=aiohttp.ClientError("connection refused"))
        server = self._launch_server(monkeypatch, tmp_path)

        with TestClient(server.setup_webserver()):
            pass

        assert not (tmp_path / "condition" / "switchyard-stats.json").exists()

    def test_unwritable_condition_dir_does_not_fail_startup(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        """Provenance is worth a warning, not an outage: a bad dir must not stop serving."""
        _install_fake_stats_endpoint(monkeypatch)
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("a file where the condition dir should be")
        server = self._launch_server(monkeypatch, tmp_path, condition_dir=str(blocker))

        # Startup raising would fail this test; the manifest is the only casualty.
        with TestClient(server.setup_webserver()):
            pass

        assert blocker.read_text() == "a file where the condition dir should be"

    def test_missing_distribution_yields_null_version(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        """A source checkout without dist metadata still gets a manifest, with version null."""
        _install_fake_stats_endpoint(monkeypatch)

        def not_installed(name):
            raise app_module.importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(app_module.importlib.metadata, "version", not_installed)
        server = self._launch_server(monkeypatch, tmp_path)

        with TestClient(server.setup_webserver()):
            manifest = json.loads((tmp_path / "condition" / "switchyard-condition.json").read_text())

        assert manifest["nemo_switchyard_version"] is None

    def test_stats_snapshot_needs_a_proxy_to_ask(self, monkeypatch: MonkeyPatch, tmp_path) -> None:
        """Before hosting starts there is no proxy URL; the snapshot must no-op, not guess."""
        _install_fake_stats_endpoint(monkeypatch)
        server = self._launch_server(monkeypatch, tmp_path)

        assert server.proxy_root_url() is None
        asyncio.run(server.snapshot_proxy_stats())

        assert not (tmp_path / "condition").exists()


class TestRolloutCorrelationCheck:
    """forward_session_id only works when requests arrive with the /ng-rollout prefix.

    Agents add that prefix only when observability or token capture is enabled, so a run without
    either silently loses the documented correlation. Startup makes that explicit: an error when
    the user asked for forwarding, a warning when it is merely the default.
    """

    def _server(self, global_config: object, **overrides) -> SwitchyardModel:
        config = SwitchyardModelConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="test_switchyard_model",
            switchyard_model="policy-model",
            switchyard_base_url="http://127.0.0.1:4000/v1",
            **overrides,
        )
        return SwitchyardModel(
            config=config, server_client=MagicMock(spec=ServerClient, global_config_dict=global_config)
        )

    def test_explicit_forwarding_without_capture_fails_startup(self) -> None:
        server = self._server({}, forward_session_id=True)

        with pytest.raises(RuntimeError, match="observability_enabled"):
            with TestClient(server.setup_webserver()):
                pass  # pragma: no cover - startup raises before the body runs

    def test_default_forwarding_without_capture_warns(self, caplog) -> None:
        server = self._server({})

        with TestClient(server.setup_webserver()):
            pass

        assert "no session id will reach Switchyard" in caplog.text

    def test_observability_enables_correlation_silently(self, caplog) -> None:
        server = self._server({"observability_enabled": True}, forward_session_id=True)

        server.check_rollout_correlation()

        assert "no session id will reach Switchyard" not in caplog.text

    def test_token_capture_enables_correlation_silently(self, caplog) -> None:
        server = self._server({"token_id_capture": {"enabled": True}}, forward_session_id=True)

        server.check_rollout_correlation()

        assert "no session id will reach Switchyard" not in caplog.text

    def test_forwarding_disabled_needs_no_capture(self, caplog) -> None:
        server = self._server({}, forward_session_id=False)

        server.check_rollout_correlation()

        assert "no session id will reach Switchyard" not in caplog.text

    def test_unknown_global_config_is_left_alone(self, caplog) -> None:
        """Without a config dict to consult there is nothing to validate against."""
        server = self._server(None, forward_session_id=True)

        server.check_rollout_correlation()

        assert "no session id will reach Switchyard" not in caplog.text

    def test_shipped_config_does_not_restate_the_forwarding_default(self) -> None:
        """The yaml restating forward_session_id would make every run look like an explicit
        request for it, turning the intended startup warning into a refusal for any eval that
        runs without observability. Found live; pinned here."""
        import yaml

        config_path = Path(__file__).parents[1] / "configs" / "switchyard_model.yaml"
        block = yaml.safe_load(config_path.read_text())["policy_model"]["responses_api_models"]["switchyard_model"]

        assert "forward_session_id" not in block


@pytest.mark.skipif(
    importlib.util.find_spec("switchyard_rust") is None,
    reason="nemo-switchyard is not installed",
)
class TestConditionRecordIntegration:
    """Real wheel, real traffic: the record survives the proxy's shutdown.

    Model calls go to the proxy directly over urllib, but the stats snapshot rides Gym's real
    shared aiohttp path -- which binds the process-singleton client to this test's event loop, so
    the singleton is torn down afterwards for the classes that bind their own.
    """

    def test_manifest_and_stats_written_across_a_run(self, tmp_path, monkeypatch: MonkeyPatch) -> None:
        # A present config dict keeps the aiohttp client stack from booting Hydra mid-request.
        monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", "test_switchyard_model: {}\n")
        monkeypatch.setenv("SWITCHYARD_TEST_API_KEY", "stub-upstream-key")  # pragma: allowlist secret
        _StubUpstream.requests = []
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubUpstream)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        deployment = tmp_path / "routes.toml"
        deployment.write_text(
            f"""
schema_version = 1

[llm_clients.upstream]
format = "openai_chat"
base_url = "http://127.0.0.1:{httpd.server_address[1]}/v1"
api_key_env = "SWITCHYARD_TEST_API_KEY" # pragma: allowlist secret

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
                condition_dir=str(tmp_path / "condition"),
            ),
            server_client=MagicMock(spec=ServerClient, global_config_dict={}),
        )

        try:
            with TestClient(server.setup_webserver()):
                request = urllib.request.Request(
                    f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                    data=json.dumps(
                        {"model": "policy-model", "messages": [{"role": "user", "content": "hi"}]}
                    ).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    assert response.status == 200
        finally:
            httpd.shutdown()
            thread.join(timeout=5)
            # The snapshot built the process-singleton aiohttp client on this test's loop; leaving
            # it behind would break later classes that bind it to theirs.
            try:
                server_utils.global_aiohttp_client_exit()
            except Exception:
                server_utils._GLOBAL_AIOHTTP_CLIENT = None

        manifest = json.loads((tmp_path / "condition" / "switchyard-condition.json").read_text())
        assert manifest["route"] == "policy-model"
        assert manifest["nemo_switchyard_version"] == "0.2.0"

        snapshot = json.loads((tmp_path / "condition" / "switchyard-stats.json").read_text())
        assert snapshot["mode"] == "hosted"
        assert "upstream/model" in json.dumps(snapshot["stats"])


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
api_key_env = "SWITCHYARD_TEST_API_KEY" # pragma: allowlist secret

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
api_key_env = "SWITCHYARD_TEST_API_KEY" # pragma: allowlist secret

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
