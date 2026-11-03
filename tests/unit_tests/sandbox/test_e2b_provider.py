# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Unit tests for the e2b sandbox provider (SDK faked; no network)."""

import asyncio
import inspect
import json
import logging
import re
import sys
import tomllib
import types
from importlib.metadata import version
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from packaging.requirements import Requirement

from nemo_gym.config.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME
from nemo_gym.package_info import __version__ as nemo_gym_version
from nemo_gym.sandbox import attribution
from nemo_gym.sandbox.api import AsyncSandbox, Sandbox
from nemo_gym.sandbox.providers import e2b as e2b_pkg
from nemo_gym.sandbox.providers.base import (
    ConnectableProvider,
    SandboxHandle,
    SandboxSnapshotNotFoundError,
    SandboxSpec,
    SandboxStatus,
    SupportsSandboxPauseResume,
    SupportsSandboxSnapshot,
)
from nemo_gym.sandbox.providers.e2b import _sdk as e2b_sdk
from nemo_gym.sandbox.providers.e2b import provider as e2b_provider
from nemo_gym.sandbox.providers.e2b.provider import (
    _API_PARAM_KEYS,
    E2BCreateError,
    E2BProvider,
    E2BSandboxUnreachableError,
    E2BSnapshotError,
)
from nemo_gym.sandbox.providers.registry import get_provider_class


pytestmark = pytest.mark.sandbox

_REAL_PROVIDER_REQUIRE_E2B_SDK = e2b_provider._require_e2b_sdk


# --------------------------------------------------------------------------
# Fake e2b SDK
# --------------------------------------------------------------------------


class FakeSandboxNotFound(Exception):
    pass


class FakeTimeout(Exception):
    pass


class FakeRateLimit(Exception):
    pass


class FakeCommandExit(Exception):
    def __init__(self, exit_code: int, stdout: str = "", stderr: str = "") -> None:
        super().__init__(f"exit {exit_code}")
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class FakeCommandResult:
    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class FakeConnectionConfig:
    integrations: list[str] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @classmethod
    def set_integration(cls, integration: str) -> None:
        cls.integrations.append(integration)


class FakeHttpResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self.text = str(payload)
        self._payload = payload

    def json(self) -> object:
        return self._payload


class FakeApiClient:
    """The SDK's REST API client: records each POST and GET and answers with ``response`` and ``get_response``."""

    calls: list[dict] = []
    response = FakeHttpResponse(201, {"sandboxID": "sbx-rest"})
    get_calls: list[str] = []
    get_response: FakeHttpResponse | BaseException = FakeHttpResponse(200, {"state": "running"})

    def __init__(self, config: FakeConnectionConfig) -> None:
        self.config = config

    def get_async_httpx_client(self) -> "FakeApiClient":
        return self

    async def post(self, url: str, json: dict) -> FakeHttpResponse:
        FakeApiClient.calls.append({"url": url, "json": json, "config": self.config.kwargs})
        return FakeApiClient.response

    async def get(self, url: str) -> FakeHttpResponse:
        FakeApiClient.get_calls.append(url)
        if isinstance(FakeApiClient.get_response, BaseException):
            raise FakeApiClient.get_response
        return FakeApiClient.get_response


def _unparsed_404() -> json.JSONDecodeError:
    """What the SDK raises for AgentENV's empty-body 404: it parses the body before it checks the status."""
    try:
        json.loads("")
    except json.JSONDecodeError as error:
        return error
    raise AssertionError("an empty body must not parse")


# The real SDK's sandbox-scoped methods take `request_timeout` only -- they are
# bound to an already-connected sandbox. Mirroring that here (rather than a
# permissive **kwargs) is what catches connection params leaking onto them:
# the SDK raises `TypeError: unexpected keyword argument 'api_key'`.
_SANDBOX_SCOPED_ALLOWED = {"request_timeout"}


def _reject_connection_params(method: str, kwargs: dict) -> None:
    leaked = sorted((set(kwargs) & set(_API_PARAM_KEYS)) - _SANDBOX_SCOPED_ALLOWED)
    if leaked:
        raise TypeError(f"{method}() got an unexpected keyword argument {leaked[0]!r}")


class FakeCommandHandle:
    """Background handle: `wait()` replays a scripted sequence of outcomes.

    Falls back to the sandbox's `exec_behaviour` so a test can script an
    outcome once and have it apply in either exec mode.
    """

    def __init__(self, pid: int, outcomes: list, fallback=None) -> None:
        self.pid = pid
        self._outcomes = outcomes
        self._fallback = fallback
        self.waits = 0
        self.stdout = ""
        self.stderr = ""

    async def wait(self):
        self.waits += 1
        if self._outcomes:
            outcome = self._outcomes.pop(0)
        else:
            outcome = self._fallback or FakeCommandResult(stdout="ok")
        if isinstance(outcome, Exception):
            self.stdout = getattr(outcome, "stdout", "")
            self.stderr = getattr(outcome, "stderr", "")
            raise outcome
        return outcome


class FakeCommands:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self._sandbox = sandbox

    async def run(self, **kwargs):
        _reject_connection_params("Commands.run", kwargs)
        self._sandbox.exec_calls.append(kwargs)
        behaviour = self._sandbox.exec_behaviour
        if kwargs.get("background"):
            return FakeCommandHandle(self._sandbox.pid, self._sandbox.wait_outcomes, behaviour)
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour or FakeCommandResult(stdout="ok")

    async def connect(self, pid, timeout=None, request_timeout=None):
        self._sandbox.connect_calls.append({"pid": pid, "timeout": timeout, "request_timeout": request_timeout})
        if self._sandbox.connect_error is not None:
            raise self._sandbox.connect_error
        return FakeCommandHandle(pid, self._sandbox.wait_outcomes, self._sandbox.exec_behaviour)


class FakeFiles:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self._sandbox = sandbox

    async def write(self, path, data, **kwargs):
        _reject_connection_params("Filesystem.write", kwargs)
        self._sandbox.file_write_calls.append({"path": path, "data": data, **kwargs})
        self._sandbox.files_written[path] = data
        return None

    async def read(self, path, **kwargs):
        _reject_connection_params("Filesystem.read", kwargs)
        if path not in self._sandbox.files_written:
            raise FakeSandboxNotFound(path)
        data = self._sandbox.files_written[path]
        return data if isinstance(data, bytes) else str(data).encode()


class FakeSdkError(Exception):
    """Shaped like the SDK's SandboxException: carries the HTTP status of the response that produced it."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeUnreachable(FakeTimeout):
    """Like the SDK's SandboxUnreachableException, a TimeoutException subclass."""


class FakeSnapshotPaginator:
    def __init__(self, pages: list[list]) -> None:
        self._pages = list(pages)

    @property
    def has_next(self) -> bool:
        return bool(self._pages)

    async def next_items(self):
        return self._pages.pop(0)


class FakeSandbox:
    instances: list["FakeSandbox"] = []
    # snapshot id -> {"names": [...], "sandbox_id": ...}; ids starting "missing" fail create as not found.
    snapshots: dict[str, dict] = {}
    delete_snapshot_calls: list[str] = []
    delete_snapshot_outcomes: list[object] = []

    def __init__(self, sandbox_id: str = "sbx-1", **create_kwargs) -> None:
        self.sandbox_id = sandbox_id
        self.create_kwargs = create_kwargs
        self.exec_calls: list[dict] = []
        self.file_write_calls: list[dict] = []
        self.files_written: dict[str, object] = {}
        self.killed = False
        self.kill_calls: list[dict] = []
        self.kill_outcomes: list[object] = []
        self.running = True
        self.paused = False
        self.pause_calls: list[dict] = []
        self.exec_behaviour = None
        self.pid = 4242
        self.wait_outcomes: list = []
        self.connect_calls: list[dict] = []
        self.connect_error = None
        self.commands = FakeCommands(self)
        self.files = FakeFiles(self)
        self.connection_config = types.SimpleNamespace(request_timeout=create_kwargs.get("request_timeout"))
        self.snapshot_names: list[str | None] = []
        self.set_timeout_calls: list[int] = []
        self.set_timeout_outcomes: list[object] = []
        FakeSandbox.instances.append(self)

    @classmethod
    async def create(cls, **kwargs):
        template = kwargs.get("template", "")
        if template.startswith("missing-typed"):
            raise FakeSandboxNotFound(f"template {template} not found")
        if template.startswith("missing-bad-request"):
            # AgentENV answers a template or snapshot id it does not have with HTTP 400.
            raise FakeSdkError(f"400: template {template} not found", status_code=400)
        if template.startswith("invalid"):
            raise FakeSdkError("400: invalid metadata", status_code=400)
        if template.startswith("missing"):
            # AgentENV's plain-text 404, which the SDK raises with no message but the status.
            raise FakeSdkError("404: b'template not found'", status_code=404)
        return cls(**kwargs)

    async def create_snapshot(self, name=None, **kwargs):
        _reject_connection_params("Sandbox.create_snapshot", kwargs)
        self.snapshot_names.append(name)
        snapshot_id = f"tmpl{len(FakeSandbox.snapshots) + 1}:default"
        names = [f"team-slug/{name}:default"] if name else []
        FakeSandbox.snapshots[snapshot_id] = {"names": names, "sandbox_id": self.sandbox_id}
        return types.SimpleNamespace(snapshot_id=snapshot_id, names=names)

    async def set_timeout(self, timeout, **kwargs):
        _reject_connection_params("Sandbox.set_timeout", kwargs)
        self.set_timeout_calls.append(timeout)
        if self.set_timeout_outcomes:
            outcome = self.set_timeout_outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome

    @classmethod
    async def delete_snapshot(cls, snapshot_id, **kwargs):
        cls.delete_snapshot_calls.append(snapshot_id)
        if cls.delete_snapshot_outcomes:
            outcome = cls.delete_snapshot_outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
        return cls.snapshots.pop(snapshot_id, None) is not None

    @classmethod
    def list_snapshots(cls, **kwargs):
        infos = [
            types.SimpleNamespace(snapshot_id=snapshot_id, names=list(entry["names"]))
            for snapshot_id, entry in cls.snapshots.items()
        ]
        # Two pages, to exercise pagination.
        return FakeSnapshotPaginator([infos[:1], infos[1:]] if len(infos) > 1 else [infos])

    @classmethod
    async def connect(cls, sandbox_id, **kwargs):
        return cls(sandbox_id=sandbox_id, **kwargs)

    async def is_running(self, **kwargs):
        _reject_connection_params("Sandbox.is_running", kwargs)
        return self.running

    async def pause(self, **kwargs: object) -> bool:
        self.pause_calls.append(kwargs)
        was_paused = self.paused
        self.paused = True
        return not was_paused

    async def get_info(self, **kwargs: object) -> types.SimpleNamespace:
        _reject_connection_params("Sandbox.get_info", kwargs)
        if not self.running:
            raise FakeSandboxNotFound(self.sandbox_id)
        return types.SimpleNamespace(state="paused" if self.paused else "running")

    async def kill(self, **kwargs):
        self.kill_calls.append(kwargs)
        if self.kill_outcomes:
            outcome = self.kill_outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome is False:
                return False
        self.killed = True
        return True


def _fake_sdk() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        AsyncSandbox=FakeSandbox,
        ConnectionConfig=FakeConnectionConfig,
        api=types.SimpleNamespace(client_async=types.SimpleNamespace(get_api_client=FakeApiClient)),
        SandboxNotFoundException=FakeSandboxNotFound,
        NotFoundException=FakeSandboxNotFound,
        TimeoutException=FakeTimeout,
        RateLimitException=FakeRateLimit,
        CommandExitException=FakeCommandExit,
        SandboxUnreachableException=FakeUnreachable,
        AuthenticationException=type("FakeAuth", (Exception,), {}),
        InvalidArgumentException=type("FakeInvalid", (Exception,), {}),
    )


def _fake_sdk_module() -> types.ModuleType:
    module = types.ModuleType("e2b")
    module.__dict__.update(vars(_fake_sdk()))
    return module


@pytest.fixture(autouse=True)
def fake_e2b_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandbox.instances.clear()
    FakeSandbox.snapshots.clear()
    FakeSandbox.delete_snapshot_calls.clear()
    FakeSandbox.delete_snapshot_outcomes.clear()
    monkeypatch.setattr(e2b_provider, "KILL_RECHECK_DELAY_S", 0.0)
    FakeConnectionConfig.integrations.clear()
    FakeApiClient.calls.clear()
    FakeApiClient.get_calls.clear()
    # Most tests model a sandbox past its new-sandbox grace period; _with_grace() opts a test back in.
    monkeypatch.setattr(E2BProvider, "_start_grace", lambda self, sandbox_id: None)
    monkeypatch.setattr(e2b_provider, "NEW_SANDBOX_KILL_RETRY_S", 0.0)
    monkeypatch.setattr(e2b_provider, "_resource_warning_logged", False)
    monkeypatch.setattr(FakeApiClient, "get_response", FakeHttpResponse(200, {"state": "running"}))
    monkeypatch.setattr(e2b_provider, "_require_e2b_sdk", _fake_sdk)


def _spec(**kwargs) -> SandboxSpec:
    return SandboxSpec(**kwargs)


_REAL_START_GRACE = E2BProvider._start_grace


def _with_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give new sandboxes their grace period (``operations.new_sandbox_grace_s``), as outside these tests."""
    monkeypatch.setattr(E2BProvider, "_start_grace", _REAL_START_GRACE)


_NETWORK = {"deny_out": ["0.0.0.0/0"], "allow_out": ["pypi.org", "deb.debian.org"]}


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_provider_is_registered_as_builtin() -> None:
    assert get_provider_class("e2b") is E2BProvider
    assert E2BProvider.name == "e2b"
    assert e2b_pkg.E2BProvider is E2BProvider


def test_sdk_install_contract_matches_package_metadata_and_guidance() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    with (repo_root / "pyproject.toml").open("rb") as source:
        dependencies = tomllib.load(source)["project"]["optional-dependencies"]["sandbox"]

    assert e2b_sdk.E2B_SDK_CONSTRAINT in dependencies
    provider_dir = repo_root / "nemo_gym/sandbox/providers/e2b"
    for relative_path in ("README.md", "configs/e2b.yaml"):
        assert e2b_sdk.E2B_SDK_CONSTRAINT in (provider_dir / relative_path).read_text(encoding="utf-8")


async def test_runtime_loader_sets_integration_once_per_sdk_module(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk_module = _fake_sdk_module()
    configured_transports = []
    monkeypatch.setitem(sys.modules, "e2b", sdk_module)
    monkeypatch.setattr(e2b_sdk, "_CONFIGURED_SDK_MODULES", {})
    monkeypatch.setattr(e2b_sdk, "_configure_async_http", lambda: configured_transports.append(True))
    monkeypatch.setattr(e2b_provider, "_require_e2b_sdk", _REAL_PROVIDER_REQUIRE_E2B_SDK)

    provider = E2BProvider(create={"template": "base"})
    handle = await provider.create(_spec())
    assert await provider.status(handle) is SandboxStatus.RUNNING

    assert FakeConnectionConfig.integrations == [f"nemo-gym/{nemo_gym_version}"]
    assert configured_transports == [True]


@pytest.mark.parametrize("configured", [False, True])
async def test_runtime_loader_routes_e2b_httpx_through_global_aiohttp(
    monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    import httpx

    from nemo_gym.sandbox.providers._http_transport import GymAiohttpTransport

    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
    if configured:
        monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "{}")
    else:
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)

    sdk_module = _fake_sdk_module()
    sdk_module.__path__ = []
    api_module = types.ModuleType("e2b.api")
    api_module.__path__ = []
    client_async_module = types.ModuleType("e2b.api.client_async")
    client_async_module.limits = httpx.Limits(max_connections=10)
    client_async_module.connection_retries = 2
    client_async_module.get_transport = lambda config, http2=True: object()
    client_async_module.get_envd_transport = lambda config, http2=True: object()
    original_control_transport = client_async_module.get_transport
    original_envd_transport = client_async_module.get_envd_transport
    sandbox_async_module = types.ModuleType("e2b.sandbox_async")
    sandbox_async_module.__path__ = []
    sandbox_main_module = types.ModuleType("e2b.sandbox_async.main")
    sandbox_main_module.get_transport = client_async_module.get_envd_transport

    sdk_module.api = api_module
    sdk_module.sandbox_async = sandbox_async_module
    api_module.client_async = client_async_module
    sandbox_async_module.main = sandbox_main_module
    for name, module in {
        "e2b": sdk_module,
        "e2b.api": api_module,
        "e2b.api.client_async": client_async_module,
        "e2b.sandbox_async": sandbox_async_module,
        "e2b.sandbox_async.main": sandbox_main_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(e2b_sdk, "_CONFIGURED_SDK_MODULES", {})

    e2b_sdk.require_e2b_sdk("Testing the e2b provider")
    if not configured:
        assert client_async_module.get_transport is original_control_transport
        assert sandbox_main_module.get_transport is original_envd_transport
        return
    config = types.SimpleNamespace(proxy=None)
    control_transport = client_async_module.get_transport(config)
    envd_transport = sandbox_main_module.get_transport(config)

    assert isinstance(control_transport, GymAiohttpTransport)
    assert isinstance(envd_transport, GymAiohttpTransport)
    # E2B 2.5x+ passes these keywords to the factory it calls.
    assert isinstance(
        client_async_module.get_transport(config, http_version="1.1", retry_connect=False, for_streaming=True),
        GymAiohttpTransport,
    )
    proxy = httpx.Proxy("http://proxy.example:8080", auth=("user", "password"))
    assert client_async_module.get_transport(types.SimpleNamespace(proxy=proxy)).proxy is proxy
    assert str(client_async_module.get_transport(types.SimpleNamespace(proxy="http://proxy.example")).proxy.url) == (
        "http://proxy.example"
    )
    with pytest.raises(ValueError, match="HTTP or HTTPS proxy"):
        client_async_module.get_transport(types.SimpleNamespace(proxy="socks5h://proxy.example:1080"))


def test_loader_reports_missing_optional_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "e2b", None)

    with pytest.raises(ImportError, match=r"pip install 'e2b>=2\.46\.0,<3\.0\.0'"):
        e2b_sdk.require_e2b_sdk("Testing the e2b provider")


async def test_real_sdk_user_agent_and_call_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    e2b = pytest.importorskip("e2b", reason="e2b optional sandbox dependency is not installed")
    from e2b.api import client_async
    from e2b.sandbox_async import main as sandbox_async

    from nemo_gym.sandbox.providers._http_transport import GymAiohttpTransport

    monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "{}")
    monkeypatch.setattr(e2b_sdk, "_CONFIGURED_SDK_MODULES", {})

    assert version("e2b") in Requirement(e2b_sdk.E2B_SDK_CONSTRAINT).specifier
    assert set(_API_PARAM_KEYS) <= set(e2b.ApiParams.__annotations__)

    e2b_sdk.require_e2b_sdk("Testing the e2b provider")
    connection = e2b.ConnectionConfig()
    products = connection.headers["User-Agent"].split()
    assert f"nemo-gym/{nemo_gym_version}" in products
    envd_client = sandbox_async.get_envd_api(connection, "https://sandbox.example")
    # The REST create path posts through the SDK's API client.
    api_client = client_async.get_api_client(
        e2b.ConnectionConfig(api_key="k", api_url="http://gw:8080", api_headers={"Authorization": "Bearer t"})
    ).get_async_httpx_client()
    assert str(api_client.base_url) == "http://gw:8080"
    assert (api_client.headers["X-API-KEY"], api_client.headers["Authorization"]) == ("k", "Bearer t")
    for transport in (client_async.get_transport(connection), envd_client._transport, api_client._transport):
        assert isinstance(transport, GymAiohttpTransport)
    await envd_client.aclose()
    await api_client.aclose()

    inspect.signature(e2b.AsyncSandbox.create).bind(
        template="base",
        timeout=60,
        request_timeout=30,
        envs={},
        metadata={},
        secure=True,
        allow_internet_access=True,
        network={"allow_out": ["pypi.org"], "deny_out": ["0.0.0.0/0"]},
        lifecycle={"on_timeout": "pause", "auto_resume": True},
    )
    inspect.signature(e2b.AsyncSandbox.connect).bind("sbx-existing", timeout=60, request_timeout=30)
    inspect.signature(e2b.AsyncSandbox.kill).bind(
        object(),
        **dict.fromkeys(_API_PARAM_KEYS),
        request_timeout=30,
    )
    # AsyncSandbox.kill(sandbox_id, ...) called on the class dispatches here.
    inspect.signature(e2b.AsyncSandbox._cls_kill).bind("sbx-id", **dict.fromkeys(_API_PARAM_KEYS), request_timeout=30)
    # get_info/pause take **ApiParams, so signature().bind() accepts any keyword; check the key itself.
    assert "request_timeout" in e2b.ApiParams.__annotations__
    assert inspect.signature(e2b.AsyncSandbox.pause).parameters["keep_memory"].default is True
    assert e2b.ConnectionConfig(request_timeout=7).request_timeout == 7
    inspect.signature(e2b.AsyncSandbox.get_info).bind(object(), request_timeout=30)
    if hasattr(e2b.AsyncSandbox, "create_snapshot"):
        # The snapshot and lease calls (absent from e2b releases before snapshots shipped).
        inspect.signature(e2b.AsyncSandbox.create).bind(template="t", network={"deny_out": ["0.0.0.0/0"]})
        inspect.signature(e2b.AsyncSandbox.create_snapshot).bind(object(), None, request_timeout=30)
        inspect.signature(e2b.AsyncSandbox.set_timeout).bind(object(), 60, request_timeout=30)
        inspect.signature(e2b.AsyncSandbox.delete_snapshot).bind("tmpl:default", **dict.fromkeys(_API_PARAM_KEYS))
        # Called on the class it dispatches to the class variant, whose signature introspection does not show.
        inspect.signature(e2b.AsyncSandbox._cls_list_snapshots).bind(**dict.fromkeys(_API_PARAM_KEYS))
        info = e2b.SnapshotInfo(snapshot_id="tmpl:default", names=["team/ng-x:default"])
        assert (info.snapshot_id, info.names) == ("tmpl:default", ["team/ng-x:default"])
    inspect.signature(e2b.AsyncTemplate.exists).bind("base")
    inspect.signature(e2b.AsyncTemplate.build).bind(
        object(),
        name="base",
        cpu_count=2,
        memory_mb=1024,
    )


async def test_real_sdk_pause_resume_and_status_use_only_the_control_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real SDK parses states into ``SandboxState`` enums and maps 409 to False; pin both."""
    e2b = pytest.importorskip("e2b", reason="e2b optional sandbox dependency is not installed")
    import httpx
    from e2b.api import client_async

    state = {"value": "running"}
    requests: list[tuple[str, str, bytes]] = []
    record = {"sandboxID": "sbx-1", "templateID": "base", "clientID": "c", "envdVersion": "0.4.0"}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, request.content))
        if request.url.path == "/sandboxes/sbx-1/pause":
            if state["value"] == "paused":
                return httpx.Response(409, json={"code": 409, "message": "already paused"})
            state["value"] = "paused"
            return httpx.Response(204)
        if request.url.path == "/sandboxes/sbx-1/connect":
            state["value"] = "running"
            return httpx.Response(200, json=record)
        detail = {"cpuCount": 1, "memoryMB": 512, "diskSizeMB": 1024, "state": state["value"]}
        times = {"startedAt": "2026-01-01T00:00:00Z", "endAt": "2026-01-01T01:00:00Z"}
        return httpx.Response(200, json={**record, **detail, **times})

    # envd HTTP clients also take get_transport; get_envd_transport is its deprecated alias.
    for factory in ("get_transport", "get_envd_transport"):
        monkeypatch.setattr(client_async, factory, lambda *args, **kwargs: httpx.MockTransport(handler))
    monkeypatch.setattr(e2b_provider, "_require_e2b_sdk", lambda: e2b)
    provider = E2BProvider(connection={"api_key": "k", "api_url": "http://api.test"}, create={"template": "base"})

    handle = await provider.connect({"sandbox_id": "sbx-1"})
    requests.clear()
    assert await provider.status(handle) is SandboxStatus.RUNNING
    await provider.pause(handle)
    with pytest.raises(RuntimeError, match="refused pause with 409"):  # A 409 is never confirmed as paused.
        await provider.pause(handle)
    assert await provider.status(handle) is SandboxStatus.PAUSED
    await provider.resume(handle)
    assert await provider.status(handle) is SandboxStatus.RUNNING

    # No envd request (an is_running health check would add GET /health): don't wake a paused guest.
    assert [(method, path) for method, path, _ in requests] == [
        ("GET", "/sandboxes/sbx-1"),
        ("POST", "/sandboxes/sbx-1/pause"),
        ("POST", "/sandboxes/sbx-1/pause"),
        ("GET", "/sandboxes/sbx-1"),
        ("POST", "/sandboxes/sbx-1/connect"),
        ("GET", "/sandboxes/sbx-1"),
    ]
    assert json.loads(requests[4][2]) == {"timeout": 3600}


# --------------------------------------------------------------------------
# Template resolution -- E2B starts from an alias, not a registry reference
# --------------------------------------------------------------------------


class TestTemplateResolution:
    async def test_provider_options_template_wins(self) -> None:
        provider = E2BProvider(create={"template": "fallback"})
        handle = await provider.create(_spec(image="also-valid", provider_options={"template": "chosen"}))
        assert handle.raw.create_kwargs["template"] == "chosen"

    async def test_template_map_translates_registry_reference(self) -> None:
        provider = E2BProvider(create={"template_map": {"ghcr.io/acme/task:1.0": "acme-task"}})
        handle = await provider.create(_spec(image="ghcr.io/acme/task:1.0"))
        assert handle.raw.create_kwargs["template"] == "acme-task"

    async def test_image_used_directly_when_already_an_alias(self) -> None:
        provider = E2BProvider()
        handle = await provider.create(_spec(image="build-cython-ext__c9fba49d4bd3"))
        assert handle.raw.create_kwargs["template"] == "build-cython-ext__c9fba49d4bd3"

    async def test_falls_back_to_configured_template(self) -> None:
        provider = E2BProvider(create={"template": "base-template"})
        handle = await provider.create(_spec())
        assert handle.raw.create_kwargs["template"] == "base-template"

    async def test_registry_reference_without_mapping_does_not_use_fallback(self) -> None:
        # Silently starting the wrong template would corrupt a benchmark run,
        # so an unmappable image must fail loudly.
        provider = E2BProvider(create={"template": "fallback"})
        with pytest.raises(E2BCreateError, match="template_map"):
            await provider.create(_spec(image="ghcr.io/acme/task:1.0"))
        assert FakeSandbox.instances == []

    async def test_no_template_at_all_raises(self) -> None:
        provider = E2BProvider()
        with pytest.raises(E2BCreateError, match="No E2B template"):
            await provider.create(_spec())


# --------------------------------------------------------------------------
# Resources -- fixed at template build time, must not be dropped silently
# --------------------------------------------------------------------------


class TestResourceHandling:
    async def test_resource_request_warns_once_per_process(self, caplog) -> None:
        provider = E2BProvider(create={"template": "base"})
        with caplog.at_level("WARNING"):
            await provider.create(
                _spec(
                    resources={
                        "cpu": 8,
                        "memory_mib": 16384,
                        "disk_gib": 100,
                        "gpu": 1,
                        "gpu_type": "H100",
                    }
                )
            )
            await provider.create(_spec(resources={"cpu": 8}))
            await E2BProvider(create={"template": "other"}).create(_spec(resources={"cpu": 8}))
        warnings = [r for r in caplog.records if "fixes sandbox resources" in r.message]
        assert len(warnings) == 1 and "Later templates are logged at DEBUG" in warnings[0].message
        for detail in ("cpu=8", "memory_mib=16384", "disk_gib=100", "gpu=1", "gpu_type=H100"):
            assert detail in warnings[0].message

    @pytest.mark.parametrize(
        "resources",
        [
            {"cpu": 1},
            {"memory_mib": 1024},
            {"disk_gib": 10},
            {"gpu": 1},
            {"gpu_type": "H100"},
        ],
    )
    async def test_strict_resources_rejects_every_resource(self, resources: dict[str, object]) -> None:
        provider = E2BProvider(create={"template": "base", "strict_resources": True})
        with pytest.raises(E2BCreateError, match="fixes sandbox resources"):
            await provider.create(_spec(resources=resources))
        assert FakeSandbox.instances == []

    async def test_no_warning_without_resource_request(self, caplog) -> None:
        provider = E2BProvider(create={"template": "base"})
        with caplog.at_level("WARNING"):
            await provider.create(_spec())
        assert not [r for r in caplog.records if "fixes sandbox resources" in r.message]


# --------------------------------------------------------------------------
# Connection params are connection-scoped, not per-call
# --------------------------------------------------------------------------


class TestConnectionParamScoping:
    """Only create/connect/kill open a connection and accept ``ApiParams``.

    ``commands.run``, ``files.*`` and ``is_running`` run against an
    already-connected sandbox and take ``request_timeout`` only. Passing them
    the full set raises ``TypeError: Commands.run() got an unexpected keyword
    argument 'api_key'`` -- which aborted every trial of a benchmark run at the
    first exec, since the sandbox had already been created successfully.
    """

    @staticmethod
    def _provider() -> E2BProvider:
        return E2BProvider(
            connection={"api_key": "k", "api_url": "http://gw:8080", "request_timeout_s": 30.0},
            create={"template": "base"},
        )

    async def test_exec_passes_only_request_timeout(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        await provider.exec(handle, "echo hi")
        kwargs = handle.raw.exec_calls[0]
        assert not set(kwargs) & set(_API_PARAM_KEYS), "connection params must not reach commands.run"
        assert kwargs["request_timeout"] == 30.0

    async def test_file_and_status_calls_do_not_leak_connection_params(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        # Each of these would raise TypeError from the fake (as the real SDK does).
        await provider.write_file(handle, "/tmp/f", b"data")
        assert await provider.read_file(handle, "/tmp/f") == b"data"
        assert await provider.status(handle) is SandboxStatus.RUNNING

    async def test_command_and_request_timeouts_are_independent(self) -> None:
        # In E2B 2.36 request_timeout bounds opening the command stream, while
        # timeout bounds the running output stream/wait, not the remote process.
        provider = E2BProvider(
            connection={"api_key": "k", "request_timeout_s": 30.0},
            create={"template": "base"},
        )
        handle = await provider.create(_spec())
        await provider.exec(handle, "make -j8", timeout_s=1800)
        kwargs = handle.raw.exec_calls[0]
        assert kwargs["timeout"] == 1800.0
        assert kwargs["request_timeout"] == 30.0

    async def test_exec_request_timeout_override_is_honoured(self) -> None:
        provider = E2BProvider(
            connection={"request_timeout_s": 30.0},
            create={"template": "base"},
            exec={"request_timeout_s": 900.0},
        )
        handle = await provider.create(_spec())
        await provider.exec(handle, "sleep 1", timeout_s=60)
        assert handle.raw.exec_calls[0]["request_timeout"] == 900.0

    async def test_untimed_command_keeps_the_connection_request_timeout(self) -> None:
        provider = E2BProvider(
            connection={"request_timeout_s": 30.0},
            create={"template": "base"},
            exec={"default_timeout_s": None},
        )
        handle = await provider.create(_spec())
        await provider.exec(handle, "sleep forever")
        kwargs = handle.raw.exec_calls[0]
        assert kwargs["timeout"] is None
        assert kwargs["request_timeout"] == 30.0

    async def test_create_still_receives_full_connection_params(self) -> None:
        # The narrowing must not strip params from the call that needs them.
        provider = self._provider()
        handle = await provider.create(_spec())
        assert handle.raw.create_kwargs["api_key"] == "k"
        assert handle.raw.create_kwargs["api_url"] == "http://gw:8080"


# --------------------------------------------------------------------------
# Background exec -- survive a lost output stream
# --------------------------------------------------------------------------


class TestBackgroundExec:
    """A dropped stream must not destroy a command that is still running.

    `run(background=True)` returns once the process has started, so the command
    outlives the stream carrying its output and can be reattached by pid.
    """

    @staticmethod
    def _provider(**exec_opts) -> E2BProvider:
        opts = {"background": True, **exec_opts}
        return E2BProvider(create={"template": "base"}, exec=opts)

    async def test_on_by_default(self) -> None:
        # Matches Harbor's own e2b environment, which always dispatches with
        # background=True.
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        await provider.exec(handle, "echo hi")
        assert handle.raw.exec_calls[0]["background"] is True

    async def test_can_be_turned_off(self) -> None:
        provider = E2BProvider(create={"template": "base"}, exec={"background": False})
        handle = await provider.create(_spec())
        await provider.exec(handle, "echo hi")
        assert "background" not in handle.raw.exec_calls[0]

    async def test_background_flag_is_sent(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        result = await provider.exec(handle, "echo hi")
        assert handle.raw.exec_calls[0]["background"] is True
        assert result.return_code == 0

    async def test_lost_stream_is_reattached_by_pid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Start, deadline base, then the reattach 5 s in.
        monotonic_values = iter([100.0, 100.0, 105.0])
        monkeypatch.setattr(e2b_provider, "monotonic", monotonic_values.__next__)
        provider = self._provider(request_timeout_s=75.0)
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [
            ConnectionError("peer closed connection without sending complete message body"),
            FakeCommandResult(stdout="finished", exit_code=0),
        ]
        result = await provider.exec(handle, "make -j8", timeout_s=60)
        assert handle.raw.connect_calls == [{"pid": 4242, "timeout": 55.0, "request_timeout": 55.0}]
        assert result.return_code == 0
        assert result.stdout == "finished"

    async def test_reattach_preserves_output_received_before_disconnect(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        stream_error = ConnectionError("stream lost")
        stream_error.stdout = "before-out\n"
        stream_error.stderr = "before-err\n"
        handle.raw.wait_outcomes = [
            stream_error,
            FakeCommandResult(stdout="after-out\n", stderr="after-err\n", exit_code=0),
        ]

        result = await provider.exec(handle, "make -j8")

        assert result.stdout == "before-out\nafter-out\n"
        assert result.stderr == "before-err\nafter-err\n"

    async def test_lost_stream_does_not_reattach_after_deadline(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Start, deadline base, then the reattach check and the timeout report after the deadline.
        monotonic_values = iter([100.0, 100.0, 161.0, 161.0])
        monkeypatch.setattr(e2b_provider, "monotonic", monotonic_values.__next__)
        provider = self._provider()
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [ConnectionError("stream lost")]

        result = await provider.exec(handle, "make -j8", timeout_s=60)

        assert (result.return_code, result.error_type) == (124, "timeout")
        assert handle.raw.connect_calls == []

    async def test_reattach_recovers_the_real_exit_code(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        stream_error = ConnectionError("stream lost")
        stream_error.stdout = "before-out\n"
        stream_error.stderr = "before-err\n"
        handle.raw.wait_outcomes = [
            stream_error,
            FakeCommandExit(7, stdout="after-out\n", stderr="after-err\n"),
        ]
        result = await provider.exec(handle, "false")
        assert result.return_code == 7
        assert result.stdout == "before-out\nafter-out\n"
        assert result.stderr == "before-err\nafter-err\n"

    async def test_reattach_attempts_are_bounded(self) -> None:
        provider = self._provider(reconnect_attempts=2)
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [ConnectionError("a"), ConnectionError("b"), ConnectionError("c")]
        with pytest.raises(ConnectionError):
            await provider.exec(handle, "sleep 1")
        assert len(handle.raw.connect_calls) == 2

    async def test_non_zero_exit_is_not_treated_as_a_lost_stream(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [FakeCommandExit(3, stdout="out", stderr="err")]
        result = await provider.exec(handle, "false")
        assert result.return_code == 3
        assert handle.raw.connect_calls == [], "a real exit must not trigger a reattach"

    async def test_a_timeout_before_the_deadline_reattaches_with_the_remaining_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # E2B reports a dropped stream as a TimeoutException; the command keeps running in the sandbox.
        clock = iter([100.0, 100.0, 1100.0, 1100.0])  # exec start, run start, the timeout check, the reattach budget
        monkeypatch.setattr(e2b_provider, "monotonic", lambda: next(clock))
        provider = self._provider()
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [FakeTimeout("Request timed out"), FakeCommandResult(stdout="done")]
        result = await provider.exec(handle, "opencode run", timeout_s=8490)
        assert (result.return_code, result.stdout) == (0, "done")
        [reattach] = handle.raw.connect_calls
        assert reattach["pid"] == handle.raw.pid and reattach["timeout"] == pytest.approx(7490)

    async def test_an_unreachable_stream_reattaches(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [FakeUnreachable("could not be reached"), FakeCommandResult(stdout="done")]
        result = await provider.exec(handle, "opencode run", timeout_s=60)
        assert result.return_code == 0 and len(handle.raw.connect_calls) == 1

    async def test_process_gone_reports_the_original_failure(self) -> None:
        # connect() raises not-found once the command has exited, so a command
        # that finishes during the gap cannot be recovered; the transport
        # failure is the useful error to surface.
        provider = self._provider()
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [ConnectionError("stream lost")]
        handle.raw.connect_error = FakeSandboxNotFound("process with pid 4242 not found")
        with pytest.raises(ConnectionError, match="stream lost"):
            await provider.exec(handle, "quick")


# --------------------------------------------------------------------------
# Create / lifecycle
# --------------------------------------------------------------------------


class TestCreateAndLifecycle:
    async def test_spec_fields_map_onto_sdk_kwargs(self) -> None:
        provider = E2BProvider(
            connection={"api_key": "k", "api_url": "http://gw:8080", "request_timeout_s": 30.0},
            create={"template": "base", "allow_internet_access": False},
            attribution={"enabled": False},
        )
        handle = await provider.create(
            _spec(ttl_s=120, env={"FOO": "bar"}, metadata={"run": "1"}),
        )
        kwargs = handle.raw.create_kwargs
        assert kwargs["timeout"] == 120
        assert kwargs["envs"] == {"FOO": "bar"}
        assert kwargs["metadata"] == {"run": "1"}
        assert kwargs["allow_internet_access"] is False
        assert kwargs["api_key"] == "k"
        assert kwargs["api_url"] == "http://gw:8080"
        assert kwargs["request_timeout"] == 30.0
        assert handle.provider_name == "e2b"
        assert handle.sandbox_id == "sbx-1"

    async def test_ready_timeout_overrides_connection_request_timeout(self) -> None:
        provider = E2BProvider(
            connection={"request_timeout_s": 30.0},
            create={"template": "base"},
        )
        handle = await provider.create(_spec(ready_timeout_s=90))
        assert handle.raw.create_kwargs["request_timeout"] == 90.0

    @pytest.mark.parametrize("ready_timeout_s", [0, -1, float("nan"), float("inf"), True, "60"])
    async def test_invalid_ready_timeout_is_rejected_before_create(self, ready_timeout_s: object) -> None:
        provider = E2BProvider(create={"template": "base"})
        with pytest.raises(E2BCreateError, match="ready_timeout_s"):
            await provider.create(_spec(ready_timeout_s=ready_timeout_s))
        assert FakeSandbox.instances == []

    @pytest.mark.parametrize("ttl_s", [0, -1, float("nan"), float("inf"), True, "60"])
    async def test_invalid_ttl_is_rejected_before_create(self, ttl_s: object) -> None:
        provider = E2BProvider(create={"template": "base"})
        with pytest.raises(E2BCreateError, match="ttl_s"):
            await provider.create(_spec(ttl_s=ttl_s))
        assert FakeSandbox.instances == []

    async def test_entrypoint_is_rejected_before_create(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        with pytest.raises(E2BCreateError, match="entrypoint"):
            await provider.create(_spec(entrypoint=["python", "app.py"]))
        assert FakeSandbox.instances == []

    async def test_unknown_provider_options_are_rejected_before_create(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        with pytest.raises(ValueError, match="provider option"):
            await provider.create(_spec(provider_options={"template": "base", "unknown": True}))
        assert FakeSandbox.instances == []

    async def test_provider_create_leaves_spec_files_to_the_facade(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec(files={"/app/seed.txt": "hello"}))
        assert handle.raw.file_write_calls == []

    async def test_async_sandbox_uploads_spec_files_exactly_once(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        sandbox = await AsyncSandbox(
            provider,
            _spec(files={"/app/seed.txt": "hello"}),
        ).start()
        raw = FakeSandbox.instances[0]
        assert raw.file_write_calls == [{"path": "/app/seed.txt", "data": b"hello"}]
        assert raw.files_written == {"/app/seed.txt": b"hello"}
        await sandbox.stop()

    async def test_async_sandbox_cleans_up_when_spec_file_upload_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = E2BProvider(create={"template": "base"})

        async def fail_upload(handle, source_path, target_path):
            raise RuntimeError("upload failed")

        monkeypatch.setattr(provider, "upload_file", fail_upload)

        with pytest.raises(RuntimeError, match="upload failed"):
            await AsyncSandbox(
                provider,
                _spec(files={"/app/seed.txt": "hello"}),
            ).start()

        raw = FakeSandbox.instances[0]
        assert raw.killed is True
        assert len(raw.kill_calls) == 1

    async def test_create_failure_is_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(**kwargs):
            raise RuntimeError("gateway exploded")

        monkeypatch.setattr(FakeSandbox, "create", boom)
        provider = E2BProvider(create={"template": "base"}, operations={"retries": 0})
        with pytest.raises(E2BCreateError, match="gateway exploded"):
            await provider.create(_spec())

    async def test_network_is_passed_to_sdk_create_only_when_set(self) -> None:
        default = await E2BProvider(create={"template": "base"}).create(_spec())
        assert "network" not in default.raw.create_kwargs

        handle = await E2BProvider(create={"template": "base", "network": _NETWORK}).create(_spec())
        assert handle.raw.create_kwargs["network"] == _NETWORK

    async def test_extension_params_create_through_rest_api_then_connect(self) -> None:
        provider = E2BProvider(
            connection={"api_key": "secret-key", "api_url": "http://gw:8080", "request_timeout_s": 30.0},
            create={"template": "base", "network": _NETWORK, "custom_extension_params": {"sandbank": "all"}},
            attribution={"team": "t", "user": "u", "workload": "w", "run": "r"},
        )
        handle = await provider.create(
            _spec(ttl_s=120, ready_timeout_s=90, env={"FOO": "bar"}, metadata={"run": "1"}),
        )

        [call] = FakeApiClient.calls
        assert call["url"] == "/sandboxes"
        # The SDK's API client takes URL, auth headers and timeout from this config.
        assert call["config"] == {"api_key": "secret-key", "api_url": "http://gw:8080", "request_timeout": 90.0}
        assert call["json"] == {
            "templateID": "base",
            "timeout": 120,
            "secure": True,
            "allow_internet_access": True,
            "network": {"denyOut": ["0.0.0.0/0"], "allowOut": ["pypi.org", "deb.debian.org"]},
            "envVars": {"FOO": "bar"},
            "metadata": {
                "nemo-gym.nvidia.com/team": "t",
                "nemo-gym.nvidia.com/user": "u",
                "nemo-gym.nvidia.com/workload": "w",
                "nemo-gym.nvidia.com/run": "r",
                "run": "1",
            },
            "customExtensionParams": {"sandbank": "all"},
            "autoPause": False,
            "autoResume": {"enabled": False},
        }
        # Attached with connect (keeping the create timeout), not created twice.
        assert [sandbox.sandbox_id for sandbox in FakeSandbox.instances] == ["sbx-rest"]
        assert handle.sandbox_id == "sbx-rest"
        assert handle.raw.create_kwargs["timeout"] == 120
        assert handle.raw.create_kwargs["api_key"] == "secret-key"
        assert handle.raw.killed is False

    async def test_rest_create_failure_is_wrapped_and_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FakeApiClient, "response", FakeHttpResponse(400, "unknown template"))
        provider = E2BProvider(
            create={"template": "base", "custom_extension_params": {"sandbank": "all"}},
            operations={"retries": 3, "retry_delay_s": 0},
        )
        with pytest.raises(E2BCreateError, match="HTTP 400: unknown template"):
            await provider.create(_spec())

        assert len(FakeApiClient.calls) == 1
        assert FakeSandbox.instances == []

    async def test_failed_connect_kills_the_created_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        killed: list[str] = []

        async def connect(cls, sandbox_id, **kwargs):
            raise FakeTimeout("connect timed out")

        async def kill(sandbox_id, **kwargs):
            killed.append(sandbox_id)
            return True

        monkeypatch.setattr(FakeSandbox, "connect", classmethod(connect))
        monkeypatch.setattr(FakeSandbox, "kill", staticmethod(kill))
        provider = E2BProvider(create={"template": "base", "custom_extension_params": {"sandbank": "all"}})
        with pytest.raises(E2BCreateError, match="connect timed out"):
            await provider.create(_spec())

        assert killed == ["sbx-rest"]

    async def test_status_and_close(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        assert await provider.status(handle) == SandboxStatus.RUNNING
        handle.raw.running = False
        assert await provider.status(handle) == SandboxStatus.STOPPED

        handle.raw.running = True
        sandbox = handle.raw
        await provider.close(handle)
        assert sandbox.killed is True
        assert handle.raw is None
        # Closing twice must not raise.
        await provider.close(handle)
        assert len(sandbox.kill_calls) == 1

    @pytest.mark.parametrize(
        "transient_error",
        [ConnectionError("connection reset"), FakeTimeout("request timed out")],
    )
    async def test_close_retries_transient_kill_errors(self, transient_error: Exception) -> None:
        provider = E2BProvider(
            create={"template": "base"},
            operations={"retries": 1, "retry_delay_s": 0},
        )
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [transient_error, True]

        await provider.close(handle)

        assert len(sandbox.kill_calls) == 2
        assert sandbox.killed is True
        assert handle.raw is None

    async def test_close_exhausted_failure_preserves_raw_handle(self) -> None:
        provider = E2BProvider(
            create={"template": "base"},
            operations={"retries": 1, "retry_delay_s": 0},
        )
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [ConnectionError("temporary"), ConnectionError("still unavailable")]

        with pytest.raises(ConnectionError, match="still unavailable"):
            await provider.close(handle)

        assert len(sandbox.kill_calls) == 2
        assert handle.raw is sandbox

    async def test_rate_limit_is_not_retried(self) -> None:
        provider = E2BProvider(
            create={"template": "base"},
            operations={"retries": 5, "retry_delay_s": 0},
        )
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [FakeRateLimit("429 resource exhausted"), True]

        with pytest.raises(FakeRateLimit, match="resource exhausted"):
            await provider.close(handle)

        assert len(sandbox.kill_calls) == 1
        assert handle.raw is sandbox

    async def test_close_false_result_clears_handle(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [False]
        sandbox.running = False  # the re-check confirms it is gone

        await provider.close(handle)

        assert sandbox.killed is False and len(sandbox.kill_calls) == 1
        assert handle.raw is None

    async def test_a_kill_that_found_nothing_is_repeated_when_the_sandbox_still_runs(self) -> None:
        """A gateway replica that has not seen the sandbox can answer "not found" for one that still runs."""
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [False, True]

        await provider.close(handle)

        assert sandbox.killed is True and len(sandbox.kill_calls) == 2
        assert handle.raw is None

    async def test_close_of_a_sandbox_answered_with_an_empty_404_finishes(self) -> None:
        """AgentENV answers a sandbox it no longer has with an empty-body 404, which the SDK fails to parse."""
        provider = E2BProvider(create={"template": "base"}, operations={"retries": 0})
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [_unparsed_404()]
        sandbox.running = False
        FakeApiClient.get_response = FakeHttpResponse(404, "")

        await provider.close(handle)

        assert FakeApiClient.get_calls == [f"/sandboxes/{sandbox.sandbox_id}"]
        assert sandbox.killed is False and handle.raw is None

    @pytest.mark.parametrize("lookup", [FakeHttpResponse(200, {"state": "running"}), ConnectionError("reset")])
    async def test_an_unparsed_kill_reply_for_a_sandbox_not_known_gone_still_fails_the_close(
        self, lookup: object
    ) -> None:
        provider = E2BProvider(create={"template": "base"}, operations={"retries": 0})
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [_unparsed_404()]
        FakeApiClient.get_response = lookup

        with pytest.raises(json.JSONDecodeError):
            await provider.close(handle)

        assert handle.raw is sandbox  # kept, so a later close can retry

    async def test_close_tolerates_already_gone_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())

        async def gone(**kwargs):
            raise FakeSandboxNotFound("expired")

        monkeypatch.setattr(handle.raw, "kill", gone)
        await provider.close(handle)
        assert handle.raw is None

    async def test_connectable_provider_serializes_and_connects_mapping(self) -> None:
        provider = E2BProvider()
        assert isinstance(provider, ConnectableProvider)
        descriptor = await provider.serialize_handle(
            SandboxHandle(sandbox_id="sbx-existing", provider_name="e2b", raw=object())
        )
        assert descriptor == {"sandbox_id": "sbx-existing"}

        handle = await provider.connect({**descriptor, "workdir": "/repo"})
        assert handle.sandbox_id == "sbx-existing"
        assert handle.provider_name == "e2b"

    async def test_async_sandbox_serialize_connect_round_trip_preserves_workdir(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        sandbox = await AsyncSandbox(provider, _spec(workdir="/repo")).start()
        descriptor = await sandbox.serialize()
        assert descriptor == {"sandbox_id": "sbx-1", "workdir": "/repo"}

        connected_provider = E2BProvider()
        connected = await AsyncSandbox.connect(descriptor, provider=connected_provider)
        await connected.exec("pwd")
        assert FakeSandbox.instances[-1].exec_calls[-1]["cwd"] == "/repo"

    async def test_aclose_is_a_noop(self) -> None:
        assert await E2BProvider().aclose() is None


# --------------------------------------------------------------------------
# exec
# --------------------------------------------------------------------------


class TestExec:
    async def test_exec_maps_arguments_and_result(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        handle.raw.exec_behaviour = FakeCommandResult(stdout="out", stderr="err", exit_code=0)

        result = await provider.exec(handle, "echo hi", cwd="/app", env={"A": "1"}, timeout_s=42, user="root")
        assert (result.stdout, result.stderr, result.return_code) == ("out", "err", 0)
        call = handle.raw.exec_calls[-1]
        assert call["cmd"] == "echo hi"
        assert call["cwd"] == "/app"
        assert call["envs"]["A"] == "1"
        assert set(call["envs"]) == {"A", "NEMO_GYM_E2B_EXEC"}
        assert call["user"] == "root"
        assert call["timeout"] == 42.0

    async def test_default_exec_timeout_matches_the_public_facade(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        sandbox = await AsyncSandbox(provider, _spec()).start()
        await sandbox.exec("true")
        assert FakeSandbox.instances[0].exec_calls[-1]["timeout"] == 180.0

    async def test_nonzero_exit_is_a_result_not_an_exception(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        handle.raw.exec_behaviour = FakeCommandExit(exit_code=7, stdout="partial", stderr="bad")

        result = await provider.exec(handle, "false")
        assert result.return_code == 7
        assert result.stdout == "partial"
        assert result.stderr == "bad"

    async def test_timeout_is_raised_as_timeout_error(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        handle.raw.exec_behaviour = FakeTimeout("deadline exceeded")

        with pytest.raises(TimeoutError, match=r"deadline exceeded.*wait/stream budget=1s"):
            await provider.exec(handle, "sleep 999", timeout_s=1)

    @pytest.mark.parametrize("timeout_s", [-1, float("nan"), float("inf"), True, "1"])
    async def test_invalid_timeout_is_rejected_before_command_start(self, timeout_s: object) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())

        with pytest.raises(ValueError, match="timeout_s"):
            await provider.exec(handle, "true", timeout_s=timeout_s)

        assert handle.raw.exec_calls == []


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


class TestFiles:
    async def test_upload_then_download_round_trip(self, tmp_path: Path) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())

        source = tmp_path / "in.bin"
        source.write_bytes(b"payload\x00binary")
        await provider.upload_file(handle, source, "/remote/in.bin")

        target = tmp_path / "nested" / "out.bin"
        await provider.download_file(handle, "/remote/in.bin", target)
        assert target.read_bytes() == b"payload\x00binary"

    async def test_upload_missing_file_raises(self, tmp_path: Path) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        with pytest.raises(FileNotFoundError):
            await provider.upload_file(handle, tmp_path / "nope.txt", "/remote/nope.txt")


# --------------------------------------------------------------------------
# Retries
# --------------------------------------------------------------------------


class TestRetries:
    async def test_transient_create_failure_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        async def flaky(cls, **kwargs):
            calls["n"] += 1
            raise RuntimeError("502 bad gateway")

        monkeypatch.setattr(FakeSandbox, "create", classmethod(flaky))
        provider = E2BProvider(create={"template": "base"}, operations={"retries": 3, "retry_delay_s": 0})
        with pytest.raises(E2BCreateError, match="502 bad gateway"):
            await provider.create(_spec())
        assert calls["n"] == 1

    async def test_deterministic_errors_are_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        async def not_found(cls, **kwargs):
            calls["n"] += 1
            raise FakeSandboxNotFound("template missing")

        monkeypatch.setattr(FakeSandbox, "create", classmethod(not_found))
        provider = E2BProvider(create={"template": "base"}, operations={"retries": 5, "retry_delay_s": 0})
        with pytest.raises(E2BCreateError):
            await provider.create(_spec())
        assert calls["n"] == 1


# --------------------------------------------------------------------------
# Job attribution
# --------------------------------------------------------------------------

_ATTRIBUTION_ENV_VARS = (
    "NEMO_GYM_TEAM",
    "NEMO_GYM_USER",
    "NEMO_GYM_WORKLOAD",
    "NEMO_GYM_RUN_ID",
    "NEMO_GYM_CONFIG_PATH",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_USER",
    "SLURM_JOB_NAME",
)


class TestAttribution:
    @pytest.fixture(autouse=True)
    def attribution_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in _ATTRIBUTION_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("NEMO_GYM_TEAM", "gym-team")
        monkeypatch.setenv("NEMO_GYM_USER", "alice")
        monkeypatch.setenv("NEMO_GYM_WORKLOAD", "swe-bench")
        monkeypatch.setenv("NEMO_GYM_RUN_ID", "run-123")

    async def test_create_sends_attribution_by_default_and_logs_it_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(attribution, "_logged_attribution", False)
        provider = E2BProvider(connection={"api_key": "secret-key"}, create={"template": "base"})
        with caplog.at_level(logging.INFO, logger=attribution.LOGGER.name):
            first = await provider.create(_spec(metadata={"purpose": "test"}))
            second = await provider.create(_spec())

        expected = {
            "nemo-gym.nvidia.com/team": "gym-team",
            "nemo-gym.nvidia.com/user": "alice",
            "nemo-gym.nvidia.com/workload": "swe-bench",
            "nemo-gym.nvidia.com/run": "run-123",
        }
        assert first.raw.create_kwargs["metadata"] == {**expected, "purpose": "test"}
        assert second.raw.create_kwargs["metadata"] == expected
        [record] = [r for r in caplog.records if "Sandbox attribution metadata" in r.getMessage()]
        assert "run-123" in record.getMessage()
        assert "secret-key" not in caplog.text

    async def test_explicit_metadata_wins_over_attribution(self) -> None:
        provider = E2BProvider(create={"template": "base"}, attribution={"team": "cfg-team", "run": "cfg-run"})
        handle = await provider.create(_spec(metadata={"nemo-gym.nvidia.com/team": "explicit-team"}))
        assert handle.raw.create_kwargs["metadata"] == {
            "nemo-gym.nvidia.com/team": "explicit-team",
            "nemo-gym.nvidia.com/user": "alice",
            "nemo-gym.nvidia.com/workload": "swe-bench",
            "nemo-gym.nvidia.com/run": "cfg-run",
        }

    async def test_disabled_attribution_sends_only_spec_metadata(self) -> None:
        provider = E2BProvider(create={"template": "base"}, attribution={"enabled": False, "team": "cfg-team"})
        explicit = await provider.create(_spec(metadata={"purpose": "test"}))
        bare = await provider.create(_spec())
        assert explicit.raw.create_kwargs["metadata"] == {"purpose": "test"}
        assert "metadata" not in bare.raw.create_kwargs

    async def test_key_prefix_names_the_keys(self) -> None:
        provider = E2BProvider(create={"template": "base"}, attribution={"key_prefix": ""})
        handle = await provider.create(_spec())
        assert handle.raw.create_kwargs["metadata"] == {
            "team": "gym-team",
            "user": "alice",
            "workload": "swe-bench",
            "run": "run-123",
        }

    @pytest.mark.parametrize(
        ("config", "message"),
        [({"key_prefix": "Bad Prefix/"}, "key_prefix"), ({"tean": "typo"}, "Unknown AttributionConfig keys")],
    )
    def test_invalid_attribution_config_is_rejected(self, config: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            E2BProvider(attribution=config)


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------


def test_unknown_config_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown E2BCreateConfig keys"):
        E2BProvider(create={"template": "base", "nope": 1})


@pytest.mark.parametrize(
    "create",
    [
        {"template": ""},
        {"template_map": {"": "template"}},
        {"template_map": {"image": ""}},
        {"template_map": []},
    ],
)
def test_invalid_template_config_is_rejected(create: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError), match=r"create\.template"):
        E2BProvider(create=create)


@pytest.mark.parametrize(
    ("create", "message"),
    [
        ({"network": ["0.0.0.0/0"]}, "create.network must be a mapping"),
        ({"network": {"allow": ["pypi.org"]}}, "Unknown create.network keys: allow"),
        ({"network": {"deny_out": "0.0.0.0/0"}}, "create.network.deny_out must be a list of strings"),
        ({"custom_extension_params": "sandbank"}, "create.custom_extension_params must be a mapping"),
    ],
)
def test_invalid_network_config_is_rejected(create: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=re.escape(message)):
        E2BProvider(create=create)


@pytest.mark.parametrize(
    ("section", "config", "message"),
    [
        ("connection", {"request_timeout_s": -1}, "connection.request_timeout_s must be >= 0"),
        ("connection", {"request_timeout_s": float("nan")}, "connection.request_timeout_s must be >= 0"),
        ("create", {"timeout_s": 0}, "create.timeout_s must be > 0"),
        ("create", {"timeout_s": float("inf")}, "create.timeout_s must be > 0"),
        ("exec", {"default_timeout_s": -1}, "exec.default_timeout_s must be >= 0"),
        ("exec", {"request_timeout_s": -1}, "exec.request_timeout_s must be >= 0"),
        ("exec", {"reconnect_attempts": -1}, "exec.reconnect_attempts must be >= 0"),
        ("exec", {"reconnect_attempts": 1.5}, "exec.reconnect_attempts must be >= 0"),
        ("operations", {"retries": -1}, "operations.retries must be >= 0"),
        ("operations", {"retries": True}, "operations.retries must be >= 0"),
        ("operations", {"retry_delay_s": -1}, "operations.retry_delay_s must be >= 0"),
        ("operations", {"retry_delay_s": None}, "operations.retry_delay_s must be >= 0"),
        ("operations", {"retry_max_delay_s": -1}, "operations.retry_max_delay_s must be >= 0"),
        ("operations", {"retry_max_delay_s": float("nan")}, "operations.retry_max_delay_s must be >= 0"),
    ],
)
def test_invalid_config_values_are_rejected(section: str, config: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        E2BProvider(**{section: config})


async def test_public_pause_resume_refreshes_e2b_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BProvider(
        connection={"api_key": "test-key", "api_url": "https://gateway.example", "request_timeout_s": 12},
        create={"template": "base"},
    )
    assert isinstance(provider, SupportsSandboxPauseResume)
    async with AsyncSandbox(provider, _spec()) as sandbox:
        await sandbox.start()
        original = FakeSandbox.instances[-1]
        health_check = AsyncMock(side_effect=AssertionError("Status must not wake a paused sandbox"))
        monkeypatch.setattr(original, "is_running", health_check)
        await sandbox.pause()
        assert original.pause_calls == [{"request_timeout": 12}]
        assert await sandbox.status() is SandboxStatus.PAUSED
        health_check.assert_not_called()

        refreshed = FakeSandbox(sandbox_id=original.sandbox_id)
        refreshed.exec_behaviour = FakeCommandResult(stdout="resumed")
        connect = AsyncMock(return_value=refreshed)
        monkeypatch.setattr(FakeSandbox, "connect", connect)
        await sandbox.resume()
        connect.assert_awaited_once_with(
            original.sandbox_id,
            timeout=3600,
            api_key="test-key",
            api_url="https://gateway.example",
            request_timeout=12,
        )
        assert await sandbox.status() is SandboxStatus.RUNNING
        assert (await sandbox.exec("echo resumed")).stdout == "resumed"
        assert original.exec_calls == []
    assert refreshed.killed
    assert not original.killed


def test_sync_public_pause_resume() -> None:
    with Sandbox(E2BProvider(create={"template": "base"}), _spec()) as sandbox:
        sandbox.start()
        sandbox.pause()
        assert sandbox.status() is SandboxStatus.PAUSED
        sandbox.resume()
        assert sandbox.status() is SandboxStatus.RUNNING
        assert sandbox.exec("true").return_code == 0
    assert FakeSandbox.instances[-1].killed


@pytest.mark.parametrize("operation", ["pause", "resume"])
@pytest.mark.parametrize(
    "error", [FakeTimeout("deadline"), ConnectionError("lost"), FakeSandboxNotFound("gone"), asyncio.CancelledError()]
)
async def test_lifecycle_failure_is_not_retried_and_preserves_handle(
    monkeypatch: pytest.MonkeyPatch, operation: str, error: BaseException
) -> None:
    provider = E2BProvider(create={"template": "base"}, operations={"retries": 5, "retry_delay_s": 0})
    handle = await provider.create(_spec())
    original = handle.raw
    call = AsyncMock(side_effect=error)
    if operation == "pause":
        monkeypatch.setattr(original, "pause", call)
    else:
        monkeypatch.setattr(FakeSandbox, "connect", call)
    with pytest.raises(type(error)):
        await getattr(provider, operation)(handle)
    call.assert_awaited_once()
    assert handle.raw is original
    await provider.close(handle)
    assert original.killed


@pytest.mark.parametrize("operation", ["pause", "resume"])
async def test_lifecycle_rejects_closed_handle(operation: str) -> None:
    provider = E2BProvider(create={"template": "base"})
    handle = await provider.create(_spec())
    await provider.close(handle)
    with pytest.raises(RuntimeError, match="carries no e2b sandbox object"):
        await getattr(provider, operation)(handle)
    assert len(FakeSandbox.instances) == 1


async def test_pause_raises_when_the_sandbox_did_not_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BProvider(create={"template": "base"})
    handle = await provider.create(_spec())
    # AgentENV answers 409 during a snapshot or fork, while status reports the sandbox as paused.
    handle.raw.paused = True
    monkeypatch.setattr(handle.raw, "pause", AsyncMock(return_value=False))
    with pytest.raises(RuntimeError, match="refused pause with 409"):
        await provider.pause(handle)


async def test_resume_keeps_the_create_lifetime_and_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BProvider(create={"template": "base", "timeout_s": 1800})
    handle = await provider.create(_spec(ttl_s=60, ready_timeout_s=90))
    connect = AsyncMock(return_value=FakeSandbox(sandbox_id=handle.sandbox_id))
    monkeypatch.setattr(FakeSandbox, "connect", connect)
    await provider.resume(handle)
    # The provider lifetime applies: the handle does not keep the spec's ttl_s.
    connect.assert_awaited_once_with(handle.sandbox_id, timeout=1800, request_timeout=90.0)


async def test_resume_prefers_the_configured_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BProvider(connection={"request_timeout_s": 12}, create={"template": "base"})
    handle = await provider.create(_spec(ready_timeout_s=90))
    connect = AsyncMock(return_value=FakeSandbox(sandbox_id=handle.sandbox_id))
    monkeypatch.setattr(FakeSandbox, "connect", connect)
    await provider.resume(handle)
    # connection.request_timeout_s wins over the ready_timeout_s the create handle kept.
    connect.assert_awaited_once_with(handle.sandbox_id, timeout=3600, request_timeout=12)


@pytest.mark.parametrize(("timeout_s", "sent"), [(90.4, 91), (0.2, 1)])
async def test_resume_sends_whole_second_lifetime(
    monkeypatch: pytest.MonkeyPatch, timeout_s: float, sent: int
) -> None:
    provider = E2BProvider(create={"template": "base", "timeout_s": timeout_s})
    handle = await provider.create(_spec())
    connect = AsyncMock(return_value=FakeSandbox(sandbox_id=handle.sandbox_id))
    monkeypatch.setattr(FakeSandbox, "connect", connect)
    await provider.resume(handle)
    assert connect.await_args.kwargs["timeout"] == sent
    assert type(connect.await_args.kwargs["timeout"]) is int


async def test_create_kills_at_timeout_by_default() -> None:
    # E2B 2.42+ omits lifecycle fields unless asked, and AgentENV then pauses timed-out
    # sandboxes forever; the default must request deletion explicitly.
    provider = E2BProvider(create={"template": "base"})
    handle = await provider.create(_spec(ttl_s=60))
    assert handle.raw.create_kwargs["lifecycle"] == {"on_timeout": "kill", "auto_resume": False}


@pytest.mark.parametrize("auto_resume", [None, True, False])
async def test_create_lifecycle_policy(auto_resume: bool | None) -> None:
    provider = E2BProvider(create={"template": "base", "auto_resume": auto_resume})
    handle = await provider.create(_spec(ttl_s=12.5))
    assert handle.raw.create_kwargs["timeout"] == 13
    if auto_resume is None:
        assert "lifecycle" not in handle.raw.create_kwargs
    else:
        assert handle.raw.create_kwargs["lifecycle"] == {
            "on_timeout": "pause" if auto_resume else "kill",
            "auto_resume": auto_resume,
        }


@pytest.mark.parametrize("invalid", [0, 1, "false", "true", [], {}])
def test_create_auto_resume_rejects_non_boolean(invalid: object) -> None:
    with pytest.raises(ValueError, match="create.auto_resume must be a boolean or None"):
        E2BProvider(create={"auto_resume": invalid})


@pytest.mark.parametrize("auto_resume", [True, False])
@pytest.mark.parametrize("custom_extension_params", [None, {"sandbank": "all"}])
async def test_real_sdk_serializes_lifecycle_policy(
    monkeypatch: pytest.MonkeyPatch, auto_resume: bool, custom_extension_params: dict | None
) -> None:
    pytest.importorskip("e2b")
    import json

    import httpx

    from nemo_gym.sandbox.providers._http_transport import GymAiohttpTransport

    bodies = []

    async def capture_request(self, request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        raise ConnectionError("captured before network")

    # Install the Gym transport here, so this test never depends on test order to stay off the network.
    monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "{}")
    monkeypatch.setattr(e2b_sdk, "_CONFIGURED_SDK_MODULES", {})
    monkeypatch.setattr(e2b_provider, "_require_e2b_sdk", _REAL_PROVIDER_REQUIRE_E2B_SDK)
    monkeypatch.setattr(GymAiohttpTransport, "handle_async_request", capture_request)
    provider = E2BProvider(
        connection={"api_key": "test-key", "validate_api_key": False},
        create={"template": "base", "auto_resume": auto_resume, "custom_extension_params": custom_extension_params},
    )
    # The SDK create and the REST create (custom_extension_params) send the same lifecycle.
    with pytest.raises(E2BCreateError, match="captured before network"):
        await provider.create(_spec(ttl_s=7))
    assert len(bodies) == 1
    assert bodies[0]["autoPause"] is auto_resume
    assert bodies[0]["autoResume"] == {"enabled": auto_resume}
    assert bodies[0]["timeout"] == 7


@pytest.mark.parametrize(
    ("state", "expected"),
    [("running", SandboxStatus.RUNNING), ("paused", SandboxStatus.PAUSED), ("resuming", SandboxStatus.UNKNOWN)],
)
async def test_status_reads_control_plane_without_waking_sandbox(
    monkeypatch: pytest.MonkeyPatch, state: str, expected: SandboxStatus
) -> None:
    provider = E2BProvider(connection={"request_timeout_s": 12}, create={"template": "base"})
    handle = await provider.create(_spec())

    async def get_info(**kwargs):
        assert kwargs == {"request_timeout": 12}
        return types.SimpleNamespace(state=state)

    async def forbidden_health_check(**kwargs):
        pytest.fail("Status must not send traffic that resumes a paused sandbox")

    monkeypatch.setattr(handle.raw, "get_info", get_info)
    monkeypatch.setattr(handle.raw, "is_running", forbidden_health_check)
    assert await provider.status(handle) is expected


async def test_status_control_plane_failure_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = E2BProvider(create={"template": "base"})
    handle = await provider.create(_spec())

    async def unavailable(**kwargs):
        raise ConnectionError("control plane unavailable")

    monkeypatch.setattr(handle.raw, "get_info", unavailable)
    assert await provider.status(handle) is SandboxStatus.UNKNOWN


@pytest.mark.parametrize(
    ("state", "expected"),
    [("running", SandboxStatus.RUNNING), ("paused", SandboxStatus.PAUSED), ("gone", SandboxStatus.STOPPED)],
)
async def test_real_sdk_status_only_reads_control_plane(
    monkeypatch: pytest.MonkeyPatch, state: str, expected: SandboxStatus
) -> None:
    pytest.importorskip("e2b")
    import json

    import httpx

    from nemo_gym.sandbox.providers._http_transport import GymAiohttpTransport

    sandbox = {"templateID": "base", "sandboxID": "sbx-1", "clientID": "client", "envdVersion": "0.2.0"}
    requests: list[tuple[str, str]] = []

    async def fake_api(self, request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path.endswith("/sandboxes"):
            return httpx.Response(201, json=sandbox, request=request)
        if request.method == "GET" and request.url.path.endswith("/sandboxes/sbx-1"):
            if state == "gone":
                # Killed at its timeout: the SDK's 404 must read as STOPPED, not UNKNOWN.
                return httpx.Response(404, json={"code": 404, "message": "sandbox not found"}, request=request)
            detail = {
                **sandbox,
                "startedAt": "2026-01-01T00:00:00Z",
                "endAt": "2026-01-01T01:00:00Z",
                "cpuCount": 2,
                "memoryMB": 1024,
                "diskSizeMB": 1024,
                "state": state,
            }
            return httpx.Response(200, content=json.dumps(detail), request=request)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    # Install the Gym transport here, so this test never depends on test order to stay off the network.
    monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, "{}")
    monkeypatch.setattr(e2b_sdk, "_CONFIGURED_SDK_MODULES", {})
    monkeypatch.setattr(e2b_provider, "_require_e2b_sdk", _REAL_PROVIDER_REQUIRE_E2B_SDK)
    monkeypatch.setattr(GymAiohttpTransport, "handle_async_request", fake_api)
    provider = E2BProvider(connection={"api_key": "test-key", "validate_api_key": False}, create={"template": "base"})
    handle = await provider.create(_spec())
    requests.clear()

    assert await provider.status(handle) is expected
    # Only the control-plane GET: an envd health check would resume a paused sandbox.
    assert [method for method, _ in requests] == ["GET"]
    assert requests[0][1].endswith("/sandboxes/sbx-1")


# --------------------------------------------------------------------------
# Snapshots -- AgentENV saves the whole VM, processes included
# --------------------------------------------------------------------------


class TestSnapshots:
    def test_the_provider_snapshots_and_keeps_processes(self) -> None:
        provider = E2BProvider()
        assert isinstance(provider, SupportsSandboxSnapshot)
        assert getattr(provider, "snapshot_keeps_processes", False) is True

    async def test_snapshot_returns_the_id_and_sends_no_name_by_default(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())

        snapshot_id = await provider.snapshot(handle, name="ng-run-ckpt-key-1")

        assert snapshot_id == "tmpl1:default"
        assert handle.raw.snapshot_names == [None], "a name would become a team-wide template alias"
        await provider.wait_snapshot_ready(snapshot_id)  # usable at once: nothing to wait for

    async def test_snapshot_sends_the_name_when_configured(self) -> None:
        provider = E2BProvider(create={"template": "base"}, snapshots={"send_names": True})
        handle = await provider.create(_spec())

        await provider.snapshot(handle, name="ng-run-ckpt-key-1")

        assert handle.raw.snapshot_names == ["ng-run-ckpt-key-1"]

    @pytest.mark.parametrize("gone", [True, False])
    async def test_a_snapshot_answered_with_an_empty_404_names_the_sandbox_gone(
        self, monkeypatch: pytest.MonkeyPatch, gone: bool
    ) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())

        async def unparsed_reply(name=None, **kwargs):
            raise _unparsed_404()

        monkeypatch.setattr(handle.raw, "create_snapshot", unparsed_reply)
        FakeApiClient.get_response = FakeHttpResponse(404 if gone else 200, "")

        if gone:
            with pytest.raises(E2BSnapshotError, match="sbx-1' is gone"):
                await provider.snapshot(handle)
        else:
            with pytest.raises(json.JSONDecodeError):
                await provider.snapshot(handle)

    async def test_snapshot_needs_an_sdk_that_can_create_snapshots(self) -> None:
        provider = E2BProvider()
        handle = SandboxHandle(sandbox_id="sbx-old", provider_name="e2b", raw=types.SimpleNamespace())
        with pytest.raises(E2BSnapshotError, match="create_snapshot"):
            await provider.snapshot(handle)

    async def test_create_from_snapshot_starts_from_the_snapshot_template(self) -> None:
        provider = E2BProvider(create={"template": "base", "auto_resume": False}, attribution={"enabled": False})

        handle = await provider.create_from_snapshot("tmpl7:default", _spec(ttl_s=300, metadata={"run": "1"}))

        kwargs = handle.raw.create_kwargs
        assert kwargs["template"] == "tmpl7:default"
        assert kwargs["timeout"] == 300 and kwargs["metadata"] == {"run": "1"}
        assert kwargs["lifecycle"] == {"on_timeout": "kill", "auto_resume": False}

    @pytest.mark.parametrize(
        "snapshot_id", ["missing-typed:default", "missing-plain-text:default", "missing-bad-request:default"]
    )
    async def test_create_from_a_missing_snapshot_raises_snapshot_not_found(self, snapshot_id: str) -> None:
        provider = E2BProvider()
        with pytest.raises(SandboxSnapshotNotFoundError, match=snapshot_id):
            await provider.create_from_snapshot(snapshot_id, _spec())

    async def test_another_bad_request_from_a_snapshot_is_not_a_missing_snapshot(self) -> None:
        provider = E2BProvider()
        with pytest.raises(E2BCreateError, match="invalid metadata") as raised:
            await provider.create_from_snapshot("invalid:default", _spec())
        assert not isinstance(raised.value, SandboxSnapshotNotFoundError)

    async def test_restoring_through_the_sandbox_facade_forks_the_snapshot(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        source = await provider.create(_spec())
        snapshot_id = await provider.snapshot(source)

        sandbox = await AsyncSandbox(provider, _spec(), owns_provider=False).start(snapshot_id=snapshot_id)

        assert FakeSandbox.instances[-1].create_kwargs["template"] == snapshot_id
        assert sandbox.sandbox_id == FakeSandbox.instances[-1].sandbox_id

    async def test_delete_snapshot_deletes_and_tolerates_a_snapshot_already_gone(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        snapshot_id = await provider.snapshot(await provider.create(_spec()))

        await provider.delete_snapshot(snapshot_id)
        await provider.delete_snapshot(snapshot_id)  # the SDK answers False for a 404
        FakeSandbox.delete_snapshot_outcomes.append(FakeSdkError("404: b'not found'", status_code=404))
        await provider.delete_snapshot(snapshot_id)  # AgentENV's plain-text 404

        assert FakeSandbox.snapshots == {}
        assert FakeSandbox.delete_snapshot_calls == [snapshot_id] * 3

    async def test_delete_snapshot_retries_a_transient_failure_and_raises_a_persistent_one(self) -> None:
        provider = E2BProvider(operations={"retries": 1, "retry_delay_s": 0.0})
        FakeSandbox.delete_snapshot_outcomes.extend([ConnectionError("reset"), False])
        await provider.delete_snapshot("tmpl1:default")

        FakeSandbox.delete_snapshot_outcomes.extend([FakeSdkError("500: boom", 500), FakeSdkError("500: boom", 500)])
        with pytest.raises(FakeSdkError, match="500"):
            await provider.delete_snapshot("tmpl1:default")

    async def test_list_snapshots_filters_names_without_namespace_or_tag(self) -> None:
        provider = E2BProvider(create={"template": "base"}, snapshots={"send_names": True})
        handle = await provider.create(_spec())
        kept = await provider.snapshot(handle, name="ng-run1-a")
        await provider.snapshot(handle, name="other-run")
        FakeSandbox.snapshots["tmpl-unnamed:default"] = {"names": [], "sandbox_id": "x"}

        assert await provider.list_snapshots("ng-run1-") == [{"id": kept, "name": "ng-run1-a"}]
        assert {entry["id"] for entry in await provider.list_snapshots("")} == set(FakeSandbox.snapshots)


# --------------------------------------------------------------------------
# Lease -- a sandbox whose creator died ends one lease later
# --------------------------------------------------------------------------


class TestLease:
    async def test_without_a_lease_the_sandbox_gets_its_full_lifetime_and_no_renewal(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec(ttl_s=600))
        assert handle.raw.create_kwargs["timeout"] == 600
        await provider.renew_leases()
        assert handle.raw.set_timeout_calls == []

    async def test_a_leased_sandbox_is_created_with_the_lease_and_renewed_to_it(self) -> None:
        provider = E2BProvider(create={"template": "base", "lease_s": 60.0})
        handle = await provider.create(_spec(ttl_s=600))
        try:
            assert handle.raw.create_kwargs["timeout"] == 60
            await provider.renew_leases()
            assert handle.raw.set_timeout_calls == [60]
        finally:
            await provider.aclose()

    async def test_renewal_never_extends_past_the_sandbox_lifetime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = [1000.0]
        monkeypatch.setattr(e2b_provider, "monotonic", lambda: clock[0])
        provider = E2BProvider(create={"template": "base", "lease_s": 60.0})
        handle = await provider.create(_spec(ttl_s=100))
        try:
            clock[0] += 70  # 30 s of lifetime left
            await provider.renew_leases()
            clock[0] += 40  # lifetime over: no renewal, the lease is dropped
            await provider.renew_leases()
            assert handle.raw.set_timeout_calls == [30]
            assert provider._leases == {}
        finally:
            await provider.aclose()

    async def test_a_sandbox_gone_or_closed_is_no_longer_renewed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ids = iter(["sbx-gone", "sbx-closed", "sbx-kept"])

        async def create(**kwargs):
            return FakeSandbox(sandbox_id=next(ids), **kwargs)

        monkeypatch.setattr(FakeSandbox, "create", create)
        provider = E2BProvider(create={"template": "base", "lease_s": 60.0})
        gone = await provider.create(_spec())
        closed = await provider.create(_spec())
        kept = await provider.create(_spec())
        gone_sandbox, closed_sandbox = gone.raw, closed.raw
        try:
            gone_sandbox.set_timeout_outcomes.append(FakeSandboxNotFound("gone"))
            kept.raw.set_timeout_outcomes.append(ConnectionError("blip"))  # retried next round
            await provider.close(closed)
            await provider.renew_leases()
            await provider.renew_leases()
            assert gone_sandbox.set_timeout_calls == [60]
            assert closed_sandbox.set_timeout_calls == []
            assert kept.raw.set_timeout_calls == [60, 60]
        finally:
            await provider.aclose()

    @pytest.mark.parametrize("gone", [True, False])
    async def test_a_renewal_answered_with_an_empty_404_stops_only_for_a_sandbox_gone(self, gone: bool) -> None:
        provider = E2BProvider(create={"template": "base", "lease_s": 60.0})
        handle = await provider.create(_spec())
        try:
            handle.raw.set_timeout_outcomes.append(_unparsed_404())
            FakeApiClient.get_response = FakeHttpResponse(404 if gone else 200, "")
            await provider.renew_leases()
            await provider.renew_leases()
            assert handle.raw.set_timeout_calls == ([60] if gone else [60, 60])
        finally:
            await provider.aclose()

    async def test_aclose_stops_the_renewal_loop(self) -> None:
        provider = E2BProvider(create={"template": "base", "lease_s": 60.0})
        await provider.create(_spec())
        task = provider._lease_task
        assert task is not None and not task.done()
        await provider.aclose()
        assert task.done() and provider._leases == {}

    async def test_the_renewal_loop_renews_every_third_of_a_lease(self) -> None:
        provider = E2BProvider(create={"template": "base", "lease_s": 0.03})
        handle = await provider.create(_spec())
        try:
            for _ in range(200):
                if len(handle.raw.set_timeout_calls) >= 2:
                    break
                await asyncio.sleep(0.01)
            assert len(handle.raw.set_timeout_calls) >= 2
        finally:
            await provider.aclose()

    def test_lease_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="lease_s"):
            E2BProvider(create={"lease_s": 0})


# --------------------------------------------------------------------------
# Provider options the SWE servers and OpenSandbox-shaped configs carry
# --------------------------------------------------------------------------


class TestProviderOptions:
    async def test_resource_requests_are_reported_once_and_not_sent(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = E2BProvider(create={"template": "base"})
        options = {"resource_requests": {"cpu": 2}}
        with caplog.at_level("WARNING"):
            first = await provider.create(_spec(provider_options=options))
            await provider.create(_spec(provider_options=options))
        assert sum("resource_requests" in record.getMessage() for record in caplog.records) == 1
        assert "resource_requests" not in first.raw.create_kwargs

    async def test_a_deny_by_default_network_policy_maps_to_deny_all_but_the_allowed(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        policy = {"defaultAction": "deny", "egress": [{"action": "allow", "target": "10.0.0.5"}]}
        handle = await provider.create(_spec(provider_options={"network_policy": policy}))
        assert handle.raw.create_kwargs["network"] == {"deny_out": ["0.0.0.0/0"], "allow_out": ["10.0.0.5"]}

    async def test_an_allow_by_default_network_policy_denies_only_the_denied(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        policy = {"defaultAction": "allow", "egress": [{"action": "deny", "target": "169.254.169.254"}]}
        handle = await provider.create(_spec(provider_options={"network_policy": policy}))
        assert handle.raw.create_kwargs["network"] == {"deny_out": ["169.254.169.254"]}

    @pytest.mark.parametrize(
        "policy",
        [{"defaultAction": "block"}, {"egress": [{"action": "allow"}]}, {"ingress": []}, "deny"],
    )
    async def test_a_network_policy_it_cannot_express_is_refused(self, policy: object) -> None:
        provider = E2BProvider(create={"template": "base"})
        with pytest.raises(ValueError, match="network_policy"):
            await provider.create(_spec(provider_options={"network_policy": policy}))
        assert FakeSandbox.instances == []


# --------------------------------------------------------------------------
# Command deadlines, leftovers and unreachable sandboxes
# --------------------------------------------------------------------------


class TestCommandDeadline:
    async def test_every_command_carries_a_marker_for_its_leftover_processes(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        await provider.exec(handle, "true")
        await provider.exec(handle, "true")
        markers = [call["envs"]["NEMO_GYM_E2B_EXEC"] for call in handle.raw.exec_calls]
        assert len(set(markers)) == 2 and all(re.fullmatch(r"[0-9a-f]{32}", marker) for marker in markers)

    async def test_a_command_that_runs_out_its_deadline_exits_124_and_its_leftovers_are_killed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exec start, run start, then the timeout seen at 99.5% of 60 s by the run (not reattached) and by exec.
        clock = iter([0.0, 0.0, 59.7, 59.7])
        monkeypatch.setattr(e2b_provider, "monotonic", lambda: next(clock))
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [FakeTimeout("deadline exceeded")]

        result = await provider.exec(handle, "sleep 999", timeout_s=60)

        assert (result.return_code, result.error_type) == (124, "timeout")
        assert handle.raw.connect_calls == [], "a timeout at the deadline is final, not reattached"
        marker = handle.raw.exec_calls[0]["envs"]["NEMO_GYM_E2B_EXEC"]
        cleanup = handle.raw.exec_calls[1]
        assert cleanup["user"] == "root" and f"NEMO_GYM_E2B_EXEC={marker}" in cleanup["cmd"]
        assert "kill -- -" not in cleanup["cmd"], "never by process group: envd shares it"
        assert "NEMO_GYM_E2B_EXEC" not in (cleanup.get("envs") or {}), "the cleanup must not kill itself"

    async def test_the_leftover_kill_script_kills_only_marked_processes(self, tmp_path: Path) -> None:
        """Run the real script against this machine's /proc: a marked child dies, an unmarked one survives."""
        import os
        import shutil
        import subprocess

        if shutil.which("bash") is None or not Path("/proc/self/environ").exists():
            pytest.skip("needs bash and /proc")
        marker = "NEMO_GYM_E2B_EXEC=test" + os.urandom(4).hex()
        marked = subprocess.Popen(["sleep", "60"], env={**os.environ, marker.split("=")[0]: marker.split("=")[1]})
        unmarked = subprocess.Popen(["sleep", "60"])
        try:
            script = e2b_provider._KILL_MARKED_SCRIPT.format(marker=marker)
            subprocess.run(["bash", "-c", script], check=True, timeout=30)
            assert marked.wait(timeout=10) == -9
            assert unmarked.poll() is None
        finally:
            for process in (marked, unmarked):
                process.kill()
                process.wait()

    async def test_early_timeouts_past_the_reattach_attempts_stay_an_error(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [FakeTimeout("stream open timed out")] * 3
        with pytest.raises(TimeoutError, match="stream open timed out"):
            await provider.exec(handle, "true", timeout_s=60)
        assert len(handle.raw.connect_calls) == 2
        assert len(handle.raw.exec_calls) == 1, "no leftover kill for a command that did not run out its deadline"

    async def test_an_unreachable_sandbox_is_its_own_error_on_exec(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [FakeUnreachable("sandbox could not be reached")] * 3
        with pytest.raises(E2BSandboxUnreachableError, match="could not reach"):
            await provider.exec(handle, "true")
        assert len(handle.raw.connect_calls) == 2, "reattached by pid, never started again"
        assert len(handle.raw.exec_calls) == 1

    async def test_idempotent_operations_retry_an_unreachable_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = E2BProvider(operations={"retries": 2, "retry_delay_s": 0.0})
        calls = []

        async def flaky_connect(sandbox_id, **kwargs):
            calls.append(sandbox_id)
            if len(calls) == 1:
                raise FakeUnreachable("sandbox could not be reached")
            if len(calls) == 2:
                raise FakeSdkError("502: sandbox could not be reached", status_code=502)
            return FakeSandbox(sandbox_id=sandbox_id)

        monkeypatch.setattr(FakeSandbox, "connect", flaky_connect)
        handle = await provider.connect({"sandbox_id": "sbx-9"})
        assert handle.sandbox_id == "sbx-9" and len(calls) == 3


async def test_create_waits_until_a_new_sandbox_is_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    # AgentENV can report a sandbox created while its gateway still answers 502 "could not be reached".
    monkeypatch.setattr(e2b_provider, "REACHABLE_RETRY_DELAY_S", 0.0)
    outcomes = [FakeUnreachable("502: The sandbox could not be reached"), FakeUnreachable("502: unreachable")]
    original_run = FakeCommands.run

    async def flaky_run(self, **kwargs):
        if kwargs.get("cmd") == "true" and outcomes:
            self._sandbox.exec_calls.append(kwargs)
            raise outcomes.pop(0)
        return await original_run(self, **kwargs)

    monkeypatch.setattr(FakeCommands, "run", flaky_run)
    provider = E2BProvider(create={"template": "base", "reachable_timeout_s": 30})
    handle = await provider.create(_spec(ttl_s=60))
    assert [call["cmd"] for call in handle.raw.exec_calls] == ["true", "true", "true"]
    assert not handle.raw.killed


async def test_a_sandbox_still_unreachable_at_the_deadline_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e2b_provider, "REACHABLE_RETRY_DELAY_S", 0.0)
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(e2b_provider, "monotonic", lambda: next(clock))

    async def unreachable(self, **kwargs):
        self._sandbox.exec_calls.append(kwargs)
        raise FakeUnreachable("502: The sandbox could not be reached")

    monkeypatch.setattr(FakeCommands, "run", unreachable)
    provider = E2BProvider(create={"template": "base", "reachable_timeout_s": 25})
    with pytest.raises(E2BCreateError, match="could not be reached within 25s"):
        await provider.create(_spec(ttl_s=60))
    sandbox = FakeSandbox.instances[-1]
    assert sandbox.killed
    assert 1 <= len(sandbox.exec_calls) <= 3


async def test_only_unreachable_is_waited_out_after_a_create(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing(self, **kwargs):
        raise FakeSdkError("500: envd crashed", status_code=500)

    monkeypatch.setattr(FakeCommands, "run", failing)
    provider = E2BProvider(create={"template": "base", "reachable_timeout_s": 30})
    with pytest.raises(E2BCreateError, match="envd crashed"):
        await provider.create(_spec(ttl_s=60))
    assert FakeSandbox.instances[-1].killed


async def test_without_a_reachable_timeout_create_runs_no_command() -> None:
    handle = await E2BProvider(create={"template": "base"}).create(_spec(ttl_s=60))
    assert handle.raw.exec_calls == []


# --------------------------------------------------------------------------
# A new sandbox's first seconds: AgentENV can answer 404 "assignment not found" for a sandbox that runs
# --------------------------------------------------------------------------


class TestNewSandboxGrace:
    async def test_the_probe_waits_out_a_404_and_create_keeps_the_lease(self, monkeypatch, caplog) -> None:
        _with_grace(monkeypatch)
        monkeypatch.setattr(e2b_provider, "REACHABLE_RETRY_DELAY_S", 0.0)
        outcomes = [FakeSandboxNotFound("sandbox assignment not found"), _unparsed_404()]
        original_run = FakeCommands.run

        async def flaky_run(self, **kwargs):
            if kwargs.get("cmd") == "true" and outcomes:
                self._sandbox.exec_calls.append(kwargs)
                raise outcomes.pop(0)
            return await original_run(self, **kwargs)

        monkeypatch.setattr(FakeCommands, "run", flaky_run)
        provider = E2BProvider(create={"template": "base", "reachable_timeout_s": 30, "lease_s": 60})
        try:
            with caplog.at_level("INFO"):
                handle = await provider.create(_spec(ttl_s=600))
            assert [call["cmd"] for call in handle.raw.exec_calls] == ["true", "true", "true"]
            assert not handle.raw.killed and handle.sandbox_id in provider._leases
            assert sum("not routable yet" in r.getMessage() for r in caplog.records) == 2
        finally:
            await provider.aclose()

    async def test_an_unreachable_new_sandbox_is_killed_once_the_kill_finds_it(self, monkeypatch, caplog) -> None:
        _with_grace(monkeypatch)
        monkeypatch.setattr(e2b_provider, "REACHABLE_RETRY_DELAY_S", 0.0)

        async def not_found(self, **kwargs):
            raise FakeSdkError("500: envd crashed", status_code=500)

        monkeypatch.setattr(FakeCommands, "run", not_found)
        original_create = FakeSandbox.create

        async def create(**kwargs):
            sandbox = await original_create(**kwargs)
            sandbox.kill_outcomes = [_unparsed_404(), _unparsed_404(), True]
            return sandbox

        monkeypatch.setattr(FakeSandbox, "create", create)
        provider = E2BProvider(create={"template": "base", "reachable_timeout_s": 30}, operations={"retries": 0})
        with caplog.at_level("WARNING"), pytest.raises(E2BCreateError, match="could not be reached"):
            await provider.create(_spec(ttl_s=60))
        sandbox = FakeSandbox.instances[-1]
        assert sandbox.killed and len(sandbox.kill_calls) == 3
        assert FakeApiClient.get_calls == [], "no gone verdict from a lookup during the grace period"
        assert any(
            "did not become reachable" in r.getMessage() and "envd crashed" in r.getMessage() for r in caplog.records
        )

    async def test_close_in_the_grace_period_retries_a_kill_answered_with_an_empty_404(self, monkeypatch) -> None:
        _with_grace(monkeypatch)
        provider = E2BProvider(create={"template": "base"}, operations={"retries": 0})
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [_unparsed_404(), False, True]
        FakeApiClient.get_response = FakeHttpResponse(404, "")

        await provider.close(handle)

        assert sandbox.killed and len(sandbox.kill_calls) == 3 and handle.raw is None
        assert FakeApiClient.get_calls == []

    async def test_after_the_grace_period_an_empty_404_lookup_settles_gone(self, monkeypatch) -> None:
        _with_grace(monkeypatch)
        provider = E2BProvider(create={"template": "base"}, operations={"retries": 0, "new_sandbox_grace_s": 0.0})
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [_unparsed_404()]
        sandbox.running = False
        FakeApiClient.get_response = FakeHttpResponse(404, "")

        await provider.close(handle)

        assert len(sandbox.kill_calls) == 1 and not sandbox.killed and handle.raw is None

    async def test_a_restored_sandbox_gets_the_grace_period(self, monkeypatch) -> None:
        _with_grace(monkeypatch)
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create_from_snapshot("tmpl7:default", _spec())
        sandbox = handle.raw
        # Not found at first, and its status says so too: only the grace period makes close try again.
        sandbox.kill_outcomes = [False, True]
        sandbox.running = False

        await provider.close(handle)

        assert handle.raw is None and sandbox.killed and len(sandbox.kill_calls) == 2

    async def test_status_of_a_new_sandbox_not_found_is_unknown(self, monkeypatch) -> None:
        _with_grace(monkeypatch)
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        handle.raw.running = False
        assert await provider.status(handle) is SandboxStatus.UNKNOWN

    async def test_a_snapshot_in_the_grace_period_is_not_declared_gone(self, monkeypatch) -> None:
        _with_grace(monkeypatch)
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())

        async def unparsed_reply(name=None, **kwargs):
            raise _unparsed_404()

        monkeypatch.setattr(handle.raw, "create_snapshot", unparsed_reply)
        FakeApiClient.get_response = FakeHttpResponse(404, "")
        with pytest.raises(json.JSONDecodeError):
            await provider.snapshot(handle)
        assert FakeApiClient.get_calls == []


class TestLeaseRenewalNoise:
    async def test_a_lease_dropped_during_a_round_is_not_renewed(self) -> None:
        provider = E2BProvider(create={"template": "base", "lease_s": 60.0})
        handle = await provider.create(_spec())
        try:
            [(sandbox_id, lease)] = list(provider._leases.items())  # the round's snapshot of held leases
            provider._leases.pop(sandbox_id)  # then a close drops it before its batch comes up
            await provider._renew_lease(sandbox_id, lease)
            assert handle.raw.set_timeout_calls == []
        finally:
            await provider.aclose()

    async def test_a_sandbox_being_killed_drops_its_lease_at_info(self, caplog) -> None:
        provider = E2BProvider(create={"template": "base", "lease_s": 60.0})
        handle = await provider.create(_spec())
        try:
            handle.raw.set_timeout_outcomes.append(
                FakeSdkError(f"500: sandbox {handle.sandbox_id} is in invalid state Killing", status_code=500)
            )
            with caplog.at_level("INFO"):
                await provider.renew_leases()
                await provider.renew_leases()
            assert handle.raw.set_timeout_calls == [60]
            assert not [r for r in caplog.records if r.levelname == "WARNING"]
            assert any("is being killed" in r.getMessage() for r in caplog.records)
        finally:
            await provider.aclose()

    async def test_a_renewal_not_found_in_the_grace_period_keeps_the_lease(self, monkeypatch) -> None:
        _with_grace(monkeypatch)
        provider = E2BProvider(create={"template": "base", "lease_s": 60.0})
        handle = await provider.create(_spec())
        try:
            handle.raw.set_timeout_outcomes.append(FakeSandboxNotFound("sandbox assignment not found"))
            await provider.renew_leases()
            await provider.renew_leases()
            assert handle.raw.set_timeout_calls == [60, 60]
        finally:
            await provider.aclose()
