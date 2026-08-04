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
import builtins
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.daytona import provider as daytona_provider


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        importlib.util.find_spec("tenacity") is None,
        reason="tenacity optional sandbox dependency is not installed",
    ),
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeImageParams(FakeParams):
    pass


class FakeSnapshotParams(FakeParams):
    pass


class FakeResources(FakeParams):
    pass


class FakeVolumeMount(FakeParams):
    pass


class FakeDaytonaConfig(FakeParams):
    pass


class FakeAsyncDaytona:
    instances: list["FakeAsyncDaytona"] = []

    def __init__(self, config: FakeDaytonaConfig | None = None) -> None:
        self.config = config
        self.closed = False
        FakeAsyncDaytona.instances.append(self)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_daytona_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        daytona_provider,
        "_require_daytona_sdk",
        lambda: (
            FakeAsyncDaytona,
            FakeDaytonaConfig,
            FakeImageParams,
            FakeSnapshotParams,
            FakeResources,
            FakeVolumeMount,
        ),
    )


@pytest.fixture
def fake_daytona_module(monkeypatch: pytest.MonkeyPatch) -> None:
    class DaytonaError(Exception):
        pass

    module = SimpleNamespace(
        AsyncDaytona=FakeAsyncDaytona,
        CreateSandboxFromImageParams=FakeImageParams,
        CreateSandboxFromSnapshotParams=FakeSnapshotParams,
        DaytonaConfig=FakeDaytonaConfig,
        Resources=FakeResources,
        VolumeMount=FakeVolumeMount,
        DaytonaAuthenticationError=type("DaytonaAuthenticationError", (DaytonaError,), {}),
        DaytonaAuthorizationError=type("DaytonaAuthorizationError", (DaytonaError,), {}),
        DaytonaConflictError=type("DaytonaConflictError", (DaytonaError,), {}),
        DaytonaConnectionError=type("DaytonaConnectionError", (DaytonaError,), {}),
        DaytonaError=DaytonaError,
        DaytonaNotFoundError=type("DaytonaNotFoundError", (DaytonaError,), {}),
        DaytonaRateLimitError=type("DaytonaRateLimitError", (DaytonaError,), {}),
        DaytonaTimeoutError=type("DaytonaTimeoutError", (DaytonaError,), {}),
        DaytonaValidationError=type("DaytonaValidationError", (DaytonaError,), {}),
    )
    monkeypatch.setitem(sys.modules, "daytona", module)


def test_sdk_import_helpers_and_retry_classification(
    fake_daytona_module: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(daytona_provider._require_daytona_sdk()) == 6
    assert len(daytona_provider._require_tenacity()) == 4
    error_types = daytona_provider._daytona_error_types()
    assert set(error_types) == {
        "authentication",
        "authorization",
        "conflict",
        "connection",
        "error",
        "not_found",
        "rate_limit",
        "timeout",
        "validation",
    }

    class StatusCodeError(Exception):
        status_code = 429

    assert daytona_provider._exception_status_code(StatusCodeError("busy")) == 429
    assert daytona_provider._exception_status_code(RuntimeError("HTTP status code: 503")) == 503
    assert daytona_provider._exception_status_code(RuntimeError("plain")) is None
    assert daytona_provider._has_retryable_error_marker(RuntimeError("gateway timeout")) is True
    assert daytona_provider._is_retryable_create_error(StatusCodeError("busy")) is True
    not_retryable_status = RuntimeError("bad request")
    not_retryable_status.status_code = 400
    assert daytona_provider._is_retryable_create_error(not_retryable_status) is False

    assert daytona_provider._is_retryable_create_error(daytona_provider.DaytonaCreateVerificationError("probe"))
    assert daytona_provider._is_retryable_create_error(daytona_provider.DaytonaCreateTimeoutError("timeout")) is False
    assert daytona_provider._is_retryable_create_error(daytona_provider.DaytonaCreateError("create failed"))
    assert daytona_provider._is_retryable_create_error(TimeoutError("ambiguous timeout")) is False
    assert daytona_provider._is_retryable_create_error(ConnectionError("reset")) is True
    assert daytona_provider._is_retryable_create_error(error_types["validation"]("bad")) is False
    assert daytona_provider._is_retryable_create_error(error_types["conflict"]("busy")) is True
    assert daytona_provider._is_retryable_create_error(error_types["timeout"]("ambiguous")) is False

    retryable_api_error = error_types["error"]("server failed")
    retryable_api_error.status_code = 503
    assert daytona_provider._is_retryable_create_error(retryable_api_error) is True
    nonretryable_api_error = error_types["error"]("bad request")
    nonretryable_api_error.status_code = 400
    assert daytona_provider._is_retryable_create_error(nonretryable_api_error) is False
    assert daytona_provider._is_retryable_create_error(RuntimeError("rate limit")) is True

    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = ConnectionError("connection reset")
    assert daytona_provider._is_retryable_operation_error(wrapped) is True
    assert daytona_provider._is_retryable_operation_error(TimeoutError("command timeout")) is False
    assert daytona_provider._is_retryable_operation_error(error_types["rate_limit"]("slow down")) is True
    assert daytona_provider._is_retryable_operation_error(error_types["not_found"]("gone")) is False

    assert daytona_provider._is_missing_sandbox_delete_error(RuntimeError("sandbox abc not found")) is True
    missing = RuntimeError("missing")
    missing.status_code = 404
    assert daytona_provider._is_missing_sandbox_delete_error(missing) is True
    assert daytona_provider._is_missing_sandbox_delete_error(RuntimeError("delete failed")) is False

    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "daytona":
            raise ImportError("daytona missing")
        if name == "tenacity":
            raise ModuleNotFoundError("tenacity missing")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.delitem(sys.modules, "daytona", raising=False)
    monkeypatch.delitem(sys.modules, "tenacity", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModuleNotFoundError, match="Daytona SDK is required"):
        daytona_provider._require_daytona_sdk()
    with pytest.raises(ModuleNotFoundError, match="tenacity is required"):
        daytona_provider._require_tenacity()
    assert daytona_provider._daytona_error_types() == {}


def test_config_unknown_keys_fail_fast() -> None:
    with pytest.raises(ValueError, match="Unsupported Daytona DaytonaCreateConfig settings: typo"):
        daytona_provider.DaytonaProvider(create={"typo": "value"})


def test_conversion_helpers_and_config_validation(fake_daytona_sdk: None) -> None:
    create_config = daytona_provider.DaytonaCreateConfig(timeout_s=1)
    assert daytona_provider.DaytonaCreateConfig.from_mapping(create_config) is create_config
    with pytest.raises(TypeError, match="must be a mapping"):
        daytona_provider.DaytonaCreateConfig.from_mapping(object())

    with pytest.raises(ValueError, match="must be a boolean"):
        daytona_provider._coerce_bool("daytona.public", "maybe")
    assert daytona_provider._coerce_bool("daytona.public", "yes") is True
    assert daytona_provider._coerce_bool("daytona.public", "0") is False
    with pytest.raises(ValueError, match="must be an integer"):
        daytona_provider._coerce_int("daytona.auto_stop_interval", True)
    assert daytona_provider._coerce_int("daytona.auto_stop_interval", 2.0) == 2
    assert daytona_provider._coerce_int("daytona.auto_stop_interval", "3") == 3

    assert daytona_provider._quantity_to_int("cpu", 2) == 2
    assert daytona_provider._quantity_to_int("cpu", 2.0) == 2
    assert daytona_provider._quantity_to_int("cpu", "500m") == 1
    assert daytona_provider._quantity_to_int("memory", "2048Mi") == 2
    assert daytona_provider._quantity_to_int("disk", "1.1Gi") == 2
    with pytest.raises(ValueError, match="integer-like"):
        daytona_provider._quantity_to_int("cpu", True)
    with pytest.raises(ValueError, match="integer-like"):
        daytona_provider._quantity_to_int("cpu", "abc")
    with pytest.raises(ValueError, match="unsupported unit"):
        daytona_provider._quantity_to_int("cpu", "1Ti")

    spec = SandboxSpec(
        provider_options={
            "extensions": {"daytona.name": "sandbox-name"},
            "volumes": [{"name": "workspace"}],
            "snapshot_id": "snapshot-1",
        }
    )
    assert daytona_provider._extension_value(spec, "name") == "sandbox-name"
    assert daytona_provider._spec_snapshot_id(spec) == "snapshot-1"
    assert daytona_provider._spec_volumes(spec) == [{"name": "workspace"}]
    assert daytona_provider._to_volume_mounts([{"name": "workspace"}])[0].kwargs == {"name": "workspace"}
    assert daytona_provider.DaytonaProviderOptions.from_mapping(None) == daytona_provider.DaytonaProviderOptions()
    with pytest.raises(TypeError, match="extensions"):
        daytona_provider._spec_extensions(SandboxSpec(provider_options={"extensions": []}))
    with pytest.raises(TypeError, match="volumes"):
        daytona_provider._spec_volumes(SandboxSpec(provider_options={"volumes": {}}))
    with pytest.raises(TypeError, match="snapshot_id"):
        daytona_provider.DaytonaProviderOptions.from_mapping({"snapshot_id": 123})
    with pytest.raises(TypeError, match="provider_options"):
        daytona_provider.DaytonaProviderOptions.from_mapping([])
    with pytest.raises(ValueError, match="Unknown Daytona provider option.*platform"):
        daytona_provider.DaytonaProviderOptions.from_mapping({"platform": {"os": "linux"}})

    assert daytona_provider._to_resources({}) is None
    assert daytona_provider._to_resources({"gpu": "1"}).kwargs == {"gpu": 1}
    assert daytona_provider._to_resources(SandboxSpec(resources={"memory_mib": 2048}).resources).kwargs == {
        "memory": 2
    }
    with pytest.raises(ValueError, match="gpu_type"):
        daytona_provider._to_resources(SandboxSpec(resources={"gpu_type": "a100"}).resources)
    assert daytona_provider._to_sandbox_status("creating") == SandboxStatus.STARTING
    assert daytona_provider._to_sandbox_status("deleted") == SandboxStatus.STOPPED
    assert daytona_provider._to_sandbox_status("failed") == SandboxStatus.ERROR
    assert daytona_provider._to_sandbox_status("mystery") == SandboxStatus.UNKNOWN

    invalid_provider_kwargs = [
        {"connection": {"connection_pool_maxsize": 0}},
        {"create": {"timeout_s": -1}},
        {"create": {"retries": -1}},
        {"create": {"retry_delay_s": -1}},
        {"create": {"retry_max_delay_s": -1}},
        {"probe": {"timeout_s": 0}},
        {"probe": {"deadline_s": 0}},
        {"operations": {"retries": -1}},
        {"operations": {"retry_delay_s": -1}},
        {"operations": {"retry_max_delay_s": -1}},
        {"operations": {"command_retries": -1}},
        {"operations": {"command_timeout_margin_s": -1}},
        {"operations": {"file_timeout_s": 0}},
        {"operations": {"close_timeout_s": 0}},
        {"batch": {"concurrency": 0}},
    ]
    for kwargs in invalid_provider_kwargs:
        with pytest.raises(ValueError):
            daytona_provider.DaytonaProvider(**kwargs)

    with pytest.raises(ValueError, match="Unsupported Daytona DaytonaConnectionConfig settings: typo"):
        daytona_provider.DaytonaProvider(connection={"typo": True})


def test_snapshot_creation_rejects_ignored_resources(fake_daytona_sdk: None) -> None:
    provider = daytona_provider.DaytonaProvider(probe={"command": None})
    spec = SandboxSpec(
        provider_options={"snapshot_id": "snapshot-1"},
        resources={"cpu": 2},
    )

    with pytest.raises(ValueError, match="snapshot creation does not support resource overrides"):
        provider._to_create_params(spec)


def test_image_creation_keeps_resource_overrides(fake_daytona_sdk: None) -> None:
    provider = daytona_provider.DaytonaProvider(probe={"command": None})
    params = provider._to_create_params(
        SandboxSpec(
            image="python:3.12",
            resources={"cpu": 2, "memory_mib": 2048, "disk_gib": 9},
        )
    )

    assert isinstance(params, FakeImageParams)
    assert params.kwargs["resources"].kwargs == {"cpu": 2, "memory": 2, "disk": 9}


async def test_client_lifecycle_create_kwargs_and_status_fallbacks(fake_daytona_sdk: None) -> None:
    FakeAsyncDaytona.instances.clear()
    provider = daytona_provider.DaytonaProvider(
        connection={
            "api_key": "key",  # pragma: allowlist secret
            "jwt_token": "jwt",
            "organization_id": "org",
            "api_url": "https://api.example",
            "server_url": "https://server.example",
            "target": "us",
            "connection_pool_maxsize": 8,
            "otel_enabled": False,
        },
        create={
            "language": "python",
            "os_user": "agent",
            "public": False,
            "auto_stop_interval": 10,
            "auto_archive_interval": 20,
            "auto_delete_interval": 30,
            "ephemeral": True,
            "network_block_all": False,
            "network_allow_list": "github.com",
        },
    )
    client = provider._client()
    assert provider._client() is client
    assert client.config.kwargs == {
        "api_key": "key",  # pragma: allowlist secret
        "jwt_token": "jwt",
        "organization_id": "org",
        "api_url": "https://api.example",
        "server_url": "https://server.example",
        "target": "us",
        "connection_pool_maxsize": 8,
        "otel_enabled": False,
    }
    await provider.aclose()
    assert client.closed is True
    assert await provider.aclose() is None

    FakeAsyncDaytona.instances.clear()
    unlimited_provider = daytona_provider.DaytonaProvider(connection={"connection_pool_maxsize": None})
    unlimited_client = unlimited_provider._client()
    assert unlimited_client.config.kwargs == {"connection_pool_maxsize": None}
    await unlimited_provider.aclose()

    kwargs = provider._base_create_kwargs(
        SandboxSpec(
            env={"A": "B"},
            metadata={"label": "value"},
            provider_options={
                "extensions": {
                    "daytona.name": "sandbox-name",
                    "daytona.language": "go",
                    "daytona.os_user": "root",
                    "daytona.public": "true",
                    "daytona.auto_stop_interval": "11",
                    "daytona.auto_archive_interval": "22",
                    "daytona.auto_delete_interval": "33",
                    "daytona.ephemeral": "false",
                    "daytona.network_block_all": "true",
                    "daytona.network_allow_list": "example.com",
                },
                "volumes": [{"name": "workspace"}],
            },
        )
    )
    assert {key: value for key, value in kwargs.items() if key != "volumes"} == {
        "name": "sandbox-name",
        "language": "go",
        "os_user": "root",
        "env_vars": {"A": "B"},
        "labels": {"label": "value"},
        "public": True,
        "auto_stop_interval": 11,
        "auto_archive_interval": 22,
        "auto_delete_interval": 33,
        "ephemeral": False,
        "network_block_all": True,
        "network_allow_list": "example.com",
    }
    assert kwargs["volumes"][0].kwargs == {"name": "workspace"}

    with pytest.raises(ValueError, match="both image and snapshot_id"):
        provider._to_create_params(SandboxSpec(image="image", provider_options={"snapshot_id": "snapshot"}))
    assert isinstance(
        provider._to_create_params(SandboxSpec(provider_options={"snapshot_id": "snapshot"})), FakeSnapshotParams
    )
    assert daytona_provider.DaytonaProvider()._to_create_params(SandboxSpec()) is None

    status_handle = SandboxHandle(
        sandbox_id="sandbox-status",
        provider_name="daytona",
        raw=SimpleNamespace(status=SimpleNamespace(state="terminated")),
    )
    assert await provider.status(status_handle) == SandboxStatus.STOPPED

    class InfoRaw:
        async def get_info(self) -> Any:
            return SimpleNamespace(status=SimpleNamespace(state="failed"))

    assert await provider.status(SandboxHandle(sandbox_id="sandbox-info", provider_name="daytona", raw=InfoRaw())) == (
        SandboxStatus.ERROR
    )
    assert await provider.status(SandboxHandle(sandbox_id="sandbox-bare", provider_name="daytona", raw=object())) == (
        SandboxStatus.UNKNOWN
    )


async def test_status_uses_daytona_refresh_data() -> None:
    class FakeRaw:
        def __init__(self) -> None:
            self.state = "creating"
            self.refresh_calls = 0

        async def refresh_data(self) -> None:
            self.refresh_calls += 1
            self.state = "started"

    provider = daytona_provider.DaytonaProvider(operations={"retries": 0})
    raw = FakeRaw()
    status = await provider.status(SandboxHandle(sandbox_id="sandbox-1", provider_name="daytona", raw=raw))

    assert status == SandboxStatus.RUNNING
    assert raw.refresh_calls == 1


async def test_await_operation_retry_and_timeout_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = daytona_provider.DaytonaProvider(operations={"retries": 1, "retry_delay_s": 0, "retry_max_delay_s": 0})
    assert (
        await provider._await_operation(
            lambda: _return_value("ok"),
            operation="op",
            sandbox_id="sandbox-1",
            timeout_s=None,
            retries=0,
        )
        == "ok"
    )
    assert (
        await provider._await_operation(
            lambda: _return_value("ok"),
            operation="op",
            sandbox_id="sandbox-1",
            timeout_s=0,
            retries=0,
        )
        == "ok"
    )

    async def never() -> None:
        await asyncio.get_running_loop().create_future()

    with pytest.raises(TimeoutError, match="Timed out during Daytona op"):
        await provider._await_operation(
            never,
            operation="op",
            sandbox_id="sandbox-1",
            timeout_s=0.01,
            retries=0,
        )

    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(daytona_provider.asyncio, "sleep", record_sleep)
    attempts = 0

    async def retry_then_succeed() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary")
        return "ok"

    assert (
        await provider._await_operation(
            retry_then_succeed,
            operation="exec",
            sandbox_id="sandbox-1",
            timeout_s=None,
        )
        == "ok"
    )
    assert attempts == 2
    assert sleeps == [0]

    with pytest.raises(ValueError, match="bad"):
        await provider._await_operation(
            lambda: _raise(ValueError("bad")),
            operation="exec",
            sandbox_id="sandbox-1",
            timeout_s=None,
        )


async def test_exec_maps_stderr_only_output() -> None:
    class FakeProcess:
        async def exec(self, *_args: Any, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                result=None,
                artifacts=SimpleNamespace(stdout=None, stderr="err\n"),
                exit_code=3,
            )

    raw = SimpleNamespace(process=FakeProcess())
    provider = daytona_provider.DaytonaProvider(operations={"retries": 0})
    result = await provider.exec(SandboxHandle(sandbox_id="sandbox-1", provider_name="daytona", raw=raw), "cmd")

    assert result.stdout is None
    assert result.stderr == "err\n"
    assert result.return_code == 3


async def test_exec_maps_daytona_command_error_to_sandbox_result(fake_daytona_module: None) -> None:
    error_types = daytona_provider._daytona_error_types()

    class FakeProcess:
        async def exec(self, *_args: Any, **_kwargs: Any) -> Any:
            raise error_types["error"]("Failed to execute command:")

    raw = SimpleNamespace(process=FakeProcess())
    provider = daytona_provider.DaytonaProvider(operations={"retries": 0})
    result = await provider.exec(SandboxHandle(sandbox_id="sandbox-1", provider_name="daytona", raw=raw), "cmd")

    assert result.stdout is None
    assert result.return_code == 124
    assert result.error_type == "timeout"
    assert result.stderr is not None
    assert "HTTP request timed out before Daytona returned a response" in result.stderr
    assert "The command may not have run" in result.stderr


async def test_exec_maps_daytona_command_timeout_to_timeout_result(fake_daytona_module: None) -> None:
    error_types = daytona_provider._daytona_error_types()

    class FakeProcess:
        async def exec(self, *_args: Any, **_kwargs: Any) -> Any:
            raise error_types["error"]("Failed to execute command: request timeout: command execution timeout")

    raw = SimpleNamespace(process=FakeProcess())
    provider = daytona_provider.DaytonaProvider(operations={"retries": 0})
    result = await provider.exec(SandboxHandle(sandbox_id="sandbox-1", provider_name="daytona", raw=raw), "cmd")

    assert result.return_code == 124
    assert result.error_type == "timeout"
    assert result.stderr is not None
    assert "command execution timeout" in result.stderr


async def test_exec_retries_empty_daytona_command_error_when_enabled(
    fake_daytona_module: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_types = daytona_provider._daytona_error_types()
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    class FakeProcess:
        def __init__(self) -> None:
            self.calls = 0

        async def exec(self, *_args: Any, **_kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise error_types["error"]("Failed to execute command:")
            return SimpleNamespace(result="ok", exit_code=0)

    monkeypatch.setattr(daytona_provider.asyncio, "sleep", record_sleep)
    process = FakeProcess()
    raw = SimpleNamespace(process=process)
    provider = daytona_provider.DaytonaProvider(
        operations={
            "retries": 0,
            "retry_delay_s": 0,
            "retry_max_delay_s": 0,
            "command_retries": 1,
        }
    )

    result = await provider.exec(SandboxHandle(sandbox_id="sandbox-1", provider_name="daytona", raw=raw), "cmd")

    assert result.stdout == "ok"
    assert result.return_code == 0
    assert process.calls == 2
    assert sleeps == [0]


async def test_probe_create_connect_file_and_close_paths(
    fake_daytona_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeSandbox:
        def __init__(self, sandbox_id: str) -> None:
            self.id = sandbox_id
            self.process = SimpleNamespace(exec=lambda *_args, **_kwargs: _return_value(SimpleNamespace(result="ok")))

    class FakeClient:
        def __init__(self) -> None:
            self.create_calls = 0
            self.deleted: list[str] = []

        async def create(self, _params: Any, *, timeout: float) -> FakeSandbox:
            self.create_calls += 1
            assert timeout == 2
            return FakeSandbox("sandbox-created")

        async def get(self, sandbox_id: str) -> FakeSandbox:
            return FakeSandbox(sandbox_id)

        async def delete(self, raw: Any, *, timeout: float) -> None:
            del timeout
            if raw == "missing":
                raise RuntimeError("sandbox missing not found")
            if raw == "failed":
                raise RuntimeError("delete failed")
            self.deleted.append(raw.id)

    client = FakeClient()
    provider = daytona_provider.DaytonaProvider(create={"timeout_s": 2}, probe={"command": None})
    monkeypatch.setattr(provider, "_client", lambda: client)

    handle = await provider._create_once(SandboxSpec(image="python:3.12"))
    assert handle.sandbox_id == "sandbox-created"
    assert client.create_calls == 1
    connected = await provider.connect("sandbox-existing")
    assert connected.sandbox_id == "sandbox-existing"

    provider = daytona_provider.DaytonaProvider(
        probe={"command": "probe", "expected_stdout": None, "timeout_s": 1, "deadline_s": 0.01}
    )
    good_probe_calls = 0

    async def good_probe(*_args: Any, **_kwargs: Any) -> daytona_provider.SandboxExecResult:
        nonlocal good_probe_calls
        good_probe_calls += 1
        return daytona_provider.SandboxExecResult(stdout="ready", stderr=None, return_code=0)

    monkeypatch.setattr(provider, "_exec", good_probe)
    await provider._verify_created_handle(handle)
    assert good_probe_calls == 1

    async def bad_probe(*_args: Any, **_kwargs: Any) -> daytona_provider.SandboxExecResult:
        return daytona_provider.SandboxExecResult(stdout="not ready", stderr="nope", return_code=1)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(provider, "_exec", bad_probe)
    monkeypatch.setattr(daytona_provider.asyncio, "sleep", no_sleep)
    with pytest.raises(daytona_provider.DaytonaCreateVerificationError):
        await provider._verify_created_handle(handle)

    async def raising_probe(*_args: Any, **_kwargs: Any) -> daytona_provider.SandboxExecResult:
        raise RuntimeError("transient probe failure")

    monkeypatch.setattr(provider, "_exec", raising_probe)
    with pytest.raises(daytona_provider.DaytonaCreateVerificationError):
        await provider._verify_created_handle(handle)

    async def cancelled_probe(*_args: Any, **_kwargs: Any) -> daytona_provider.SandboxExecResult:
        raise asyncio.CancelledError()

    monkeypatch.setattr(provider, "_exec", cancelled_probe)
    with pytest.raises(asyncio.CancelledError):
        await provider._verify_created_handle(handle)

    cleanup_calls: list[tuple[str, bool]] = []
    provider = daytona_provider.DaytonaProvider(create={"timeout_s": 2}, probe={"command": "probe"})
    monkeypatch.setattr(provider, "_client", lambda: client)

    async def fail_verify(_handle: SandboxHandle) -> None:
        raise RuntimeError("probe failed")

    async def close_for_cleanup(cleanup_handle: SandboxHandle, *, delete: bool) -> None:
        cleanup_calls.append((cleanup_handle.sandbox_id, delete))

    monkeypatch.setattr(provider, "_verify_created_handle", fail_verify)
    monkeypatch.setattr(provider, "close", close_for_cleanup)
    with pytest.raises(RuntimeError, match="probe failed"):
        await provider._create_once(SandboxSpec(image="python:3.12"))
    assert cleanup_calls == [("sandbox-created", True)]

    async def close_cleanup_failure(cleanup_handle: SandboxHandle, *, delete: bool) -> None:
        cleanup_calls.append((cleanup_handle.sandbox_id, delete))
        raise RuntimeError("cleanup failed")

    cleanup_calls.clear()
    monkeypatch.setattr(provider, "close", close_cleanup_failure)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="probe failed"):
            await provider._create_once(SandboxSpec(image="python:3.12"))
    assert cleanup_calls == [("sandbox-created", True)]
    assert "Failed to delete Daytona sandbox after create probe failure" in caplog.text
    assert "cleanup failed" in caplog.text

    class TimeoutClient(FakeClient):
        async def create(self, _params: Any, *, timeout: float) -> FakeSandbox:
            del timeout
            await asyncio.get_running_loop().create_future()
            return FakeSandbox("never")

    timeout_provider = daytona_provider.DaytonaProvider(create={"timeout_s": 0.01}, probe={"command": None})
    monkeypatch.setattr(timeout_provider, "_client", lambda: TimeoutClient())
    with pytest.raises(daytona_provider.DaytonaCreateTimeoutError, match="Timed out creating Daytona sandbox"):
        await timeout_provider._create_once(SandboxSpec(image="python:3.12"))

    class FakeFs:
        def __init__(self) -> None:
            self.uploads: list[tuple[Any, ...]] = []
            self.downloads: list[tuple[Any, ...]] = []

        async def upload_file(self, *args: Any, **kwargs: Any) -> None:
            self.uploads.append((*args, kwargs))

        async def download_file(self, *args: Any, **kwargs: Any) -> str:
            self.downloads.append((*args, kwargs))
            if len(args) == 1 or (len(args) == 2 and isinstance(args[1], int)):
                return "contents"
            Path(args[1]).write_text("downloaded", encoding="utf-8")
            return ""

    fs = FakeFs()
    file_provider = daytona_provider.DaytonaProvider(operations={"retries": 0, "file_timeout_s": 5})
    file_handle = SandboxHandle(sandbox_id="sandbox-files", provider_name="daytona", raw=SimpleNamespace(fs=fs))
    await file_provider.write_file(file_handle, "/remote/write.txt", "text")
    assert await file_provider.read_file(file_handle, "/remote/read.txt") == b"contents"
    local_upload = tmp_path / "upload.txt"
    local_upload.write_text("upload", encoding="utf-8")
    await file_provider.upload_file(file_handle, local_upload, "/remote/upload.txt")
    local_download = tmp_path / "nested" / "download.txt"
    await file_provider.download_file(file_handle, "/remote/download.txt", local_download)
    assert fs.uploads[0] == (b"text", "/remote/write.txt", {"timeout": 5})
    assert fs.uploads[1] == (str(local_upload), "/remote/upload.txt", {"timeout": 5})
    assert fs.downloads[0] == ("/remote/read.txt", 5, {})
    assert fs.downloads[1] == ("/remote/download.txt", str(local_download), 5, {})
    assert local_download.read_text(encoding="utf-8") == "downloaded"

    close_provider = daytona_provider.DaytonaProvider(operations={"close_timeout_s": 5})
    monkeypatch.setattr(close_provider, "_client", lambda: client)
    await close_provider.close(SandboxHandle("sandbox-retained", "daytona", raw=FakeSandbox("retained")), delete=False)
    await close_provider.close(SandboxHandle("sandbox-default-delete", "daytona", raw=FakeSandbox("default-delete")))
    await close_provider.close(SandboxHandle("sandbox-delete", "daytona", raw=FakeSandbox("delete-me")), delete=True)
    await close_provider.close(SandboxHandle("sandbox-missing", "daytona", raw="missing"), delete=True)
    with pytest.raises(RuntimeError, match="delete failed"):
        await close_provider.close(SandboxHandle("sandbox-failed", "daytona", raw="failed"), delete=True)
    assert client.deleted[-1] == "delete-me"


async def test_command_retries_default_to_zero() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.calls = 0

        async def exec(self, *_args: Any, **_kwargs: Any) -> Any:
            self.calls += 1
            raise ConnectionError("response lost after command may have run")

    process = FakeProcess()
    raw = SimpleNamespace(process=process)
    provider = daytona_provider.DaytonaProvider(operations={"retries": 3})

    with pytest.raises(ConnectionError):
        await provider.exec(SandboxHandle(sandbox_id="sandbox-1", provider_name="daytona", raw=raw), "cmd")

    assert process.calls == 1


def test_per_command_user_switching_is_not_supported() -> None:
    assert daytona_provider.DaytonaProvider._effective_command("whoami", None) == "whoami"
    assert daytona_provider.DaytonaProvider._effective_command("whoami", "root") == "whoami"
    assert daytona_provider.DaytonaProvider._effective_command("whoami", 0) == "whoami"

    with pytest.raises(NotImplementedError, match="per-command user switching"):
        daytona_provider.DaytonaProvider._effective_command("whoami", "agent")


async def test_create_timeout_is_not_retried() -> None:
    provider = daytona_provider.DaytonaProvider(create={"retries": 3}, probe={"command": None})
    calls = 0

    async def create_once(_spec: SandboxSpec) -> SandboxHandle:
        nonlocal calls
        calls += 1
        raise daytona_provider.DaytonaCreateTimeoutError("ambiguous create timeout")

    provider._create_once = create_once  # type: ignore[method-assign]

    with pytest.raises(daytona_provider.DaytonaCreateTimeoutError):
        await provider.create(SandboxSpec(image="python:3.12"))

    assert calls == 1


async def test_create_retries_retryable_verification_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = daytona_provider.DaytonaProvider(create={"retries": 1, "retry_delay_s": 0}, probe={"command": None})
    calls = 0
    names: list[str | None] = []
    labels: list[str | None] = []
    sleeps: list[float] = []

    async def create_once(spec: SandboxSpec) -> SandboxHandle:
        nonlocal calls
        calls += 1
        names.append(daytona_provider._extension_value(spec, "name"))
        labels.append(spec.metadata.get("sandbox_id"))
        if calls == 1:
            raise daytona_provider.DaytonaCreateVerificationError("probe failed")
        return SandboxHandle("sandbox-created", "daytona", raw=object())

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(daytona_provider.asyncio, "sleep", record_sleep)
    provider._create_once = create_once  # type: ignore[method-assign]

    assert (await provider.create(SandboxSpec(image="python:3.12"))).sandbox_id == "sandbox-created"
    assert calls == 2
    assert names[0] != names[1]
    assert labels == names
    assert all(name is not None and name.startswith("nemo-gym-") for name in names)
    assert sleeps == [0]


async def test_create_retries_preserve_explicit_name(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = daytona_provider.DaytonaProvider(create={"retries": 1, "retry_delay_s": 0}, probe={"command": None})
    names: list[str | None] = []

    async def create_once(spec: SandboxSpec) -> SandboxHandle:
        names.append(daytona_provider._extension_value(spec, "name"))
        if len(names) == 1:
            raise daytona_provider.DaytonaCreateVerificationError("probe failed")
        return SandboxHandle("sandbox-created", "daytona", raw=object())

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(daytona_provider.asyncio, "sleep", no_sleep)
    provider._create_once = create_once  # type: ignore[method-assign]

    spec = SandboxSpec(
        image="python:3.12",
        provider_options={"extensions": {"daytona.name": "custom-sandbox"}},
    )
    assert (await provider.create(spec)).sandbox_id == "sandbox-created"
    assert names == ["custom-sandbox", "custom-sandbox"]


async def test_create_batch_returns_all_partial_successes_and_aggregates_cleanup_errors() -> None:
    provider = daytona_provider.DaytonaProvider(batch={"concurrency": 4}, probe={"command": None})
    handles = [
        SandboxHandle(sandbox_id="sandbox-1", provider_name="daytona", raw=object()),
        SandboxHandle(sandbox_id="sandbox-2", provider_name="daytona", raw=object()),
    ]
    responses: list[SandboxHandle | Exception] = [handles[0], RuntimeError("create failed"), handles[1]]

    async def create(_spec: SandboxSpec) -> SandboxHandle:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    provider.create = create  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="count must be >= 1"):
        await provider.create_batch(SandboxSpec(image="python:3.12"), 0)
    assert await provider.create_batch(SandboxSpec(image="python:3.12"), 3, allow_partial=True) == handles

    responses = [handles[0], handles[1]]
    assert await provider.create_batch(SandboxSpec(image="python:3.12"), 2) == handles

    responses = [handles[0], RuntimeError("create failed"), handles[1]]
    cleanup_calls: list[str] = []

    async def close(handle: SandboxHandle, *, delete: bool) -> None:
        assert delete is True
        cleanup_calls.append(handle.sandbox_id)
        raise RuntimeError(f"cleanup failed for {handle.sandbox_id}")

    provider.create = create  # type: ignore[method-assign]
    provider.close = close  # type: ignore[method-assign]

    with pytest.raises(daytona_provider.DaytonaCreateError, match="cleanup_errors=.*sandbox-1.*sandbox-2"):
        await provider.create_batch(SandboxSpec(image="python:3.12"), 3)

    assert cleanup_calls == ["sandbox-1", "sandbox-2"]


async def _return_value(value: Any) -> Any:
    return value


async def _raise(error: BaseException) -> None:
    raise error
