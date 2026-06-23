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
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSpec, SandboxStatus


pytest.importorskip("tenacity", reason="tenacity optional sandbox dependency is not installed")

from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


@dataclass(frozen=True)
class FakePlatformSpec:
    os: str
    arch: str


class FakeConnectionConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@dataclass(frozen=True)
class FakeVolume:
    name: str


class FakeSandbox:
    created_kwargs: dict[str, Any] = {}
    connected_args: tuple[Any, ...] = ()
    connected_kwargs: dict[str, Any] = {}

    def __init__(self, sandbox_id: str = "sandbox-1") -> None:
        self.id = sandbox_id

    @classmethod
    async def create(cls, *_args: Any, **kwargs: Any) -> "FakeSandbox":
        cls.created_kwargs = kwargs
        return cls()

    @classmethod
    async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
        cls.connected_args = args
        cls.connected_kwargs = kwargs
        return cls()


@pytest.fixture
def fake_opensandbox_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    def require_sdk() -> tuple[Any, Any, Any, Any, Any]:
        return FakeSandbox, FakeConnectionConfig, object, FakePlatformSpec, object

    monkeypatch.setattr(opensandbox_provider, "_require_opensandbox_sdk", require_sdk)


def test_sdk_import_helpers_and_retry_classification() -> None:
    assert len(opensandbox_provider._require_opensandbox_sdk()) == 5
    assert len(opensandbox_provider._require_tenacity()) == 4

    class StatusCodeError(Exception):
        status_code = 429

    assert opensandbox_provider._exception_status_code(StatusCodeError("rate limited")) == 429
    assert opensandbox_provider._is_retryable_create_error(
        opensandbox_provider.OpenSandboxCreateError("create failed")
    )

    from opensandbox.exceptions import (  # noqa: PLC0415
        InvalidArgumentException,
        SandboxApiException,
        SandboxException,
        SandboxInternalException,
    )

    assert opensandbox_provider._is_retryable_create_error(InvalidArgumentException("bad input")) is False
    assert opensandbox_provider._is_retryable_create_error(SandboxInternalException("server failed")) is True

    retryable_api_error = SandboxApiException("busy")
    retryable_api_error.status_code = 503
    assert opensandbox_provider._is_retryable_create_error(retryable_api_error) is True

    nonretryable_api_error = SandboxApiException("not found")
    nonretryable_api_error.status_code = 404
    assert opensandbox_provider._is_retryable_create_error(nonretryable_api_error) is False
    assert opensandbox_provider._is_retryable_create_error(SandboxException("gateway timeout")) is True

    retry_state = SimpleNamespace(
        outcome=SimpleNamespace(exception=lambda: RuntimeError("temporary")),
        next_action=SimpleNamespace(sleep=0.5),
        attempt_number=2,
    )
    opensandbox_provider._log_create_retry(retry_state)


def test_missing_optional_dependency_import_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def block_imports(*blocked_names: str) -> None:
        def fake_import(
            name: str,
            globals_: dict[str, Any] | None = None,
            locals_: dict[str, Any] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if any(name == blocked or name.startswith(f"{blocked}.") for blocked in blocked_names):
                raise ModuleNotFoundError(name)
            return real_import(name, globals_, locals_, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    block_imports("opensandbox")
    with pytest.raises(ModuleNotFoundError, match="OpenSandbox SDK is required"):
        opensandbox_provider._require_opensandbox_sdk()

    block_imports("tenacity")
    with pytest.raises(ModuleNotFoundError, match="tenacity is required"):
        opensandbox_provider._require_tenacity()

    block_imports("opensandbox.exceptions")
    assert opensandbox_provider._is_retryable_create_error(RuntimeError("gateway timeout")) is True


async def test_provider_conversion_helpers(
    fake_opensandbox_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection_config = opensandbox_provider.OpenSandboxConnectionConfig(domain="sandbox.example")
    assert (
        opensandbox_provider._coerce_config(connection_config, opensandbox_provider.OpenSandboxConnectionConfig)
        is connection_config
    )

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, object, FakePlatformSpec, FakeVolume),
    )
    assert opensandbox_provider._to_volumes([{"name": "workspace"}]) == [FakeVolume(name="workspace")]


async def test_direct_create_passes_platform_to_sdk_create(
    fake_opensandbox_sdk: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
    )

    handle = await provider.create(
        SandboxSpec(
            image="mirror.gcr.io/astral/uv:python3.12-bookworm-slim",
            provider_options={"platform": {"os": "linux", "arch": "amd64"}},
        ),
    )

    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.created_kwargs["platform"] == FakePlatformSpec(
        os="linux",
        arch="amd64",
    )


def test_provider_validation_and_retry_helpers() -> None:
    with pytest.raises(ValueError, match="image_pull_policy"):
        opensandbox_provider.validate_image_pull_policy("Sometimes")
    with pytest.raises(TypeError, match="must be a bool"):
        opensandbox_provider._provider_option_bool({"skip_health_check": "true"}, "skip_health_check")

    assert opensandbox_provider._to_sandbox_status("starting") == SandboxStatus.STARTING
    assert opensandbox_provider._to_sandbox_status("terminated") == SandboxStatus.STOPPED
    assert opensandbox_provider._to_sandbox_status("failed") == SandboxStatus.ERROR
    assert opensandbox_provider._to_sandbox_status(None) == SandboxStatus.UNKNOWN

    invalid_kwargs = [
        {"create": {"timeout_s": 0}},
        {"probe": {"timeout_s": 0}},
        {"probe": {"deadline_s": 0}},
        {"probe": {"stable_count": 0}},
        {"probe": {"stable_delay_s": -1}},
        {"create": {"retries": -1}},
        {"create": {"retry_delay_s": -1}},
        {"create": {"retry_max_delay_s": -1}},
        {"operations": {"retries": -1}},
        {"operations": {"retry_delay_s": -1}},
        {"operations": {"retry_max_delay_s": -1}},
        {"operations": {"command_retries": -1}},
        {"operations": {"close_timeout_s": 0}},
        {"create": {"connect_attempt_timeout_s": 0}},
        {"create": {"connect_poll_s": 0}},
        {"create": {"image_pull_policy": "Sometimes"}},
    ]
    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            opensandbox_provider.OpenSandboxProvider(**kwargs)
    with pytest.raises(TypeError):
        opensandbox_provider.OpenSandboxProvider(**{"batch_" + "create_retries": 1})
    with pytest.raises(TypeError):
        opensandbox_provider.OpenSandboxProvider(connection=object())

    assert opensandbox_provider._exception_status_code(RuntimeError("HTTP status code: 503")) == 503
    assert opensandbox_provider._exception_status_code(RuntimeError("plain error")) is None
    attrs = opensandbox_provider._sdk_error_attributes(
        RuntimeError("HTTP 502 bad gateway"),
        operation="exec",
        sandbox_id="sandbox-1",
        attempt_number=2,
        max_attempts=3,
        sleep_s=0.5,
    )
    assert attrs["status_code"] == 502
    assert attrs["attempt_number"] == 2
    assert attrs["next_sleep_s"] == 0.5


def test_connection_config_and_image_policy(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "sandbox.example",
            "api_key": "key",  # pragma: allowlist secret
            "protocol": "https",
            "request_timeout_s": 10,
            "use_server_proxy": True,
        }
    )

    config = provider._connection_config()
    assert config.kwargs == {
        "domain": "sandbox.example",
        "api_key": "key",  # pragma: allowlist secret
        "protocol": "https",
        "request_timeout": timedelta(seconds=10),
        "use_server_proxy": True,
    }
    short_timeout_config = provider._connection_config(request_timeout_s=3)
    assert short_timeout_config.kwargs["request_timeout"] == timedelta(seconds=3)

    spec = SandboxSpec(image="image:tag", provider_options={"extensions": {"imagePullPolicy": "Never"}})
    updated = provider._with_default_image_pull_policy(spec)
    extensions = updated.provider_options["extensions"]
    assert extensions["imagePullPolicy"] == "Never"
    assert extensions["opensandbox.extensions.image-pull-policy"] == "Never"

    no_policy_provider = opensandbox_provider.OpenSandboxProvider(create={"image_pull_policy": None})
    assert no_policy_provider._with_default_image_pull_policy(spec) is spec


async def test_exec_file_operations_and_reference_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeLog:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeCommands:
        def __init__(self) -> None:
            self.calls: list[tuple[str, FakeRunCommandOpts]] = []

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            self.calls.append((command, opts))
            if "fail" in command:
                return SimpleNamespace(
                    logs=SimpleNamespace(stdout=[], stderr=[FakeLog("stderr")]),
                    error=SimpleNamespace(name="CommandError", value="failed"),
                    exit_code=None,
                )
            return SimpleNamespace(
                logs=SimpleNamespace(stdout=[FakeLog("stdout")], stderr=[]),
                error=None,
                exit_code=None,
            )

    class FakeFiles:
        def __init__(self) -> None:
            self.writes: list[tuple[str, str | bytes]] = []

        async def write_file(self, target_path: str, data: str | bytes) -> None:
            self.writes.append((target_path, data))

        async def read_bytes(self, source_path: str) -> bytes:
            return f"bytes:{source_path}".encode()

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()
            self.files = FakeFiles()

        async def get_info(self) -> Any:
            return SimpleNamespace(status=SimpleNamespace(state="RUNNING"))

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
    )
    raw = FakeRaw()
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=raw)

    result = await provider.exec(
        handle,
        "echo hello",
        cwd="/repo",
        env={"A": "B"},
        timeout_s=2,
        user=1000,
    )
    assert result == opensandbox_provider.SandboxExecResult(stdout="stdout", stderr=None, return_code=0)
    command, opts = raw.commands.calls[0]
    assert command == "echo hello"
    assert opts.kwargs == {
        "working_directory": "/repo",
        "envs": {"A": "B"},
        "timeout": timedelta(seconds=2),
        "uid": 1000,
    }

    result = await provider.exec(handle, "fail", user="agent")
    assert result.return_code == 125
    assert result.error_type == "sandbox"
    assert result.stderr == "stderr\nCommandError: failed"
    assert raw.commands.calls[1][0] == "su -s /bin/sh -c fail agent"

    upload_path = tmp_path / "upload.txt"
    upload_path.write_text("upload", encoding="utf-8")
    await provider.upload_file(handle, upload_path, "/remote/upload.txt")
    download_path = tmp_path / "nested" / "download.txt"
    await provider.download_file(handle, "/remote/download.txt", download_path)
    assert raw.files.writes == [("/remote/upload.txt", b"upload")]
    assert download_path.read_bytes() == b"bytes:/remote/download.txt"
    assert await provider.status(handle) == SandboxStatus.RUNNING
    bare_handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-2", provider_name="opensandbox", raw=object())
    assert await provider.status(bare_handle) == SandboxStatus.UNKNOWN


async def test_provider_create_probe_and_close_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"connect_poll_s": 0.01},
        probe={
            "command": "probe",
            "expected_stdout": "ready",
            "timeout_s": 1,
            "deadline_s": 0.01,
        },
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=object())

    async def bad_probe(*_args: Any, **_kwargs: Any) -> opensandbox_provider.SandboxExecResult:
        return opensandbox_provider.SandboxExecResult(stdout="not ready", stderr="bad", return_code=1)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(provider, "_exec", bad_probe)
    with pytest.raises(opensandbox_provider.OpenSandboxCreateVerificationError):
        await provider._verify_created_handle(handle)

    provider = opensandbox_provider.OpenSandboxProvider(
        probe={"command": "probe", "expected_stdout": None, "stable_count": 2, "stable_delay_s": 0.01},
    )
    sleep_calls: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    async def good_probe(*_args: Any, **_kwargs: Any) -> opensandbox_provider.SandboxExecResult:
        return opensandbox_provider.SandboxExecResult(stdout="ready", stderr=None, return_code=0)

    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(provider, "_exec", good_probe)
    await provider._verify_created_handle(handle)
    assert sleep_calls == [0.01]

    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": "probe"})

    async def cancelled_probe(*_args: Any, **_kwargs: Any) -> opensandbox_provider.SandboxExecResult:
        raise asyncio.CancelledError()

    monkeypatch.setattr(provider, "_exec", cancelled_probe)
    with pytest.raises(asyncio.CancelledError):
        await provider._verify_created_handle(handle)

    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    async def close_raises(_handle: Any) -> None:
        raise RuntimeError("close failed")

    monkeypatch.setattr(provider, "close", close_raises)
    await provider._cleanup_failed_create_handle(handle)
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    class StopAlreadyGoneRaw:
        async def kill(self) -> None:
            raise RuntimeError("sandbox sandbox-1 not found")

        async def close(self) -> None:
            return None

    await provider.close(
        opensandbox_provider.SandboxHandle(
            sandbox_id="sandbox-1",
            provider_name="opensandbox",
            raw=StopAlreadyGoneRaw(),
        ),
    )

    class StopAndCloseFailRaw:
        async def kill(self) -> None:
            raise RuntimeError("stop failed")

        async def close(self) -> None:
            raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="Failed to stop and close"):
        await provider.close(
            opensandbox_provider.SandboxHandle(
                sandbox_id="sandbox-2",
                provider_name="opensandbox",
                raw=StopAndCloseFailRaw(),
            ),
        )

    class StopFailsCloseSucceedsRaw:
        async def kill(self) -> None:
            raise RuntimeError("stop failed")

        async def close(self) -> None:
            return None

    with pytest.raises(RuntimeError, match="stop failed"):
        await provider.close(
            opensandbox_provider.SandboxHandle(
                sandbox_id="sandbox-3",
                provider_name="opensandbox",
                raw=StopFailsCloseSucceedsRaw(),
            ),
        )


async def test_create_once_and_connect_after_create_error_paths(
    fake_opensandbox_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"timeout_s": 1, "skip_health_check": True},
        probe={"command": None},
    )
    monkeypatch.setattr(opensandbox_provider, "_to_volumes", lambda volumes: volumes)
    spec = SandboxSpec(
        image="image:tag",
        ttl_s=10,
        ready_timeout_s=20,
        resources=SandboxResources(cpu=2, memory_mib=8192, disk_gib=20, gpu=1, gpu_type="H100"),
        entrypoint=["/bin/sh"],
        provider_options={
            "snapshot_id": "snapshot-1",
            "platform": {"os": "linux", "arch": "amd64"},
            "volumes": [{"name": "workspace"}],
            "skip_health_check": False,
        },
    )
    handle = await provider._create_once(spec)
    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.created_kwargs["snapshot_id"] == "snapshot-1"
    assert FakeSandbox.created_kwargs["timeout"] == timedelta(seconds=10)
    assert FakeSandbox.created_kwargs["ready_timeout"] == timedelta(seconds=20)
    assert FakeSandbox.created_kwargs["resource"] == {
        "cpu": "2",
        "memory": "8192Mi",
        "ephemeral-storage": "20Gi",
        "gpu": "1",
        "gpu_type": "H100",
    }
    assert FakeSandbox.created_kwargs["entrypoint"] == ["/bin/sh"]
    assert FakeSandbox.created_kwargs["platform"] == FakePlatformSpec(os="linux", arch="amd64")
    assert FakeSandbox.created_kwargs["volumes"] == [{"name": "workspace"}]
    assert FakeSandbox.created_kwargs["skip_health_check"] is True

    class FailingConnectSandbox(FakeSandbox):
        @classmethod
        async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
            del args, kwargs
            raise ConnectionError("pod may still be starting")

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FailingConnectSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"connect_attempt_timeout_s": 0.01, "connect_poll_s": 0.01},
        probe={"command": None},
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", no_sleep)
    with pytest.raises(opensandbox_provider.OpenSandboxCreateTimeoutError):
        await provider._connect_after_create(
            opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=None),
            SandboxSpec(image="image:tag"),
        )

    class CancelledConnectSandbox(FakeSandbox):
        @classmethod
        async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
            del args, kwargs
            raise asyncio.CancelledError()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (CancelledConnectSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with pytest.raises(asyncio.CancelledError):
        await provider._connect_after_create(
            opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=None),
            SandboxSpec(image="image:tag", ready_timeout_s=1),
        )

    class NonRetryableConnectSandbox(FakeSandbox):
        @classmethod
        async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
            del args, kwargs
            raise ValueError("bad connection request")

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (NonRetryableConnectSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with pytest.raises(ValueError, match="bad connection request"):
        await provider._connect_after_create(
            opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=None),
            SandboxSpec(image="image:tag", ready_timeout_s=1),
        )

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FakeSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 3},
        probe={"command": None},
    )
    handle = await provider._create_once(SandboxSpec(image="image:tag", provider_options={"skip_health_check": True}))
    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.created_kwargs["skip_health_check"] is True

    class TimeoutSandbox(FakeSandbox):
        @classmethod
        async def create(cls, **_kwargs: Any) -> "FakeSandbox":
            await asyncio.get_running_loop().create_future()
            return cls()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (TimeoutSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"timeout_s": 0.01},
        probe={"command": None},
    )
    with pytest.raises(opensandbox_provider.OpenSandboxCreateTimeoutError):
        await provider._create_once(SandboxSpec(image="image:tag"))

    class EmptyCreateSandbox(FakeSandbox):
        @classmethod
        async def create(cls, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (EmptyCreateSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with pytest.raises(RuntimeError, match="returned no sandbox handle"):
        await provider._create_once(SandboxSpec(image="image:tag"))

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FakeSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": "probe"})
    cleanup_calls: list[str] = []

    async def fail_verify(_handle: opensandbox_provider.SandboxHandle) -> None:
        raise RuntimeError("probe failed")

    async def cleanup(handle: opensandbox_provider.SandboxHandle) -> None:
        cleanup_calls.append(handle.sandbox_id)

    monkeypatch.setattr(provider, "_verify_created_handle", fail_verify)
    monkeypatch.setattr(provider, "_cleanup_failed_create_handle", cleanup)
    with pytest.raises(RuntimeError, match="probe failed"):
        await provider._create_once(SandboxSpec(image="image:tag"))
    assert cleanup_calls == ["sandbox-1"]


async def test_retry_classification_and_await_sdk_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 0},
        probe={"command": None},
    )
    assert await provider.aclose() is None
    assert await provider._await_sdk_call(_return_value("ok"), operation="op", sandbox_id="sandbox-1", timeout_s=None)
    assert opensandbox_provider._is_retryable_sdk_operation_error(TimeoutError("command timeout")) is False
    assert opensandbox_provider._is_retryable_sdk_operation_error(ConnectionError("connection failed")) is True
    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = ConnectionError("connection reset")
    assert opensandbox_provider._is_retryable_sdk_operation_error(wrapped) is True
    wrapped.__cause__ = wrapped
    assert opensandbox_provider._is_retryable_sdk_operation_error(wrapped) is False

    from opensandbox.exceptions import SandboxApiException  # noqa: PLC0415

    cyclic_api_error = SandboxApiException("proxy failed")
    cyclic_api_error.status_code = 500
    cyclic_api_error.__cause__ = cyclic_api_error
    assert opensandbox_provider._is_retryable_sdk_operation_error(cyclic_api_error) is True

    async def cancelled() -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await provider._await_sdk_operation(
            cancelled,
            operation="cancelled",
            sandbox_id="sandbox-1",
            timeout_s=None,
        )


async def _return_value(value: Any) -> Any:
    return value
