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

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)


pytestmark = pytest.mark.sandbox


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
    with pytest.raises(TypeError, match="extensions"):
        opensandbox_provider.OpenSandboxProviderOptions.from_mapping({"extensions": ["not", "a", "mapping"]})
    with pytest.raises(TypeError, match="must be a bool"):
        opensandbox_provider.OpenSandboxProviderOptions.from_mapping({"skip_health_check": "true"})

    assert opensandbox_provider._resource_map(SandboxResources(cpu=2.0))["cpu"] == "2"
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


def test_provider_options_from_mapping() -> None:
    options_cls = opensandbox_provider.OpenSandboxProviderOptions

    assert options_cls.from_mapping(None) == options_cls()

    parsed = options_cls.from_mapping(
        {
            "platform": {"os": "linux", "arch": "amd64"},
            "snapshot_id": "snap-1",
            "volumes": [{"name": "workspace"}],
            "skip_health_check": True,
            "extensions": {"imagePullPolicy": "Never"},
        }
    )
    assert parsed.platform == {"os": "linux", "arch": "amd64"}
    assert parsed.snapshot_id == "snap-1"
    assert parsed.volumes == ({"name": "workspace"},)
    assert parsed.skip_health_check is True
    assert parsed.extensions == {"imagePullPolicy": "Never"}

    with pytest.raises(ValueError, match="Unknown OpenSandbox provider option"):
        options_cls.from_mapping({"bogus": 1})
    with pytest.raises(TypeError, match="provider_options must be a mapping"):
        options_cls.from_mapping(["not", "a", "mapping"])
    with pytest.raises(TypeError, match="'platform' must be a mapping"):
        options_cls.from_mapping({"platform": "linux/amd64"})
    with pytest.raises(TypeError, match="'snapshot_id' must be a string"):
        options_cls.from_mapping({"snapshot_id": 123})
    with pytest.raises(TypeError, match="'volumes' must be a list of mappings"):
        options_cls.from_mapping({"volumes": ["workspace"]})


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

    extensions = provider._resolve_extensions({"imagePullPolicy": "Never"})
    assert extensions["imagePullPolicy"] == "Never"
    assert extensions["opensandbox.extensions.image-pull-policy"] == "Never"

    no_policy_provider = opensandbox_provider.OpenSandboxProvider(create={"image_pull_policy": None})
    assert no_policy_provider._resolve_extensions({"imagePullPolicy": "Never"}) == {"imagePullPolicy": "Never"}


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


async def test_retry_loop_empty_iterator_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyAsyncRetrying:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __aiter__(self) -> "EmptyAsyncRetrying":
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_tenacity",
        lambda: (EmptyAsyncRetrying, lambda predicate: predicate, lambda attempts: attempts, lambda **kwargs: kwargs),
    )

    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with pytest.raises(RuntimeError, match="SDK operation retry loop did not run"):
        await provider._await_sdk_operation(
            lambda: _return_value("ok"),
            operation="noop",
            sandbox_id="sandbox-1",
            timeout_s=None,
        )
    with pytest.raises(opensandbox_provider.OpenSandboxCreateError, match="create retry loop did not run"):
        await provider._create_with_retries(SandboxSpec(image="image:tag"))


async def _return_value(value: Any) -> Any:
    return value


def test_opensandbox_sdk_create_receives_default_image_pull_policy(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_sdk_create_receives_default_image_pull_policy(monkeypatch))


async def _assert_opensandbox_sdk_create_receives_default_image_pull_policy(monkeypatch) -> None:
    class FakeSDKSandbox:
        create_calls: list[dict[str, Any]] = []

        def __init__(self, sandbox_id: str) -> None:
            self.id = sandbox_id

        @classmethod
        async def create(cls, **kwargs: Any) -> "FakeSDKSandbox":
            cls.create_calls.append(kwargs)
            return cls("sdk-sandbox-1")

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FakeSDKSandbox, object, object, object, object),
    )

    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    monkeypatch.setattr(provider, "_connection_config", lambda request_timeout_s=None: object())

    handle = await provider.create(
        SandboxSpec(
            image="image:tag",
            metadata={
                "harbor_instance_id": "swebench::django__django-10880",
                "long": f"bad:{'x' * 80}:",
            },
        )
    )

    assert handle.sandbox_id == "sdk-sandbox-1"
    metadata = FakeSDKSandbox.create_calls[0]["metadata"]
    assert metadata["harbor_instance_id"] == "swebench_django__django-10880"
    assert metadata["long"] == ("bad_" + "x" * 59)
    extensions = FakeSDKSandbox.create_calls[0]["extensions"]
    assert extensions[opensandbox_provider.IMAGE_PULL_POLICY_EXTENSION_KEY] == "IfNotPresent"
    assert extensions[opensandbox_provider.IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY] == "IfNotPresent"


def test_opensandbox_connect_after_create_preserves_request_timeout(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_connect_after_create_preserves_request_timeout(monkeypatch))


async def _assert_opensandbox_connect_after_create_preserves_request_timeout(monkeypatch) -> None:
    class FakeConnectionConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeSDKSandbox:
        connect_calls: list[dict[str, Any]] = []

        def __init__(self, sandbox_id: str) -> None:
            self.id = sandbox_id

        @classmethod
        async def connect(cls, sandbox_id: str, **kwargs: Any) -> "FakeSDKSandbox":
            cls.connect_calls.append({"sandbox_id": sandbox_id, **kwargs})
            return cls(sandbox_id)

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FakeSDKSandbox, FakeConnectionConfig, object, object, object),
    )

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"domain": "sandbox.example", "protocol": "https", "request_timeout_s": 300},
        create={"connect_attempt_timeout_s": 1},
        probe={"command": None},
    )
    handle = await provider._connect_after_create(
        SandboxHandle(sandbox_id="sdk-sandbox-1", provider_name="opensandbox", raw=None),
        SandboxSpec(image="image:tag", ready_timeout_s=10),
    )

    assert handle.sandbox_id == "sdk-sandbox-1"
    assert isinstance(handle.raw, FakeSDKSandbox)
    connect_call = FakeSDKSandbox.connect_calls[0]
    assert connect_call["skip_health_check"] is True
    assert connect_call["connection_config"].kwargs == {
        "domain": "sandbox.example",
        "protocol": "https",
        "request_timeout": timedelta(seconds=300),
    }


def test_opensandbox_create_probe_can_require_stable_successes(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_create_probe_can_require_stable_successes(monkeypatch))


async def _assert_opensandbox_create_probe_can_require_stable_successes(monkeypatch) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        probe={
            "command": "true",
            "expected_stdout": None,
            "stable_count": 3,
            "stable_delay_s": 0,
        },
    )
    calls: list[dict[str, Any]] = []

    async def fake_exec(
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        calls.append(
            {
                "handle": handle,
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_s": timeout_s,
                "user": user,
            }
        )
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(provider, "_exec", fake_exec)
    handle = SandboxHandle(sandbox_id="sdk-sandbox-0", provider_name="opensandbox", raw=object())

    await provider._verify_created_handle(handle)

    assert [call["command"] for call in calls] == ["true", "true", "true"]
    assert all(call["timeout_s"] == 30 for call in calls)
    assert all(call["user"] == "root" for call in calls)


def test_opensandbox_create_probe_polls_same_sandbox_after_transient_errors(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_create_probe_polls_same_sandbox_after_transient_errors(monkeypatch))


async def _assert_opensandbox_create_probe_polls_same_sandbox_after_transient_errors(monkeypatch) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"connect_poll_s": 0.01},
        probe={
            "command": "true",
            "expected_stdout": None,
            "timeout_s": 1,
            "deadline_s": 2,
            "stable_count": 2,
            "stable_delay_s": 0,
        },
    )
    attempts = 0
    handles: list[SandboxHandle] = []

    async def fake_exec(
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        del command, cwd, env, timeout_s, user
        nonlocal attempts
        attempts += 1
        handles.append(handle)
        if attempts <= 2:
            raise ConnectionError("direct execd endpoint is not accepting connections yet")
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(provider, "_exec", fake_exec)
    handle = SandboxHandle(sandbox_id="sdk-sandbox-0", provider_name="opensandbox", raw=object())

    await provider._verify_created_handle(handle)

    assert attempts == 4
    assert {seen_handle.sandbox_id for seen_handle in handles} == {"sdk-sandbox-0"}


def test_opensandbox_create_probe_failures_are_retryable() -> None:
    error = opensandbox_provider.OpenSandboxCreateVerificationError("pod sdk-sandbox-0 failed create probe")

    assert isinstance(error, SandboxCreateError)
    assert opensandbox_provider._is_retryable_create_error(error) is True


def test_opensandbox_starting_pod_endpoint_errors_are_retryable() -> None:
    error = RuntimeError(
        "Get endpoint for sandbox sdk-sandbox-0 port 44772 failed: "
        "Pod IP is not yet available. The Pod may still be starting."
    )

    assert opensandbox_provider._is_retryable_create_error(error) is True


def test_opensandbox_exec_retries_retryable_sdk_failures(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_exec_retries_retryable_sdk_failures(monkeypatch))


async def _assert_opensandbox_exec_retries_retryable_sdk_failures(monkeypatch) -> None:
    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeLog:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeLogs:
        stdout = [FakeLog("ok")]
        stderr: list[FakeLog] = []

    class FakeExecution:
        logs = FakeLogs()
        error = None
        exit_code = 0

    class FakeCommands:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> FakeExecution:
            del command, opts
            self.calls += 1
            if self.calls <= 2:
                raise ConnectionError("transient connection failure")
            return FakeExecution()

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )

    provider = opensandbox_provider.OpenSandboxProvider(
        operations={
            "retries": 2,
            "retry_delay_s": 0,
            "retry_max_delay_s": 0,
            "command_retries": 2,
        },
        probe={"command": None},
    )
    raw = FakeRaw()
    handle = SandboxHandle(sandbox_id="sdk-sandbox-1", provider_name="opensandbox", raw=raw)

    result = await provider.exec(handle, "echo hello", timeout_s=30)

    assert result.stdout == "ok"
    assert result.return_code == 0
    assert raw.commands.calls == 3


def test_opensandbox_command_retries_default_to_disabled(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_command_retries_default_to_disabled(monkeypatch))


async def _assert_opensandbox_command_retries_default_to_disabled(monkeypatch) -> None:
    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> None:
            del command, opts
            self.calls += 1
            raise ConnectionError("transient connection failure")

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )

    provider = opensandbox_provider.OpenSandboxProvider(
        operations={
            "retries": 2,
            "retry_delay_s": 0,
            "retry_max_delay_s": 0,
        },
        probe={"command": None},
    )
    raw = FakeRaw()
    handle = SandboxHandle(sandbox_id="sdk-sandbox-1", provider_name="opensandbox", raw=raw)

    try:
        await provider.exec(handle, "echo hello", timeout_s=30)
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected provider.exec to propagate the command failure")

    assert raw.commands.calls == 1


def test_opensandbox_close_timeout_does_not_fail_after_stop() -> None:
    asyncio.run(_assert_opensandbox_close_timeout_does_not_fail_after_stop())


async def _assert_opensandbox_close_timeout_does_not_fail_after_stop() -> None:
    class SlowCloseRaw:
        def __init__(self) -> None:
            self.killed = False

        async def kill(self) -> None:
            self.killed = True

        async def close(self) -> None:
            await asyncio.sleep(60)

    raw = SlowCloseRaw()
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"close_timeout_s": 0.01},
        probe={"command": None},
    )
    handle = SandboxHandle(sandbox_id="sdk-sandbox-1", provider_name="opensandbox", raw=raw)

    await provider.close(handle)

    assert raw.killed is True


def test_opensandbox_close_propagates_stop_failure() -> None:
    asyncio.run(_assert_opensandbox_close_propagates_stop_failure())


async def _assert_opensandbox_close_propagates_stop_failure() -> None:
    class StopFailureRaw:
        async def kill(self) -> None:
            raise RuntimeError("stop failed")

        async def close(self) -> None:
            return None

    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"close_timeout_s": 0.01},
        probe={"command": None},
    )
    handle = SandboxHandle(sandbox_id="sdk-sandbox-1", provider_name="opensandbox", raw=StopFailureRaw())

    with pytest.raises(RuntimeError, match="stop failed"):
        await provider.close(handle)
