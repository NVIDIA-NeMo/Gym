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
import importlib.util
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tenacity") is None,
    reason="tenacity optional sandbox dependency is not installed",
)


@dataclass(frozen=True)
class FakePlatformSpec:
    os: str
    arch: str


class FakeConnectionConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


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


@dataclass
class FakePoolCreationSpec:
    image: str
    entrypoint: list[str] | None = None
    resource: dict[str, str] | None = None
    env: dict[str, str] | None = None
    metadata: dict[str, str] | None = None
    extensions: dict[str, str] | None = None
    platform: Any | None = None
    volumes: list[Any] | None = None


class FakeAcquirePolicy:
    FAIL_FAST = "fail_fast"


class FakeStateStore:
    pass


class FakeSnapshot:
    idle_count = 1
    state = None


class FakeSandboxPoolAsync:
    received_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        self.received_kwargs = kwargs
        type(self).received_kwargs = kwargs

    async def start(self) -> None:
        return None

    async def snapshot(self) -> FakeSnapshot:
        return FakeSnapshot()

    async def resize(self, _count: int) -> None:
        return None

    async def acquire(
        self,
        *,
        sandbox_timeout: timedelta | None,
        policy: str,
    ) -> FakeSandbox:
        del sandbox_timeout, policy
        creation_spec = self.received_kwargs["creation_spec"]
        return await FakeSandbox.create(
            creation_spec.image,
            platform=creation_spec.platform,
        )

    async def shutdown(self, *, graceful: bool) -> None:
        del graceful

    async def release_all_idle(self) -> None:
        return None


@pytest.fixture
def fake_opensandbox_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    def require_sdk() -> tuple[Any, Any, Any, Any, Any]:
        return FakeSandbox, FakeConnectionConfig, object, FakePlatformSpec, object

    def require_sdk_pool() -> tuple[Any, Any, Any, Any]:
        return (
            FakeAcquirePolicy,
            FakeStateStore,
            FakePoolCreationSpec,
            FakeSandboxPoolAsync,
        )

    monkeypatch.setattr(opensandbox_provider, "_require_opensandbox_sdk", require_sdk)
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk_pool",
        require_sdk_pool,
    )


async def test_sdk_pool_passes_platform_through_pool_creation_spec(
    fake_opensandbox_sdk: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
    )

    handles = await provider.create_batch(
        SandboxSpec(
            image="mirror.gcr.io/astral/uv:python3.12-bookworm-slim",
            platform={"os": "linux", "arch": "amd64"},
        ),
        1,
    )

    assert len(handles) == 1
    assert handles[0].sandbox_id == "sandbox-1"
    assert "sandbox_factory" not in FakeSandboxPoolAsync.received_kwargs
    assert FakeSandbox.created_kwargs["platform"] == FakePlatformSpec(
        os="linux",
        arch="amd64",
    )


async def test_connect_passes_configured_connect_timeout(
    fake_opensandbox_sdk: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"connect_timeout_s": 300, "request_timeout_s": 10},
        probe={"command": None},
    )

    handle = await provider.connect("sandbox-123")

    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.connected_args == ("sandbox-123",)
    assert FakeSandbox.connected_kwargs["connect_timeout"] == timedelta(seconds=300)


def test_provider_validation_and_retry_helpers() -> None:
    with pytest.raises(ValueError, match="image_pull_policy"):
        opensandbox_provider.validate_image_pull_policy("Sometimes")

    invalid_kwargs = [
        {"pool": {"concurrency": 0}},
        {"connection": {"connect_timeout_s": 0}},
        {"pool": {"progress_timeout_s": 0}},
        {"create": {"timeout_s": 0}},
        {"probe": {"timeout_s": 0}},
        {"probe": {"deadline_s": 0}},
        {"probe": {"sample_count": 0}},
        {"probe": {"stable_count": 0}},
        {"probe": {"stable_delay_s": -1}},
        {"create": {"retries": -1}},
        {"create": {"retry_delay_s": -1}},
        {"create": {"retry_max_delay_s": -1}},
        {"operations": {"retries": -1}},
        {"operations": {"retry_delay_s": -1}},
        {"operations": {"retry_max_delay_s": -1}},
        {"operations": {"command_retries": -1}},
        {"pool": {"reconcile_interval_s": 0}},
        {"pool": {"acquire_poll_interval_s": 0}},
        {"pool": {"idle_timeout_s": 0}},
        {"pool": {"primary_lock_ttl_s": 0}},
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
    assert opensandbox_provider._seconds_to_timedelta(None) is None
    assert opensandbox_provider._seconds_to_timedelta(1.5) == timedelta(seconds=1.5)


def test_connection_config_exec_proxy_and_image_policy(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "sandbox.example",
            "api_key": "key",  # pragma: allowlist secret
            "protocol": "https",
            "use_server_proxy": True,
            "exec_use_server_proxy": False,
            "request_timeout_s": 10,
        }
    )

    config = provider._connection_config()
    assert config.kwargs == {
        "domain": "sandbox.example",
        "api_key": "key",  # pragma: allowlist secret
        "protocol": "https",
        "use_server_proxy": True,
        "request_timeout": timedelta(seconds=10),
    }
    exec_config = provider._exec_connection_config(request_timeout_s=3)
    assert exec_config.kwargs["use_server_proxy"] is False
    assert exec_config.kwargs["request_timeout"] == timedelta(seconds=3)

    spec = SandboxSpec(image="image:tag", extensions={"imagePullPolicy": "Never"})
    updated = provider._with_default_image_pull_policy(spec)
    assert updated.extensions["imagePullPolicy"] == "Never"
    assert updated.extensions["opensandbox.extensions.image-pull-policy"] == "Never"

    no_policy_provider = opensandbox_provider.OpenSandboxProvider(create={"image_pull_policy": None})
    assert no_policy_provider._with_default_image_pull_policy(spec) is spec


async def test_wait_sdk_pool_idle_success_partial_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class Snapshot:
        def __init__(self, idle_count: int) -> None:
            self.idle_count = idle_count
            self.state = SimpleNamespace(value="warming")

    class FakePool:
        def __init__(self, counts: list[int]) -> None:
            self.counts = counts
            self.index = 0
            self._config = SimpleNamespace(pool_name="pool-1")

        async def snapshot(self) -> Snapshot:
            count = self.counts[min(self.index, len(self.counts) - 1)]
            self.index += 1
            return Snapshot(count)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", no_sleep)

    provider = opensandbox_provider.OpenSandboxProvider(
        pool={"acquire_poll_interval_s": 0.01},
        probe={"command": None},
    )
    assert (
        await provider._wait_sdk_pool_idle(
            FakePool([0, 1, 2]),
            spec=SandboxSpec(image="image:tag"),
            requested=2,
            timeout_s=1,
            allow_partial=False,
        )
        == 2
    )
    assert (
        await provider._wait_sdk_pool_idle(
            FakePool([1]),
            spec=SandboxSpec(image="image:tag"),
            requested=2,
            timeout_s=0,
            allow_partial=True,
        )
        == 1
    )
    with pytest.raises(opensandbox_provider.OpenSandboxCreateTimeoutError):
        await provider._wait_sdk_pool_idle(
            FakePool([0]),
            spec=SandboxSpec(image="image:tag"),
            requested=2,
            timeout_s=0,
            allow_partial=False,
        )


async def test_exec_file_operations_and_batch_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    assert result.return_code == 1
    assert result.stderr == "stderr\nCommandError: failed"
    assert raw.commands.calls[1][0] == "su -s /bin/sh -c fail agent"

    await provider.write_file(handle, "/tmp/file.txt", "contents")
    assert await provider.read_file(handle, "/tmp/file.txt") == b"bytes:/tmp/file.txt"
    upload_path = tmp_path / "upload.txt"
    upload_path.write_text("upload", encoding="utf-8")
    await provider.upload_file(handle, upload_path, "/remote/upload.txt")
    download_path = tmp_path / "nested" / "download.txt"
    await provider.download_file(handle, "/remote/download.txt", download_path)
    assert raw.files.writes == [("/tmp/file.txt", "contents"), ("/remote/upload.txt", b"upload")]
    assert download_path.read_bytes() == b"bytes:/remote/download.txt"

    with pytest.raises(ValueError, match="count"):
        await provider._create_batch_sdk(SandboxSpec(image="image:tag"), 0)
    with pytest.raises(ValueError, match="count"):
        await provider.create_batch(SandboxSpec(image="image:tag"), 0)
    with pytest.raises(ValueError, match="snapshot_id"):
        provider._validate_sdk_pool_spec(SandboxSpec(image="image:tag", snapshot_id="snapshot"))
    with pytest.raises(ValueError, match="Unsupported"):
        await provider.materialize_handle({"kind": "other"})


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

    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": "probe"})

    async def fail_verify(_handle: Any) -> None:
        raise RuntimeError("probe failed")

    monkeypatch.setattr(provider, "_verify_created_handle", fail_verify)
    with pytest.raises(opensandbox_provider.OpenSandboxCreateVerificationError):
        await provider._verify_created_handles([handle, handle])

    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    await provider._verify_created_handles([])

    async def close_raises(_handle: Any, *, delete: bool) -> None:
        del delete
        raise RuntimeError("close failed")

    monkeypatch.setattr(provider, "close", close_raises)
    await provider._cleanup_failed_create_handle(handle)
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    class DeleteAlreadyGoneRaw:
        async def kill(self) -> None:
            raise RuntimeError("sandbox sandbox-1 not found")

        async def close(self) -> None:
            return None

    await provider.close(
        opensandbox_provider.SandboxHandle(
            sandbox_id="sandbox-1",
            provider_name="opensandbox",
            raw=DeleteAlreadyGoneRaw(),
        ),
        delete=True,
    )

    class DeleteAndCloseFailRaw:
        async def kill(self) -> None:
            raise RuntimeError("delete failed")

        async def close(self) -> None:
            raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="Failed to delete and close"):
        await provider.close(
            opensandbox_provider.SandboxHandle(
                sandbox_id="sandbox-2",
                provider_name="opensandbox",
                raw=DeleteAndCloseFailRaw(),
            ),
            delete=True,
        )


async def test_create_once_and_connect_after_create_error_paths(
    fake_opensandbox_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"use_server_proxy": False},
        probe={"command": None},
    )
    with pytest.raises(ValueError, match="pooled creation"):
        await provider._create_once(SandboxSpec(image="image:tag", extensions={"poolRef": "pool"}))

    provider = opensandbox_provider.OpenSandboxProvider(
        create={"timeout_s": 1, "skip_health_check": True},
        probe={"command": None},
    )
    monkeypatch.setattr(opensandbox_provider, "_to_volumes", lambda volumes: volumes)
    spec = SandboxSpec(
        image="image:tag",
        snapshot_id="snapshot-1",
        timeout_s=10,
        ready_timeout_s=20,
        entrypoint=["/bin/sh"],
        platform={"os": "linux", "arch": "amd64"},
        volumes=[{"name": "workspace"}],
        skip_health_check=False,
    )
    handle = await provider._create_once(spec)
    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.created_kwargs["snapshot_id"] == "snapshot-1"
    assert FakeSandbox.created_kwargs["timeout"] == timedelta(seconds=10)
    assert FakeSandbox.created_kwargs["ready_timeout"] == timedelta(seconds=20)
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


async def test_retry_classification_and_await_sdk_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 0},
        probe={"command": None},
    )
    assert await provider.aclose() is None
    assert await provider._await_sdk_call(_return_value("ok"), operation="op", sandbox_id="sandbox-1", timeout_s=None)
    assert opensandbox_provider._is_retryable_sdk_operation_error(TimeoutError("command timeout")) is False
    assert opensandbox_provider._is_retryable_sdk_operation_error(ConnectionError("proxy failed")) is True
    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = ConnectionError("connection reset")
    assert opensandbox_provider._is_retryable_sdk_operation_error(wrapped) is True

    class FakeHttpxConnectError(Exception):
        pass

    monkeypatch.setattr(opensandbox_provider, "_httpx_retryable_types", lambda: (FakeHttpxConnectError,))
    assert opensandbox_provider._is_retryable_create_error(FakeHttpxConnectError("temporary")) is True
    assert opensandbox_provider._is_retryable_sdk_operation_error(FakeHttpxConnectError("temporary")) is True

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
