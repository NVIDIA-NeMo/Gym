# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import asyncio
import importlib.util
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from nemo_gym.sandbox import (
    AsyncSandbox,
    Sandbox,
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    get_provider_class,
    list_providers,
    register_provider,
    rewrite_image,
)
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider_module
from nemo_gym.sandbox.providers.opensandbox.provider import (
    IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY,
    IMAGE_PULL_POLICY_EXTENSION_KEY,
    OpenSandboxCreateVerificationError,
    OpenSandboxProvider,
)
from responses_api_agents.mini_swe_agent_2.sandbox_environment import MiniSWESandboxEnvironment


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


requires_tenacity = pytest.mark.skipif(
    not _has_module("tenacity"),
    reason="tenacity optional sandbox dependency is not installed",
)


class FakeSandboxProvider:
    name = "fake"
    last_instance: "FakeSandboxProvider | None" = None

    def __init__(self, marker: str = "default") -> None:
        self.marker = marker
        self.created_specs: list[SandboxSpec] = []
        self.exec_calls: list[dict[str, Any]] = []
        self.write_calls: list[tuple[SandboxHandle, str, str | bytes]] = []
        self.read_calls: list[tuple[SandboxHandle, str]] = []
        self.upload_calls: list[tuple[SandboxHandle, Path, str]] = []
        self.download_calls: list[tuple[SandboxHandle, str, Path]] = []
        self.closed: list[tuple[SandboxHandle, bool]] = []
        self.aclosed = False
        FakeSandboxProvider.last_instance = self

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self.created_specs.append(spec)
        return SandboxHandle(sandbox_id="fake-1", provider_name=self.name, raw={"spec": spec})

    async def create_batch(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        del allow_partial
        return [await self.create(spec) for _ in range(count)]

    async def connect(self, sandbox_id: str) -> SandboxHandle:
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw={})

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        self.exec_calls.append(
            {
                "handle": handle,
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_s": timeout_s,
                "user": user,
            }
        )
        return SandboxExecResult(stdout="ok", stderr=None, return_code=0)

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        self.write_calls.append((handle, target_path, data))

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        self.read_calls.append((handle, source_path))
        return f"read:{source_path}".encode()

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        self.upload_calls.append((handle, source_path, target_path))

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        self.download_calls.append((handle, source_path, target_path))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"downloaded")

    async def close(self, handle: SandboxHandle, *, delete: bool) -> None:
        self.closed.append((handle, delete))

    async def aclose(self) -> None:
        self.aclosed = True

    def handle_reference(self, handle: SandboxHandle) -> dict[str, str]:
        return {"kind": "fake", "sandbox_id": handle.sandbox_id}

    async def materialize_handle(self, value: Any) -> SandboxHandle:
        return SandboxHandle(sandbox_id=value["sandbox_id"], provider_name=self.name, raw={"materialized": True})


def test_sandbox_facade_uses_public_provider_api() -> None:
    asyncio.run(_assert_sandbox_facade_uses_public_provider_api())


async def _assert_sandbox_facade_uses_public_provider_api() -> None:
    provider_name = f"fake-{uuid4().hex}"
    register_provider(provider_name, FakeSandboxProvider)

    sandbox = AsyncSandbox({provider_name: {"marker": "configured"}})
    handle = await sandbox.create(SandboxSpec(image="image:tag", metadata={"suite": "unit"}))

    provider = FakeSandboxProvider.last_instance
    assert provider is not None
    assert provider.marker == "configured"
    assert provider.created_specs[0].image == "image:tag"
    assert provider.created_specs[0].metadata == {"suite": "unit"}

    result = await sandbox.exec(handle, "pytest -q", cwd="/repo", timeout_s=60, user="agent")
    assert result == SandboxExecResult(stdout="ok", stderr=None, return_code=0)
    assert provider.exec_calls[0] == {
        "handle": handle,
        "command": "pytest -q",
        "cwd": "/repo",
        "env": None,
        "timeout_s": 60,
        "user": "agent",
    }

    await sandbox.delete(handle)
    assert provider.closed[0] == (handle, True)
    assert sandbox.handle_reference(handle) == {"kind": "fake", "sandbox_id": "fake-1"}
    assert await sandbox.materialize_handle({"sandbox_id": "fake-2"}) == SandboxHandle(
        sandbox_id="fake-2", provider_name="fake", raw={"materialized": True}
    )
    async with AsyncSandbox(provider) as context_sandbox:
        assert context_sandbox.provider_name == "fake"
    await sandbox.shutdown()
    assert provider.aclosed is True


def test_rewrite_image_and_materialize_handle_validation() -> None:
    asyncio.run(_assert_rewrite_image_and_materialize_handle_validation())


def test_provider_registry_validation_and_listing() -> None:
    provider_name = f"fake-{uuid4().hex}"
    register_provider(provider_name, FakeSandboxProvider)

    assert get_provider_class(provider_name) is FakeSandboxProvider
    assert provider_name in list_providers()
    with pytest.raises(ValueError, match="must be non-empty"):
        register_provider("", FakeSandboxProvider)
    with pytest.raises(ValueError, match="already registered"):
        register_provider(provider_name, FakeSandboxProvider)
    with pytest.raises(ValueError, match="Unknown sandbox provider"):
        get_provider_class(f"missing-{uuid4().hex}")


async def _assert_rewrite_image_and_materialize_handle_validation() -> None:
    assert rewrite_image(None, []) is None
    assert rewrite_image("image:tag", [{"from": "other/", "to": "mirror/"}]) == "image:tag"

    class BadMaterializeProvider(FakeSandboxProvider):
        async def materialize_handle(self, value: Any) -> object:
            del value
            return object()

    sandbox = AsyncSandbox(BadMaterializeProvider())
    try:
        await sandbox.materialize_handle({"sandbox_id": "bad"})
    except TypeError as e:
        assert "must return SandboxHandle" in str(e)
    else:
        raise AssertionError("expected invalid materialize_handle return type to fail")


def test_async_sandbox_batch_file_and_fallback_reference_operations(tmp_path: Path) -> None:
    asyncio.run(_assert_async_sandbox_batch_file_and_fallback_reference_operations(tmp_path))


async def _assert_async_sandbox_batch_file_and_fallback_reference_operations(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    sandbox = AsyncSandbox(provider)

    handles = await sandbox.create_batch(SandboxSpec(image="image:tag"), 2, allow_partial=True)
    connected = await sandbox.connect("connected-1")
    await sandbox.write_file(connected, "/tmp/file.txt", "contents")
    assert await sandbox.read_file(connected, "/tmp/file.txt") == b"read:/tmp/file.txt"
    source_path = tmp_path / "source.txt"
    target_path = tmp_path / "nested" / "target.txt"
    source_path.write_text("local", encoding="utf-8")
    await sandbox.upload_file(connected, source_path, "/remote/source.txt")
    await sandbox.download_file(connected, "/remote/source.txt", target_path)
    await sandbox.close(connected)

    assert [handle.sandbox_id for handle in handles] == ["fake-1", "fake-1"]
    assert provider.write_calls == [(connected, "/tmp/file.txt", "contents")]
    assert provider.read_calls == [(connected, "/tmp/file.txt")]
    assert provider.upload_calls == [(connected, source_path, "/remote/source.txt")]
    assert provider.download_calls == [(connected, "/remote/source.txt", target_path)]
    assert target_path.read_bytes() == b"downloaded"

    plain_provider = FakeSandboxProvider()
    plain_provider.handle_reference = None  # type: ignore[method-assign]
    plain_provider.materialize_handle = None  # type: ignore[method-assign]
    plain_sandbox = AsyncSandbox(plain_provider)
    plain_handle = SandboxHandle(sandbox_id="plain-1", provider_name="fake", raw={})
    assert plain_sandbox.handle_reference(plain_handle) is plain_handle
    assert await plain_sandbox.materialize_handle(plain_handle) is plain_handle
    try:
        await plain_sandbox.materialize_handle({"sandbox_id": "plain-2"})
    except ValueError as e:
        assert "cannot materialize" in str(e)
    else:
        raise AssertionError("expected materialize_handle without provider support to fail")


def test_sync_sandbox_facade_uses_public_provider_api() -> None:
    provider_name = f"fake-{uuid4().hex}"
    register_provider(provider_name, FakeSandboxProvider)

    with Sandbox({provider_name: {"marker": "configured"}}) as sandbox:
        handle = sandbox.create(SandboxSpec(image="image:tag", metadata={"suite": "unit"}))

        provider = FakeSandboxProvider.last_instance
        assert provider is not None
        assert provider.marker == "configured"
        assert provider.created_specs[0].image == "image:tag"
        assert provider.created_specs[0].metadata == {"suite": "unit"}

        result = sandbox.exec(handle, "pytest -q", cwd="/repo", timeout_s=60, user="agent")
        assert result == SandboxExecResult(stdout="ok", stderr=None, return_code=0)
        assert provider.exec_calls[0] == {
            "handle": handle,
            "command": "pytest -q",
            "cwd": "/repo",
            "env": None,
            "timeout_s": 60,
            "user": "agent",
        }

        sandbox.delete(handle)
        assert provider.closed[0] == (handle, True)
        assert sandbox.handle_reference(handle) == {"kind": "fake", "sandbox_id": "fake-1"}
        assert sandbox.materialize_handle({"sandbox_id": "fake-3"}).sandbox_id == "fake-3"
        assert sandbox.provider_name == "fake"
        assert len(sandbox.create_batch(SandboxSpec(image="image:tag"), 2)) == 2
        sandbox.shutdown()
        sandbox.shutdown()
        assert provider.aclosed is True
        try:
            sandbox.provider_name
        except RuntimeError as e:
            assert "sync loop is closed" in str(e)
        else:
            raise AssertionError("expected closed sync sandbox to reject further calls")


def test_sync_sandbox_file_operations(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    with Sandbox(provider) as sandbox:
        handle = sandbox.connect("sync-1")
        sandbox.write_file(handle, "/tmp/file.txt", b"contents")
        assert sandbox.read_file(handle, "/tmp/file.txt") == b"read:/tmp/file.txt"
        source_path = tmp_path / "source.txt"
        target_path = tmp_path / "target.txt"
        source_path.write_text("local", encoding="utf-8")
        sandbox.upload_file(handle, source_path, "/remote/source.txt")
        sandbox.download_file(handle, "/remote/source.txt", target_path)

    assert provider.write_calls == [(handle, "/tmp/file.txt", b"contents")]
    assert provider.read_calls == [(handle, "/tmp/file.txt")]
    assert provider.upload_calls == [(handle, source_path, "/remote/source.txt")]
    assert provider.download_calls == [(handle, "/remote/source.txt", target_path)]
    assert target_path.read_bytes() == b"downloaded"


def test_sync_sandbox_facade_rejects_async_context() -> None:
    provider_name = f"fake-{uuid4().hex}"
    register_provider(provider_name, FakeSandboxProvider)

    async def _create_sync_sandbox_in_async_context() -> None:
        Sandbox({provider_name: {}})

    try:
        asyncio.run(_create_sync_sandbox_in_async_context())
    except RuntimeError as e:
        assert "use AsyncSandbox in async code" in str(e)
    else:
        raise AssertionError("expected sync Sandbox to reject async context")


@requires_tenacity
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
        opensandbox_provider_module,
        "_require_opensandbox_sdk",
        lambda: (FakeSDKSandbox, object, object, object, object),
    )

    provider = OpenSandboxProvider(probe={"command": None})
    monkeypatch.setattr(provider, "_connection_config", lambda request_timeout_s=None, use_server_proxy=None: object())

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
    assert extensions[IMAGE_PULL_POLICY_EXTENSION_KEY] == "IfNotPresent"
    assert extensions[IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY] == "IfNotPresent"


@requires_tenacity
def test_opensandbox_connect_after_create_can_use_direct_exec_endpoint(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_connect_after_create_can_use_direct_exec_endpoint(monkeypatch))


async def _assert_opensandbox_connect_after_create_can_use_direct_exec_endpoint(monkeypatch) -> None:
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
        opensandbox_provider_module,
        "_require_opensandbox_sdk",
        lambda: (FakeSDKSandbox, FakeConnectionConfig, object, object, object),
    )

    provider = OpenSandboxProvider(
        connection={"use_server_proxy": True, "exec_use_server_proxy": False},
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
    assert connect_call["connection_config"].kwargs["use_server_proxy"] is False


@requires_tenacity
def test_opensandbox_create_probe_can_require_stable_successes(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_create_probe_can_require_stable_successes(monkeypatch))


async def _assert_opensandbox_create_probe_can_require_stable_successes(monkeypatch) -> None:
    provider = OpenSandboxProvider(
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
        timeout_s: int | None = None,
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


@requires_tenacity
def test_opensandbox_create_probe_polls_same_sandbox_after_transient_errors(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_create_probe_polls_same_sandbox_after_transient_errors(monkeypatch))


async def _assert_opensandbox_create_probe_polls_same_sandbox_after_transient_errors(monkeypatch) -> None:
    provider = OpenSandboxProvider(
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
        timeout_s: int | None = None,
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
    error = OpenSandboxCreateVerificationError("pod sdk-sandbox-0 failed create probe")

    assert isinstance(error, SandboxCreateError)
    assert opensandbox_provider_module._is_retryable_create_error(error) is True


def test_opensandbox_starting_pod_endpoint_errors_are_retryable() -> None:
    error = RuntimeError(
        "Get endpoint for sandbox sdk-sandbox-0 port 44772 failed: "
        "Pod IP is not yet available. The Pod may still be starting."
    )

    assert opensandbox_provider_module._is_retryable_create_error(error) is True


@requires_tenacity
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
                raise ConnectionError("transient proxy failure")
            return FakeExecution()

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider_module,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )

    provider = OpenSandboxProvider(
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


@requires_tenacity
def test_opensandbox_command_retries_can_be_disabled(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_command_retries_can_be_disabled(monkeypatch))


async def _assert_opensandbox_command_retries_can_be_disabled(monkeypatch) -> None:
    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> None:
            del command, opts
            self.calls += 1
            raise ConnectionError("transient proxy failure")

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider_module,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )

    provider = OpenSandboxProvider(
        operations={
            "retries": 2,
            "retry_delay_s": 0,
            "retry_max_delay_s": 0,
            "command_retries": 0,
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


@requires_tenacity
def test_opensandbox_close_timeout_does_not_fail_after_delete() -> None:
    asyncio.run(_assert_opensandbox_close_timeout_does_not_fail_after_delete())


async def _assert_opensandbox_close_timeout_does_not_fail_after_delete() -> None:
    class SlowCloseRaw:
        def __init__(self) -> None:
            self.killed = False

        async def kill(self) -> None:
            self.killed = True

        async def close(self) -> None:
            await asyncio.sleep(60)

    raw = SlowCloseRaw()
    provider = OpenSandboxProvider(
        operations={"close_timeout_s": 0.01},
        probe={"command": None},
    )
    handle = SandboxHandle(sandbox_id="sdk-sandbox-1", provider_name="opensandbox", raw=raw)

    await provider.close(handle, delete=True)

    assert raw.killed is True


@requires_tenacity
def test_opensandbox_close_timeout_still_fails_without_delete() -> None:
    asyncio.run(_assert_opensandbox_close_timeout_still_fails_without_delete())


async def _assert_opensandbox_close_timeout_still_fails_without_delete() -> None:
    class SlowCloseRaw:
        async def close(self) -> None:
            await asyncio.sleep(60)

    provider = OpenSandboxProvider(
        operations={"close_timeout_s": 0.01},
        probe={"command": None},
    )
    handle = SandboxHandle(sandbox_id="sdk-sandbox-1", provider_name="opensandbox", raw=SlowCloseRaw())

    try:
        await provider.close(handle, delete=False)
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected close timeout to fail when delete=False")


def test_mini_swe_sandbox_environment_owns_conda_setup(monkeypatch) -> None:
    provider_name = f"fake-{uuid4().hex}"
    register_provider(provider_name, FakeSandboxProvider)
    monkeypatch.setenv("FORWARDED_KEY", "forwarded-value")

    env = MiniSWESandboxEnvironment(
        image="upstream/image:tag",
        cwd="/testbed",
        provider={provider_name: {"marker": "configured"}},
        spec={
            "image_rewrites": [{"from": "upstream/", "to": "mirror/"}],
            "metadata": {"suite": "unit"},
            "resources": {"cpu": "1"},
        },
        env={"STATIC_KEY": "static-value"},
        forward_env=["FORWARDED_KEY"],
        conda_env="testbed",
        activate_conda=True,
        user="agent",
        delete=True,
    )

    try:
        assert env.get_template_vars(extra="value")["extra"] == "value"
        serialized = env.serialize()
        assert serialized["info"]["config"]["environment_type"].endswith("MiniSWESandboxEnvironment")
        env.config.activate_conda = False
        assert env._command("echo plain", "/tmp/work") == "echo plain"
        env.config.activate_conda = True

        provider = FakeSandboxProvider.last_instance
        assert provider is not None
        assert provider.marker == "configured"
        assert provider.created_specs[0].image == "mirror/image:tag"
        assert provider.created_specs[0].env == {
            "FORWARDED_KEY": "forwarded-value",
            "STATIC_KEY": "static-value",
        }

        result = env.execute("pytest -q", is_eval=True)
        assert result == {"output": "ok", "returncode": 0, "exception_info": ""}
        exec_call = provider.exec_calls[0]
        assert exec_call["cwd"] == "/"
        assert exec_call["timeout_s"] == 1800
        assert exec_call["user"] == "agent"
        assert "conda activate testbed" in exec_call["command"]
        assert exec_call["command"].endswith("pytest -q")
    finally:
        env.cleanup()
        env.cleanup()

    assert FakeSandboxProvider.last_instance is not None
    assert FakeSandboxProvider.last_instance.closed[0][1] is True


def test_mini_swe_sandbox_environment_validation_and_context_manager() -> None:
    with pytest.raises(ValueError, match="requires provider"):
        MiniSWESandboxEnvironment(image="image:tag")

    provider_name = f"fake-{uuid4().hex}"
    register_provider(provider_name, FakeSandboxProvider)
    with MiniSWESandboxEnvironment(
        image="image:tag",
        provider={provider_name: {}},
        delete=False,
    ) as env:
        assert env._handle is not None

    assert FakeSandboxProvider.last_instance is not None
    assert FakeSandboxProvider.last_instance.closed[-1][1] is False
