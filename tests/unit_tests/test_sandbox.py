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
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from nemo_gym.sandbox import (
    AsyncSandbox,
    Sandbox,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    register_provider,
    rewrite_image,
)
from nemo_gym.sandbox.observability import (
    SandboxRecorder,
    aperf_config_from_extensions,
    aperf_record_command,
    build_recorder_from_env,
    use_recorder,
)
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider_module
from nemo_gym.sandbox.providers.opensandbox.provider import (
    IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY,
    IMAGE_PULL_POLICY_EXTENSION_KEY,
    OpenSandboxCreateVerificationError,
    OpenSandboxProvider,
)
from responses_api_agents.mini_swe_agent.sandbox_environment import MiniSWESandboxEnvironment


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


def _test_recorder(output_dir: Path) -> SandboxRecorder:
    return SandboxRecorder(
        output_dir=output_dir,
        otel={
            "enabled": False,
            "endpoint": None,
            "service_name": "nemo-gym-test",
        },
    )


def _mini_swe_command_titles() -> dict[str, Any]:
    return {
        "strip_prefixes": [
            "cd /testbed && source $(conda info --base)/etc/profile.d/conda.sh && conda activate testbed &&"
        ],
        "rules": [
            {
                "line_starts_with": ["pytest ", "python -m pytest ", "./tests/runtests.py "],
                "search": "last",
                "title": "run verifier: {line}",
            }
        ],
    }


def _otel_attributes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attrs = {}
    for row in rows:
        value = row["value"]
        if "stringValue" in value:
            attrs[row["key"]] = value["stringValue"]
        elif "boolValue" in value:
            attrs[row["key"]] = value["boolValue"]
        elif "intValue" in value:
            attrs[row["key"]] = int(value["intValue"])
        elif "doubleValue" in value:
            attrs[row["key"]] = value["doubleValue"]
    return attrs


def _otel_spans(output_dir: Path) -> list[dict[str, Any]]:
    trace_payload = json.loads((output_dir / "traces" / "otel_traces.json").read_text())
    return [
        span
        for resource_span in trace_payload["resourceSpans"]
        for scope_span in resource_span["scopeSpans"]
        for span in scope_span["spans"]
    ]


def _otel_resource_service_names(output_dir: Path) -> set[str]:
    trace_payload = json.loads((output_dir / "traces" / "otel_traces.json").read_text())
    service_names = set()
    for resource_span in trace_payload["resourceSpans"]:
        attrs = _otel_attributes(resource_span["resource"]["attributes"])
        service_names.add(attrs["service.name"])
    return service_names


def test_sandbox_facade_uses_public_provider_api() -> None:
    asyncio.run(_assert_sandbox_facade_uses_public_provider_api())


async def _assert_sandbox_facade_uses_public_provider_api() -> None:
    provider_name = f"fake-{uuid4().hex}"
    register_provider(provider_name, FakeSandboxProvider)

    sandbox = AsyncSandbox({"name": provider_name, "kwargs": {"marker": "configured"}})
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

    with Sandbox({"name": provider_name, "kwargs": {"marker": "configured"}}) as sandbox:
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
        Sandbox({"name": provider_name})

    try:
        asyncio.run(_create_sync_sandbox_in_async_context())
    except RuntimeError as e:
        assert "use AsyncSandbox in async code" in str(e)
    else:
        raise AssertionError("expected sync Sandbox to reject async context")


def test_sandbox_facade_owns_operation_observability(tmp_path: Path) -> None:
    asyncio.run(_assert_sandbox_facade_owns_operation_observability(tmp_path))


async def _assert_sandbox_facade_owns_operation_observability(tmp_path: Path) -> None:
    provider_name = f"fake-{uuid4().hex}"
    register_provider(provider_name, FakeSandboxProvider)
    recorder = _test_recorder(tmp_path)

    with use_recorder(recorder):
        sandbox = AsyncSandbox({"name": provider_name})
        handle = await sandbox.create(
            SandboxSpec(
                image="image:tag",
                metadata={
                    "benchmark": "swebench-verified",
                    "instance_id": "django__django-12345",
                    "nemo_gym_agent": "mini_swe_agent",
                },
            )
        )
        await sandbox.exec(handle, "pytest -q", cwd="/repo", timeout_s=60, user="agent")
        await sandbox.close(handle, delete=True)

    recorder.finalize()
    span_attrs = {
        span["name"]: _otel_attributes(span["attributes"])
        for span in _otel_spans(recorder.output_dir)
        if span["name"] in {"sandbox.create: image:tag", "exec: pytest -q", "sandbox.cleanup: fake-1"}
    }

    assert set(span_attrs) == {"sandbox.create: image:tag", "exec: pytest -q", "sandbox.cleanup: fake-1"}
    assert span_attrs["sandbox.create: image:tag"]["trajectory_id"] == "django__django-12345"
    assert span_attrs["sandbox.create: image:tag"]["harness"] == "mini_swe_agent"
    assert span_attrs["sandbox.create: image:tag"]["benchmark"] == "swebench-verified"
    assert span_attrs["sandbox.create: image:tag"]["operation.name"] == "sandbox.start"
    assert span_attrs["exec: pytest -q"]["sandbox_id"] == "fake-1"
    assert span_attrs["exec: pytest -q"]["command"] == "pytest -q"
    assert span_attrs["exec: pytest -q"]["operation.name"] == "trajectory.tool"
    assert span_attrs["exec: pytest -q"]["span.section"] == "rollout"
    assert "command_class" not in span_attrs["exec: pytest -q"]
    assert "command_hash" not in span_attrs["exec: pytest -q"]
    assert span_attrs["sandbox.cleanup: fake-1"]["delete"] is True
    assert {"sandbox.create", "sandbox.exec", "sandbox.cleanup"}.issubset(
        _otel_resource_service_names(recorder.output_dir)
    )
    forbidden_prefix = "nemo" + "_rl."
    forbidden_hash_attr = "nemo" + "_gym.sandbox_id_hash"
    assert all(
        not key.startswith(forbidden_prefix) and key != forbidden_hash_attr
        for attrs in span_attrs.values()
        for key in attrs
    )


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

    provider = OpenSandboxProvider(create_probe_command=None)
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
        create_probe_command=None,
        use_server_proxy=True,
        exec_use_server_proxy=False,
        connect_after_create_attempt_timeout_s=1,
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


def test_opensandbox_create_probe_can_require_stable_successes(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_create_probe_can_require_stable_successes(monkeypatch))


async def _assert_opensandbox_create_probe_can_require_stable_successes(monkeypatch) -> None:
    provider = OpenSandboxProvider(
        create_probe_command="true",
        create_probe_expected_stdout=None,
        create_probe_stable_count=3,
        create_probe_stable_delay_s=0,
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


def test_opensandbox_create_probe_polls_same_sandbox_after_transient_errors(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_create_probe_polls_same_sandbox_after_transient_errors(monkeypatch))


async def _assert_opensandbox_create_probe_polls_same_sandbox_after_transient_errors(monkeypatch) -> None:
    provider = OpenSandboxProvider(
        create_probe_command="true",
        create_probe_expected_stdout=None,
        create_probe_timeout_s=1,
        create_probe_deadline_s=2,
        create_probe_stable_count=2,
        create_probe_stable_delay_s=0,
        connect_after_create_poll_s=0.01,
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

    assert opensandbox_provider_module._is_retryable_create_error(error) is True


def test_opensandbox_starting_pod_endpoint_errors_are_retryable() -> None:
    error = RuntimeError(
        "Get endpoint for sandbox sdk-sandbox-0 port 44772 failed: "
        "Pod IP is not yet available. The Pod may still be starting."
    )

    assert opensandbox_provider_module._is_retryable_create_error(error) is True


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
        create_probe_command=None,
        operation_retries=2,
        operation_retry_delay_s=0,
        operation_retry_max_delay_s=0,
        command_retries=2,
    )
    raw = FakeRaw()
    handle = SandboxHandle(sandbox_id="sdk-sandbox-1", provider_name="opensandbox", raw=raw)

    result = await provider.exec(handle, "echo hello", timeout_s=30)

    assert result.stdout == "ok"
    assert result.return_code == 0
    assert raw.commands.calls == 3


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
        create_probe_command=None,
        operation_retries=2,
        operation_retry_delay_s=0,
        operation_retry_max_delay_s=0,
        command_retries=0,
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
        create_probe_command=None,
        close_timeout_s=0.01,
    )
    handle = SandboxHandle(sandbox_id="sdk-sandbox-1", provider_name="opensandbox", raw=raw)

    await provider.close(handle, delete=True)

    assert raw.killed is True


def test_opensandbox_close_timeout_still_fails_without_delete() -> None:
    asyncio.run(_assert_opensandbox_close_timeout_still_fails_without_delete())


async def _assert_opensandbox_close_timeout_still_fails_without_delete() -> None:
    class SlowCloseRaw:
        async def close(self) -> None:
            await asyncio.sleep(60)

    provider = OpenSandboxProvider(
        create_probe_command=None,
        close_timeout_s=0.01,
    )
    handle = SandboxHandle(sandbox_id="sdk-sandbox-1", provider_name="opensandbox", raw=SlowCloseRaw())

    try:
        await provider.close(handle, delete=False)
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected close timeout to fail when delete=False")


def test_observability_finalize_exports_only_otel_traces(tmp_path: Path) -> None:
    recorder = SandboxRecorder(
        output_dir=tmp_path / "observability",
        run_id="run-1",
        run_span_name="unit-job",
        otel={
            "enabled": False,
            "service_name": "nemo-gym-test",
        },
    )
    with recorder.sync_span(
        "trajectory.tool",
        phase="exec",
        attributes={"trajectory_id": "task-1", "sandbox_id": "sandbox-1"},
    ):
        pass

    recorder.finalize()

    assert (recorder.output_dir / "traces" / "otel_traces.json").exists()
    assert not (recorder.output_dir / "traces" / "chrome_trace.json").exists()
    assert not (recorder.output_dir / "summary.json").exists()
    trace_payload = json.loads((recorder.output_dir / "traces" / "otel_traces.json").read_text())
    span_names = [
        span["name"]
        for resource_span in trace_payload["resourceSpans"]
        for scope_span in resource_span["scopeSpans"]
        for span in scope_span["spans"]
    ]
    assert "eval: unit-job" in span_names
    assert "exec: <empty>" in span_names
    assert "rollout: task-1" in span_names
    assert _otel_resource_service_names(recorder.output_dir) == {
        "nemo-gym.eval",
        "nemo-gym.rollout",
        "sandbox.exec",
    }


def test_observability_otlp_exporter_does_not_require_local_artifacts(monkeypatch, tmp_path: Path) -> None:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    exporter_kwargs = []

    class FakeOTLPSpanExporter(SpanExporter):
        def __init__(self, **kwargs: Any) -> None:
            exporter_kwargs.append(kwargs)

        def export(self, spans: Any) -> SpanExportResult:
            del spans
            return SpanExportResult.SUCCESS

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            del timeout_millis
            return True

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", FakeOTLPSpanExporter)

    recorder = SandboxRecorder(
        output_dir=None,
        run_id="run-1",
        otel={
            "enabled": True,
            "service_name": "sandbox-test",
            "traces_exporter": "otlp_http",
            "metrics_exporter": "none",
            "traces_endpoint": "http://collector:4318",
            "traces_headers": {"x-scope-orgid": "sandbox"},
            "traces_timeout_s": 3,
        },
    )
    with recorder.sync_span("trajectory.tool", phase="execution", attributes={"trajectory_id": "task-1"}):
        pass
    recorder.finalize()

    assert exporter_kwargs == [
        {
            "endpoint": "http://collector:4318/v1/traces",
            "headers": {"x-scope-orgid": "sandbox"},
            "timeout": 3.0,
        }
    ]
    assert not (tmp_path / "traces").exists()


def test_observability_env_can_enable_recorder_without_output_dir(monkeypatch) -> None:
    monkeypatch.delenv("NEMO_GYM_SANDBOX_OBSERVABILITY_DIR", raising=False)
    monkeypatch.setenv("NEMO_GYM_SANDBOX_OBSERVABILITY_TRACES_EXPORTER", "none")
    monkeypatch.setenv("NEMO_GYM_SANDBOX_OBSERVABILITY_JOB_NAME", "unit-job")

    recorder = build_recorder_from_env()

    assert recorder is not None
    assert recorder.output_dir is None
    assert recorder.run_span_name == "eval: unit-job"
    recorder.finalize()


def test_sandbox_lifecycle_aperf_diagnostic_overlaps_sandbox_lifetime(tmp_path: Path) -> None:
    asyncio.run(_assert_sandbox_lifecycle_aperf_diagnostic_overlaps_sandbox_lifetime(tmp_path))


async def _assert_sandbox_lifecycle_aperf_diagnostic_overlaps_sandbox_lifetime(tmp_path: Path) -> None:
    provider_name = f"fake-{uuid4().hex}"
    register_provider(provider_name, FakeSandboxProvider)
    sandbox = AsyncSandbox({"name": provider_name})
    handle = await sandbox.create(
        SandboxSpec(
            image="image:tag",
            timeout_s=600,
            metadata={"trajectory_id": "task-1"},
            extensions={
                "observability.aperf.enabled": "true",
                "observability.aperf.interval_s": "1",
                "observability.aperf.local_output_dir": str(tmp_path / "aperf"),
                "observability.aperf.install_url": "https://example.test/aperf.tgz",
            },
        )
    )
    provider = FakeSandboxProvider.last_instance
    assert provider is not None

    assert len(provider.exec_calls) == 1
    start_command = provider.exec_calls[0]["command"]
    assert "aperf record -r task-1 -i 1 -p 600" in start_command
    assert "nohup aperf record" in start_command
    assert "aperf.pid" in start_command

    await sandbox.exec(handle, "pytest -q")
    await sandbox.close(handle, delete=True)

    assert len(provider.exec_calls) == 3
    assert provider.exec_calls[1]["command"] == "pytest -q"
    stop_command = provider.exec_calls[2]["command"]
    assert "kill -INT" in stop_command
    assert "nemo-gym-aperf.tgz" in stop_command
    assert provider.download_calls[0][1] == "/tmp/nemo-gym-aperf.tgz"
    assert provider.download_calls[0][2].name == "fake-1.aperf_artifacts.tgz"
    assert provider.download_calls[0][2].exists()


def test_aperf_diagnostic_command_is_explicit_opt_in() -> None:
    assert aperf_record_command(None) is None
    assert aperf_record_command({"enabled": False, "run_name": "task-1"}) is None
    assert (
        aperf_record_command(
            {
                "enabled": True,
                "run_name": "task-1",
                "interval_s": 2,
                "period_s": 5,
                "dont_collect": ["perf_stat"],
                "profile": True,
            }
        )
        == "aperf record -r task-1 -i 2 -p 5 --dont-collect perf_stat --profile"
    )
    assert aperf_config_from_extensions(
        {
            "observability.aperf.enabled": "true",
            "observability.aperf.run_name": "task:1",
        },
        timeout_s=120,
    ) == {
        "enabled": True,
        "run_name": "task_1",
        "output_dir": "/tmp/nemo-gym-aperf",
        "tmp_dir": "/tmp/nemo-gym-aperf/tmp",
        "period_s": 120,
    }


def test_observability_attributes_are_configurable(tmp_path: Path) -> None:
    recorder = SandboxRecorder(
        output_dir=tmp_path / "observability",
        otel={
            "enabled": False,
            "attribute_aliases": {"trajectory_id": "custom.trajectory_id"},
            "local_service_name_strategy": "preserve",
            "resource_attributes": {"deployment": "unit-test"},
            "service_name": "sandbox-test",
        },
    )

    with recorder.sync_span("trajectory.tool", phase="execution", attributes={"trajectory_id": "task-1"}):
        pass
    recorder.finalize()

    spans = _otel_spans(recorder.output_dir)
    tool_span = next(span for span in spans if span["name"] == "exec: <empty>")
    attrs = _otel_attributes(tool_span["attributes"])
    resource_attrs = _otel_attributes(
        json.loads((recorder.output_dir / "traces" / "otel_traces.json").read_text())["resourceSpans"][0]["resource"][
            "attributes"
        ]
    )

    assert attrs["trajectory_id"] == "task-1"
    assert attrs["operation.name"] == "trajectory.tool"
    assert attrs["span.section"] == "rollout"
    assert attrs["custom.trajectory_id"] == "task-1"
    assert ("nemo" + "_rl.trajectory_id") not in attrs
    assert resource_attrs["deployment"] == "unit-test"
    assert resource_attrs["service.name"] == "sandbox-test"


def test_observability_command_span_titles_prefer_verifier_command(tmp_path: Path) -> None:
    recorder = SandboxRecorder(
        output_dir=tmp_path / "observability",
        otel={
            "enabled": False,
            "command_titles": _mini_swe_command_titles(),
        },
    )

    command = """cd /testbed && source $(conda info --base)/etc/profile.d/conda.sh && conda activate testbed &&
set -xo pipefail
cd /testbed
git status
pytest -rA testing/test_collection.py
git checkout base testing/test_collection.py
"""
    with recorder.sync_span(
        "trajectory.tool",
        phase="execution",
        attributes={"trajectory_id": "task-1", "command": command},
    ):
        pass
    recorder.finalize()

    spans = _otel_spans(recorder.output_dir)
    tool_span = next(
        span for span in spans if span["name"] == "exec: run verifier: pytest -rA testing/test_collection.py"
    )
    attrs = _otel_attributes(tool_span["attributes"])

    assert attrs["operation.name"] == "trajectory.tool"
    assert attrs["span.section"] == "rollout"


def test_observability_command_span_titles_do_not_use_builtin_task_heuristics(tmp_path: Path) -> None:
    recorder = SandboxRecorder(output_dir=tmp_path / "observability", otel={"enabled": False})

    command = """cd /testbed && source $(conda info --base)/etc/profile.d/conda.sh && conda activate testbed &&
set -xo pipefail
pytest -q
"""
    with recorder.sync_span(
        "trajectory.tool",
        phase="execution",
        attributes={"trajectory_id": "task-1", "command": command},
    ):
        pass
    recorder.finalize()

    spans = _otel_spans(recorder.output_dir)
    assert any(span["name"].startswith("exec: cd /testbed && source $(conda info --base)") for span in spans)
    assert not any(span["name"].startswith("exec: run verifier:") for span in spans)


def test_observability_splits_rollout_llm_and_verifier_sections(tmp_path: Path) -> None:
    recorder = SandboxRecorder(output_dir=tmp_path / "observability", otel={"enabled": False})

    with recorder.sync_span("llm.request", phase="llm", attributes={"trajectory_id": "task-1", "model": "qwen"}):
        pass
    with recorder.sync_span(
        "trajectory.tool",
        phase="execution",
        attributes={"trajectory_id": "task-1", "command": "ls -la /testbed"},
    ):
        pass
    with recorder.sync_span(
        "trajectory.tool",
        phase="execution",
        attributes={
            "trajectory_id": "task-1",
            "command": "pytest -q",
            "execution.section": "verifier",
            "span.section": "verifier",
        },
    ):
        pass
    recorder.record_event("trajectory", "trajectory.complete", attributes={"trajectory_id": "task-1", "reward": 1.0})
    recorder.finalize()

    spans = _otel_spans(recorder.output_dir)
    span_attrs = {span["name"]: _otel_attributes(span["attributes"]) for span in spans}
    span_by_name = {span["name"]: span for span in spans}

    assert "rollout: task-1" in span_attrs
    assert "verifier: task-1" in span_attrs
    assert span_by_name["verifier: task-1"]["parentSpanId"] == span_by_name["rollout: task-1"]["spanId"]
    assert span_attrs["llm.request"]["span.section"] == "rollout"
    assert span_attrs["exec: ls -la /testbed"]["span.section"] == "rollout"
    assert span_attrs["exec: pytest -q"]["span.section"] == "verifier"
    assert {
        "nemo-gym.rollout",
        "nemo-gym.verifier",
        "llm.request",
        "sandbox.exec",
        "verifier.exec",
    }.issubset(_otel_resource_service_names(recorder.output_dir))


def test_observability_can_record_exception_without_stacktrace(tmp_path: Path) -> None:
    recorder = SandboxRecorder(output_dir=tmp_path / "observability", otel={"enabled": False})

    with pytest.raises(RuntimeError):
        with recorder.sync_span(
            "llm.request",
            phase="llm",
            attributes={"trajectory_id": "task-1", "_record_exception_stacktrace": False},
        ):
            raise RuntimeError("format retry")
    recorder.finalize()

    llm_span = next(span for span in _otel_spans(recorder.output_dir) if span["name"] == "llm.request")
    attrs = _otel_attributes(llm_span["attributes"])
    events = llm_span.get("events") or []

    assert attrs["status"] == "error"
    assert attrs["error_type"] == "RuntimeError"
    assert events
    assert events[0]["name"] == "exception"
    event_attrs = _otel_attributes(events[0]["attributes"])
    assert event_attrs["exception.type"] == "RuntimeError"
    assert "exception.stacktrace" not in event_attrs


def test_mini_swe_sandbox_environment_owns_conda_setup(monkeypatch, tmp_path: Path) -> None:
    provider_name = f"fake-{uuid4().hex}"
    register_provider(provider_name, FakeSandboxProvider)
    monkeypatch.setenv("FORWARDED_KEY", "forwarded-value")
    recorder = SandboxRecorder(
        output_dir=tmp_path,
        otel={
            "enabled": False,
            "endpoint": None,
            "service_name": "nemo-gym-test",
            "command_titles": _mini_swe_command_titles(),
        },
    )

    with use_recorder(recorder):
        env = MiniSWESandboxEnvironment(
            image="upstream/image:tag",
            cwd="/testbed",
            provider={"name": provider_name, "kwargs": {"marker": "configured"}},
            spec={
                "image_rewrites": [{"from": "upstream/", "to": "mirror/"}],
                "metadata": {"suite": "unit"},
                "resources": {"cpu": "1"},
            },
            env={"STATIC_KEY": "static-value"},
            forward_env=["FORWARDED_KEY"],
            cache_dir_template="/tmp/{instance_id}.sif",
            conda_env="testbed",
            activate_conda=True,
            user="agent",
            delete=True,
        )

        try:
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

    recorder.finalize()
    spans = _otel_spans(recorder.output_dir)
    span_attrs = {span["name"]: _otel_attributes(span["attributes"]) for span in spans}

    assert FakeSandboxProvider.last_instance is not None
    assert FakeSandboxProvider.last_instance.closed[0][1] is True
    assert "verifier: unknown" in span_attrs
    assert span_attrs["exec: run verifier: pytest -q"]["span.section"] == "verifier"
