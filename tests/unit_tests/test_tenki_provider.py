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

"""Unit tests for the Tenki sandbox provider."""

import asyncio
import builtins
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.providers.base import ConnectableProvider, SandboxEndpoint, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.registry import get_provider_class
from nemo_gym.sandbox.providers.tenki import provider as tenki_provider
from nemo_gym.sandbox.providers.tenki.provider import (
    TenkiConnectionConfig,
    TenkiCreateConfig,
    TenkiCreateError,
    TenkiCreateVerificationError,
    TenkiOperationsConfig,
    TenkiProvider,
    TenkiProviderOptions,
    _TenkiSandbox,
)


pytestmark = pytest.mark.sandbox


class FakeCommandTimeoutError(Exception):
    pass


class FakeSessionNotFoundError(Exception):
    pass


class FakeSessionTerminatedError(Exception):
    pass


class FakeFS:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.mkdirs: list[str] = []
        self.removes: list[str] = []

    async def mkdir(self, path: str) -> None:
        self.mkdirs.append(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def read_bytes(self, path: str) -> bytes:
        return self.files[path]

    async def remove(self, path: str) -> None:
        self.removes.append(path)
        self.files.pop(path, None)


class FakeSandbox:
    def __init__(self, sandbox_id: str = "sb-1") -> None:
        self.id = sandbox_id
        self.state = "RUNNING"
        self.fs = FakeFS()
        self.exec_calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.exec_results: list[Any] = []
        self.expose_calls: list[tuple[int, dict[str, Any]]] = []
        self.exposed = []
        self.refresh_results: list[Any] = []
        self.close_calls = 0
        self.close_error: Exception | None = None

    async def exec(self, *argv: str, **kwargs: Any) -> Any:
        self.exec_calls.append((argv, kwargs))
        if self.exec_results:
            result = self.exec_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return SimpleNamespace(stdout=b"nemo-gym-tenki-ready", stderr=b"", exit_code=0)

    async def expose_port(self, port: int, **kwargs: Any) -> Any:
        self.expose_calls.append((port, kwargs))
        exposed = SimpleNamespace(port=port, url=f"https://{port}.example.test")
        self.exposed.append(exposed)
        return exposed

    async def list_exposed_ports(self) -> list[Any]:
        return self.exposed

    async def close_if_open(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error
        self.state = "TERMINATING"

    async def refresh(self) -> Any:
        if self.refresh_results:
            value = self.refresh_results.pop(0)
            if isinstance(value, BaseException):
                raise value
            self.state = value
        elif self.state == "TERMINATING":
            self.state = "TERMINATED"
        return SimpleNamespace(state=self.state)


class FakeClient:
    def __init__(self, sandbox: FakeSandbox | None = None) -> None:
        self.sandbox = sandbox or FakeSandbox()
        self.create_calls: list[dict[str, Any]] = []
        self.create_error: Exception | None = None
        self.close_calls = 0
        self.get_calls: list[str] = []

    async def create(self, **kwargs: Any) -> FakeSandbox:
        self.create_calls.append(kwargs)
        if self.create_error is not None:
            raise self.create_error
        return self.sandbox

    async def get(self, sandbox_id: str) -> FakeSandbox:
        self.get_calls.append(sandbox_id)
        return self.sandbox

    async def close(self) -> None:
        self.close_calls += 1


class FakeWaitReadyError(Exception):
    def __init__(self, sandbox: FakeSandbox) -> None:
        super().__init__("not ready")
        self.sandbox = sandbox


def fake_sdk_types() -> tuple[Any, Any, Any, Any]:
    return object, FakeCommandTimeoutError, FakeSessionNotFoundError, FakeSessionTerminatedError


@pytest.fixture
def provider_and_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TenkiProvider, FakeClient]:
    monkeypatch.setattr(tenki_provider, "_require_tenki_sdk", fake_sdk_types)
    client = FakeClient()
    provider = TenkiProvider(attribution={"team": "rl", "user": "test", "workload": "unit", "run": "run-1"})
    provider._client_instance = client
    return provider, client


def make_handle(sandbox: FakeSandbox, *, workdir: str | None = None) -> Any:
    return SimpleNamespace(sandbox_id=sandbox.id, provider_name="tenki", raw=_TenkiSandbox(sandbox, workdir))


def test_registry_resolves_tenki() -> None:
    assert get_provider_class("tenki") is TenkiProvider


def test_sdk_call_shapes_match_installed_version() -> None:
    tenki = pytest.importorskip("tenki", reason="tenki optional sandbox dependency is not installed")
    assert tenki_provider._require_tenki_sdk()[0] is tenki.AsyncClient
    inspect.signature(tenki.AsyncClient).bind(
        auth_token="token", base_url="https://api", gateway_url="https://gateway", timeout=60
    )
    inspect.signature(tenki.AsyncClient.create).bind(
        object(),
        workspace_id=None,
        name="nemo-gym-test",
        wait=True,
        timeout=300,
        allow_inbound=True,
        allow_outbound=True,
        max_duration=3600,
        idle_timeout_minutes=None,
        metadata={},
        tags=None,
        env=None,
        image=None,
        from_template_spec=None,
        snapshot_id=None,
        volumes=None,
        sticky=False,
        wait_for_runtime=False,
        cpu_cores=2,
        memory_mb=2048,
        disk_size_gb=20,
    )
    inspect.signature(tenki.AsyncClient.get).bind(object(), "sandbox-id")
    inspect.signature(tenki.AsyncSandbox.exec).bind(
        object(), "bash", "-lc", "true", cwd=None, env=None, timeout=30, privileged=False
    )
    inspect.signature(tenki.AsyncSandbox.expose_port).bind(object(), 8000, ttl=3600)
    inspect.signature(tenki.AsyncSandbox.close_if_open).bind(object())


def test_missing_sdk_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fail_tenki_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tenki":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_tenki_import)
    with pytest.raises(ModuleNotFoundError, match=r"nemo-gym\[sandbox\]"):
        tenki_provider._require_tenki_sdk()


@pytest.mark.parametrize(
    ("block", "value"),
    [
        ("connection", {"timeout_s": 0}),
        ("create", {"ready_timeout_s": 0}),
        ("create", {"max_duration_s": 0}),
        ("create", {"probe_timeout_s": 0}),
        ("operations", {"close_timeout_s": 0}),
        ("operations", {"close_poll_interval_s": 0}),
        ("operations", {"transfer_timeout_s": 0}),
    ],
)
def test_config_validation(block: str, value: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TenkiProvider(**{block: value})


def test_config_strict_mapping_and_instances() -> None:
    connection = TenkiConnectionConfig(timeout_s=10)
    provider = TenkiProvider(connection=connection, create=TenkiCreateConfig(), operations=TenkiOperationsConfig())
    assert provider._connection is connection
    with pytest.raises(TypeError, match="TenkiConnectionConfig"):
        TenkiProvider(connection=42)
    with pytest.raises(ValueError, match="unknown"):
        TenkiProvider(connection={"unknown": True})
    with pytest.raises(TypeError, match="attribution"):
        TenkiProvider(attribution=42)
    with pytest.raises(ValueError, match="extra"):
        TenkiProvider(attribution={"extra": "x"})
    with pytest.raises(TypeError, match="probe_command"):
        TenkiProvider(create={"probe_command": 3})
    with pytest.raises(TypeError, match="probe_expected_stdout"):
        TenkiProvider(create={"probe_expected_stdout": 3})


def test_bundled_config_resolves_to_tenki_provider() -> None:
    root = Path(__file__).parents[2]
    config = yaml.safe_load((root / "nemo_gym/sandbox/providers/tenki/configs/tenki.yaml").read_text(encoding="utf-8"))
    resolved = resolve_provider_config("sandbox", config)
    assert resolved["tenki"]["create"]["max_duration_s"] == 3600
    assert resolve_provider_metadata("sandbox", config) == {"sandbox-api": "tenki-sdk"}


def test_provider_options_validation() -> None:
    assert TenkiProviderOptions.from_mapping(None) == TenkiProviderOptions()
    options = TenkiProviderOptions.from_mapping({"tags": "one", "volumes": [{"volume_id": "v"}]})
    assert options.tags == ("one",)
    assert options.volumes == ({"volume_id": "v"},)
    assert TenkiProviderOptions.from_mapping(options.__dict__) == options
    with pytest.raises(TypeError, match="mapping"):
        TenkiProviderOptions.from_mapping([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown"):
        TenkiProviderOptions.from_mapping({"typo": True})
    with pytest.raises(TypeError, match="volumes"):
        TenkiProviderOptions.from_mapping({"volumes": ["bad"]})
    with pytest.raises(TypeError, match="tags"):
        TenkiProviderOptions.from_mapping({"tags": 3})
    with pytest.raises(ValueError, match="name"):
        TenkiProviderOptions.from_mapping({"name": ""})
    with pytest.raises(ValueError, match="idle_timeout_minutes"):
        TenkiProviderOptions.from_mapping({"idle_timeout_minutes": 0})
    with pytest.raises(TypeError, match="idle_timeout_minutes"):
        TenkiProviderOptions.from_mapping({"idle_timeout_minutes": True})
    with pytest.raises(TypeError, match="sticky"):
        TenkiProviderOptions.from_mapping({"sticky": "yes"})


def test_client_is_built_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        tenki_provider,
        "_require_tenki_sdk",
        lambda: (FakeAsyncClient, FakeCommandTimeoutError, FakeSessionNotFoundError, FakeSessionTerminatedError),
    )
    provider = TenkiProvider(connection={"auth_token": "token", "timeout_s": 10})
    assert provider._client() is provider._client()
    assert calls == [{"auth_token": "token", "timeout": 10}]


@pytest.mark.asyncio
async def test_serialize_and_connect_round_trip(provider_and_client: tuple[TenkiProvider, FakeClient]) -> None:
    provider, client = provider_and_client
    assert isinstance(provider, ConnectableProvider)
    original = make_handle(client.sandbox, workdir="/work")
    descriptor = await provider.serialize_handle(original, scope="ignored")
    assert descriptor == {"sandbox_id": "sb-1", "workdir": "/work"}
    connected = await provider.connect(descriptor)
    assert connected.sandbox_id == "sb-1"
    assert connected.raw.workdir == "/work"
    assert client.get_calls == ["sb-1"]

    with pytest.raises(TypeError, match="mapping"):
        await provider.connect([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sandbox_id"):
        await provider.connect({})
    with pytest.raises(TypeError, match="workdir"):
        await provider.connect({"sandbox_id": "sb-1", "workdir": 3})


@pytest.mark.asyncio
async def test_create_maps_spec_and_default_lifetime(provider_and_client: tuple[TenkiProvider, FakeClient]) -> None:
    provider, client = provider_and_client
    handle = await provider.create(
        SandboxSpec(
            image="workspace/base:latest",
            ready_timeout_s=45,
            workdir="/work",
            env={"A": "1"},
            metadata={"purpose": "test"},
            resources={"cpu": 1.2, "memory_mib": 2048, "disk_gib": 30},
            provider_options={
                "workspace_id": "ws-1",
                "name": "gym-one",
                "allow_inbound": False,
                "tags": ["gym"],
                "idle_timeout_minutes": 10,
            },
        )
    )
    assert handle.sandbox_id == "sb-1"
    call = client.create_calls[0]
    assert call["image"] == "workspace/base:latest"
    assert call["timeout"] == 45
    assert call["max_duration"] == 3600
    assert call["cpu_cores"] == 2
    assert call["memory_mb"] == 2048
    assert call["disk_size_gb"] == 30
    assert call["workspace_id"] == "ws-1"
    assert call["allow_inbound"] is False
    assert call["tags"] == ["gym"]
    assert call["metadata"] == {"team": "rl", "user": "test", "workload": "unit", "run": "run-1", "purpose": "test"}
    assert client.sandbox.exec_calls[0][1]["cwd"] == "/work"


@pytest.mark.asyncio
async def test_create_maps_optional_tenki_sources_and_ports(
    provider_and_client: tuple[TenkiProvider, FakeClient],
) -> None:
    provider, client = provider_and_client
    handle = await provider.create(
        SandboxSpec(
            ttl_s=90,
            ports=[8000],
            provider_options={
                "template": "python-template",
                "volumes": [{"volume_id": "vol-1", "mount_path": "/data"}],
                "sticky": True,
                "wait_for_runtime": True,
            },
        )
    )
    call = client.create_calls[0]
    assert call["from_template_spec"] == "python-template"
    assert call["volumes"] == [{"volume_id": "vol-1", "mount_path": "/data"}]
    assert call["max_duration"] == 90
    assert client.sandbox.expose_calls == [(8000, {"ttl": 90.0})]
    assert await provider.endpoint(handle, 8000) == SandboxEndpoint("https://8000.example.test")


@pytest.mark.asyncio
async def test_endpoint_recovers_from_sdk(provider_and_client: tuple[TenkiProvider, FakeClient]) -> None:
    provider, client = provider_and_client
    client.sandbox.exposed = [SimpleNamespace(port=9000, url="https://port.example.test")]
    handle = make_handle(client.sandbox)
    assert await provider.endpoint(handle, 9000) == SandboxEndpoint("https://port.example.test")
    with pytest.raises(ValueError, match="no exposed port"):
        await provider.endpoint(handle, 9001)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec",
    [
        SandboxSpec(entrypoint=["sleep", "1"]),
        SandboxSpec(ttl_s=True),
        SandboxSpec(ready_timeout_s=True),
        SandboxSpec(image="image", provider_options={"template": "template"}),
        SandboxSpec(resources={"gpu": 1}),
        SandboxSpec(resources={"gpu_type": "a100"}),
        SandboxSpec(resources={"cpu": 0}),
        SandboxSpec(resources={"memory_mib": 0}),
        SandboxSpec(resources={"disk_gib": 0}),
        SandboxSpec(ttl_s=-1),
        SandboxSpec(ttl_s=0),
        SandboxSpec(ttl_s=float("inf")),
        SandboxSpec(ready_timeout_s=-1),
        SandboxSpec(ready_timeout_s=0),
        SandboxSpec(ready_timeout_s=float("nan")),
    ],
)
async def test_create_rejects_unsupported_or_invalid_spec(
    provider_and_client: tuple[TenkiProvider, FakeClient], spec: SandboxSpec
) -> None:
    provider, _ = provider_and_client
    with pytest.raises(ValueError):
        await provider.create(spec)


@pytest.mark.asyncio
async def test_create_failure_terminates_admitted_sandbox(
    provider_and_client: tuple[TenkiProvider, FakeClient],
) -> None:
    provider, client = provider_and_client
    partial = FakeSandbox("partial")
    client.create_error = FakeWaitReadyError(partial)
    with pytest.raises(TenkiCreateError) as error:
        await provider.create(SandboxSpec())
    assert partial.close_calls == 1
    assert isinstance(error.value.__cause__, FakeWaitReadyError)


@pytest.mark.asyncio
async def test_cancelled_create_waits_for_admission_then_terminates(
    provider_and_client: tuple[TenkiProvider, FakeClient],
) -> None:
    provider, _ = provider_and_client
    sandbox = FakeSandbox("cancelled")
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowClient(FakeClient):
        async def create(self, **kwargs: Any) -> FakeSandbox:
            self.create_calls.append(kwargs)
            started.set()
            await release.wait()
            return self.sandbox

    provider._client_instance = SlowClient(sandbox)
    task = asyncio.create_task(provider.create(SandboxSpec()))
    await started.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sandbox.close_calls == 1
    assert sandbox.state == "TERMINATED"


@pytest.mark.asyncio
async def test_cancelled_create_cleans_sdk_partial_failure(
    provider_and_client: tuple[TenkiProvider, FakeClient],
) -> None:
    provider, _ = provider_and_client
    partial = FakeSandbox("cancelled-partial")
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowFailingClient(FakeClient):
        async def create(self, **kwargs: Any) -> FakeSandbox:
            started.set()
            await release.wait()
            raise FakeWaitReadyError(partial)

    provider._client_instance = SlowFailingClient()
    task = asyncio.create_task(provider.create(SandboxSpec()))
    await started.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert partial.close_calls == 1
    assert partial.state == "TERMINATED"


@pytest.mark.asyncio
async def test_failed_create_cleanup_errors_are_logged(
    provider_and_client: tuple[TenkiProvider, FakeClient], caplog: pytest.LogCaptureFixture
) -> None:
    provider, _ = provider_and_client
    caplog.set_level(logging.WARNING)

    close_failure = FakeSandbox("close-failure")
    close_failure.close_error = RuntimeError("close failed")
    await provider._terminate_partial(close_failure, close_failure.id)
    assert "Failed to terminate Tenki sandbox" in caplog.text

    refresh_failure = FakeSandbox("refresh-failure")
    refresh_failure.refresh_results = [RuntimeError("refresh failed")]
    await provider._terminate_partial(refresh_failure, refresh_failure.id)
    assert "Failed to verify Tenki sandbox termination" in caplog.text

    provider._operations = TenkiOperationsConfig(close_timeout_s=0.001, close_poll_interval_s=0.001)
    timeout = FakeSandbox("timeout")
    timeout.refresh_results = ["RUNNING", "RUNNING", "RUNNING"]
    await provider._terminate_partial(timeout, timeout.id)
    assert "did not terminate after create failure" in caplog.text


@pytest.mark.asyncio
async def test_probe_and_port_failures_terminate_sandbox(
    provider_and_client: tuple[TenkiProvider, FakeClient],
) -> None:
    provider, client = provider_and_client
    client.sandbox.exec_results = [SimpleNamespace(stdout=b"bad", stderr=b"probe", exit_code=1)]
    with pytest.raises(TenkiCreateVerificationError):
        await provider.create(SandboxSpec())
    assert client.sandbox.close_calls == 1

    client.sandbox = FakeSandbox("port-fail")

    async def fail_expose(port: int, **kwargs: Any) -> Any:
        raise RuntimeError(f"cannot expose {port}: {kwargs}")

    client.sandbox.expose_port = fail_expose  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="cannot expose"):
        await provider.create(SandboxSpec(ports=[8080]))
    assert client.sandbox.close_calls == 1


@pytest.mark.asyncio
async def test_create_can_disable_probe(provider_and_client: tuple[TenkiProvider, FakeClient]) -> None:
    provider, client = provider_and_client
    provider._create = TenkiCreateConfig(probe_command=None)
    await provider.create(SandboxSpec())
    assert client.sandbox.exec_calls == []


@pytest.mark.asyncio
async def test_exec_maps_shell_user_and_results(provider_and_client: tuple[TenkiProvider, FakeClient]) -> None:
    provider, client = provider_and_client
    handle = make_handle(client.sandbox, workdir="/default")
    client.sandbox.exec_results = [SimpleNamespace(stdout=b"hello\xff", stderr=b"warn", exit_code=3)]
    result = await provider.exec(handle, "echo hi", env={"X": "2"}, timeout_s=12, user="root")
    assert result.stdout == "hello�"
    assert result.stderr == "warn"
    assert result.return_code == 3
    assert client.sandbox.exec_calls[0] == (
        ("bash", "-lc", "echo hi"),
        {"cwd": "/default", "env": {"X": "2"}, "timeout": 12, "privileged": True},
    )
    with pytest.raises(NotImplementedError, match="default tenki user"):
        await provider.exec(handle, "true", user="nobody")
    with pytest.raises(NotImplementedError, match="boolean"):
        await provider.exec(handle, "true", user=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "return_code", "error_type"),
    [
        (FakeCommandTimeoutError("late"), 124, "timeout"),
        (FakeSessionNotFoundError("gone"), 125, "sandbox"),
        (FakeSessionTerminatedError("done"), 125, "sandbox"),
    ],
)
async def test_exec_maps_sdk_runtime_errors(
    provider_and_client: tuple[TenkiProvider, FakeClient],
    error: Exception,
    return_code: int,
    error_type: str,
) -> None:
    provider, client = provider_and_client
    client.sandbox.exec_results = [error]
    result = await provider.exec(make_handle(client.sandbox), "true")
    assert result.return_code == return_code
    assert result.error_type == error_type


@pytest.mark.asyncio
async def test_upload_home_and_privileged_paths(
    provider_and_client: tuple[TenkiProvider, FakeClient], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, client = provider_and_client
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload\x00")
    handle = make_handle(client.sandbox, workdir="/home/tenki/work")
    await provider.upload_file(handle, source, "relative.bin")
    assert client.sandbox.fs.files["/home/tenki/work/relative.bin"] == b"payload\x00"

    monkeypatch.setattr(tenki_provider.uuid, "uuid4", lambda: SimpleNamespace(hex="stage-id"))
    await provider.upload_file(handle, source, "/opt/data/file.bin")
    assert client.sandbox.exec_calls[-1][1]["privileged"] is True
    assert client.sandbox.exec_calls[-1][0] == (
        "bash",
        "-lc",
        "mkdir -p -- /opt/data && cp -- /home/tenki/.nemo-gym-transfers/stage-id /opt/data/file.bin",
    )
    assert client.sandbox.fs.removes == ["/home/tenki/.nemo-gym-transfers/stage-id"]


@pytest.mark.asyncio
async def test_upload_reports_copy_failure_and_cleans_staging(
    provider_and_client: tuple[TenkiProvider, FakeClient], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, client = provider_and_client
    source = tmp_path / "source"
    source.write_text("x")
    monkeypatch.setattr(tenki_provider.uuid, "uuid4", lambda: SimpleNamespace(hex="stage-id"))
    client.sandbox.exec_results = [SimpleNamespace(stdout=b"", stderr=b"denied", exit_code=1)]
    with pytest.raises(RuntimeError, match="denied"):
        await provider.upload_file(make_handle(client.sandbox), source, "/root/file")
    assert client.sandbox.fs.removes == ["/home/tenki/.nemo-gym-transfers/stage-id"]


@pytest.mark.asyncio
async def test_upload_ignores_staging_cleanup_failure(
    provider_and_client: tuple[TenkiProvider, FakeClient], tmp_path: Path
) -> None:
    provider, client = provider_and_client
    source = tmp_path / "source"
    source.write_text("x")

    async def fail_remove(path: str) -> None:
        raise RuntimeError(f"cannot remove {path}")

    client.sandbox.fs.remove = fail_remove  # type: ignore[method-assign]
    await provider.upload_file(make_handle(client.sandbox), source, "/root/file")


@pytest.mark.asyncio
async def test_download_home_and_privileged_paths(
    provider_and_client: tuple[TenkiProvider, FakeClient], tmp_path: Path
) -> None:
    provider, client = provider_and_client
    handle = make_handle(client.sandbox)
    client.sandbox.fs.files["/home/tenki/a"] = b"home"
    home_target = tmp_path / "nested" / "home"
    await provider.download_file(handle, "/home/tenki/a", home_target)
    assert home_target.read_bytes() == b"home"

    client.sandbox.exec_results = [SimpleNamespace(stdout=b"root\x00", stderr=b"", exit_code=0)]
    root_target = tmp_path / "root"
    await provider.download_file(handle, "/root/a", root_target)
    assert root_target.read_bytes() == b"root\x00"
    assert client.sandbox.exec_calls[-1] == (
        ("cat", "--", "/root/a"),
        {"timeout": 300.0, "privileged": True},
    )


@pytest.mark.asyncio
async def test_download_reports_privileged_failure(
    provider_and_client: tuple[TenkiProvider, FakeClient], tmp_path: Path
) -> None:
    provider, client = provider_and_client
    client.sandbox.exec_results = [SimpleNamespace(stdout=b"", stderr=b"missing", exit_code=1)]
    with pytest.raises(RuntimeError, match="missing"):
        await provider.download_file(make_handle(client.sandbox), "/root/missing", tmp_path / "target")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("RUNNING", SandboxStatus.RUNNING),
        ("PROVISIONING", SandboxStatus.STARTING),
        ("TERMINATED", SandboxStatus.STOPPED),
        ("FAILED", SandboxStatus.ERROR),
        ("MYSTERY", SandboxStatus.UNKNOWN),
    ],
)
async def test_status_mapping(
    provider_and_client: tuple[TenkiProvider, FakeClient], state: str, expected: SandboxStatus
) -> None:
    provider, client = provider_and_client
    client.sandbox.refresh_results = [state]
    assert await provider.status(make_handle(client.sandbox)) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [FakeSessionNotFoundError("gone"), FakeSessionTerminatedError("done")])
async def test_status_missing_is_stopped(
    provider_and_client: tuple[TenkiProvider, FakeClient], error: Exception
) -> None:
    provider, client = provider_and_client
    client.sandbox.refresh_results = [error]
    assert await provider.status(make_handle(client.sandbox)) is SandboxStatus.STOPPED


@pytest.mark.asyncio
async def test_close_waits_for_terminal_state(provider_and_client: tuple[TenkiProvider, FakeClient]) -> None:
    provider, client = provider_and_client
    client.sandbox.refresh_results = ["TERMINATING", "TERMINATED"]
    await provider.close(make_handle(client.sandbox))
    assert client.sandbox.close_calls == 1


@pytest.mark.asyncio
async def test_close_treats_missing_as_stopped(provider_and_client: tuple[TenkiProvider, FakeClient]) -> None:
    provider, client = provider_and_client
    client.sandbox.close_error = FakeSessionNotFoundError("gone")
    await provider.close(make_handle(client.sandbox))


@pytest.mark.asyncio
async def test_close_refresh_missing_is_stopped(provider_and_client: tuple[TenkiProvider, FakeClient]) -> None:
    provider, client = provider_and_client
    client.sandbox.refresh_results = [FakeSessionTerminatedError("done")]
    await provider.close(make_handle(client.sandbox))


@pytest.mark.asyncio
async def test_close_timeout(provider_and_client: tuple[TenkiProvider, FakeClient]) -> None:
    provider, client = provider_and_client
    provider._operations = TenkiOperationsConfig(close_timeout_s=0.001, close_poll_interval_s=0.001)
    client.sandbox.refresh_results = ["RUNNING", "RUNNING", "RUNNING"]
    with pytest.raises(TimeoutError, match="did not terminate"):
        await provider.close(make_handle(client.sandbox))


@pytest.mark.asyncio
async def test_aclose_closes_and_releases_client(provider_and_client: tuple[TenkiProvider, FakeClient]) -> None:
    provider, client = provider_and_client
    await provider.aclose()
    assert client.close_calls == 1
    assert provider._client_instance is None
    await provider.aclose()
