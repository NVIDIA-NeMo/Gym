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

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.registry import get_provider_class


pytestmark = pytest.mark.sandbox


pytest.importorskip("openshell", reason="openshell optional sandbox dependency is not installed")

import grpc  # noqa: E402  (grpcio ships with the openshell SDK)
from openshell import SandboxError  # noqa: E402

from nemo_gym.sandbox.providers.openshell import provider as openshell_provider  # noqa: E402
from nemo_gym.sandbox.providers.openshell.provider import (  # noqa: E402
    PHASE_DELETING,
    PHASE_ERROR,
    PHASE_PROVISIONING,
    PHASE_READY,
    PHASE_UNKNOWN,
    PHASE_UNSPECIFIED,
    SANDBOX_LABEL,
    SANDBOX_NAME_PREFIX,
    SANDBOX_RUNTIME_RETURN_CODE,
    OpenShellConnectionConfig,
    OpenShellCreateConfig,
    OpenShellCreateError,
    OpenShellCreateVerificationError,
    OpenShellExecConfig,
    OpenShellOperationsConfig,
    OpenShellProbeConfig,
    OpenShellProvider,
    _OpenShellSandbox,
)


class FakeRpcError(grpc.RpcError):
    """Minimal stand-in for the SDK's raised RPC errors (grpc.RpcError with a code())."""

    def __init__(self, code: grpc.StatusCode, details: str = "fake rpc error") -> None:
        super().__init__(details)
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


def make_ref(phase: int, *, sandbox_id: str = "sbx-1", name: str = "nemo-gym-test") -> SimpleNamespace:
    return SimpleNamespace(id=sandbox_id, name=name, phase=phase)


def make_exec_result(exit_code: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(exit_code=exit_code, stdout=stdout, stderr=stderr)


class FakeClient:
    """Records SDK calls and replays queued results (a queued Exception is raised; the last entry repeats)."""

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.exec_calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.close_calls = 0
        self.create_results: list[Any] = [make_ref(PHASE_PROVISIONING)]
        self.get_results: list[Any] = [make_ref(PHASE_READY)]
        self.exec_results: list[Any] = [make_exec_result()]
        self.delete_results: list[Any] = [True]

    @staticmethod
    def _next(queue: list[Any]) -> Any:
        result = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(result, Exception):
            raise result
        return result

    def create(self, *, spec: Any, name: str, labels: dict[str, str]) -> Any:
        self.create_calls.append({"spec": spec, "name": name, "labels": labels})
        result = self._next(self.create_results)
        return SimpleNamespace(id=result.id, name=name, phase=result.phase)

    def get(self, name: str) -> Any:
        self.get_calls.append(name)
        return self._next(self.get_results)

    def exec(self, sandbox_id: str, command: list[str], **kwargs: Any) -> Any:
        self.exec_calls.append({"sandbox_id": sandbox_id, "command": command, **kwargs})
        return self._next(self.exec_results)

    def delete(self, name: str) -> Any:
        self.delete_calls.append(name)
        return self._next(self.delete_results)

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def make_provider(monkeypatch: pytest.MonkeyPatch, fake_client: FakeClient):
    def factory(**overrides: Any) -> OpenShellProvider:
        monkeypatch.setattr(OpenShellProvider, "_build_client", lambda self: fake_client)
        kwargs: dict[str, Any] = {
            "create": {"poll_interval_s": 0.01},
            "exec": {"concurrency": 2},
            "probe": {"command": None},
            "operations": {"poll_interval_s": 0.01, "close_timeout_s": 0.5},
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(kwargs.get(key), dict):
                kwargs[key] = {**kwargs[key], **value}
            else:
                kwargs[key] = value
        return OpenShellProvider(**kwargs)

    return factory


def _make_handle(
    *,
    name: str = "nemo-gym-test",
    sandbox_id: str = "sbx-1",
    env: dict[str, str] | None = None,
    workdir: str | None = None,
) -> SandboxHandle:
    return SandboxHandle(
        sandbox_id=sandbox_id,
        provider_name="openshell",
        raw=_OpenShellSandbox(name=name, sandbox_id=sandbox_id, image="img", env=env or {}, workdir=workdir),
    )


def test_registry_resolves_openshell() -> None:
    assert get_provider_class("openshell") is OpenShellProvider


def test_constructor_builds_real_client_and_config_coercion() -> None:
    provider = OpenShellProvider(connection={"endpoint": "localhost:1", "request_timeout_s": 5})
    assert provider._connection == OpenShellConnectionConfig(endpoint="localhost:1", request_timeout_s=5)
    assert provider._client is not None


@pytest.mark.parametrize(
    ("group", "kwargs"),
    [
        ("connection", {"endpoint": ""}),
        ("connection", {"request_timeout_s": 0}),
        ("connection", {"tls_cert_path": "/tmp/cert.pem"}),
        ("create", {"ready_timeout_s": 0}),
        ("create", {"poll_interval_s": 0}),
        ("exec", {"default_timeout_s": 0}),
        ("exec", {"concurrency": 0}),
        ("exec", {"exec_shell": ""}),
        ("probe", {"timeout_s": 0}),
        ("probe", {"deadline_s": 0}),
        ("probe", {"stable_count": 0}),
        ("probe", {"stable_delay_s": -1}),
        ("operations", {"close_timeout_s": 0}),
        ("operations", {"poll_interval_s": 0}),
    ],
)
def test_config_validation(group: str, kwargs: dict[str, Any]) -> None:
    config_cls = {
        "connection": OpenShellConnectionConfig,
        "create": OpenShellCreateConfig,
        "exec": OpenShellExecConfig,
        "probe": OpenShellProbeConfig,
        "operations": OpenShellOperationsConfig,
    }[group]
    with pytest.raises(ValueError):
        config_cls(**kwargs)


def test_invalid_config_type_raises() -> None:
    with pytest.raises(TypeError, match="OpenShellExecConfig"):
        OpenShellProvider(exec=42)


async def test_create_success_maps_spec(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    spec = SandboxSpec(
        image="docker://python:3.12-slim",
        env={"FOO": "bar"},
        metadata={"task": "demo"},
        workdir="/workspace",
        resources={"gpu": 2},
        provider_options={"providers": ["nvidia"]},
    )
    handle = await provider.create(spec)

    call = fake_client.create_calls[0]
    assert call["name"].startswith(SANDBOX_NAME_PREFIX)
    assert call["labels"] == {SANDBOX_LABEL: "1", "task": "demo"}
    pb_spec = call["spec"]
    assert pb_spec.template.image == "python:3.12-slim"
    assert dict(pb_spec.environment) == {"FOO": "bar"}
    assert list(pb_spec.providers) == ["nvidia"]
    assert pb_spec.resource_requirements.gpu.count == 2

    assert handle.provider_name == "openshell"
    assert handle.sandbox_id == "sbx-1"
    assert handle.raw.name == call["name"]
    assert handle.raw.env == {"FOO": "bar"}
    assert handle.raw.workdir == "/workspace"


async def test_create_without_image_uses_gateway_default(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    await provider.create(SandboxSpec())
    pb_spec = fake_client.create_calls[0]["spec"]
    assert not pb_spec.HasField("template")


async def test_create_warns_on_unsupported_resources(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    with caplog.at_level("WARNING"):
        await provider.create(SandboxSpec(resources={"cpu": 2, "memory_mib": 1024}))
    assert "cpu, memory_mib" in caplog.text


async def test_create_entrypoint_raises(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    with pytest.raises(OpenShellCreateError, match="entrypoint"):
        await provider.create(SandboxSpec(entrypoint=["/bin/bash"]))
    assert not fake_client.create_calls


async def test_create_ttl_warns(make_provider, caplog) -> None:
    provider = make_provider()
    with caplog.at_level("WARNING"):
        await provider.create(SandboxSpec(ttl_s=60))
    assert "ttl_s is not enforced" in caplog.text


async def test_create_polls_until_ready(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.get_results = [
        make_ref(PHASE_PROVISIONING),
        FakeRpcError(grpc.StatusCode.UNAVAILABLE),
        make_ref(PHASE_PROVISIONING),
        make_ref(PHASE_READY),
    ]
    await provider.create(SandboxSpec())
    assert len(fake_client.get_calls) == 4


async def test_create_error_phase_cleans_up(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.get_results = [make_ref(PHASE_ERROR)]
    with pytest.raises(OpenShellCreateError, match="ERROR phase"):
        await provider.create(SandboxSpec())
    assert fake_client.delete_calls == [fake_client.create_calls[0]["name"]]


async def test_create_ready_timeout_cleans_up(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.get_results = [make_ref(PHASE_PROVISIONING)]
    with pytest.raises(OpenShellCreateError, match="was not READY"):
        await provider.create(SandboxSpec(ready_timeout_s=0.05))
    assert fake_client.delete_calls


async def test_create_rpc_failure_wrapped(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.create_results = [FakeRpcError(grpc.StatusCode.UNAVAILABLE)]
    with pytest.raises(OpenShellCreateError, match="CreateSandbox failed"):
        await provider.create(SandboxSpec())


async def test_create_probe_passes_after_retry(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(probe={"command": "printf ready", "expected_stdout": "ready", "deadline_s": 5})
    fake_client.exec_results = [make_exec_result(exit_code=1, stderr="not yet"), make_exec_result(stdout="ready")]
    handle = await provider.create(SandboxSpec())
    assert handle.sandbox_id == "sbx-1"
    assert len(fake_client.exec_calls) == 2


async def test_create_probe_failure_cleans_up(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(probe={"command": "printf ready", "expected_stdout": "ready", "deadline_s": 0.05})
    fake_client.exec_results = [make_exec_result(exit_code=1, stderr="broken")]
    with pytest.raises(OpenShellCreateVerificationError, match="readiness probe"):
        await provider.create(SandboxSpec())
    assert fake_client.delete_calls


async def test_exec_maps_command_and_result(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    handle = _make_handle(env={"A": "1", "B": "2"}, workdir="/workspace")
    fake_client.exec_results = [make_exec_result(exit_code=3, stdout="out", stderr="err")]

    result = await provider.exec(handle, "echo hi", env={"B": "override"})

    call = fake_client.exec_calls[0]
    assert call["sandbox_id"] == "sbx-1"
    assert call["command"] == ["/bin/sh", "-c", "echo hi"]
    assert call["workdir"] == "/workspace"
    assert call["env"] == {"A": "1", "B": "override"}
    assert call["timeout_seconds"] == 180
    assert result == openshell_provider.SandboxExecResult(stdout="out", stderr="err", return_code=3)


async def test_exec_cwd_overrides_workdir_and_timeout_rounds_up(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    handle = _make_handle(workdir="/workspace")
    await provider.exec(handle, "true", cwd="/tmp", timeout_s=2.5)
    call = fake_client.exec_calls[0]
    assert call["workdir"] == "/tmp"
    assert call["timeout_seconds"] == 3


async def test_exec_no_env_passes_none(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    await provider.exec(_make_handle(), "true")
    assert fake_client.exec_calls[0]["env"] is None


async def test_exec_user_warns(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    with caplog.at_level("WARNING"):
        await provider.exec(_make_handle(), "true", user="root")
    assert "no user field" in caplog.text


async def test_exec_grpc_deadline_maps_to_timeout(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.exec_results = [FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)]
    result = await provider.exec(_make_handle(), "sleep 999")
    assert result.error_type == "timeout"
    assert result.return_code == SANDBOX_RUNTIME_RETURN_CODE


@pytest.mark.parametrize(
    "error",
    [FakeRpcError(grpc.StatusCode.UNAVAILABLE), FakeRpcError(grpc.StatusCode.NOT_FOUND), SandboxError("boom")],
)
async def test_exec_runtime_failures_map_to_sandbox(make_provider, fake_client: FakeClient, error: Exception) -> None:
    provider = make_provider()
    fake_client.exec_results = [error]
    result = await provider.exec(_make_handle(), "true")
    assert result.error_type == "sandbox"
    assert result.return_code == SANDBOX_RUNTIME_RETURN_CODE


async def test_exec_unexpected_error_raises(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.exec_results = [ValueError("bug")]
    with pytest.raises(ValueError, match="bug"):
        await provider.exec(_make_handle(), "true")


async def test_upload_file_streams_stdin(make_provider, fake_client: FakeClient, tmp_path: Path) -> None:
    provider = make_provider()
    source = tmp_path / "payload.bin"
    payload = b"\x00\x01binary\nbytes"
    source.write_bytes(payload)

    await provider.upload_file(_make_handle(), source, "/data/dir/payload.bin")

    call = fake_client.exec_calls[0]
    assert call["command"] == ["/bin/sh", "-c", "mkdir -p /data/dir && cat > /data/dir/payload.bin"]
    assert call["stdin"] == payload


async def test_upload_file_without_parent_dir(make_provider, fake_client: FakeClient, tmp_path: Path) -> None:
    provider = make_provider()
    source = tmp_path / "f.txt"
    source.write_text("hi")
    await provider.upload_file(_make_handle(), source, "f.txt")
    assert fake_client.exec_calls[0]["command"] == ["/bin/sh", "-c", "cat > f.txt"]


async def test_upload_file_failure_raises(make_provider, fake_client: FakeClient, tmp_path: Path) -> None:
    provider = make_provider()
    source = tmp_path / "f.txt"
    source.write_text("hi")
    fake_client.exec_results = [make_exec_result(exit_code=1, stderr="disk full")]
    with pytest.raises(RuntimeError, match="disk full"):
        await provider.upload_file(_make_handle(), source, "/data/f.txt")


async def test_download_file_roundtrips_base64(make_provider, fake_client: FakeClient, tmp_path: Path) -> None:
    provider = make_provider()
    payload = b"\x00\xffbinary payload"
    encoded = base64.encodebytes(payload).decode()  # multi-line, like `base64` line-wrapping
    fake_client.exec_results = [make_exec_result(stdout=encoded)]

    target = tmp_path / "nested" / "out.bin"
    await provider.download_file(_make_handle(), "/data/out.bin", target)

    assert fake_client.exec_calls[0]["command"] == ["/bin/sh", "-c", "base64 /data/out.bin"]
    assert target.read_bytes() == payload


async def test_download_file_failure_raises(make_provider, fake_client: FakeClient, tmp_path: Path) -> None:
    provider = make_provider()
    fake_client.exec_results = [make_exec_result(exit_code=1, stderr="No such file")]
    with pytest.raises(RuntimeError, match="No such file"):
        await provider.download_file(_make_handle(), "/missing", tmp_path / "out.bin")


async def test_download_file_invalid_base64_raises(make_provider, fake_client: FakeClient, tmp_path: Path) -> None:
    provider = make_provider()
    fake_client.exec_results = [make_exec_result(stdout="not!base64@@")]
    with pytest.raises(RuntimeError, match="invalid base64"):
        await provider.download_file(_make_handle(), "/data/out.bin", tmp_path / "out.bin")


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        (PHASE_UNSPECIFIED, SandboxStatus.UNKNOWN),
        (PHASE_PROVISIONING, SandboxStatus.STARTING),
        (PHASE_READY, SandboxStatus.RUNNING),
        (PHASE_ERROR, SandboxStatus.ERROR),
        (PHASE_DELETING, SandboxStatus.STOPPED),
        (PHASE_UNKNOWN, SandboxStatus.UNKNOWN),
        (99, SandboxStatus.UNKNOWN),
    ],
)
async def test_status_phase_mapping(
    make_provider, fake_client: FakeClient, phase: int, expected: SandboxStatus
) -> None:
    provider = make_provider()
    fake_client.get_results = [make_ref(phase)]
    assert await provider.status(_make_handle()) == expected


async def test_status_not_found_is_stopped(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.get_results = [FakeRpcError(grpc.StatusCode.NOT_FOUND)]
    assert await provider.status(_make_handle()) == SandboxStatus.STOPPED


@pytest.mark.parametrize("error", [FakeRpcError(grpc.StatusCode.UNAVAILABLE), SandboxError("boom")])
async def test_status_runtime_failure_is_unknown(make_provider, fake_client: FakeClient, error: Exception) -> None:
    provider = make_provider()
    fake_client.get_results = [error]
    assert await provider.status(_make_handle()) == SandboxStatus.UNKNOWN


async def test_status_unexpected_error_raises(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.get_results = [ValueError("bug")]
    with pytest.raises(ValueError, match="bug"):
        await provider.status(_make_handle())


async def test_close_deletes_and_waits(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.get_results = [
        make_ref(PHASE_DELETING),
        FakeRpcError(grpc.StatusCode.NOT_FOUND),
    ]
    await provider.close(_make_handle(name="nemo-gym-x"))
    assert fake_client.delete_calls == ["nemo-gym-x"]
    assert fake_client.get_calls == ["nemo-gym-x", "nemo-gym-x"]


async def test_close_without_wait(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(operations={"close_wait_deleted": False})
    await provider.close(_make_handle())
    assert fake_client.delete_calls
    assert not fake_client.get_calls


async def test_close_missing_sandbox_is_success(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.delete_results = [FakeRpcError(grpc.StatusCode.NOT_FOUND)]
    await provider.close(_make_handle())
    assert not fake_client.get_calls


async def test_close_delete_failure_raises(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.delete_results = [FakeRpcError(grpc.StatusCode.UNAVAILABLE)]
    with pytest.raises(RuntimeError, match="delete failed"):
        await provider.close(_make_handle())


async def test_close_wait_deleted_timeout_raises(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(operations={"close_timeout_s": 0.05})
    fake_client.get_results = [make_ref(PHASE_DELETING)]
    with pytest.raises(RuntimeError, match="was not deleted"):
        await provider.close(_make_handle())


async def test_aclose_idempotent(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    await provider.aclose()
    await provider.aclose()
    assert fake_client.close_calls == 1
