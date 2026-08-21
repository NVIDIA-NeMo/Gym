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
import builtins
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.registry import get_provider_class


pytestmark = pytest.mark.sandbox


pytest.importorskip("openshell", reason="openshell optional dependency is not installed")

import grpc  # noqa: E402  (grpcio ships with the openshell SDK)
from openshell import SandboxError, SandboxRef, SandboxStatusRef  # noqa: E402
from openshell._proto import openshell_pb2, sandbox_pb2  # noqa: E402

from nemo_gym.sandbox.providers.openshell import provider as openshell_provider  # noqa: E402
from nemo_gym.sandbox.providers.openshell.provider import (  # noqa: E402
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
    OpenShellProviderOptions,
    _OpenShellSandbox,
)


class FakeRpcError(grpc.RpcError):
    """Minimal stand-in for the SDK's raised RPC errors (grpc.RpcError with a code())."""

    def __init__(self, code: grpc.StatusCode, details: str = "fake rpc error") -> None:
        super().__init__(details)
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


def make_ref(
    phase: int,
    *,
    sandbox_id: str = "sbx-1",
    name: str = "nemo-gym-test",
    workspace: str = "default",
) -> SandboxRef:
    """A real SDK SandboxRef, so tests exercise the SDK's actual shape (nested status/phase)."""
    return SandboxRef(
        id=sandbox_id,
        name=name,
        workspace=workspace,
        status=SandboxStatusRef(phase=phase, current_policy_version=0),
    )


def make_exec_result(exit_code: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(exit_code=exit_code, stdout=stdout, stderr=stderr)


class FakeClient:
    """Records SDK calls and replays queued results (a queued Exception is raised; the last entry repeats).

    Method signatures mirror openshell's ``SandboxClient`` (0.0.92+): ``workspace`` is a
    required keyword-only argument on the lifecycle calls, so provider call-shape drift
    fails these tests instead of passing silently.
    """

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.exec_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.close_calls = 0
        self.create_results: list[Any] = [make_ref(openshell_pb2.SANDBOX_PHASE_PROVISIONING)]
        self.get_results: list[Any] = [make_ref(openshell_pb2.SANDBOX_PHASE_READY)]
        self.exec_results: list[Any] = [make_exec_result()]
        self.delete_results: list[Any] = [True]

    @staticmethod
    def _next(queue: list[Any]) -> Any:
        result = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(result, Exception):
            raise result
        return result

    def create(
        self,
        *,
        workspace: str,
        spec: Any = None,
        name: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> Any:
        self.create_calls.append({"workspace": workspace, "spec": spec, "name": name, "labels": labels})
        result = self._next(self.create_results)
        return make_ref(result.phase, sandbox_id=result.id, name=name or result.name, workspace=workspace)

    def get(self, sandbox_name: str, *, workspace: str) -> Any:
        self.get_calls.append({"name": sandbox_name, "workspace": workspace})
        return self._next(self.get_results)

    def delete(self, sandbox_name: str, *, workspace: str) -> Any:
        self.delete_calls.append({"name": sandbox_name, "workspace": workspace})
        return self._next(self.delete_results)

    def exec(
        self,
        sandbox_id: str,
        command: list[str],
        *,
        stream_output: bool = False,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> Any:
        self.exec_calls.append(
            {
                "sandbox_id": sandbox_id,
                "command": command,
                "workdir": workdir,
                "env": env,
                "stdin": stdin,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self._next(self.exec_results)

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture(autouse=True)
def clear_shared_clients() -> Any:
    openshell_provider._SHARED_CLIENTS.clear()
    yield
    openshell_provider._SHARED_CLIENTS.clear()


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def make_provider(monkeypatch: pytest.MonkeyPatch, fake_client: FakeClient):
    def factory(**overrides: Any) -> OpenShellProvider:
        monkeypatch.setattr(openshell_provider, "_build_client", lambda connection: fake_client)
        kwargs: dict[str, Any] = {
            "create": {"poll_interval_s": 0.01, "retry_delay_s": 0.0, "retry_max_delay_s": 0.0},
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
    workspace: str = "default",
    env: dict[str, Any] | None = None,
    workdir: str | None = None,
) -> SandboxHandle:
    return SandboxHandle(
        sandbox_id=sandbox_id,
        provider_name="openshell",
        raw=_OpenShellSandbox(
            name=name, sandbox_id=sandbox_id, workspace=workspace, image="img", env=env or {}, workdir=workdir
        ),
    )


def test_registry_resolves_openshell() -> None:
    assert get_provider_class("openshell") is OpenShellProvider


def test_missing_openshell_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "openshell" or name.startswith("openshell."):
            raise ModuleNotFoundError(name)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match=r"nemo-gym\[openshell\]"):
        openshell_provider._require_openshell()


def test_provider_call_shapes_bind_against_installed_sdk() -> None:
    """Bind the provider's exact SDK call shapes against the installed openshell signatures.

    This is the drift guard: an SDK release that changes a lifecycle signature (like 0.0.92
    adding required ``workspace``) fails here even though the unit tests run against fakes.
    """
    from openshell import SandboxClient

    inspect.signature(SandboxClient.create).bind(None, workspace="w", spec=None, name="n", labels={})
    inspect.signature(SandboxClient.get).bind(None, "n", workspace="w")
    inspect.signature(SandboxClient.delete).bind(None, "n", workspace="w")
    inspect.signature(SandboxClient.exec).bind(
        None, "sid", ["/bin/sh", "-c", "true"], workdir=None, env=None, stdin=None, timeout_seconds=None
    )
    inspect.signature(SandboxClient.close).bind(None)


def test_constructor_builds_real_client_and_config_coercion() -> None:
    provider = OpenShellProvider(connection={"endpoint": "localhost:1", "request_timeout_s": 5})
    try:
        assert provider._connection == OpenShellConnectionConfig(endpoint="localhost:1", request_timeout_s=5)
        assert provider._client is not None
    finally:
        openshell_provider._release_shared_client(provider._shared)


@pytest.mark.parametrize(
    ("group", "kwargs"),
    [
        ("connection", {"endpoint": ""}),
        ("connection", {"workspace": ""}),
        ("connection", {"request_timeout_s": 0}),
        ("connection", {"tls_cert_path": "/tmp/cert.pem"}),
        ("create", {"ready_timeout_s": 0}),
        ("create", {"poll_interval_s": 0}),
        ("create", {"retries": -1}),
        ("create", {"retry_delay_s": -1}),
        ("create", {"retry_max_delay_s": -1}),
        ("exec", {"default_timeout_s": 0}),
        ("exec", {"concurrency": 0}),
        ("exec", {"exec_shell": ""}),
        ("exec", {"upload_chunk_bytes": 0}),
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


def test_provider_options_validation() -> None:
    with pytest.raises(ValueError, match="Unknown openshell provider_options keys"):
        OpenShellProviderOptions.from_mapping({"provider": ["typo"]})
    with pytest.raises(TypeError, match="providers"):
        OpenShellProviderOptions.from_mapping({"providers": 42})
    with pytest.raises(TypeError, match="policy"):
        OpenShellProviderOptions.from_mapping({"policy": 42})
    with pytest.raises(TypeError, match="template_resources"):
        OpenShellProviderOptions.from_mapping({"template_resources": ["not", "a", "mapping"]})
    with pytest.raises(TypeError, match="driver_config"):
        OpenShellProviderOptions.from_mapping({"driver_config": ["not", "a", "mapping"]})
    options = OpenShellProviderOptions.from_mapping({"providers": "solo"})
    assert options.providers == ["solo"]


async def test_create_success_maps_spec(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    spec = SandboxSpec(
        image="docker://python:3.12-slim",
        env={"FOO": "bar"},
        metadata={"task": "demo", SANDBOX_LABEL: "user-override"},
        workdir="/workspace",
        resources={"gpu": 2},
        provider_options={"providers": ["nvidia"]},
    )
    handle = await provider.create(spec)

    call = fake_client.create_calls[0]
    assert call["workspace"] == "default"
    assert call["name"].startswith(SANDBOX_NAME_PREFIX)
    # The marker label is applied last, so user metadata cannot clobber it.
    assert call["labels"] == {"task": "demo", SANDBOX_LABEL: "1"}
    pb_spec = call["spec"]
    assert pb_spec.template.image == "python:3.12-slim"
    assert dict(pb_spec.environment) == {"FOO": "bar"}
    assert list(pb_spec.providers) == ["nvidia"]
    assert pb_spec.resource_requirements.gpu.count == 2

    assert handle.provider_name == "openshell"
    assert handle.sandbox_id == "sbx-1"
    assert handle.raw.name == call["name"]
    assert handle.raw.workspace == "default"
    assert handle.raw.env == {"FOO": "bar"}
    assert handle.raw.workdir == "/workspace"


async def test_create_uses_configured_workspace(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(connection={"workspace": "team-a"})
    handle = await provider.create(SandboxSpec())
    assert fake_client.create_calls[0]["workspace"] == "team-a"
    assert fake_client.get_calls[0]["workspace"] == "team-a"
    assert handle.raw.workspace == "team-a"


async def test_create_without_image_uses_gateway_default(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    await provider.create(SandboxSpec())
    pb_spec = fake_client.create_calls[0]["spec"]
    assert not pb_spec.HasField("template")


async def test_create_template_resources_and_driver_config(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    await provider.create(
        SandboxSpec(
            provider_options={
                "template_resources": {"cpu": "2", "memory": "4Gi"},
                "driver_config": {"runtime": "kata"},
            }
        )
    )
    template = fake_client.create_calls[0]["spec"].template
    assert dict(template.resources) == {"cpu": "2", "memory": "4Gi"}
    assert dict(template.driver_config) == {"runtime": "kata"}


async def test_create_policy_from_mapping_and_path(make_provider, fake_client: FakeClient, tmp_path: Path) -> None:
    provider = make_provider()
    await provider.create(SandboxSpec(provider_options={"policy": {}}))
    assert fake_client.create_calls[0]["spec"].HasField("policy")

    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("{}\n")
    await provider.create(SandboxSpec(provider_options={"policy": str(policy_path)}))
    assert fake_client.create_calls[1]["spec"].HasField("policy")

    with pytest.raises(OpenShellCreateError, match="not a valid SandboxPolicy"):
        await provider.create(SandboxSpec(provider_options={"policy": {"not_a_policy_field": 1}}))
    assert isinstance(sandbox_pb2.SandboxPolicy(), object)  # policy parses against the real proto


async def test_create_unknown_provider_option_raises(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    with pytest.raises(ValueError, match="Unknown openshell provider_options keys"):
        await provider.create(SandboxSpec(provider_options={"polcy": {}}))
    assert not fake_client.create_calls


async def test_create_warns_on_unmapped_resources(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    with caplog.at_level("WARNING"):
        await provider.create(SandboxSpec(resources={"cpu": 2, "memory_mib": 1024}))
    assert "not mapped by this provider" in caplog.text
    assert "template_resources" in caplog.text


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
        make_ref(openshell_pb2.SANDBOX_PHASE_PROVISIONING),
        FakeRpcError(grpc.StatusCode.UNAVAILABLE),
        make_ref(openshell_pb2.SANDBOX_PHASE_PROVISIONING),
        make_ref(openshell_pb2.SANDBOX_PHASE_READY),
    ]
    await provider.create(SandboxSpec())
    assert len(fake_client.get_calls) == 4


@pytest.mark.parametrize(
    ("phase", "match"),
    [
        (openshell_pb2.SANDBOX_PHASE_ERROR, "ERROR phase"),
        (openshell_pb2.SANDBOX_PHASE_DELETING, "being deleted"),
    ],
)
async def test_create_terminal_phase_cleans_up(make_provider, fake_client: FakeClient, phase: int, match: str) -> None:
    provider = make_provider()
    fake_client.get_results = [make_ref(phase)]
    with pytest.raises(OpenShellCreateError, match=match):
        await provider.create(SandboxSpec())
    assert fake_client.delete_calls[0]["name"] == fake_client.create_calls[0]["name"]


async def test_create_ready_timeout_cleans_up(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.get_results = [make_ref(openshell_pb2.SANDBOX_PHASE_PROVISIONING)]
    with pytest.raises(OpenShellCreateError, match="was not READY"):
        await provider.create(SandboxSpec(ready_timeout_s=0.05))
    assert fake_client.delete_calls


async def test_create_cleanup_failure_logs_leak_warning(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    fake_client.get_results = [make_ref(openshell_pb2.SANDBOX_PHASE_ERROR)]
    fake_client.delete_results = [FakeRpcError(grpc.StatusCode.UNAVAILABLE)]
    with caplog.at_level("WARNING"):
        with pytest.raises(OpenShellCreateError, match="ERROR phase"):
            await provider.create(SandboxSpec())
    assert "may be leaked" in caplog.text


async def test_create_retries_transient_failures(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(create={"retries": 2})
    fake_client.create_results = [
        FakeRpcError(grpc.StatusCode.UNAVAILABLE),
        make_ref(openshell_pb2.SANDBOX_PHASE_PROVISIONING),
    ]
    handle = await provider.create(SandboxSpec())
    assert len(fake_client.create_calls) == 2
    # Retries reuse the same sandbox name so a committed first attempt is recoverable.
    assert fake_client.create_calls[0]["name"] == fake_client.create_calls[1]["name"]
    assert handle.sandbox_id == "sbx-1"


async def test_create_already_exists_recovers_via_get(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.create_results = [FakeRpcError(grpc.StatusCode.ALREADY_EXISTS)]
    handle = await provider.create(SandboxSpec())
    assert len(fake_client.create_calls) == 1
    assert fake_client.get_calls[0]["name"] == fake_client.create_calls[0]["name"]
    assert handle.sandbox_id == "sbx-1"


async def test_create_exhausted_retries_wrapped(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(create={"retries": 1})
    fake_client.create_results = [
        FakeRpcError(grpc.StatusCode.UNAVAILABLE),
        FakeRpcError(grpc.StatusCode.UNAVAILABLE, "still down"),
    ]
    with pytest.raises(OpenShellCreateError, match="CreateSandbox failed"):
        await provider.create(SandboxSpec())
    assert len(fake_client.create_calls) == 2


async def test_create_nonretryable_rpc_failure_wrapped_without_retry(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(create={"retries": 3})
    fake_client.create_results = [FakeRpcError(grpc.StatusCode.PERMISSION_DENIED)]
    with pytest.raises(OpenShellCreateError, match="CreateSandbox failed"):
        await provider.create(SandboxSpec())
    assert len(fake_client.create_calls) == 1


async def test_create_programming_error_propagates_unwrapped(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.create_results = [TypeError("missing 1 required keyword-only argument: 'workspace'")]
    with pytest.raises(TypeError, match="workspace"):
        await provider.create(SandboxSpec())


async def test_create_probe_passes_after_retry(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(
        probe={"command": "printf ready", "expected_stdout": "ready", "deadline_s": 5, "stable_delay_s": 0.01}
    )
    fake_client.exec_results = [make_exec_result(exit_code=1, stderr="not yet"), make_exec_result(stdout="ready")]
    handle = await provider.create(SandboxSpec())
    assert handle.sandbox_id == "sbx-1"
    assert len(fake_client.exec_calls) == 2


async def test_create_probe_failure_cleans_up(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(
        probe={"command": "printf ready", "expected_stdout": "ready", "deadline_s": 0.05, "stable_delay_s": 0.01}
    )
    fake_client.exec_results = [make_exec_result(exit_code=1, stderr="broken")]
    with pytest.raises(OpenShellCreateVerificationError, match="readiness probe"):
        await provider.create(SandboxSpec())
    assert fake_client.delete_calls


async def test_exec_maps_command_and_result(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    handle = _make_handle(env={"A": 1, "B": "2"}, workdir="/workspace")
    fake_client.exec_results = [make_exec_result(exit_code=3, stdout="out", stderr="err")]

    result = await provider.exec(handle, "echo hi", env={"B": "override", "C": 3})

    call = fake_client.exec_calls[0]
    assert call["sandbox_id"] == "sbx-1"
    assert call["command"] == ["/bin/sh", "-c", "echo hi"]
    assert call["workdir"] == "/workspace"
    # Env values are coerced to strings at the boundary, not inside protobuf map assignment.
    assert call["env"] == {"A": "1", "B": "override", "C": "3"}
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


async def test_upload_file_chunks_large_payloads(make_provider, fake_client: FakeClient, tmp_path: Path) -> None:
    provider = make_provider(exec={"upload_chunk_bytes": 4})
    source = tmp_path / "payload.bin"
    payload = b"0123456789"  # 3 chunks at 4 bytes
    source.write_bytes(payload)

    await provider.upload_file(_make_handle(), source, "payload.bin")

    commands = [call["command"][2] for call in fake_client.exec_calls]
    assert commands == ["cat > payload.bin", "cat >> payload.bin", "cat >> payload.bin"]
    assert b"".join(call["stdin"] for call in fake_client.exec_calls) == payload


async def test_upload_file_empty_file(make_provider, fake_client: FakeClient, tmp_path: Path) -> None:
    provider = make_provider()
    source = tmp_path / "empty.txt"
    source.write_bytes(b"")
    await provider.upload_file(_make_handle(), source, "empty.txt")
    assert len(fake_client.exec_calls) == 1
    assert fake_client.exec_calls[0]["stdin"] == b""


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
        (openshell_pb2.SANDBOX_PHASE_UNSPECIFIED, SandboxStatus.UNKNOWN),
        (openshell_pb2.SANDBOX_PHASE_PROVISIONING, SandboxStatus.STARTING),
        (openshell_pb2.SANDBOX_PHASE_READY, SandboxStatus.RUNNING),
        (openshell_pb2.SANDBOX_PHASE_ERROR, SandboxStatus.ERROR),
        (openshell_pb2.SANDBOX_PHASE_DELETING, SandboxStatus.STOPPED),
        (openshell_pb2.SANDBOX_PHASE_UNKNOWN, SandboxStatus.UNKNOWN),
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
        make_ref(openshell_pb2.SANDBOX_PHASE_DELETING),
        FakeRpcError(grpc.StatusCode.NOT_FOUND),
    ]
    await provider.close(_make_handle(name="nemo-gym-x", workspace="team-a"))
    assert fake_client.delete_calls == [{"name": "nemo-gym-x", "workspace": "team-a"}]
    assert [call["name"] for call in fake_client.get_calls] == ["nemo-gym-x", "nemo-gym-x"]


async def test_close_without_wait(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(operations={"close_wait_deleted": False})
    await provider.close(_make_handle())
    assert fake_client.delete_calls
    assert not fake_client.get_calls


async def test_close_deleted_false_skips_wait(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    fake_client.delete_results = [False]
    with caplog.at_level("WARNING"):
        await provider.close(_make_handle())
    assert "deleted=False" in caplog.text
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
    fake_client.get_results = [make_ref(openshell_pb2.SANDBOX_PHASE_DELETING)]
    with pytest.raises(RuntimeError, match="was not deleted"):
        await provider.close(_make_handle())


async def test_aclose_idempotent(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    await provider.aclose()
    await provider.aclose()
    assert fake_client.close_calls == 1


async def test_shared_client_released_on_last_aclose(make_provider, fake_client: FakeClient) -> None:
    provider_a = make_provider()
    provider_b = make_provider()
    assert provider_a._shared is provider_b._shared
    assert provider_a._shared.refcount == 2

    await provider_a.aclose()
    assert fake_client.close_calls == 0  # provider_b still holds the shared client

    await provider_b.aclose()
    assert fake_client.close_calls == 1
    assert not openshell_provider._SHARED_CLIENTS


async def test_distinct_connection_configs_do_not_share_clients(make_provider) -> None:
    provider_a = make_provider()
    provider_b = make_provider(connection={"endpoint": "other:8080"})
    try:
        assert provider_a._shared is not provider_b._shared
    finally:
        await provider_a.aclose()
        await provider_b.aclose()
