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
import base64
import builtins
import concurrent.futures
import dataclasses
import inspect
import re
import threading
import time
from collections.abc import Callable
from os import getpid as os_getpid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_gym.sandbox import attribution
from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.registry import get_provider_class


pytestmark = pytest.mark.sandbox


pytest.importorskip("openshell", reason="openshell optional dependency is not installed")

import grpc  # noqa: E402  (grpcio ships with the openshell SDK)
from openshell import SandboxError, SandboxRef, SandboxStatusRef, TlsConfig
from openshell._proto import openshell_pb2  # noqa: E402
from openshell.mutations import DeletionOutcome, DeletionResult  # noqa: E402
from openshell.sandbox import SandboxConditionRef  # noqa: E402

from nemo_gym.sandbox.providers.openshell import provider as openshell_provider  # noqa: E402
from nemo_gym.sandbox.providers.openshell.provider import (  # noqa: E402
    MAX_SANDBOX_NAME_LENGTH,
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
    _generate_sandbox_name,
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

    Method signatures mirror openshell's ``SandboxClient`` (0.1.x): ``workspace`` is a
    required keyword-only argument on the lifecycle calls, so provider call-shape drift
    fails these tests instead of passing silently.
    """

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.exec_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.list_results: list[Any] = [[]]
        self.gateway_labels: dict[str, dict[str, str]] | None = None  # name -> labels, when set
        self.close_calls = 0
        self.create_results: list[Any] = [make_ref(openshell_pb2.SANDBOX_PHASE_PROVISIONING)]
        self.get_results: list[Any] = [make_ref(openshell_pb2.SANDBOX_PHASE_READY)]
        self.exec_results: list[Any] = [make_exec_result()]
        self.delete_gate: threading.Event | None = None  # set -> delete blocks until the test opens it
        self.create_gate: threading.Event | None = None  # set -> create blocks (after recording) until opened
        self.exec_delay_s = 0.0
        self.deleted: set[str] = set()  # names a delete took; get() then answers NOT_FOUND unless get_results is set
        self._default_get_results = self.get_results
        self.delete_results: list[Any] = [DeletionResult(DeletionOutcome.COMPLETED, "sbx-1")]

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
        if self.gateway_labels is not None and name is not None:
            self.gateway_labels[name] = dict(labels or {})
        if self.create_gate is not None:
            self.create_gate.wait(timeout=5)
        result = self._next(self.create_results)
        return make_ref(result.phase, sandbox_id=result.id, name=name or result.name, workspace=workspace)

    def get(self, sandbox_name: str, *, workspace: str) -> Any:
        self.get_calls.append({"name": sandbox_name, "workspace": workspace})
        if sandbox_name in self.deleted and self.get_results is self._default_get_results:
            raise FakeRpcError(grpc.StatusCode.NOT_FOUND)
        return self._next(self.get_results)

    def delete(self, sandbox_name: str, *, workspace: str, allow_missing: bool = False) -> Any:
        self.delete_calls.append({"name": sandbox_name, "workspace": workspace, "allow_missing": allow_missing})
        if self.delete_gate is not None:
            self.delete_gate.wait(timeout=5)
        result = self._next(self.delete_results)
        self.deleted.add(sandbox_name)
        return result

    def exec(
        self,
        sandbox: str,
        command: list[str],
        *,
        workspace: str,
        stream_output: bool = False,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> Any:
        self.exec_calls.append(
            {
                "sandbox": sandbox,
                "workspace": workspace,
                "command": command,
                "workdir": workdir,
                "env": env,
                "stdin": stdin,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.exec_delay_s:
            time.sleep(self.exec_delay_s)
        return self._next(self.exec_results)

    def list_all(self, *, workspace: str, label_selector: str | None = None, **_: Any) -> list[Any]:
        self.list_calls.append({"workspace": workspace, "label_selector": label_selector})
        if self.gateway_labels is not None:  # filter like the gateway: every k=v of the selector must match
            wanted = dict(term.split("=", 1) for term in (label_selector or "").split(",") if term)
            return [
                make_ref(openshell_pb2.SANDBOX_PHASE_READY, name=name)
                for name, labels in self.gateway_labels.items()
                if all(labels.get(k) == v for k, v in wanted.items())
            ]
        return self._next(self.list_results)

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


@pytest.fixture(autouse=True)
def reset_delete_state():
    """Delete workers and the live/pending registries are process-wide; isolate each test."""
    yield
    for future in list(openshell_provider._PENDING_DELETES.values()):
        concurrent.futures.wait([future], timeout=5)
    for state in openshell_provider._DELETE_STATES.values():
        state.executor.shutdown(wait=True)
    openshell_provider._DELETE_STATES.clear()
    openshell_provider._LIVE_SANDBOXES.clear()
    openshell_provider._PENDING_DELETES.clear()


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
        None, "n", ["/bin/sh", "-c", "true"], workspace="w", workdir=None, env=None, stdin=None, timeout_seconds=None
    )
    inspect.signature(SandboxClient.close).bind(None)
    inspect.signature(SandboxClient.delete).bind(None, "n", workspace="w", allow_missing=True)
    inspect.signature(SandboxClient.list_all).bind(None, workspace="w", label_selector="a=b")
    inspect.signature(SandboxClient).bind("localhost:1", tls=None, bearer_token=None, timeout=5.0)
    inspect.signature(TlsConfig).bind(ca_path=None, cert_path=None, key_path=None)
    assert {"COMPLETED", "ACCEPTED", "ALREADY_ABSENT"} <= {member.name for member in DeletionOutcome}


def test_every_sdk_sandbox_phase_has_a_status_mapping() -> None:
    mapped = openshell_provider._phase_to_status_map()
    phases = {number for name, number in openshell_pb2.SandboxPhase.items()}
    assert phases <= set(mapped), f"unmapped SandboxPhase values: {sorted(phases - set(mapped))}"
    assert mapped[openshell_pb2.SANDBOX_PHASE_STOPPING] == SandboxStatus.STOPPED
    assert mapped[openshell_pb2.SANDBOX_PHASE_SUSPENDING] == SandboxStatus.STOPPED
    assert mapped[openshell_pb2.SANDBOX_PHASE_SUSPENDED] == SandboxStatus.STOPPED


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


def test_generated_sandbox_names_fit_routable_name_limit() -> None:
    names = [_generate_sandbox_name() for _ in range(100)]

    assert len(set(names)) == len(names)
    assert all(len(name) == MAX_SANDBOX_NAME_LENGTH for name in names)
    assert all(re.fullmatch(rf"{re.escape(SANDBOX_NAME_PREFIX)}[0-9a-f]+", name) for name in names)


async def _drain_deletes_only() -> None:
    """Wait for every delete handed to a worker; a done callback that never clears an entry fails after 10 s."""

    async def drain() -> None:
        while openshell_provider._PENDING_DELETES:
            pending = [f for f in openshell_provider._PENDING_DELETES.values() if not f.cancelled()]
            await asyncio.to_thread(concurrent.futures.wait, pending, 5)
            await asyncio.sleep(0)

    await asyncio.wait_for(drain(), 10)


async def _drain(provider: OpenShellProvider) -> None:
    """Wait for every delete handed to a worker, then release the provider."""
    await _drain_deletes_only()
    await provider.aclose()


def _create_labels(fake_client: FakeClient) -> dict[str, str]:
    return fake_client.create_calls[0]["labels"]


async def test_create_adds_attribution_labels_with_spec_keys_winning(
    make_provider, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEMO_GYM_TEAM", "team-a")
    monkeypatch.setenv("NEMO_GYM_USER", "alice")
    monkeypatch.setenv("NEMO_GYM_WORKLOAD", "bench")
    monkeypatch.setenv("NEMO_GYM_RUN_ID", "run-1")
    provider = make_provider()
    await provider.create(SandboxSpec(metadata={"nemo-gym.nvidia.com/team": "explicit", "task": "demo"}))
    assert _create_labels(fake_client) == {
        "nemo-gym.nvidia.com/team": "explicit",
        "nemo-gym.nvidia.com/user": "alice",
        "nemo-gym.nvidia.com/workload": "bench",
        "nemo-gym.nvidia.com/run": "run-1",
        "nemo-gym.nvidia.com/process": openshell_provider.resolve_process_id(),
        "task": "demo",
        SANDBOX_LABEL: "1",
    }


async def test_create_attribution_disabled_adds_nothing(
    make_provider, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEMO_GYM_TEAM", "team-a")
    provider = make_provider(attribution={"enabled": False})
    await provider.create(SandboxSpec(metadata={"task": "demo"}))
    assert _create_labels(fake_client) == {"task": "demo", SANDBOX_LABEL: "1"}


async def test_create_attribution_custom_prefix_and_run_key_present(
    make_provider, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NEMO_GYM_RUN_ID", raising=False)
    provider = make_provider(attribution={"key_prefix": "", "team": "t"})
    await provider.create(SandboxSpec())
    labels = _create_labels(fake_client)
    assert labels["team"] == "t"
    assert labels["run"]  # generated per process when NEMO_GYM_RUN_ID is unset


async def test_create_labels_are_valid_kubernetes_label_values(
    make_provider, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEMO_GYM_WORKLOAD", "responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml")
    provider = make_provider()
    await provider.create(SandboxSpec(metadata={"task": "x" * 200, "odd": " /weird value/ "}))
    labels = _create_labels(fake_client)
    value_re = re.compile(r"[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?")
    assert labels
    for key, value in labels.items():
        assert len(value) <= 63 and value_re.fullmatch(value), (key, value)
    assert labels["nemo-gym.nvidia.com/workload"].startswith("responses_api_agents_mini_swe_agent_2")
    assert labels["task"] == "x" * 63
    assert labels["odd"] == "weird_value"


def test_attribution_invalid_key_prefix_rejected() -> None:
    with pytest.raises(ValueError, match="key_prefix"):
        OpenShellProvider(attribution={"key_prefix": "Bad Prefix/"})


async def test_create_success_maps_spec(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(attribution={"enabled": False})
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
    assert len(call["name"]) <= MAX_SANDBOX_NAME_LENGTH
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


async def test_create_network_policies_mapping_builds_a_valid_sandbox_policy(
    make_provider, fake_client: FakeClient
) -> None:
    policy_mapping = {
        "version": 1,
        "network_policies": {
            "egress": {
                "name": "egress",
                "endpoints": [
                    {"host": "10.1.2.3", "port": 8000, "allowed_ips": ["10.1.2.3/32"]},
                    {"host": "model.example", "port": 443},
                ],
                "binaries": [{"path": "/**"}],
            }
        },
    }
    await make_provider().create(SandboxSpec(provider_options={"policy": policy_mapping}))

    rule = fake_client.create_calls[0]["spec"].policy.network_policies["egress"]
    assert [(e.host, e.port, list(e.allowed_ips)) for e in rule.endpoints] == [
        ("10.1.2.3", 8000, ["10.1.2.3/32"]),
        ("model.example", 443, []),
    ]
    assert [b.path for b in rule.binaries] == ["/**"]


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


async def test_create_error_phase_reports_the_gateway_conditions(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    ref = make_ref(openshell_pb2.SANDBOX_PHASE_ERROR)
    failed = dataclasses.replace(
        ref,
        status=dataclasses.replace(
            ref.status,
            conditions=(
                SandboxConditionRef(type="Ready", status="False", reason="ImagePull", message="no such image"),
            ),
        ),
    )
    fake_client.get_results = [failed]
    with pytest.raises(OpenShellCreateError, match=r"ERROR phase.*Ready=False \(ImagePull: no such image\)"):
        await provider.create(SandboxSpec())
    await _drain(provider)


@pytest.mark.parametrize(
    ("phase", "match"),
    [
        (openshell_pb2.SANDBOX_PHASE_ERROR, "ERROR phase"),
        (openshell_pb2.SANDBOX_PHASE_DELETING, "being deleted"),
        (openshell_pb2.SANDBOX_PHASE_STOPPED, "deleted or stopped"),
        (openshell_pb2.SANDBOX_PHASE_STOPPING, "deleted or stopped"),
        (openshell_pb2.SANDBOX_PHASE_SUSPENDING, "deleted or stopped"),
        (openshell_pb2.SANDBOX_PHASE_SUSPENDED, "deleted or stopped"),
        (openshell_pb2.SANDBOX_PHASE_COMPLETED, "deleted or stopped"),
    ],
)
async def test_create_terminal_phase_cleans_up(make_provider, fake_client: FakeClient, phase: int, match: str) -> None:
    provider = make_provider()
    fake_client.get_results = [make_ref(phase)]
    with pytest.raises(OpenShellCreateError, match=match):
        await provider.create(SandboxSpec())
    await _drain(provider)
    assert fake_client.delete_calls[0]["name"] == fake_client.create_calls[0]["name"]


async def test_create_ready_timeout_cleans_up(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.get_results = [make_ref(openshell_pb2.SANDBOX_PHASE_PROVISIONING)]
    with pytest.raises(OpenShellCreateError, match="was not READY"):
        await provider.create(SandboxSpec(ready_timeout_s=0.05))
    await _drain(provider)
    assert fake_client.delete_calls


async def test_create_ready_timeout_reports_gateway_conditions(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    ref = make_ref(openshell_pb2.SANDBOX_PHASE_PROVISIONING)
    stuck = dataclasses.replace(
        ref,
        status=dataclasses.replace(
            ref.status,
            conditions=(SandboxConditionRef(type="Ready", status="False", reason="Pulling", message="boom"),),
        ),
    )
    fake_client.get_results = [stuck]
    with pytest.raises(OpenShellCreateError, match=r"conditions: Ready=False \(Pulling: boom\)"):
        await provider.create(SandboxSpec(ready_timeout_s=0.05))
    await provider.aclose()


async def test_create_cleanup_failure_logs_leak_warning(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    fake_client.get_results = [make_ref(openshell_pb2.SANDBOX_PHASE_ERROR)]
    fake_client.delete_results = [FakeRpcError(grpc.StatusCode.UNAVAILABLE)]
    with caplog.at_level("WARNING"):
        with pytest.raises(OpenShellCreateError, match="ERROR phase"):
            await provider.create(SandboxSpec())
        await _drain(provider)
    assert "may be leaked" in caplog.text
    assert fake_client.create_calls[0]["name"] in caplog.text


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
    await _drain(provider)
    assert fake_client.delete_calls


async def test_exec_maps_command_and_result(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    handle = _make_handle(env={"A": 1, "B": "2"}, workdir="/workspace")
    fake_client.exec_results = [make_exec_result(exit_code=3, stdout="out", stderr="err")]

    result = await provider.exec(handle, "echo hi", env={"B": "override", "C": 3})

    call = fake_client.exec_calls[0]
    assert call["sandbox"] == "nemo-gym-test"
    assert call["workspace"] == "default"
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


async def test_exec_gateway_timeout_maps_to_timeout(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.exec_delay_s = 1.0
    fake_client.exec_results = [make_exec_result(exit_code=124, stdout="partial", stderr="")]
    result = await provider.exec(_make_handle(), "sleep 999", timeout_s=1)
    assert result.error_type == "timeout"
    assert result.return_code == 124
    assert result.stdout == "partial"
    assert "timed out after 1s" in result.stderr
    assert "did not kill the process" in result.stderr


@pytest.mark.parametrize(
    ("return_code", "elapsed_s", "timeout_s", "expected"),
    [
        (124, 30.0, 30, True),
        (124, 29.6, 30, True),  # within the slack below the timeout
        (124, 2.0, 30, False),  # exits 124 on its own, early
        (124, 29.0, 30, False),  # slack is capped at EXEC_TIMEOUT_SLACK_S
        (124, 0.95, 1, True),  # slack is a fraction of a short timeout
        (124, 0.5, 1, False),
        (1, 30.0, 30, False),
        (124, None, 30, False),
        (124, 30.0, None, False),
    ],
)
def test_exec_timed_out(return_code: int, elapsed_s: float | None, timeout_s: int | None, expected: bool) -> None:
    assert openshell_provider._exec_timed_out(return_code, elapsed_s, timeout_s) is expected


async def test_exec_own_exit_124_is_not_a_timeout(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.exec_results = [make_exec_result(exit_code=124, stderr="mine")]
    result = await provider.exec(_make_handle(), "exit 124", timeout_s=30)
    assert result == openshell_provider.SandboxExecResult(stdout="", stderr="mine", return_code=124)


async def test_exec_slow_non_124_exit_is_not_a_timeout(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.exec_delay_s = 1.0
    fake_client.exec_results = [make_exec_result(exit_code=1)]
    result = await provider.exec(_make_handle(), "false", timeout_s=1)
    assert result.error_type is None and result.return_code == 1


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
        (openshell_pb2.SANDBOX_PHASE_STARTING, SandboxStatus.STARTING),
        (openshell_pb2.SANDBOX_PHASE_STOPPED, SandboxStatus.STOPPED),
        (openshell_pb2.SANDBOX_PHASE_COMPLETED, SandboxStatus.STOPPED),
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
    await _drain(provider)
    assert [(c["name"], c["workspace"], c["allow_missing"]) for c in fake_client.delete_calls] == [
        ("nemo-gym-x", "team-a", True)
    ]
    assert [call["name"] for call in fake_client.get_calls] == ["nemo-gym-x", "nemo-gym-x"]


async def test_close_returns_while_delete_is_blocked_on_the_gateway(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    gate = fake_client.delete_gate = threading.Event()
    fake_client.get_results = [FakeRpcError(grpc.StatusCode.NOT_FOUND)]
    await provider.close(_make_handle(name="nemo-gym-x"))
    # The delete is already on a worker: close() returned and the gateway call has been (or is being) issued.
    for _ in range(100):
        if fake_client.delete_calls:
            break
        await asyncio.sleep(0.01)
    assert fake_client.delete_calls and not fake_client.get_calls
    assert not openshell_provider._PENDING_DELETES[("default", "nemo-gym-x")].done()
    gate.set()
    await _drain(provider)
    assert [call["name"] for call in fake_client.get_calls] == ["nemo-gym-x"]
    assert not openshell_provider._PENDING_DELETES


async def test_close_twice_schedules_one_delete(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.delete_gate = threading.Event()
    handle = _make_handle()
    await provider.close(handle)
    await provider.close(handle)
    await asyncio.sleep(0.05)
    fake_client.delete_gate.set()
    await _drain(provider)
    assert len(fake_client.delete_calls) == 1


async def test_close_delete_deadline_exceeded_waits_for_the_gateway_delete(
    make_provider, fake_client: FakeClient, caplog
) -> None:
    # The gateway finishes a delete it has started after the client deadline, so a deadline goes to the
    # deletion wait and does not issue another DeleteSandbox when the sandbox disappears.
    provider = make_provider()
    fake_client.delete_results = [FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)]
    fake_client.get_results = [
        make_ref(openshell_pb2.SANDBOX_PHASE_DELETING),
        FakeRpcError(grpc.StatusCode.NOT_FOUND),
    ]
    with caplog.at_level("ERROR"):
        await provider.close(_make_handle(name="nemo-gym-x"))
        await _drain(provider)
    assert len(fake_client.delete_calls) == 1
    assert len(fake_client.get_calls) == 2
    assert "may be leaked" not in caplog.text


async def test_close_delete_deadline_exceeded_reissues_when_the_sandbox_stays(
    make_provider, fake_client: FakeClient, caplog
) -> None:
    # A delete that timed out waiting on a lifecycle lock never started: the sandbox stays READY, so re-issue.
    provider = make_provider(operations={"close_timeout_s": 0.05})
    fake_client.delete_results = [
        FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED),
        DeletionResult(DeletionOutcome.COMPLETED),
    ]
    # get() answers READY until the second delete lands, then NOT_FOUND.
    with caplog.at_level("ERROR"):
        await provider.close(_make_handle(name="nemo-gym-x"))
        await _drain(provider)
    assert len(fake_client.delete_calls) == 2
    assert "may be leaked" not in caplog.text


async def test_close_delete_deadline_exceeded_every_attempt_logs_failure(
    make_provider, fake_client: FakeClient, caplog
) -> None:
    provider = make_provider(operations={"close_timeout_s": 0.05})
    fake_client.delete_results = [FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)]
    fake_client.get_results = [make_ref(openshell_pb2.SANDBOX_PHASE_READY)]
    with caplog.at_level("ERROR"):
        await provider.close(_make_handle(name="nemo-gym-x"))
        await _drain(provider)
    assert len(fake_client.delete_calls) == openshell_provider.MAX_DELETE_ATTEMPTS
    assert "'nemo-gym-x'" in caplog.text and "may be leaked" in caplog.text


async def test_close_delete_deadline_exceeded_never_gone_logs_failure(
    make_provider, fake_client: FakeClient, caplog
) -> None:
    provider = make_provider(operations={"close_timeout_s": 0.05})
    fake_client.delete_results = [FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)]
    fake_client.get_results = [make_ref(openshell_pb2.SANDBOX_PHASE_DELETING)]
    with caplog.at_level("ERROR"):
        await provider.close(_make_handle(name="nemo-gym-x"))
        await _drain(provider)
    assert "'nemo-gym-x'" in caplog.text and "may be leaked" in caplog.text


async def test_close_accepted_outcome_waits_for_the_sandbox_to_go(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.delete_results = [DeletionResult(DeletionOutcome.ACCEPTED)]
    fake_client.get_results = [make_ref(openshell_pb2.SANDBOX_PHASE_DELETING), FakeRpcError(grpc.StatusCode.NOT_FOUND)]
    await provider.close(_make_handle(name="nemo-gym-x"))
    await _drain(provider)
    assert len(fake_client.get_calls) == 2


async def test_close_non_rpc_delete_error_is_logged_as_failure(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    fake_client.delete_results = [ValueError("bad")]
    with caplog.at_level("ERROR"):
        await provider.close(_make_handle(name="nemo-gym-x"))
        await _drain(provider)
    assert "'nemo-gym-x'" in caplog.text and "may be leaked" in caplog.text


async def test_close_wait_non_rpc_error_is_logged_as_failure(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    fake_client.get_results = [ValueError("bad")]
    with caplog.at_level("ERROR"):
        await provider.close(_make_handle(name="nemo-gym-x"))
        await _drain(provider)
    assert "'nemo-gym-x'" in caplog.text and "may be leaked" in caplog.text


async def test_close_without_wait(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(operations={"close_wait_deleted": False})
    await provider.close(_make_handle())
    await _drain(provider)
    assert fake_client.delete_calls
    assert not fake_client.get_calls


async def test_close_unknown_outcome_skips_wait(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    fake_client.delete_results = [DeletionResult(DeletionOutcome.UNSPECIFIED)]
    with caplog.at_level("WARNING"):
        await provider.close(_make_handle())
        await _drain(provider)
    assert "skipping the deletion wait" in caplog.text
    assert not fake_client.get_calls


async def test_close_already_absent_skips_wait(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.delete_results = [DeletionResult(DeletionOutcome.ALREADY_ABSENT)]
    await provider.close(_make_handle())
    await _drain(provider)
    assert not fake_client.get_calls


async def test_close_missing_sandbox_is_success(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    fake_client.delete_results = [FakeRpcError(grpc.StatusCode.NOT_FOUND)]
    with caplog.at_level("ERROR"):
        await provider.close(_make_handle())
        await _drain(provider)
    assert not fake_client.get_calls
    assert "may be leaked" not in caplog.text


@pytest.mark.parametrize("code", [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.FAILED_PRECONDITION])
async def test_close_retries_transient_delete_failures_in_the_worker(
    make_provider, fake_client: FakeClient, caplog, code: grpc.StatusCode
) -> None:
    provider = make_provider()
    fake_client.delete_results = [FakeRpcError(code), DeletionResult(DeletionOutcome.ALREADY_ABSENT)]
    with caplog.at_level("ERROR"):
        await provider.close(_make_handle(name="nemo-gym-x"))
        await _drain(provider)
    assert len(fake_client.delete_calls) == 2
    assert "may be leaked" not in caplog.text


async def test_close_gives_up_on_a_delete_that_keeps_failing(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider(operations={"close_timeout_s": 0.1})
    fake_client.delete_results = [FakeRpcError(grpc.StatusCode.UNAVAILABLE)]
    with caplog.at_level("ERROR"):
        await provider.close(_make_handle(name="nemo-gym-x"))
        await _drain(provider)
    assert len(fake_client.delete_calls) >= 2
    assert "'nemo-gym-x'" in caplog.text and "may be leaked" in caplog.text


async def test_close_delete_failure_is_logged_and_retryable(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    fake_client.delete_results = [
        SandboxError("boom"),
        DeletionResult(DeletionOutcome.ALREADY_ABSENT),
    ]
    handle = _make_handle(name="nemo-gym-x")
    with caplog.at_level("ERROR"):
        await provider.close(handle)
        await asyncio.to_thread(concurrent.futures.wait, list(openshell_provider._PENDING_DELETES.values()))
        assert "'nemo-gym-x' failed" in caplog.text
        # The failed delete is forgotten, so a later close() retries it.
        await provider.close(handle)
        await _drain(provider)
    assert len(fake_client.delete_calls) == 2


async def test_aclose_does_not_wait_for_or_cancel_a_pending_delete(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    gate = fake_client.delete_gate = threading.Event()
    fake_client.get_results = [FakeRpcError(grpc.StatusCode.NOT_FOUND)]
    await provider.close(_make_handle(name="nemo-gym-x"))
    await asyncio.wait_for(provider.aclose(), timeout=1)  # returns although DeleteSandbox is still blocked
    assert not openshell_provider._PENDING_DELETES[("default", "nemo-gym-x")].done()
    assert fake_client.close_calls == 1  # the main client; the delete worker and its client are not released
    gate.set()
    await _drain(provider)
    # The queued delete ran to the end (delete, then the wait for NOT_FOUND) after the provider was released.
    assert [call["name"] for call in fake_client.get_calls] == ["nemo-gym-x"]


def test_delete_completes_after_the_event_loop_is_gone(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    gate = fake_client.delete_gate = threading.Event()
    fake_client.get_results = [FakeRpcError(grpc.StatusCode.NOT_FOUND)]

    async def stop_and_exit() -> None:
        await provider.close(_make_handle(name="nemo-gym-x"))
        await provider.aclose()

    with caplog.at_level("INFO"):
        asyncio.run(stop_and_exit())  # like `stop()` right before the process ends: the loop closes here
        assert not openshell_provider._PENDING_DELETES[("default", "nemo-gym-x")].done()
        gate.set()
        concurrent.futures.wait(list(openshell_provider._PENDING_DELETES.values()), timeout=5)
    assert [call["name"] for call in fake_client.delete_calls] == ["nemo-gym-x"]
    assert "'nemo-gym-x' deleted" in caplog.text


async def test_close_tracks_live_sandboxes(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    handle = await provider.create(SandboxSpec())
    assert ("default", handle.raw.name) in openshell_provider._LIVE_SANDBOXES
    await provider.close(handle)
    assert ("default", handle.raw.name) not in openshell_provider._LIVE_SANDBOXES
    await _drain(provider)


async def test_aclose_reaps_orphaned_sandboxes_of_this_process(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.list_results = [
        [
            make_ref(openshell_pb2.SANDBOX_PHASE_READY, name="orphan"),
            make_ref(openshell_pb2.SANDBOX_PHASE_READY, name="pending"),
        ]
    ]
    in_flight: concurrent.futures.Future[None] = concurrent.futures.Future()  # a delete already handed to a worker
    openshell_provider._PENDING_DELETES[("default", "pending")] = in_flight
    await provider.aclose()
    in_flight.set_result(None)
    openshell_provider._PENDING_DELETES.pop(("default", "pending"))
    await _drain_deletes_only()
    assert fake_client.list_calls == [
        {
            "workspace": "default",
            "label_selector": f"nemo-gym.nvidia.com/process={attribution.resolve_process_id()},{SANDBOX_LABEL}=1",
        }
    ]
    assert [call["name"] for call in fake_client.delete_calls] == ["orphan"]


async def test_reap_never_deletes_another_process_sandboxes_of_the_same_run(
    make_provider, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two processes of one job share the run id (NEMO_GYM_RUN_ID is inherited), so the run label must not
    # decide what this process may delete: process B's live sandbox shares the run but not the process label.
    monkeypatch.setenv("NEMO_GYM_RUN_ID", "shared-run")
    monkeypatch.setattr(openshell_provider, "resolve_process_id", lambda: "process-a")
    provider = make_provider()
    prefix = "nemo-gym.nvidia.com/"
    fake_client.gateway_labels = {
        "a-orphan": {f"{prefix}run": "shared-run", f"{prefix}process": "process-a", SANDBOX_LABEL: "1"},
        "b-live": {f"{prefix}run": "shared-run", f"{prefix}process": "process-b", SANDBOX_LABEL: "1"},
        "unlabeled": {},
    }
    await provider.aclose()
    await _drain_deletes_only()
    assert [call["name"] for call in fake_client.delete_calls] == ["a-orphan"]


async def test_create_process_label_differs_per_process(
    make_provider, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEMO_GYM_RUN_ID", "shared-run")
    provider = make_provider()
    monkeypatch.setattr(openshell_provider, "resolve_process_id", lambda: "process-a")
    await provider.create(SandboxSpec())
    monkeypatch.setattr(openshell_provider, "resolve_process_id", lambda: "process-b")
    await provider.create(SandboxSpec())
    first, second = (call["labels"] for call in fake_client.create_calls)
    assert first["nemo-gym.nvidia.com/run"] == second["nemo-gym.nvidia.com/run"] == "shared-run"
    assert (first["nemo-gym.nvidia.com/process"], second["nemo-gym.nvidia.com/process"]) == ("process-a", "process-b")


def test_process_id_is_stable_in_a_process_and_new_after_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    first = attribution.resolve_process_id()
    assert attribution.resolve_process_id() == first
    monkeypatch.setattr(attribution.os, "getpid", lambda: os_getpid() + 1)
    assert attribution.resolve_process_id() != first


async def test_aclose_reap_skips_when_a_sandbox_is_still_live_in_this_process(
    make_provider, fake_client: FakeClient
) -> None:
    provider = make_provider()
    fake_client.list_results = [[make_ref(openshell_pb2.SANDBOX_PHASE_READY, name="live")]]
    openshell_provider._LIVE_SANDBOXES.add(("default", "live"))
    await provider.aclose()
    assert not fake_client.list_calls and not fake_client.delete_calls


async def test_reap_never_deletes_a_live_sandbox(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    fake_client.list_results = [
        [
            make_ref(openshell_pb2.SANDBOX_PHASE_READY, name="live"),
            make_ref(openshell_pb2.SANDBOX_PHASE_READY, name="orphan"),
        ]
    ]
    openshell_provider._LIVE_SANDBOXES.add(("default", "live"))
    await provider._reap_orphans()
    await _drain_deletes_only()
    assert [call["name"] for call in fake_client.delete_calls] == ["orphan"]
    await provider.aclose()


async def test_reap_skipped_without_attribution(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(attribution={"enabled": False})
    await provider.aclose()
    assert not fake_client.list_calls


async def test_reap_list_failure_is_logged_not_raised(make_provider, fake_client: FakeClient, caplog) -> None:
    provider = make_provider()
    fake_client.list_results = [FakeRpcError(grpc.StatusCode.UNAVAILABLE)]
    with caplog.at_level("WARNING"):
        await provider.aclose()
    assert "Could not list orphaned" in caplog.text


@pytest.mark.parametrize(
    "metadata",
    [
        {"nemo-gym.nvidia.com/process": "mine", "nemo-gym.nvidia.com/run": "task-run"},
        {SANDBOX_LABEL: "0"},
    ],
)
async def test_caller_metadata_cannot_replace_the_process_or_marker_label(
    make_provider, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch, metadata: dict[str, str]
) -> None:
    monkeypatch.setattr(openshell_provider, "resolve_process_id", lambda: "process-a")
    provider = make_provider()
    await provider.create(SandboxSpec(metadata=metadata))
    labels = _create_labels(fake_client)
    assert labels["nemo-gym.nvidia.com/process"] == "process-a"
    assert labels[SANDBOX_LABEL] == "1"
    assert labels["nemo-gym.nvidia.com/run"] == metadata.get(
        "nemo-gym.nvidia.com/run", labels["nemo-gym.nvidia.com/run"]
    )


async def test_reap_leaves_a_create_in_flight_alone(make_provider, fake_client: FakeClient) -> None:
    # The gateway lists a sandbox as soon as it stores it, before CreateSandbox returns; another episode's
    # aclose() in the same process must not delete it.
    fake_client.gateway_labels = {}
    create_gate = fake_client.create_gate = threading.Event()
    creating = make_provider()
    other = make_provider()
    task = asyncio.create_task(creating.create(SandboxSpec()))
    for _ in range(200):
        if fake_client.create_calls:
            break
        await asyncio.sleep(0.01)
    assert fake_client.gateway_labels  # listable on the gateway while its create has not returned
    await other.aclose()
    assert not fake_client.delete_calls
    create_gate.set()
    handle = await task
    assert handle.raw.name == fake_client.create_calls[0]["name"]
    await creating.close(handle)
    await _drain(creating)


async def test_create_cancelled_while_waiting_for_ready_submits_the_delete(
    make_provider, fake_client: FakeClient
) -> None:
    provider = make_provider()
    fake_client.get_results = [make_ref(openshell_pb2.SANDBOX_PHASE_PROVISIONING)]
    task = asyncio.create_task(provider.create(SandboxSpec()))
    for _ in range(200):
        if fake_client.get_calls:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    name = fake_client.create_calls[0]["name"]
    assert ("default", name) not in openshell_provider._LIVE_SANDBOXES
    await _drain(provider)
    assert [call["name"] for call in fake_client.delete_calls] == [name]


async def test_create_cancelled_during_the_rpc_deletes_a_late_commit(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider()
    create_gate = fake_client.create_gate = threading.Event()
    task = asyncio.create_task(provider.create(SandboxSpec()))
    for _ in range(200):
        if fake_client.create_calls:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    create_gate.set()  # the create commits after the caller is gone
    for _ in range(200):
        if len(fake_client.delete_calls) >= 2:
            break
        await asyncio.sleep(0.01)
    await _drain(provider)
    name = fake_client.create_calls[0]["name"]
    assert {call["name"] for call in fake_client.delete_calls} == {name}
    assert fake_client.delete_calls[-1]["allow_missing"] is True


async def test_failed_create_rpc_submits_an_idempotent_delete(make_provider, fake_client: FakeClient) -> None:
    provider = make_provider(create={"retries": 0})
    fake_client.create_results = [FakeRpcError(grpc.StatusCode.UNAVAILABLE)]
    fake_client.delete_results = [DeletionResult(DeletionOutcome.ALREADY_ABSENT)]
    with pytest.raises(OpenShellCreateError):
        await provider.create(SandboxSpec())
    await _drain(provider)
    assert [call["name"] for call in fake_client.delete_calls] == [fake_client.create_calls[0]["name"]]
    assert not openshell_provider._LIVE_SANDBOXES


async def test_close_deletes_synchronously_when_the_executor_refuses_work(
    make_provider, fake_client: FakeClient
) -> None:
    provider = make_provider()
    handle = _make_handle(name="nemo-gym-x")
    openshell_provider._LIVE_SANDBOXES.add(("default", "nemo-gym-x"))

    def refuse(*_: Any, **__: Any) -> Any:
        raise RuntimeError("cannot schedule new futures after interpreter shutdown")

    provider._delete_shared.executor.submit = refuse  # type: ignore[method-assign]
    await provider.close(handle)
    assert [call["name"] for call in fake_client.delete_calls] == ["nemo-gym-x"]
    assert ("default", "nemo-gym-x") not in openshell_provider._LIVE_SANDBOXES


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


async def test_delete_uses_its_own_long_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[float] = []
    monkeypatch.setattr(
        openshell_provider,
        "_build_client",
        lambda connection: built.append(connection.request_timeout_s) or FakeClient(),
    )
    provider = OpenShellProvider(
        connection={"request_timeout_s": 30}, operations={"delete_rpc_timeout_s": 90}, probe={"command": None}
    )
    await provider.aclose()
    assert sorted(built) == [30, 90]


@pytest.mark.asyncio
async def test_reconnect_preserves_workspace_without_serializing_credentials(
    make_provider: Callable[..., OpenShellProvider], fake_client: FakeClient
) -> None:
    import json

    from nemo_gym.sandbox.api import AsyncSandbox

    workspace = "review-workspace"
    provider = make_provider(connection={"workspace": workspace, "bearer_token": "test-only-secret"})
    owner = AsyncSandbox(provider, SandboxSpec(image="img", env={"OMP_NUM_THREADS": "2"}, workdir="/workspace/output"))
    connected = None
    try:
        await owner.start()
        descriptor = json.loads(json.dumps(await owner.serialize()))
        assert "test-only-secret" not in json.dumps(descriptor)
        assert "env" not in descriptor
        assert "OMP_NUM_THREADS" not in json.dumps(descriptor)
        fake_client.get_results = [
            make_ref(
                openshell_pb2.SANDBOX_PHASE_READY,
                sandbox_id=descriptor["sandbox_id"],
                name=descriptor["name"],
                workspace=workspace,
            )
        ]
        receiver = make_provider(connection={"workspace": workspace, "bearer_token": "test-only-secret"})
        connected = await AsyncSandbox.connect(descriptor, provider=receiver)
        assert fake_client.get_calls[-1] == {"name": descriptor["name"], "workspace": workspace}
        await connected.exec("pwd")
        assert fake_client.exec_calls[-1]["workspace"] == workspace
        assert fake_client.exec_calls[-1]["workdir"] == "/workspace/output"
        assert fake_client.exec_calls[-1]["env"] is None
        await connected.disconnect()
        assert not fake_client.delete_calls
        fake_client.exec_results = [make_exec_result(stdout="owner alive")]
        result = await owner.exec("echo owner alive")
        assert result.return_code == 0 and result.stdout == "owner alive"
        assert fake_client.exec_calls[-1]["workspace"] == workspace
        assert fake_client.close_calls == 0
    finally:
        if connected is not None:
            await connected.disconnect()
        fake_client.get_results = [FakeRpcError(grpc.StatusCode.NOT_FOUND)]
        await owner.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replacement, lookup_expected",
    [
        ({"provider": "other"}, False),
        ({"endpoint": "other:8080"}, False),
        ({"workspace": "other"}, False),
        ({"sandbox_id": "replaced"}, True),
        ({"name": "replaced"}, True),
    ],
)
async def test_reconnect_rejects_changed_gateway_or_identity(make_provider, fake_client, replacement, lookup_expected):
    provider = make_provider()
    descriptor = await provider.serialize_handle(_make_handle())
    try:
        message = "sandbox identity changed" if lookup_expected else "same gateway and workspace"
        with pytest.raises(ValueError, match=message):
            await provider.connect({**descriptor, **replacement})
        assert len(fake_client.get_calls) == int(lookup_expected)
    finally:
        await provider.aclose()
