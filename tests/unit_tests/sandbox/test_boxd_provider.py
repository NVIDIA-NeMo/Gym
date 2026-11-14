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

"""Unit tests for the boxd sandbox provider (SDK faked; no network)."""

import asyncio
import inspect
import shlex
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.core.failure_kinds import (
    PROVIDER_COMMAND_OUTCOME_UNKNOWN,
    PROVIDER_CREATE_TIMEOUT,
    PROVIDER_CREDENTIALS_REJECTED,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_NOT_READY,
    PROVIDER_QUOTA_EXHAUSTED,
    PROVIDER_REQUEST_FAILED,
    PROVIDER_SETUP_FAILED,
    PROVIDER_UNAVAILABLE,
)
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers import boxd as boxd_pkg
from nemo_gym.sandbox.providers.base import (
    ConnectableProvider,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
    SupportsSandboxPauseResume,
)
from nemo_gym.sandbox.providers.boxd import provider as boxd_provider
from nemo_gym.sandbox.providers.boxd.provider import (
    CONTAINER_NAME,
    BoxdCreateConfig,
    BoxdMachine,
    BoxdProvider,
    machine_size,
)
from nemo_gym.sandbox.providers.registry import create_provider, get_provider_class


pytestmark = pytest.mark.sandbox

_REAL_REQUIRE_BOXD_SDK = boxd_provider._require_boxd_sdk

REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# Fake boxd SDK
# --------------------------------------------------------------------------


class BoxdError(Exception):
    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class AuthenticationError(BoxdError):
    pass


class PermissionDeniedError(BoxdError):
    pass


class NotFoundError(BoxdError):
    pass


class RateLimitError(BoxdError):
    pass


class APIConnectionError(BoxdError):
    pass


class APIStatusError(BoxdError):
    pass


@dataclass
class FakeExecResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class FakeMachine:
    id: str
    name: str
    status: str = "running"
    labels: dict[str, str] = field(default_factory=dict)


class FakeFiles:
    def __init__(self, sdk: "FakeSdk") -> None:
        self.sdk = sdk
        self.store: dict[str, bytes] = {}
        self.uploads: list[str] = []
        self.downloads: list[str] = []
        self.writable_prefixes = ("/tmp/", "/home/boxd/")
        self.upload_errors: list[BaseException] = []

    async def upload(self, machine_id: str, path: str, data: bytes) -> int:
        self.uploads.append(path)
        if self.upload_errors:
            raise self.upload_errors.pop(0)
        if not path.startswith(self.writable_prefixes):
            raise APIStatusError(f"upload failed: cannot write {path}")
        self.store[path] = data
        return len(data)

    async def download(self, machine_id: str, path: str) -> bytes:
        self.downloads.append(path)
        if path not in self.store:
            if not path.startswith(self.writable_prefixes):
                raise APIStatusError(f"download failed: cannot read {path}")
            raise NotFoundError(f"no such file: {path}", code="not_found")
        return self.store[path]


class FakeMachines:
    def __init__(self, sdk: "FakeSdk") -> None:
        self.sdk = sdk
        self.files = FakeFiles(sdk)
        self.machines: dict[str, FakeMachine] = {}
        self.create_calls: list[tuple[str, dict[str, Any]]] = []
        self.exec_calls: list[tuple[str, str, float | None]] = []
        self.deleted: list[str] = []
        self.hibernated: list[str] = []
        self.woken: list[str] = []
        self.create_error: BaseException | None = None
        # Fail the create call after the machine was made, as a lost answer would.
        self.create_error_after_create: BaseException | None = None
        # With create_error_after_create: the machine shows up only at this lookup by name.
        self.create_lands_at_lookup = 0
        self.pending: FakeMachine | None = None
        self.lookups = 0
        # The create call makes the machine and then never answers.
        self.create_never_answers = False
        self.transient_errors: dict[str, list[BaseException]] = {}
        self.ready_error: BaseException | None = None
        self.delete_errors: list[BaseException] = []
        self.get_error: BaseException | None = None
        # Queue of results or exceptions for the next exec calls; default is exit 0.
        self.exec_results: list[FakeExecResult | BaseException] = []

    async def create(self, name: str, **kwargs: Any) -> FakeMachine:
        self.create_calls.append((name, kwargs))
        if self.create_error is not None:
            raise self.create_error
        machine = FakeMachine(id=f"m-{len(self.machines) + 1}", name=name, labels=dict(kwargs.get("labels") or {}))
        if self.create_lands_at_lookup:
            self.pending = machine
        else:
            self.machines[machine.id] = machine
        if self.create_never_answers:
            await asyncio.Event().wait()
        if self.create_error_after_create is not None:
            raise self.create_error_after_create
        return machine

    def fail_next(self, method: str) -> None:
        if self.transient_errors.get(method):
            raise self.transient_errors[method].pop(0)

    def resolve(self, machine_id: str) -> str:
        """The SDK takes a machine's id or its name."""
        for machine in self.machines.values():
            if machine.name == machine_id:
                return machine.id
        return machine_id

    async def wait_until_ready(self, machine_id: str, *, timeout: float) -> FakeMachine:
        self.fail_next("wait_until_ready")
        if self.ready_error is not None:
            raise self.ready_error
        return self.machines[machine_id]

    async def get(self, machine_id: str) -> FakeMachine:
        if self.get_error is not None:
            raise self.get_error
        self.lookups += 1
        if self.pending is not None and self.lookups >= self.create_lands_at_lookup:
            self.machines[self.pending.id], self.pending = self.pending, None
        machine_id = self.resolve(machine_id)
        if machine_id not in self.machines:
            raise NotFoundError(f"machine {machine_id} not found", code="not_found")
        return self.machines[machine_id]

    async def delete(self, machine_id: str) -> None:
        if self.delete_errors:
            raise self.delete_errors.pop(0)
        if machine_id not in self.machines:
            raise NotFoundError(f"machine {machine_id} not found", code="not_found")
        del self.machines[machine_id]
        self.deleted.append(machine_id)

    async def hibernate(self, machine_id: str) -> None:
        self.fail_next("hibernate")
        self.hibernated.append(machine_id)
        self.machines[machine_id].status = "hibernated"

    async def wake(self, machine_id: str) -> None:
        self.fail_next("wake")
        self.woken.append(machine_id)
        self.machines[machine_id].status = "running"

    async def exec(self, machine_id: str, command: str, *, timeout: float | None = None) -> FakeExecResult:
        self.exec_calls.append((machine_id, command, timeout))
        if self.exec_results:
            result = self.exec_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        if "command -v bash" in command:
            return FakeExecResult(stdout="bash\n")
        return FakeExecResult()

    def commands(self) -> list[str]:
        return [command for _, command, _ in self.exec_calls]


class FakeClient:
    def __init__(self, sdk: "FakeSdk", **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.machines = sdk.machines
        self.closed = False
        sdk.clients.append(self)

    async def close(self) -> None:
        self.closed = True


class FakeSdk(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.clients: list[FakeClient] = []
        self.machines = FakeMachines(self)
        self.errors = types.SimpleNamespace(
            BoxdError=BoxdError,
            AuthenticationError=AuthenticationError,
            PermissionDeniedError=PermissionDeniedError,
            NotFoundError=NotFoundError,
            RateLimitError=RateLimitError,
            APIConnectionError=APIConnectionError,
            APIStatusError=APIStatusError,
        )
        self.AsyncBoxd = lambda **kwargs: FakeClient(self, **kwargs)


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> FakeSdk:
    fake = FakeSdk()
    monkeypatch.setattr(boxd_provider, "_require_boxd_sdk", lambda: fake)
    monkeypatch.setattr(boxd_provider, "ORPHAN_LOOKUP_S", 0.2)
    monkeypatch.setattr(boxd_provider, "ORPHAN_LOOKUP_INTERVAL_S", 0.01)
    return fake


def _provider(**kwargs: Any) -> BoxdProvider:
    kwargs.setdefault("attribution", {"enabled": False})
    kwargs.setdefault("operations", {"retry_delay_s": 0})
    return BoxdProvider(**kwargs)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Registration and configuration
# --------------------------------------------------------------------------


def test_registry_resolves_boxd() -> None:
    assert get_provider_class("boxd") is BoxdProvider
    provider = create_provider({"boxd": {"create": {"isolated": True}}})
    assert isinstance(provider, BoxdProvider)
    assert boxd_pkg.BoxdProvider is BoxdProvider


def test_provider_declares_optional_capabilities() -> None:
    provider = _provider()
    assert isinstance(provider, SupportsSandboxPauseResume)
    assert isinstance(provider, ConnectableProvider)


def test_unknown_config_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown BoxdCreateConfig keys: image"):
        BoxdProvider(create={"image": "x"})
    with pytest.raises(TypeError, match="expects a mapping"):
        BoxdProvider(exec=["x"])


@pytest.mark.parametrize(
    ("create", "message"),
    [
        ({"vcpu": 3}, "must name a machine size"),
        ({"vcpu": 2, "memory": "16G"}, "must name a machine size"),
        ({"memory": "lots"}, "must look like '8G'"),
        ({"auto_suspend_timeout": -1}, "auto_suspend_timeout must be >= 0"),
        ({"ready_timeout_s": 0}, "ready_timeout_s must be > 0"),
        ({"from_snapshot": ""}, "from_snapshot must be a non-empty string"),
        ({"name_prefix": 3}, "name_prefix must be a string"),
    ],
)
def test_create_config_validation(create: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BoxdCreateConfig(**create)


def test_create_config_labels_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="labels must be a mapping"):
        BoxdCreateConfig(labels=["a"])


def test_exec_and_operation_config_validation() -> None:
    with pytest.raises(ValueError, match="default_timeout_s must be > 0"):
        BoxdProvider(exec={"default_timeout_s": 0})
    with pytest.raises(ValueError, match="retries must be >= 0"):
        BoxdProvider(operations={"retries": -1})
    with pytest.raises(ValueError, match="retry_delay_s must be >= 0"):
        BoxdProvider(operations={"retry_delay_s": -1})


def test_shipped_config_selects_the_provider_and_keeps_the_key_out() -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.load(REPO_ROOT / "nemo_gym/sandbox/providers/boxd/configs/boxd.yaml")
    block = OmegaConf.to_container(config, resolve=False)["sandbox"]
    assert set(block) == {"default_metadata", "boxd"}
    assert "api_key" not in block["boxd"]["connection"]
    assert block["boxd"]["create"]["isolated"] is True
    assert block["boxd"]["create"]["max_lifetime_s"] == 3600.0
    resolved = OmegaConf.to_container(config.sandbox.boxd, resolve=True)
    BoxdProvider(**resolved)


def test_require_boxd_sdk_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "boxd", None)
    with pytest.raises(ImportError, match=r"nemo-gym\[boxd\]"):
        _REAL_REQUIRE_BOXD_SDK()


# --------------------------------------------------------------------------
# Machine sizes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("resources", "size"),
    [
        (SandboxResources(), None),
        (SandboxResources(cpu=1), (1, 4)),
        (SandboxResources(cpu=1.5), (2, 8)),
        (SandboxResources(memory_mib=8192), (2, 8)),
        (SandboxResources(cpu=1, memory_mib=10000), (4, 16)),
        (SandboxResources(cpu=8, memory_mib=32768, disk_gib=100), (8, 32)),
    ],
)
def test_machine_size_picks_the_smallest_size_that_fits(resources: SandboxResources, size: Any) -> None:
    assert machine_size(resources) == size


@pytest.mark.parametrize(
    ("resources", "message"),
    [
        (SandboxResources(cpu=16), "No boxd machine size holds"),
        (SandboxResources(gpu=1), "no GPUs"),
        (SandboxResources(gpu_type="a100"), "no GPUs"),
        (SandboxResources(disk_gib=200), "100 GiB disk"),
    ],
)
def test_machine_size_rejects_requests_no_machine_holds(resources: SandboxResources, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        machine_size(resources)


# --------------------------------------------------------------------------
# Create
# --------------------------------------------------------------------------


def test_create_machine_mode_is_isolated_and_labelled(sdk: FakeSdk) -> None:
    provider = _provider(
        connection={"org": "acme", "region": "fra", "api_key": "k"},
        create={"labels": {"team": "rl"}},
    )
    spec = SandboxSpec(workdir="/workspace", metadata={"task": "t1"}, resources={"cpu": 4})
    handle = _run(provider.create(spec))

    name, kwargs = sdk.machines.create_calls[0]
    assert name.startswith("nemo-gym-")
    assert kwargs["isolated"] is True
    assert kwargs["auto_suspend_timeout"] == 0
    assert kwargs["org"] == "acme"
    assert kwargs["labels"] == {"team": "rl", "task": "t1"}
    assert (kwargs["vcpu"], kwargs["memory"]) == (4, "16G")
    assert sdk.clients[0].kwargs == {"api_key": "k", "region": "fra", "max_retries": 0}
    assert handle.sandbox_id == "m-1"
    assert handle.provider_name == "boxd"
    assert handle.raw.container is None
    # The workdir exists and belongs to the machine's user before the first command runs in it.
    assert sdk.machines.commands() == ["sudo -n mkdir -p /workspace && sudo -n chown boxd: /workspace"]


def test_create_uses_configured_size_and_ttl(sdk: FakeSdk) -> None:
    provider = _provider(create={"vcpu": 8, "memory": "32G"})
    _run(provider.create(SandboxSpec(ttl_s=90.5)))
    _, kwargs = sdk.machines.create_calls[0]
    assert (kwargs["vcpu"], kwargs["memory"]) == (8, "32G")
    assert kwargs["delete_after"] == 91
    assert "org" not in kwargs


def test_every_machine_gets_a_lifetime_unless_it_is_turned_off(sdk: FakeSdk) -> None:
    _run(_provider().create(SandboxSpec()))
    _run(_provider(create={"max_lifetime_s": 120.2}).create(SandboxSpec()))
    _run(_provider(create={"max_lifetime_s": 120}).create(SandboxSpec(ttl_s=30)))
    _run(_provider(create={"max_lifetime_s": None}).create(SandboxSpec()))
    lifetimes = [kwargs.get("delete_after") for _, kwargs in sdk.machines.create_calls]
    assert lifetimes == [3600, 121, 30, None]
    with pytest.raises(ValueError, match="create.max_lifetime_s must be > 0"):
        BoxdCreateConfig(max_lifetime_s=0)


def test_the_sdk_client_does_not_retry_on_its_own(sdk: FakeSdk) -> None:
    """Retries are the provider's: a create is sent once, and other calls retry in one layer only."""
    provider = _provider()
    sdk.machines.create_error = APIConnectionError("reset")
    with pytest.raises(NgFailureEvent):
        _run(provider.create(SandboxSpec()))
    assert sdk.clients[0].kwargs["max_retries"] == 0
    assert len(sdk.machines.create_calls) == 1


def test_readiness_pause_and_resume_retry_transport_failures(sdk: FakeSdk) -> None:
    sdk.machines.transient_errors = {
        "wait_until_ready": [APIConnectionError("reset")],
        "hibernate": [APIStatusError("down", code="unavailable")],
        "wake": [APIConnectionError("reset")],
    }
    provider = _provider()
    handle = _run(provider.create(SandboxSpec()))
    _run(provider.pause(handle))
    _run(provider.resume(handle))
    assert sdk.machines.hibernated == sdk.machines.woken == ["m-1"]
    assert sdk.machines.transient_errors == {"wait_until_ready": [], "hibernate": [], "wake": []}


def test_create_from_snapshot_keeps_the_snapshot_size(sdk: FakeSdk, caplog: pytest.LogCaptureFixture) -> None:
    provider = _provider(create={"from_snapshot": "base", "vcpu": 2})
    _run(provider.create(SandboxSpec(resources={"cpu": 4})))
    _, kwargs = sdk.machines.create_calls[0]
    assert kwargs["from_snapshot"] == "base"
    assert "vcpu" not in kwargs and "memory" not in kwargs
    assert "keeps the snapshot's size" in caplog.text


def test_resource_requests_are_accepted_and_the_limits_size_the_machine(sdk: FakeSdk) -> None:
    spec = SandboxSpec(resources={"cpu": 2, "memory_mib": 8192}, provider_options={"resource_requests": {"cpu": 0.5}})
    _run(_provider().create(spec))
    _, kwargs = sdk.machines.create_calls[0]
    assert (kwargs["vcpu"], kwargs["memory"]) == (2, "8G")


def test_provider_option_overrides_snapshot(sdk: FakeSdk) -> None:
    _run(_provider().create(SandboxSpec(provider_options={"from_snapshot": "warm"})))
    assert sdk.machines.create_calls[0][1]["from_snapshot"] == "warm"


def test_attribution_becomes_labels_and_spec_metadata_wins(sdk: FakeSdk) -> None:
    provider = _provider(attribution={"team": "rl", "run": "r1", "key_prefix": "nemo-gym.nvidia.com/"})
    _run(provider.create(SandboxSpec(metadata={"nemo-gym.nvidia.com/run": "mine"})))
    labels = sdk.machines.create_calls[0][1]["labels"]
    assert labels["nemo-gym.nvidia.com/team"] == "rl"
    assert labels["nemo-gym.nvidia.com/run"] == "mine"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (SandboxSpec(provider_options={"template": "x"}), "Unknown boxd provider option"),
        (SandboxSpec(entrypoint=["/start"]), "entrypoint needs SandboxSpec.image"),
        (SandboxSpec(ports=[8080]), "does not publish"),
        (SandboxSpec(ready_timeout_s=0), "ready_timeout_s must be > 0"),
        (SandboxSpec(ttl_s=-1), "ttl_s must be > 0"),
        (SandboxSpec(resources={"gpu": 1}), "no GPUs"),
    ],
)
def test_create_rejects_specs_before_allocating(sdk: FakeSdk, spec: SandboxSpec, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _run(_provider().create(spec))
    assert sdk.machines.create_calls == []


def test_create_container_mode_pulls_and_starts_the_image(sdk: FakeSdk) -> None:
    spec = SandboxSpec(image="docker://ghcr.io/acme/task:1.0", workdir="/testbed", env={"A": "b c"})
    handle = _run(_provider().create(spec))
    commands = sdk.machines.commands()
    image = "ghcr.io/acme/task:1.0"
    assert commands[0] == f"docker image inspect {image} >/dev/null 2>&1 || docker pull -q {image} >/dev/null"
    run = shlex.split(commands[1])
    assert run[:8] == ["docker", "run", "-d", "--init", "--name", CONTAINER_NAME, "--network", "host"]
    assert ["-w", "/testbed"] == run[8:10]
    assert ["--env", "A=b c"] == run[10:12]
    assert run[12:] == ["--entrypoint", "/bin/sh", image, "-c", "while :; do sleep 2147483647; done"]
    assert "command -v bash" in commands[2]
    assert handle.raw.container == CONTAINER_NAME
    assert handle.raw.shell == "bash"


def test_create_container_mode_with_entrypoint_and_always_pull(sdk: FakeSdk) -> None:
    provider = _provider(create={"always_pull": True}, exec={"shell": "sh"})
    _run(provider.create(SandboxSpec(image="img", entrypoint=["/serve", "--port", "1"])))
    commands = sdk.machines.commands()
    assert commands[0] == "docker pull -q img >/dev/null"
    assert shlex.split(commands[1])[-5:] == ["--entrypoint", "/serve", "img", "--port", "1"]
    assert len(commands) == 2  # a configured shell needs no probe


def test_create_falls_back_to_sh_without_bash(sdk: FakeSdk) -> None:
    sdk.machines.exec_results = [FakeExecResult(), FakeExecResult(), FakeExecResult(stdout="sh\n")]
    handle = _run(_provider().create(SandboxSpec(image="alpine")))
    assert handle.raw.shell == "sh"


def test_create_reports_unready_container_and_deletes_the_machine(sdk: FakeSdk) -> None:
    sdk.machines.exec_results = [FakeExecResult(), FakeExecResult(), FakeExecResult(stdout="weird")]
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().create(SandboxSpec(image="alpine")))
    assert raised.value.failure_kind == PROVIDER_NOT_READY
    assert sdk.machines.deleted == ["m-1"]


def test_failed_image_pull_is_a_setup_failure_and_deletes_the_machine(sdk: FakeSdk) -> None:
    sdk.machines.exec_results = [FakeExecResult(stderr="manifest unknown", exit_code=1)]
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().create(SandboxSpec(image="nope:1")))
    event = raised.value
    assert event.boundary == "sandbox"
    assert event.failure_kind == PROVIDER_SETUP_FAILED
    assert event.operation == "image pull"
    assert "manifest unknown" in event.failure_reason
    assert sdk.machines.deleted == ["m-1"]


def test_readiness_timeout_is_a_create_timeout_and_deletes_the_machine(sdk: FakeSdk) -> None:
    sdk.machines.ready_error = TimeoutError("not running")
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().create(SandboxSpec(ready_timeout_s=5)))
    assert raised.value.failure_kind == PROVIDER_CREATE_TIMEOUT
    assert "within 5s" in raised.value.failure_reason
    assert sdk.machines.deleted == ["m-1"]


def test_readiness_sdk_error_keeps_its_cause(sdk: FakeSdk) -> None:
    sdk.machines.ready_error = APIConnectionError("reset")
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().create(SandboxSpec()))
    assert raised.value.failure_kind == PROVIDER_UNAVAILABLE
    assert isinstance(raised.value.__cause__, APIConnectionError)


def test_cleanup_failure_after_a_failed_create_does_not_hide_the_cause(
    sdk: FakeSdk, caplog: pytest.LogCaptureFixture
) -> None:
    sdk.machines.ready_error = TimeoutError()
    sdk.machines.delete_errors = [PermissionDeniedError("no")]
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().create(SandboxSpec()))
    assert raised.value.failure_kind == PROVIDER_CREATE_TIMEOUT
    assert "Could not delete boxd machine m-1" in caplog.text


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (AuthenticationError("invalid_api_key"), PROVIDER_CREDENTIALS_REJECTED),
        (PermissionDeniedError("denied"), PROVIDER_CREDENTIALS_REJECTED),
        (RateLimitError("quota"), PROVIDER_QUOTA_EXHAUSTED),
        (APIConnectionError("reset"), PROVIDER_UNAVAILABLE),
        (APIStatusError("down", code="unavailable"), PROVIDER_UNAVAILABLE),
        (asyncio.TimeoutError(), PROVIDER_IO_TIMEOUT),
        (APIStatusError("bad", code="invalid_argument"), PROVIDER_REQUEST_FAILED),
    ],
)
def test_create_errors_are_classified_by_type(sdk: FakeSdk, error: BaseException, kind: str) -> None:
    sdk.machines.create_error = error
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().create(SandboxSpec()))
    event = raised.value
    assert event.failure_kind == kind
    assert event.operation == "create"
    assert event.details.provider == "boxd"
    assert event.__cause__ is error
    # A create is never replayed: its answer may have been lost after the machine was made.
    assert len(sdk.machines.create_calls) == 1


def test_create_that_fails_after_making_the_machine_deletes_it_by_name(sdk: FakeSdk) -> None:
    sdk.machines.create_error_after_create = APIConnectionError("answer lost")
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().create(SandboxSpec()))
    assert raised.value.failure_kind == PROVIDER_UNAVAILABLE
    assert sdk.machines.deleted == ["m-1"]
    assert sdk.machines.machines == {}


def test_machine_that_appears_after_a_failed_create_call_is_deleted(sdk: FakeSdk) -> None:
    sdk.machines.create_error_after_create = APIConnectionError("answer lost")
    sdk.machines.create_lands_at_lookup = 3
    with pytest.raises(NgFailureEvent):
        _run(_provider().create(SandboxSpec()))
    assert sdk.machines.lookups == 3
    assert sdk.machines.deleted == ["m-1"]


def test_refused_create_looks_for_a_machine_once(sdk: FakeSdk) -> None:
    sdk.machines.create_error = APIStatusError("bad size", code="invalid_argument")
    with pytest.raises(NgFailureEvent):
        _run(_provider().create(SandboxSpec()))
    assert sdk.machines.lookups == 1


@pytest.mark.parametrize("lands_at_lookup", [0, 2])
def test_cancelled_create_call_deletes_the_machine_it_made(sdk: FakeSdk, lands_at_lookup: int) -> None:
    sdk.machines.create_never_answers = True
    sdk.machines.create_lands_at_lookup = lands_at_lookup

    async def scenario() -> None:
        task = asyncio.ensure_future(_provider().create(SandboxSpec()))
        while not sdk.machines.create_calls:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _run(scenario())
    assert sdk.machines.deleted == ["m-1"]
    assert sdk.machines.machines == {}


def test_cleanup_of_a_cancelled_create_survives_a_second_cancel(sdk: FakeSdk) -> None:
    sdk.machines.create_never_answers = True
    sdk.machines.create_lands_at_lookup = 5

    async def scenario() -> None:
        provider = _provider()
        task = asyncio.ensure_future(provider.create(SandboxSpec()))
        while not sdk.machines.create_calls:
            await asyncio.sleep(0)
        task.cancel()
        while not sdk.machines.lookups:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sdk.machines.deleted == []
        await provider.aclose()

    _run(scenario())
    assert sdk.machines.deleted == ["m-1"]


def test_cancel_while_waiting_for_readiness_deletes_the_machine(sdk: FakeSdk) -> None:
    async def never_ready(machine_id: str, *, timeout: float) -> None:
        await asyncio.Event().wait()

    sdk.machines.wait_until_ready = never_ready

    async def scenario() -> None:
        task = asyncio.ensure_future(_provider().create(SandboxSpec()))
        while not sdk.machines.machines:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _run(scenario())
    assert sdk.machines.deleted == ["m-1"]


def test_create_cleanup_lookup_failure_keeps_the_create_error(sdk: FakeSdk, caplog: pytest.LogCaptureFixture) -> None:
    sdk.machines.create_error = APIConnectionError("reset")
    sdk.machines.get_error = APIConnectionError("still down")
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().create(SandboxSpec()))
    assert raised.value.failure_kind == PROVIDER_UNAVAILABLE
    assert "Could not look up boxd machine" in caplog.text


def test_refused_credentials_on_create_skip_the_cleanup(sdk: FakeSdk) -> None:
    sdk.machines.create_error = AuthenticationError("invalid_api_key")
    with pytest.raises(NgFailureEvent):
        _run(_provider().create(SandboxSpec()))
    assert sdk.machines.deleted == []


# --------------------------------------------------------------------------
# Exec
# --------------------------------------------------------------------------


def _machine_handle(**raw: Any) -> SandboxHandle:
    return SandboxHandle(sandbox_id="m-1", provider_name="boxd", raw=BoxdMachine(machine_id="m-1", name="n", **raw))


def test_exec_returns_exit_code_and_split_streams(sdk: FakeSdk) -> None:
    sdk.machines.exec_results = [FakeExecResult(stdout="out\n", stderr="err\n", exit_code=7)]
    result = _run(_provider().exec(_machine_handle(), "false", timeout_s=30))
    assert (result.stdout, result.stderr, result.return_code, result.error_type) == ("out\n", "err\n", 7, None)
    assert sdk.machines.exec_calls[0][2] == 30.0


def test_exec_machine_mode_line_runs_in_a_real_shell(sdk: FakeSdk, tmp_path: Path) -> None:
    """The line the provider sends gives the command its env, cwd and quoting when a shell runs it."""
    handle = _machine_handle(env={"SPEC_VAR": "from spec"})
    _run(_provider().exec(handle, 'echo "$SPEC_VAR|$ARG|$(pwd)|it\'s"', cwd=str(tmp_path), env={"ARG": "a b"}))
    line = sdk.machines.exec_calls[0][1]
    completed = subprocess.run(["sh", "-c", line], capture_output=True, text=True, check=True)
    assert completed.stdout == f"from spec|a b|{tmp_path}|it's\n"


def test_exec_per_command_env_overrides_spec_env(sdk: FakeSdk) -> None:
    _run(_provider().exec(_machine_handle(env={"K": "spec"}), "echo $K", env={"K": "call"}))
    completed = subprocess.run(["sh", "-c", sdk.machines.exec_calls[0][1]], capture_output=True, text=True)
    assert completed.stdout == "call\n"


@pytest.mark.parametrize(
    ("user", "prefix"),
    [
        (None, "bash -c"),
        ("boxd", "bash -c"),
        ("root", "sudo -n bash -c"),
        (0, "sudo -n bash -c"),
        ("agent", "sudo -n -u agent -H bash -c"),
        (1001, "sudo -n -u '#1001' -H bash -c"),
    ],
)
def test_exec_machine_mode_switches_user_with_sudo(sdk: FakeSdk, user: Any, prefix: str) -> None:
    _run(_provider().exec(_machine_handle(), "id", user=user))
    assert sdk.machines.exec_calls[0][1] == f"{prefix} id"


def test_exec_container_mode_uses_docker_exec(sdk: FakeSdk) -> None:
    handle = _machine_handle(container=CONTAINER_NAME, shell="bash", env={"IGNORED": "1"})
    _run(_provider().exec(handle, "ls -la | head", cwd="/testbed", env={"X": "1 2"}, user="root"))
    argv = shlex.split(sdk.machines.exec_calls[0][1])
    assert argv == [
        "docker",
        "exec",
        "-w",
        "/testbed",
        "--env",
        "X=1 2",
        "--user",
        "0",
        CONTAINER_NAME,
        "bash",
        "-c",
        "ls -la | head",
    ]


def test_exec_rejects_invalid_env_names(sdk: FakeSdk) -> None:
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        _run(_provider().exec(_machine_handle(), "true", env={"A=B": "c"}))


def test_exec_uses_the_configured_default_timeout(sdk: FakeSdk) -> None:
    _run(_provider(exec={"default_timeout_s": 42}).exec(_machine_handle(), "true"))
    assert sdk.machines.exec_calls[0][2] == 42.0
    with pytest.raises(ValueError, match="timeout_s must be > 0"):
        _run(_provider().exec(_machine_handle(), "true", timeout_s=-1))


def test_exec_timeout_returns_a_timeout_result(sdk: FakeSdk, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([100.0, 103.5])
    monkeypatch.setattr(boxd_provider, "monotonic", lambda: next(clock))
    sdk.machines.exec_results = [BoxdError("exec timed out after 3s")]
    result = _run(_provider().exec(_machine_handle(), "sleep 10", timeout_s=3))
    assert result.error_type == "timeout"
    assert result.return_code == 124
    assert result.stdout is None
    assert "timed out after 3s" in result.stderr


def test_exec_error_before_the_deadline_is_a_failure_event(sdk: FakeSdk, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([100.0, 100.2])
    monkeypatch.setattr(boxd_provider, "monotonic", lambda: next(clock))
    sdk.machines.exec_results = [APIConnectionError("stream reset")]
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().exec(_machine_handle(), "make", timeout_s=3))
    # The command may have started before the transport failed.
    assert raised.value.failure_kind == PROVIDER_COMMAND_OUTCOME_UNKNOWN
    assert raised.value.operation == "exec"
    # Commands are not replayed: the first attempt may have run.
    assert len(sdk.machines.exec_calls) == 1


def test_long_commands_get_a_keepalive_that_is_stripped(sdk: FakeSdk, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boxd_provider.uuid, "uuid4", lambda: types.SimpleNamespace(hex="ab12"))
    mark = "<nemo-gym-keepalive-ab12>"
    # The command's own NUL bytes are output like any other and stay.
    sdk.machines.exec_results = [FakeExecResult(stdout="ok\n", stderr=f"{mark}wa\0rn{mark}\n", exit_code=0)]
    result = _run(_provider().exec(_machine_handle(), "make test", timeout_s=1800))
    assert result.stderr == "wa\0rn\n"
    line = sdk.machines.exec_calls[0][1]
    assert line.startswith(f"( while sleep 300 >/dev/null 2>&1; do printf %s '{mark}' >&2; done )")
    assert line.endswith("; status=$?; kill $keepalive 2>/dev/null; exit $status")

    _run(_provider().exec(_machine_handle(), "true", timeout_s=60))
    assert "keepalive" not in sdk.machines.exec_calls[1][1]


def test_keepalive_wrapper_keeps_the_exit_code_and_ends_with_the_command(
    sdk: FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(boxd_provider, "KEEPALIVE_INTERVAL_S", 1)
    _run(_provider().exec(_machine_handle(), "sleep 2.5; echo done; exit 3", timeout_s=3600))
    started = time.monotonic()
    completed = subprocess.run(["sh", "-c", sdk.machines.exec_calls[0][1]], capture_output=True, timeout=30)
    assert time.monotonic() - started < 5
    assert completed.returncode == 3
    assert completed.stdout == b"done\n"
    assert completed.stderr.count(b"<nemo-gym-keepalive-") >= 2


def test_exec_stream_that_ends_without_a_status_is_outcome_unknown(
    sdk: FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter([100.0, 130.0])
    monkeypatch.setattr(boxd_provider, "monotonic", lambda: next(clock))
    sdk.machines.exec_results = [BoxdError("stream ended")]
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().exec(_machine_handle(), "make", timeout_s=600))
    assert raised.value.failure_kind == PROVIDER_COMMAND_OUTCOME_UNKNOWN


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (AuthenticationError("expired"), PROVIDER_CREDENTIALS_REJECTED),
        (RateLimitError("quota"), PROVIDER_QUOTA_EXHAUSTED),
        (APIStatusError("internal", code="internal"), PROVIDER_COMMAND_OUTCOME_UNKNOWN),
    ],
)
def test_exec_errors_keep_kinds_that_prove_the_command_never_ran(sdk: FakeSdk, error: Exception, kind: str) -> None:
    sdk.machines.exec_results = [error]
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().exec(_machine_handle(), "make", timeout_s=30))
    assert raised.value.failure_kind == kind


def test_exec_without_an_exit_status_is_outcome_unknown(sdk: FakeSdk) -> None:
    sdk.machines.exec_results = [FakeExecResult(stdout="partial", exit_code=-1)]
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().exec(_machine_handle(), "make", timeout_s=30))
    assert raised.value.failure_kind == PROVIDER_COMMAND_OUTCOME_UNKNOWN
    assert "without an exit status" in raised.value.failure_reason


def test_exec_needs_a_boxd_handle(sdk: FakeSdk) -> None:
    with pytest.raises(RuntimeError, match="carries no boxd machine"):
        _run(_provider().exec(SandboxHandle(sandbox_id="x", provider_name="boxd", raw=None), "true"))


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def test_upload_and_download_machine_mode(sdk: FakeSdk, tmp_path: Path) -> None:
    source = tmp_path / "in.bin"
    source.write_bytes(b"\x00\x01payload")
    provider = _provider()
    handle = _machine_handle()
    _run(provider.upload_file(handle, source, "/home/boxd/in.bin"))
    assert sdk.machines.files.store["/home/boxd/in.bin"] == b"\x00\x01payload"
    target = tmp_path / "out" / "back.bin"
    _run(provider.download_file(handle, "/home/boxd/in.bin", target))
    assert target.read_bytes() == b"\x00\x01payload"
    assert sdk.machines.exec_calls == []


def test_upload_to_a_root_owned_path_stages_and_moves_with_sudo(sdk: FakeSdk) -> None:
    _run(_provider().write_file(_machine_handle(), "/workspace/a b.txt", "hi"))
    staged = sdk.machines.files.uploads[1]
    assert staged.startswith("/tmp/nemo-gym-transfer/")
    assert sdk.machines.commands() == [
        f"sudo -n mkdir -p /workspace && sudo -n mv {staged} '/workspace/a b.txt'"
        f" || {{ status=$?; rm -f {staged}; exit $status; }}"
    ]


def test_download_of_a_root_owned_path_copies_out_with_sudo(sdk: FakeSdk) -> None:
    provider = _provider()
    sdk.machines.exec_results = [FakeExecResult(), FakeExecResult()]

    async def scenario() -> bytes:
        original = sdk.machines.exec

        async def exec_and_stage(machine_id: str, command: str, *, timeout: float | None = None) -> Any:
            if "sudo -n cp" in command:
                staged = shlex.split(command.split("&&")[1])[-1]
                sdk.machines.files.store[staged] = b"secret"
            return await original(machine_id, command, timeout=timeout)

        sdk.machines.exec = exec_and_stage
        return await provider.read_file(_machine_handle(), "/root/out.txt")

    assert _run(scenario()) == b"secret"
    commands = sdk.machines.commands()
    assert commands[0].startswith(
        "mkdir -p /tmp/nemo-gym-transfer && sudo -n cp /root/out.txt /tmp/nemo-gym-transfer/"
    )
    assert commands[1].startswith("rm -f /tmp/nemo-gym-transfer/")


def test_failed_staging_copy_still_removes_the_staged_file(sdk: FakeSdk) -> None:
    sdk.machines.exec_results = [FakeExecResult(stderr="cp: permission denied", exit_code=1), FakeExecResult()]
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().read_file(_machine_handle(), "/root/secret"))
    assert raised.value.failure_kind == PROVIDER_SETUP_FAILED
    assert sdk.machines.commands()[1].startswith("rm -f /tmp/nemo-gym-transfer/")


def test_download_of_a_missing_file_is_a_failure_event(sdk: FakeSdk) -> None:
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().read_file(_machine_handle(), "/tmp/missing"))
    assert raised.value.operation == "download"
    assert isinstance(raised.value.__cause__, NotFoundError)


def test_container_upload_goes_through_docker_cp(sdk: FakeSdk) -> None:
    handle = _machine_handle(container=CONTAINER_NAME)
    _run(_provider().write_file(handle, "/testbed/patch.diff", b"diff"))
    staged = sdk.machines.files.uploads[0]
    assert sdk.machines.files.store[staged] == b"diff"
    assert sdk.machines.commands() == [
        f"docker exec --user 0 {CONTAINER_NAME} mkdir -p /testbed && docker cp {staged} "
        f"{CONTAINER_NAME}:/testbed/patch.diff && docker exec --user 0 {CONTAINER_NAME} chown 0:0 /testbed/patch.diff; "
        f"status=$?; rm -f {staged}; exit $status"
    ]


def test_container_download_goes_through_docker_cp(sdk: FakeSdk) -> None:
    async def scenario() -> bytes:
        original = sdk.machines.exec

        async def exec_and_stage(machine_id: str, command: str, *, timeout: float | None = None) -> Any:
            if "docker cp" in command:
                sdk.machines.files.store[command.split()[-1]] = b"result"
            return await original(machine_id, command, timeout=timeout)

        sdk.machines.exec = exec_and_stage
        return await _provider().read_file(_machine_handle(container=CONTAINER_NAME), "/testbed/out.json")

    assert _run(scenario()) == b"result"
    assert f"docker cp {CONTAINER_NAME}:/testbed/out.json /tmp/nemo-gym-transfer/" in sdk.machines.commands()[0]


def test_failed_docker_cp_is_a_setup_failure(sdk: FakeSdk) -> None:
    sdk.machines.exec_results = [FakeExecResult(stderr="no such container", exit_code=1)]
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().write_file(_machine_handle(container=CONTAINER_NAME), "/x", b""))
    assert raised.value.failure_kind == PROVIDER_SETUP_FAILED
    assert raised.value.details.exit_code == 1


def test_file_transfer_retries_transport_failures(sdk: FakeSdk, monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[BaseException] = []
    monkeypatch.setattr(boxd_provider, "observe_failure", observed.append)
    sdk.machines.files.upload_errors = [APIConnectionError("reset")]
    _run(_provider().write_file(_machine_handle(), "/tmp/a", b"x"))
    assert sdk.machines.files.uploads == ["/tmp/a", "/tmp/a"]
    assert [event.failure_kind for event in observed] == [PROVIDER_UNAVAILABLE]


def test_file_transfer_gives_up_after_the_retry_budget(sdk: FakeSdk) -> None:
    sdk.machines.files.upload_errors = [APIConnectionError("reset")] * 3
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider(operations={"retries": 1, "retry_delay_s": 0}).write_file(_machine_handle(), "/tmp/a", b"x"))
    assert raised.value.failure_kind == PROVIDER_UNAVAILABLE
    assert len(sdk.machines.files.uploads) == 2


def test_staged_upload_failure_is_a_failure_event(sdk: FakeSdk) -> None:
    sdk.machines.files.upload_errors = [AuthenticationError("expired")]
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().write_file(_machine_handle(container=CONTAINER_NAME), "/x", b""))
    assert raised.value.failure_kind == PROVIDER_CREDENTIALS_REJECTED


def test_upload_file_requires_a_local_file(sdk: FakeSdk, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _run(_provider().upload_file(_machine_handle(), tmp_path / "missing", "/tmp/x"))


# --------------------------------------------------------------------------
# Status, close, pause/resume, connect
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "status"),
    [
        ("running", SandboxStatus.RUNNING),
        ("starting", SandboxStatus.STARTING),
        ("suspended", SandboxStatus.PAUSED),
        ("hibernated", SandboxStatus.PAUSED),
        ("stopped", SandboxStatus.STOPPED),
        ("failed", SandboxStatus.ERROR),
        ("something-new", SandboxStatus.UNKNOWN),
    ],
)
def test_status_maps_machine_states(sdk: FakeSdk, state: str, status: SandboxStatus) -> None:
    sdk.machines.machines["m-1"] = FakeMachine(id="m-1", name="n", status=state)
    assert _run(_provider().status(_machine_handle())) is status


def test_status_of_a_deleted_machine_is_stopped_and_errors_are_unknown(sdk: FakeSdk) -> None:
    assert _run(_provider().status(_machine_handle())) is SandboxStatus.STOPPED
    sdk.machines.get_error = APIConnectionError("reset")
    assert _run(_provider().status(_machine_handle())) is SandboxStatus.UNKNOWN


@pytest.mark.parametrize(
    ("result", "status"),
    [
        (FakeExecResult(stdout="true\n"), SandboxStatus.RUNNING),
        (FakeExecResult(stdout="false\n"), SandboxStatus.STOPPED),
        (FakeExecResult(stderr="No such object", exit_code=1), SandboxStatus.STOPPED),
        (APIConnectionError("reset"), SandboxStatus.UNKNOWN),
    ],
)
def test_status_checks_the_container_in_container_mode(sdk: FakeSdk, result: Any, status: SandboxStatus) -> None:
    sdk.machines.machines["m-1"] = FakeMachine(id="m-1", name="n")
    sdk.machines.exec_results = [result]
    assert _run(_provider().status(_machine_handle(container=CONTAINER_NAME))) is status


def test_close_deletes_the_machine_once(sdk: FakeSdk) -> None:
    provider = _provider()
    handle = _run(provider.create(SandboxSpec()))
    _run(provider.close(handle))
    _run(provider.close(handle))
    assert sdk.machines.deleted == ["m-1"]
    assert handle.raw is None


def test_close_tolerates_a_machine_that_is_already_gone(sdk: FakeSdk) -> None:
    handle = _machine_handle()
    _run(_provider().close(handle))
    assert handle.raw is None


def test_close_retries_transport_failures_and_reports_others(sdk: FakeSdk) -> None:
    sdk.machines.machines["m-1"] = FakeMachine(id="m-1", name="n")
    sdk.machines.delete_errors = [APIConnectionError("reset")]
    _run(_provider().close(_machine_handle()))
    assert sdk.machines.deleted == ["m-1"]

    sdk.machines.machines["m-1"] = FakeMachine(id="m-1", name="n")
    sdk.machines.delete_errors = [PermissionDeniedError("not yours")]
    handle = _machine_handle()
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().close(handle))
    assert raised.value.failure_kind == PROVIDER_CREDENTIALS_REJECTED
    assert handle.raw is not None  # a later close can retry


def test_pause_hibernates_and_resume_wakes(sdk: FakeSdk) -> None:
    provider = _provider()
    handle = _run(provider.create(SandboxSpec()))
    _run(provider.pause(handle))
    assert _run(provider.status(handle)) is SandboxStatus.PAUSED
    _run(provider.resume(handle))
    assert sdk.machines.hibernated == ["m-1"] and sdk.machines.woken == ["m-1"]
    assert _run(provider.status(handle)) is SandboxStatus.RUNNING


def test_pause_and_resume_failures_are_events(sdk: FakeSdk) -> None:
    provider = _provider()
    with pytest.raises(NgFailureEvent) as raised:
        _run(provider.pause(_machine_handle()))
    assert raised.value.operation == "pause"
    with pytest.raises(NgFailureEvent) as raised:
        _run(provider.resume(_machine_handle()))
    assert raised.value.operation == "resume"


def test_serialize_and_connect_round_trip(sdk: FakeSdk) -> None:
    provider = _provider()
    handle = _run(provider.create(SandboxSpec(image="img", env={"K": "v"})))
    descriptor = _run(provider.serialize_handle(handle))
    assert descriptor == {
        "machine_id": "m-1",
        "name": handle.raw.name,
        "container": CONTAINER_NAME,
        "shell": "bash",
        "env": {"K": "v"},
    }
    other = _provider()
    rebuilt = _run(other.connect(descriptor))
    assert rebuilt.sandbox_id == "m-1"
    assert rebuilt.raw == handle.raw


def test_connect_to_a_missing_machine_is_an_event(sdk: FakeSdk) -> None:
    with pytest.raises(NgFailureEvent) as raised:
        _run(_provider().connect({"machine_id": "gone"}))
    assert raised.value.operation == "connect"
    assert isinstance(raised.value.__cause__, NotFoundError)


def test_aclose_closes_the_sdk_client(sdk: FakeSdk) -> None:
    provider = _provider()
    _run(provider.status(_machine_handle()))
    client = sdk.clients[0]
    _run(provider.aclose())
    assert client.closed
    _run(provider.aclose())  # idempotent


def test_async_sandbox_facade_end_to_end(sdk: FakeSdk, tmp_path: Path) -> None:
    """The public facade drives create, file delivery, exec and stop through the provider."""

    async def scenario() -> None:
        sandbox = AsyncSandbox({"boxd": {"attribution": {"enabled": False}}}, SandboxSpec(files={"/tmp/in.txt": "hi"}))
        await sandbox.start()
        result = await sandbox.exec("cat /tmp/in.txt", timeout_s=10)
        assert result.return_code == 0
        await sandbox.stop()

    _run(scenario())
    assert sdk.machines.files.store["/tmp/in.txt"] == b"hi"
    assert sdk.machines.deleted == ["m-1"]
    assert sdk.clients[0].closed


# --------------------------------------------------------------------------
# The real SDK surface the provider calls
# --------------------------------------------------------------------------


def test_installed_sdk_has_the_calls_the_provider_makes() -> None:
    boxd = pytest.importorskip("boxd")
    machines = boxd.AsyncBoxd(api_key="unit-test-key").machines
    create = inspect.signature(machines.create).parameters
    for name in (
        "from_snapshot",
        "org",
        "isolated",
        "labels",
        "vcpu",
        "memory",
        "auto_suspend_timeout",
        "delete_after",
    ):
        assert name in create
    assert "timeout" in inspect.signature(machines.exec).parameters
    assert "timeout" in inspect.signature(machines.wait_until_ready).parameters
    for method in ("get", "delete", "hibernate", "wake"):
        assert inspect.iscoroutinefunction(getattr(machines, method))
    assert inspect.iscoroutinefunction(machines.files.upload)
    assert inspect.iscoroutinefunction(machines.files.download)
    for name in (
        "BoxdError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "RateLimitError",
        "APIConnectionError",
    ):
        assert issubclass(getattr(boxd.errors, name), Exception)
