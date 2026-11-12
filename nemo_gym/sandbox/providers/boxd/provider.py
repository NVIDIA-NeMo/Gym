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

"""Sandbox provider backed by boxd machines and the ``boxd`` Python SDK.

Each sandbox is one boxd machine: a microVM running Ubuntu 24.04 with a Docker daemon.
``SandboxSpec.image`` selects how commands run inside it:

**Container mode** (``image`` set). The provider pulls the image with the machine's
Docker daemon and starts one keep-alive container from it, the way the ``docker``
provider does on a local host. Commands run through ``docker exec`` and files move
through ``docker cp``, so tasks see exactly the image's filesystem, user and tools.

**Machine mode** (``image`` omitted). Commands run directly on the machine as its
default user ``boxd``, which has passwordless sudo. ``create.from_snapshot`` starts
the machine from a boxd snapshot instead of the default image, for example one with
images pulled or dependencies installed ahead of time.

Machines are created ``isolated`` by default: they cannot reach other machines of the
organization and carry no boxd credentials. ``close()`` deletes the machine.

Failures follow the sandbox failure-event contract: allocation, readiness, lifecycle
and transport failures raise ``NgFailureEvent`` with ``boundary="sandbox"``; command
exits, including nonzero exits, return ``SandboxExecResult``.
"""

import asyncio
import logging
import math
import posixpath
import shlex
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from time import monotonic
from typing import Any, TypeVar

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
from nemo_gym.core.failure_transport import observe_failure
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.attribution import AttributionConfig
from nemo_gym.sandbox.failures import sandbox_failure
from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

BOXD_SDK_CONSTRAINT = "boxd>=0.3.1"

# Machine sizes as (vCPU, memory GiB); the SDK takes memory as "<GiB>G".
MACHINE_SIZES: tuple[tuple[int, int], ...] = ((1, 4), (2, 8), (4, 16), (8, 32))
# Every machine has a fixed disk of this size.
MACHINE_DISK_GIB = 100

# Exit status for a command that outlived its deadline, as coreutils ``timeout`` reports it.
TIMEOUT_RETURN_CODE = 124
# Return code for a sandbox runtime failure without a process exit code.
SANDBOX_RUNTIME_RETURN_CODE = 125

CONTAINER_NAME = "nemo-gym-sandbox"
# The container's main process; ``docker exec`` runs every command beside it.
KEEPALIVE_SHELL = "/bin/sh"
KEEPALIVE_CMD = "while :; do sleep 2147483647; done"
# Files on their way into or out of a container are staged here on the machine.
STAGING_DIR = "/tmp/nemo-gym-transfer"
# The machine's own user. Commands for any other user go through sudo.
MACHINE_USER = "boxd"
# Commands whose deadline is longer than this get a keepalive: a NUL byte on stderr every
# KEEPALIVE_INTERVAL_S, so idle-connection timeouts between client and machine never end the stream
# of a command that prints nothing for a long time. The provider removes those bytes from stderr.
KEEPALIVE_AFTER_S = 900.0
KEEPALIVE_INTERVAL_S = 300

_STATUS_BY_MACHINE_STATE = {
    "pending": SandboxStatus.STARTING,
    "starting": SandboxStatus.STARTING,
    "migrating": SandboxStatus.STARTING,
    "running": SandboxStatus.RUNNING,
    "suspended": SandboxStatus.PAUSED,
    "hibernated": SandboxStatus.PAUSED,
    "stopped": SandboxStatus.STOPPED,
    "destroyed": SandboxStatus.STOPPED,
    "failed": SandboxStatus.ERROR,
}


def _require_boxd_sdk() -> Any:
    """Import the optional ``boxd`` SDK, or explain how to install it.

    Every SDK access goes through this one function, so unit tests can replace the SDK without network access.
    """
    try:
        import boxd
    except ImportError as exc:
        raise ImportError(
            "The boxd sandbox provider requires the boxd SDK. Install it with `pip install 'nemo-gym[boxd]'` "
            f"or `pip install '{BOXD_SDK_CONSTRAINT}'`."
        ) from exc
    return boxd


def _is_sdk_error(error: BaseException, name: str) -> bool:
    """Whether ``error`` is an instance of the SDK's ``boxd.errors.<name>``."""
    cls = getattr(_require_boxd_sdk().errors, name, None)
    return isinstance(cls, type) and isinstance(error, cls)


def _is_unclassified_sdk_error(error: BaseException) -> bool:
    """A plain ``BoxdError`` without a status code: the call ended without the server naming a reason."""
    return (
        type(error) is getattr(_require_boxd_sdk().errors, "BoxdError", None) and getattr(error, "code", None) is None
    )


def _config_from_mapping(cls: type[T], value: Any) -> T:
    """Build a config dataclass from a mapping, rejecting unknown keys."""
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{cls.__name__} expects a mapping, got {type(value).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} keys: {', '.join(sorted(unknown))}. Expected: {', '.join(sorted(allowed))}"
        )
    return cls(**dict(value))


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _check_positive(name: str, value: Any, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if not _is_number(value) or value <= 0:
        raise ValueError(f"{name} must be > 0")


def _check_nonnegative(name: str, value: Any) -> None:
    if not _is_number(value) or value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class BoxdConnectionConfig:
    """How to reach boxd.

    Fields left ``None`` fall back to the SDK's environment variables: ``BOXD_API_KEY``
    (or ``BOXD_TOKEN``) and ``BOXD_REGION``. ``org`` falls back to ``BOXD_ORG`` through
    the shipped YAML config.
    """

    api_key: str | None = None
    org: str | None = None
    region: str | None = None


@dataclass(frozen=True)
class BoxdCreateConfig:
    """Machine creation settings."""

    # Start every machine from this boxd snapshot instead of the default image.
    from_snapshot: str | None = None
    # Machine size when SandboxSpec.resources names none; None uses the organization's default.
    vcpu: int | None = None
    memory: str | None = None
    # Isolated machines cannot reach other machines of the organization and hold no boxd credentials.
    isolated: bool = True
    # Seconds without network traffic before the machine goes to standby; 0 keeps it running.
    auto_suspend_timeout: int = 0
    # Prefix of every machine name; the rest is random.
    name_prefix: str = "nemo-gym-"
    # Labels on every machine, next to SandboxSpec.metadata and job attribution.
    labels: dict[str, str] = field(default_factory=dict)
    # Deadline for the machine to boot and answer a first command, when SandboxSpec.ready_timeout_s is unset.
    ready_timeout_s: float = 180.0
    # Deadline for ``docker pull`` of SandboxSpec.image in container mode.
    image_pull_timeout_s: float = 1200.0
    # Pull the image even when the machine already has it (a snapshot may carry it).
    always_pull: bool = False

    def __post_init__(self) -> None:
        if self.from_snapshot is not None and (not isinstance(self.from_snapshot, str) or not self.from_snapshot):
            raise ValueError("create.from_snapshot must be a non-empty string when provided")
        if self.vcpu is not None or self.memory is not None:
            size = (self.vcpu, _memory_gib(self.memory) if self.memory is not None else None)
            if not any(size[0] in (None, vcpu) and size[1] in (None, gib) for vcpu, gib in MACHINE_SIZES):
                raise ValueError(f"create.vcpu/create.memory must name a machine size: {_sizes_text()}")
        _check_nonnegative("create.auto_suspend_timeout", self.auto_suspend_timeout)
        if not isinstance(self.name_prefix, str):
            raise ValueError("create.name_prefix must be a string")
        if not isinstance(self.labels, Mapping):
            raise TypeError("create.labels must be a mapping")
        _check_positive("create.ready_timeout_s", self.ready_timeout_s, allow_none=False)
        _check_positive("create.image_pull_timeout_s", self.image_pull_timeout_s, allow_none=False)


@dataclass(frozen=True)
class BoxdExecConfig:
    """Command execution settings."""

    # Applied when the provider receives timeout_s=None. The AsyncSandbox facade passes its own default.
    default_timeout_s: float | None = 180.0
    # Container mode shell; None uses bash when the image has it, else sh.
    shell: str | None = None

    def __post_init__(self) -> None:
        _check_positive("exec.default_timeout_s", self.default_timeout_s)


@dataclass(frozen=True)
class BoxdOperationConfig:
    """Retries for control-plane and file calls that failed in transport.

    Machine creation and commands are never retried: either may already have taken effect.
    """

    retries: int = 2
    retry_delay_s: float = 1.0
    retry_max_delay_s: float = 10.0

    def __post_init__(self) -> None:
        if isinstance(self.retries, bool) or not isinstance(self.retries, int) or self.retries < 0:
            raise ValueError("operations.retries must be >= 0")
        _check_nonnegative("operations.retry_delay_s", self.retry_delay_s)
        _check_nonnegative("operations.retry_max_delay_s", self.retry_max_delay_s)


@dataclass
class BoxdMachine:
    """Provider-owned state behind ``SandboxHandle.raw``."""

    machine_id: str
    name: str
    # The container commands run in; None in machine mode.
    container: str | None = None
    # The shell commands run under.
    shell: str = "bash"
    # SandboxSpec.env, applied to every command in machine mode (a container carries its own).
    env: dict[str, str] = field(default_factory=dict)


def _memory_gib(memory: str | int) -> int:
    text = str(memory).strip().upper().removesuffix("B").removesuffix("I")
    if text.endswith("G"):
        text = text[:-1]
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"create.memory must look like '8G', got {memory!r}") from exc


def _sizes_text() -> str:
    return ", ".join(f"{vcpu} vCPU / {gib}G" for vcpu, gib in MACHINE_SIZES)


def machine_size(resources: SandboxResources) -> tuple[int, int] | None:
    """The smallest machine size that holds ``resources``, or ``None`` when it requests no CPU or memory.

    Raises ``ValueError`` for a request no machine size holds: a GPU, more disk than a machine
    has, or more CPU or memory than the largest size.
    """
    if resources.gpu or resources.gpu_type:
        raise ValueError("boxd machines have no GPUs; remove resources.gpu and resources.gpu_type")
    disk = resources.disk_gib
    if disk is not None and disk > MACHINE_DISK_GIB:
        raise ValueError(f"boxd machines have a {MACHINE_DISK_GIB} GiB disk; resources.disk_gib={disk} does not fit")
    cpu = resources.cpu
    memory_mib = resources.memory_mib
    if cpu is None and memory_mib is None:
        return None
    for vcpu, gib in MACHINE_SIZES:
        if (cpu is None or cpu <= vcpu) and (memory_mib is None or memory_mib <= gib * 1024):
            return vcpu, gib
    raise ValueError(f"No boxd machine size holds cpu={cpu}, memory_mib={memory_mib}. Sizes: {_sizes_text()}")


def _normalize_image(image: str) -> str:
    prefix = "docker://"
    return image[len(prefix) :] if image.startswith(prefix) else image


def _user_is_machine_default(user: str | int | None) -> bool:
    return user is None or user == MACHINE_USER


def _docker_user(user: str | int) -> str:
    """``docker exec --user`` value: root becomes 0, anything else is passed as written."""
    return "0" if user in ("root", 0) else str(user)


def _env_assignments(env: Mapping[str, str]) -> list[str]:
    for key in env:
        if not key or "=" in key:
            raise ValueError(f"Invalid environment variable name: {key!r}")
    return [f"{key}={value}" for key, value in env.items()]


def _failure_kind(cause: BaseException) -> str:
    """The registered failure kind for a boxd SDK error, chosen by its type and status code."""
    if _is_sdk_error(cause, "AuthenticationError") or _is_sdk_error(cause, "PermissionDeniedError"):
        return PROVIDER_CREDENTIALS_REJECTED
    if _is_sdk_error(cause, "RateLimitError"):
        return PROVIDER_QUOTA_EXHAUSTED
    if _is_sdk_error(cause, "APIConnectionError") or getattr(cause, "code", None) == "unavailable":
        return PROVIDER_UNAVAILABLE
    if isinstance(cause, TimeoutError):
        return PROVIDER_IO_TIMEOUT
    return PROVIDER_REQUEST_FAILED


class BoxdProvider:
    """Provider backed by boxd machines."""

    name = "boxd"

    def __init__(
        self,
        *,
        connection: BoxdConnectionConfig | Mapping[str, Any] | None = None,
        create: BoxdCreateConfig | Mapping[str, Any] | None = None,
        exec: BoxdExecConfig | Mapping[str, Any] | None = None,
        operations: BoxdOperationConfig | Mapping[str, Any] | None = None,
        attribution: AttributionConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = _config_from_mapping(BoxdConnectionConfig, connection)
        self._create = _config_from_mapping(BoxdCreateConfig, create)
        self._exec = _config_from_mapping(BoxdExecConfig, exec)
        self._operations = _config_from_mapping(BoxdOperationConfig, operations)
        self._attribution = _config_from_mapping(AttributionConfig, attribution)
        self._client: Any = None

    # ---------------------------------------------------------------- helpers

    def _sdk_client(self) -> Any:
        if self._client is None:
            boxd = _require_boxd_sdk()
            kwargs = {
                key: value
                for key, value in (("api_key", self._connection.api_key), ("region", self._connection.region))
                if value
            }
            self._client = boxd.AsyncBoxd(**kwargs)
        return self._client

    def _failure(
        self,
        *,
        operation: str,
        cause: BaseException | None = None,
        sandbox_id: str | None = None,
        failure_kind: str | None = None,
        reason: str | None = None,
        exit_code: int | None = None,
    ) -> NgFailureEvent:
        """Classify a boxd SDK error by its type, then build the sandbox failure event."""
        if failure_kind is None:
            failure_kind = _failure_kind(cause) if cause is not None else PROVIDER_REQUEST_FAILED
        return sandbox_failure(
            operation=operation,
            failure_kind=failure_kind,
            sandbox_id=sandbox_id,
            cause=cause,
            reason=reason,
            provider=self.name,
            exit_code=exit_code,
        )

    @staticmethod
    def _is_transient(error: BaseException) -> bool:
        return _is_sdk_error(error, "APIConnectionError") or getattr(error, "code", None) == "unavailable"

    async def _with_retries(self, factory: Callable[[], Awaitable[T]], *, operation: str, sandbox_id: str | None) -> T:
        """Retry transport failures of a call that is safe to repeat, recording each failed attempt."""
        delay = self._operations.retry_delay_s
        for attempt in range(self._operations.retries + 1):
            try:
                return await factory()
            except Exception as exc:
                if attempt == self._operations.retries or not self._is_transient(exc):
                    raise
                observe_failure(self._failure(operation=operation, cause=exc, sandbox_id=sandbox_id))
                LOGGER.debug("boxd %s failed (attempt %d): %s", operation, attempt + 1, exc)
                await asyncio.sleep(min(delay, self._operations.retry_max_delay_s))
                delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _machine(handle: SandboxHandle) -> BoxdMachine:
        machine = handle.raw
        if not isinstance(machine, BoxdMachine):
            raise RuntimeError(f"Sandbox handle {handle.sandbox_id} carries no boxd machine")
        return machine

    async def _run(self, machine_id: str, line: str, *, timeout_s: float | None) -> Any:
        """Run one shell line on the machine through the SDK."""
        return await self._sdk_client().machines.exec(machine_id, line, timeout=timeout_s)

    async def _run_checked(self, handle: SandboxHandle, line: str, *, operation: str, timeout_s: float) -> str:
        """Run a provider-internal command; a nonzero exit is a setup failure."""
        machine = self._machine(handle)
        try:
            result = await self._run(machine.machine_id, line, timeout_s=timeout_s)
        except Exception as exc:
            raise self._failure(operation=operation, cause=exc, sandbox_id=handle.sandbox_id) from exc
        if result.exit_code != 0:
            raise self._failure(
                operation=operation,
                sandbox_id=handle.sandbox_id,
                failure_kind=PROVIDER_SETUP_FAILED,
                exit_code=result.exit_code,
                reason=f"boxd {operation} exited {result.exit_code}: {(result.stderr or result.stdout or '').strip()[-1000:]}",
            )
        return result.stdout or ""

    def _labels(self, spec: SandboxSpec) -> dict[str, str]:
        labels = {**self._attribution.metadata(), **self._create.labels, **spec.metadata}
        return {str(key): str(value) for key, value in labels.items()}

    def _size_kwargs(self, spec: SandboxSpec) -> dict[str, Any]:
        size = machine_size(spec.resources)
        if size is not None:
            return {"vcpu": size[0], "memory": f"{size[1]}G"}
        kwargs: dict[str, Any] = {}
        if self._create.vcpu is not None:
            kwargs["vcpu"] = self._create.vcpu
        if self._create.memory is not None:
            kwargs["memory"] = f"{_memory_gib(self._create.memory)}G"
        return kwargs

    # ------------------------------------------------------------- lifecycle

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a machine, wait until it runs commands and, in container mode, start the image."""
        options = dict(spec.provider_options or {})
        unknown = set(options) - {"from_snapshot", "resource_requests"}
        if unknown:
            raise ValueError(
                f"Unknown boxd provider option(s): {', '.join(sorted(unknown))}. "
                "Supported: from_snapshot, resource_requests"
            )
        # Scheduling requests below the limits mean nothing for a whole machine: its size comes from
        # SandboxSpec.resources. Accepted so configs written for request-and-limit providers run unchanged.
        options.pop("resource_requests", None)
        from_snapshot = options.get("from_snapshot", self._create.from_snapshot)
        if spec.entrypoint and not spec.image:
            raise ValueError("SandboxSpec.entrypoint needs SandboxSpec.image: it replaces the image's entrypoint")
        if spec.ports:
            raise ValueError("The boxd provider does not publish SandboxSpec.ports")
        ready_timeout_s = spec.ready_timeout_s if spec.ready_timeout_s is not None else self._create.ready_timeout_s
        _check_positive("SandboxSpec.ready_timeout_s", ready_timeout_s, allow_none=False)
        if spec.ttl_s is not None:
            _check_positive("SandboxSpec.ttl_s", spec.ttl_s)

        name = f"{self._create.name_prefix}{uuid.uuid4().hex[:12]}"
        kwargs: dict[str, Any] = {
            "isolated": self._create.isolated,
            "labels": self._labels(spec),
            "auto_suspend_timeout": self._create.auto_suspend_timeout,
        }
        if self._connection.org:
            kwargs["org"] = self._connection.org
        if from_snapshot:
            kwargs["from_snapshot"] = from_snapshot
            if machine_size(spec.resources) is not None:
                LOGGER.warning(
                    "A machine from snapshot %r keeps the snapshot's size; resources are ignored", from_snapshot
                )
        else:
            kwargs.update(self._size_kwargs(spec))
        if spec.ttl_s is not None:
            # boxd deletes the machine when its lifetime ends, even if close() never runs.
            kwargs["delete_after"] = max(1, math.ceil(spec.ttl_s))

        try:
            # Not retried: a create whose answer was lost may still have made a machine.
            created = await self._sdk_client().machines.create(name, **kwargs)
        except Exception as exc:
            failure = self._failure(operation="create", cause=exc, reason=f"boxd could not create machine {name!r}")
            if failure.failure_kind != PROVIDER_CREDENTIALS_REJECTED:
                await self._delete_orphan(name)
            raise failure from exc

        machine = BoxdMachine(machine_id=str(created.id), name=name, env={str(k): str(v) for k, v in spec.env.items()})
        handle = SandboxHandle(sandbox_id=machine.machine_id, provider_name=self.name, raw=machine)
        try:
            await self._wait_ready(handle, timeout_s=float(ready_timeout_s))
            if spec.image:
                await self._start_container(handle, spec)
            elif spec.workdir:
                await self._run_checked(
                    handle,
                    self._as_user(f"mkdir -p {shlex.quote(spec.workdir)}", "root")
                    + f" && sudo -n chown {MACHINE_USER}: {shlex.quote(spec.workdir)}",
                    operation="create workdir",
                    timeout_s=60.0,
                )
        except BaseException:
            await self._delete_quietly(handle)
            raise
        return handle

    async def _wait_ready(self, handle: SandboxHandle, *, timeout_s: float) -> None:
        try:
            await self._sdk_client().machines.wait_until_ready(handle.sandbox_id, timeout=timeout_s)
        except TimeoutError as exc:
            raise self._failure(
                operation="create",
                cause=exc,
                sandbox_id=handle.sandbox_id,
                failure_kind=PROVIDER_CREATE_TIMEOUT,
                reason=f"boxd machine {handle.sandbox_id} did not run a command within {timeout_s:g}s",
            ) from exc
        except Exception as exc:
            raise self._failure(operation="create", cause=exc, sandbox_id=handle.sandbox_id) from exc

    async def _start_container(self, handle: SandboxHandle, spec: SandboxSpec) -> None:
        """Pull ``spec.image`` and start the keep-alive container commands run in."""
        machine = self._machine(handle)
        image = _normalize_image(str(spec.image))
        quoted = shlex.quote(image)
        pull = f"docker pull -q {quoted} >/dev/null"
        if not self._create.always_pull:
            pull = f"docker image inspect {quoted} >/dev/null 2>&1 || {pull}"
        await self._run_checked(handle, pull, operation="image pull", timeout_s=self._create.image_pull_timeout_s)

        argv = ["docker", "run", "-d", "--init", "--name", CONTAINER_NAME, "--network", "host"]
        if spec.workdir:
            argv += ["-w", spec.workdir]
        for assignment in _env_assignments(spec.env):
            argv += ["--env", assignment]
        if spec.entrypoint:
            argv += ["--entrypoint", spec.entrypoint[0], image, *spec.entrypoint[1:]]
        else:
            argv += ["--entrypoint", KEEPALIVE_SHELL, image, "-c", KEEPALIVE_CMD]
        await self._run_checked(handle, shlex.join(argv), operation="container start", timeout_s=120.0)
        machine.container = CONTAINER_NAME

        if self._exec.shell:
            machine.shell = self._exec.shell
        else:
            probe = f"docker exec {CONTAINER_NAME} sh -c 'command -v bash' >/dev/null 2>&1 && echo bash || echo sh"
            machine.shell = (
                await self._run_checked(handle, probe, operation="container probe", timeout_s=60.0)
            ).strip()
        if machine.shell not in ("bash", "sh") and not self._exec.shell:
            raise self._failure(
                operation="container probe",
                sandbox_id=handle.sandbox_id,
                failure_kind=PROVIDER_NOT_READY,
                reason=f"container from {image!r} did not answer the readiness probe: {machine.shell[-200:]!r}",
            )

    async def _delete_orphan(self, name: str) -> None:
        """Delete the machine a failed create call may still have made; its name is unique to that attempt."""
        try:
            machine = await self._sdk_client().machines.get(name)
        except Exception as exc:  # noqa: BLE001 - nothing to clean up, or nothing we can find
            if not _is_sdk_error(exc, "NotFoundError"):
                LOGGER.warning("Could not look up boxd machine %r after a failed create: %s", name, exc)
            return
        await self._delete_quietly(SandboxHandle(sandbox_id=str(machine.id), provider_name=self.name, raw=None))

    async def _delete_quietly(self, handle: SandboxHandle) -> None:
        try:
            await self._delete(handle)
        except Exception as exc:  # noqa: BLE001 - the original failure is the one to report
            LOGGER.warning("Could not delete boxd machine %s after a failed create: %s", handle.sandbox_id, exc)

    async def _delete(self, handle: SandboxHandle) -> None:
        try:
            await self._with_retries(
                lambda: self._sdk_client().machines.delete(handle.sandbox_id),
                operation="delete",
                sandbox_id=handle.sandbox_id,
            )
        except Exception as exc:
            if _is_sdk_error(exc, "NotFoundError"):
                LOGGER.debug("boxd machine %s already gone on close", handle.sandbox_id)
                return
            raise

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """The machine's state; in container mode a running machine also needs a running container."""
        try:
            machine = await self._sdk_client().machines.get(handle.sandbox_id)
        except Exception as exc:  # noqa: BLE001 - status must not raise for transient issues
            if _is_sdk_error(exc, "NotFoundError"):
                return SandboxStatus.STOPPED
            return SandboxStatus.UNKNOWN
        status = _STATUS_BY_MACHINE_STATE.get(str(getattr(machine, "status", "")).lower(), SandboxStatus.UNKNOWN)
        container = self._machine(handle).container
        if status is not SandboxStatus.RUNNING or container is None:
            return status
        try:
            result = await self._run(
                handle.sandbox_id, f"docker inspect -f '{{{{.State.Running}}}}' {container}", timeout_s=30.0
            )
        except Exception:  # noqa: BLE001
            return SandboxStatus.UNKNOWN
        return SandboxStatus.RUNNING if (result.stdout or "").strip() == "true" else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        """Delete the machine. Safe to call again, and for a machine that is already gone."""
        if handle.raw is None:
            return
        try:
            await self._delete(handle)
        except Exception as exc:
            raise self._failure(operation="close", cause=exc, sandbox_id=handle.sandbox_id) from exc
        handle.raw = None

    async def aclose(self) -> None:
        client, self._client = self._client, None
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    async def pause(self, handle: SandboxHandle) -> None:
        """Hibernate the machine: memory and disk are kept, so running processes resume as they were."""
        self._machine(handle)
        try:
            await self._sdk_client().machines.hibernate(handle.sandbox_id)
        except Exception as exc:
            raise self._failure(operation="pause", cause=exc, sandbox_id=handle.sandbox_id) from exc

    async def resume(self, handle: SandboxHandle) -> None:
        """Wake a hibernated machine and wait until it runs commands again."""
        self._machine(handle)
        try:
            await self._sdk_client().machines.wake(handle.sandbox_id)
        except Exception as exc:
            raise self._failure(operation="resume", cause=exc, sandbox_id=handle.sandbox_id) from exc
        await self._wait_ready(handle, timeout_s=self._create.ready_timeout_s)

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        """A descriptor ``connect`` rebuilds the handle from, in this or another process."""
        machine = self._machine(handle)
        return {
            "machine_id": machine.machine_id,
            "name": machine.name,
            "container": machine.container,
            "shell": machine.shell,
            "env": dict(machine.env),
        }

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        machine = BoxdMachine(
            machine_id=str(descriptor["machine_id"]),
            name=str(descriptor.get("name") or descriptor["machine_id"]),
            container=descriptor.get("container"),
            shell=str(descriptor.get("shell") or "bash"),
            env={str(k): str(v) for k, v in (descriptor.get("env") or {}).items()},
        )
        try:
            await self._with_retries(
                lambda: self._sdk_client().machines.get(machine.machine_id),
                operation="connect",
                sandbox_id=machine.machine_id,
            )
        except Exception as exc:
            raise self._failure(operation="connect", cause=exc, sandbox_id=machine.machine_id) from exc
        return SandboxHandle(sandbox_id=machine.machine_id, provider_name=self.name, raw=machine)

    # -------------------------------------------------------------- commands

    @staticmethod
    def _as_user(line: str, user: str | int | None) -> str:
        """Run ``line`` on the machine as ``user`` through sudo; the machine's own user needs none."""
        if _user_is_machine_default(user):
            return line
        if user in ("root", 0):
            return f"sudo -n {line}"
        return f"sudo -n -u {shlex.quote(str(user) if not isinstance(user, int) else f'#{user}')} -H {line}"

    def _command_line(
        self,
        machine: BoxdMachine,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        user: str | int | None,
    ) -> str:
        """The shell line that runs ``command`` with ``cwd``, ``env`` and ``user`` on the machine."""
        if machine.container is not None:
            argv = ["docker", "exec"]
            if cwd is not None:
                argv += ["-w", cwd]
            for assignment in _env_assignments(env or {}):
                argv += ["--env", assignment]
            if user is not None:
                argv += ["--user", _docker_user(user)]
            return shlex.join([*argv, machine.container, machine.shell, "-c", command])
        script = f"cd {shlex.quote(cwd)} && {command}" if cwd is not None else command
        merged = {**machine.env, **(env or {})}
        argv = ["env", *_env_assignments(merged)] if merged else []
        return self._as_user(shlex.join([*argv, "bash", "-c", script]), user)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        machine = self._machine(handle)
        effective_timeout = timeout_s if timeout_s is not None else self._exec.default_timeout_s
        if effective_timeout is not None and (not _is_number(effective_timeout) or effective_timeout <= 0):
            raise ValueError("boxd command timeout_s must be > 0")
        line = self._command_line(machine, command, cwd=cwd, env=env, user=user)
        keepalive = effective_timeout is None or effective_timeout > KEEPALIVE_AFTER_S
        if keepalive:
            # The keepalive loop shares the command's process group, so a deadline stops both.
            line = (
                f"( while sleep {KEEPALIVE_INTERVAL_S} >/dev/null 2>&1; do printf '\\0' >&2; done ) >/dev/null & "
                "keepalive=$!; "
                f"{line}; status=$?; kill $keepalive 2>/dev/null; exit $status"
            )
        started = monotonic()
        try:
            result = await self._run(
                machine.machine_id, line, timeout_s=float(effective_timeout) if effective_timeout else None
            )
        except Exception as exc:
            elapsed = monotonic() - started
            # The SDK reports its own deadline as a BoxdError with no status code, after the server stopped
            # the command's process group.
            if (
                effective_timeout is not None
                and elapsed >= float(effective_timeout)
                and _is_unclassified_sdk_error(exc)
            ):
                return SandboxExecResult(
                    stdout=None,
                    stderr=f"Command timed out after {float(effective_timeout):g}s",
                    return_code=TIMEOUT_RETURN_CODE,
                    error_type="timeout",
                )
            # Refused credentials and quota mean the command never ran. Any other failure may come after it
            # started, so its outcome is unknown and it is never replayed.
            kind = _failure_kind(exc)
            if kind not in (PROVIDER_CREDENTIALS_REJECTED, PROVIDER_QUOTA_EXHAUSTED):
                kind = PROVIDER_COMMAND_OUTCOME_UNKNOWN
            raise self._failure(operation="exec", cause=exc, sandbox_id=handle.sandbox_id, failure_kind=kind) from exc
        if result.exit_code < 0:
            raise self._failure(
                operation="exec",
                sandbox_id=handle.sandbox_id,
                failure_kind=PROVIDER_COMMAND_OUTCOME_UNKNOWN,
                reason="boxd command stream ended without an exit status",
            )
        stderr = result.stderr
        if keepalive and stderr:
            stderr = stderr.replace("\0", "")
        return SandboxExecResult(
            stdout=result.stdout,
            stderr=stderr,
            return_code=int(result.exit_code),
        )

    # ----------------------------------------------------------------- files

    async def _upload_bytes(self, handle: SandboxHandle, target_path: str, data: bytes) -> None:
        machine = self._machine(handle)
        files = self._sdk_client().machines.files
        if machine.container is None:
            try:
                await self._with_retries(
                    lambda: files.upload(machine.machine_id, target_path, data),
                    operation="upload",
                    sandbox_id=handle.sandbox_id,
                )
                return
            except Exception as exc:
                if self._is_transient(exc):
                    raise self._failure(operation="upload", cause=exc, sandbox_id=handle.sandbox_id) from exc
                # The machine's user cannot write there: stage the file and move it with sudo.
        staged = f"{STAGING_DIR}/{uuid.uuid4().hex}"
        try:
            await self._with_retries(
                lambda: files.upload(machine.machine_id, staged, data),
                operation="upload",
                sandbox_id=handle.sandbox_id,
            )
        except Exception as exc:
            raise self._failure(operation="upload", cause=exc, sandbox_id=handle.sandbox_id) from exc
        parent = posixpath.dirname(target_path) or "/"
        if machine.container is None:
            line = (
                f"sudo -n mkdir -p {shlex.quote(parent)} && sudo -n mv {shlex.quote(staged)} {shlex.quote(target_path)}"
                f" || {{ status=$?; rm -f {shlex.quote(staged)}; exit $status; }}"
            )
        else:
            container = shlex.quote(machine.container)
            line = (
                f"docker exec --user 0 {container} mkdir -p {shlex.quote(parent)} && "
                f"docker cp {shlex.quote(staged)} {container}:{shlex.quote(target_path)} && "
                f"docker exec --user 0 {container} chown 0:0 {shlex.quote(target_path)}; "
                f"status=$?; rm -f {shlex.quote(staged)}; exit $status"
            )
        await self._run_checked(handle, line, operation="upload", timeout_s=300.0)

    async def _download_bytes(self, handle: SandboxHandle, source_path: str) -> bytes:
        machine = self._machine(handle)
        files = self._sdk_client().machines.files
        if machine.container is None:
            try:
                return await self._with_retries(
                    lambda: files.download(machine.machine_id, source_path),
                    operation="download",
                    sandbox_id=handle.sandbox_id,
                )
            except Exception as exc:
                if self._is_transient(exc) or _is_sdk_error(exc, "NotFoundError"):
                    raise self._failure(operation="download", cause=exc, sandbox_id=handle.sandbox_id) from exc
                # Not readable by the machine's user: copy it out with sudo first.
        staged = f"{STAGING_DIR}/{uuid.uuid4().hex}"
        if machine.container is None:
            line = (
                f"mkdir -p {STAGING_DIR} && sudo -n cp {shlex.quote(source_path)} {shlex.quote(staged)} && "
                f"sudo -n chown {MACHINE_USER}: {shlex.quote(staged)}"
            )
        else:
            line = f"mkdir -p {STAGING_DIR} && docker cp {shlex.quote(machine.container)}:{shlex.quote(source_path)} {shlex.quote(staged)}"
        try:
            await self._run_checked(handle, line, operation="download", timeout_s=300.0)
            try:
                return await self._with_retries(
                    lambda: files.download(machine.machine_id, staged),
                    operation="download",
                    sandbox_id=handle.sandbox_id,
                )
            except Exception as exc:
                raise self._failure(operation="download", cause=exc, sandbox_id=handle.sandbox_id) from exc
        finally:
            try:
                await self._run(machine.machine_id, f"rm -f {shlex.quote(staged)}", timeout_s=60.0)
            except Exception as exc:  # noqa: BLE001 - a leftover staging file does not fail the download
                LOGGER.debug("Could not remove staged download %s: %s", staged, exc)

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        await self._upload_bytes(handle, target_path, data.encode() if isinstance(data, str) else data)

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        return await self._download_bytes(handle, source_path)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        await self._upload_bytes(handle, target_path, source.read_bytes())

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        target = Path(target_path)
        data = await self._download_bytes(handle, source_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
