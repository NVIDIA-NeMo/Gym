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

"""Modal-backed implementation of the NeMo Gym sandbox provider protocol."""

import asyncio
import inspect
import math
import os
import shlex
import tempfile
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxDownloadLimitExceeded,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)


class ModalCreateError(SandboxCreateError):
    """Raised when Modal cannot create a sandbox."""


class ModalCreateTimeoutError(ModalCreateError):
    """Raised when Modal sandbox creation exceeds the configured deadline."""


class ModalCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a newly created Modal sandbox cannot run the readiness probe."""


class ModalCleanupError(RuntimeError):
    """Raised when a Modal sandbox cannot be terminated after bounded retries."""


DEFAULT_EXEC_STDOUT_LIMIT_BYTES = 16 * 1024 * 1024
DEFAULT_EXEC_STDERR_LIMIT_BYTES = 16 * 1024 * 1024
DEFAULT_EXEC_COMBINED_OUTPUT_LIMIT_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class ModalConnectionConfig:
    """Modal app lookup settings.

    Authentication stays in Modal's normal profile or ``MODAL_TOKEN_*`` environment
    variables. Token values are intentionally not accepted here.
    """

    app_name: str = "nemo-gym-sandbox"
    environment_name: str | None = None
    create_if_missing: bool = True

    def __post_init__(self) -> None:
        if not self.app_name:
            raise ValueError("Modal connection app_name must be non-empty")
        if self.environment_name is not None and not self.environment_name:
            raise ValueError("Modal connection environment_name must be non-empty when set")
        if not isinstance(self.create_if_missing, bool):
            raise TypeError("Modal connection create_if_missing must be a bool")


@dataclass(frozen=True)
class ModalCreateConfig:
    """Provider defaults for sandbox creation."""

    default_timeout_s: int | float = 21600
    ready_timeout_s: int | float = 1200
    cleanup_timeout_s: int | float = 30
    idle_timeout_s: int | float | None = None
    exec_stdout_limit_bytes: int = DEFAULT_EXEC_STDOUT_LIMIT_BYTES
    exec_stderr_limit_bytes: int = DEFAULT_EXEC_STDERR_LIMIT_BYTES
    exec_combined_output_limit_bytes: int = DEFAULT_EXEC_COMBINED_OUTPUT_LIMIT_BYTES

    def __post_init__(self) -> None:
        _seconds(self.default_timeout_s, field_name="default_timeout_s")
        _seconds(self.ready_timeout_s, field_name="ready_timeout_s")
        _seconds(self.cleanup_timeout_s, field_name="cleanup_timeout_s")
        if self.idle_timeout_s is not None:
            _seconds(self.idle_timeout_s, field_name="idle_timeout_s")
        _byte_limit(self.exec_stdout_limit_bytes, field_name="exec_stdout_limit_bytes")
        _byte_limit(self.exec_stderr_limit_bytes, field_name="exec_stderr_limit_bytes")
        _byte_limit(self.exec_combined_output_limit_bytes, field_name="exec_combined_output_limit_bytes")


@dataclass(frozen=True)
class ModalProbeConfig:
    """Readiness probe run immediately after creation."""

    command: str | None = "printf modal-sandbox-ready"
    expected_stdout: str | None = "modal-sandbox-ready"
    timeout_s: int | float = 60

    def __post_init__(self) -> None:
        if self.command is not None and not self.command:
            raise ValueError("Modal probe command must be non-empty when set")
        _seconds(self.timeout_s, field_name="probe.timeout_s")


@dataclass(frozen=True)
class ModalImageSetupStep:
    """One cached Modal image-build step."""

    run: tuple[str, ...]
    user: str | int | None = None
    shell: str = "/bin/sh"
    env: dict[str, str] | None = None
    secret_names: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModalImageSetupStep":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"Unknown Modal image setup field(s): {', '.join(sorted(unknown))}. "
                f"Supported: {', '.join(sorted(allowed))}"
            )

        commands = value.get("run")
        if isinstance(commands, str):
            commands = (commands,)
        else:
            commands = _string_sequence(commands, field_name="image_setup_steps.run")
        if not commands:
            raise ValueError("Modal image setup field 'run' must not be empty")

        user = value.get("user")
        if user is not None and not isinstance(user, (str, int)):
            raise TypeError("Modal image setup field 'user' must be a username or uid")
        if user == "":
            raise ValueError("Modal image setup field 'user' must be non-empty")
        if isinstance(user, int) and user != 0:
            raise ValueError("Modal image setup supports uid 0 or a username; non-root numeric uids are ambiguous")

        shell = value.get("shell", "/bin/sh")
        if not isinstance(shell, str) or not shell or not PurePosixPath(shell).is_absolute():
            raise ValueError("Modal image setup field 'shell' must be an absolute executable path")

        env_value = value.get("env")
        env: dict[str, str] | None = None
        if env_value is not None:
            if not isinstance(env_value, Mapping) or not all(
                isinstance(key, str) and key and isinstance(item, str) for key, item in env_value.items()
            ):
                raise TypeError("Modal image setup field 'env' must map non-empty string names to string values")
            env = dict(env_value)

        return cls(
            run=tuple(commands),
            user=user,
            shell=shell,
            env=env,
            secret_names=_string_sequence(value.get("secret_names", ()), field_name="image_setup_steps.secret_names"),
        )


@dataclass(frozen=True)
class ModalProviderOptions:
    """Validated per-sandbox options from ``SandboxSpec.provider_options``."""

    name: str | None = None
    secret_names: tuple[str, ...] = ()
    registry_secret_name: str | None = None
    block_network: bool = False
    outbound_cidr_allowlist: tuple[str, ...] | None = None
    outbound_domain_allowlist: tuple[str, ...] | None = None
    inbound_cidr_allowlist: tuple[str, ...] | None = None
    cloud: str | None = None
    region: str | tuple[str, ...] | None = None
    idle_timeout_s: int | float | None = None
    cpu_limit: int | float | None = None
    memory_limit_mib: int | None = None
    image_setup_steps: tuple[ModalImageSetupStep, ...] = ()

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any] | None) -> "ModalProviderOptions":
        if options is None:
            return cls()
        if not isinstance(options, Mapping):
            raise TypeError("Modal provider_options must be a mapping")

        allowed = set(cls.__dataclass_fields__) | {"network_allowlist"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(
                f"Unknown Modal provider option(s): {', '.join(sorted(unknown))}. "
                f"Supported: {', '.join(sorted(allowed))}"
            )

        name = _optional_string(options.get("name"), field_name="name")
        registry_secret_name = _optional_string(options.get("registry_secret_name"), field_name="registry_secret_name")
        cloud = _optional_string(options.get("cloud"), field_name="cloud")
        block_network = options.get("block_network", False)
        if not isinstance(block_network, bool):
            raise TypeError("Modal provider option 'block_network' must be a bool")

        network_allowlist = _optional_string_sequence(options.get("network_allowlist"), field_name="network_allowlist")
        outbound_domain_allowlist = _optional_string_sequence(
            options.get("outbound_domain_allowlist"), field_name="outbound_domain_allowlist"
        )
        if network_allowlist is not None and outbound_domain_allowlist is not None:
            raise ValueError("Set only one of Modal 'network_allowlist' and 'outbound_domain_allowlist'")
        outbound_domain_allowlist = outbound_domain_allowlist or network_allowlist
        outbound_cidr_allowlist = _optional_string_sequence(
            options.get("outbound_cidr_allowlist"), field_name="outbound_cidr_allowlist"
        )
        inbound_cidr_allowlist = _optional_string_sequence(
            options.get("inbound_cidr_allowlist"), field_name="inbound_cidr_allowlist"
        )
        if block_network and any(
            allowlist is not None
            for allowlist in (outbound_cidr_allowlist, outbound_domain_allowlist, inbound_cidr_allowlist)
        ):
            raise ValueError("Modal 'block_network' cannot be combined with a network allowlist")

        idle_timeout_s = options.get("idle_timeout_s")
        if idle_timeout_s is not None:
            _seconds(idle_timeout_s, field_name="idle_timeout_s")
        cpu_limit = options.get("cpu_limit")
        if cpu_limit is not None:
            _seconds(cpu_limit, field_name="cpu_limit")
        memory_limit_mib = options.get("memory_limit_mib")
        if memory_limit_mib is not None:
            if isinstance(memory_limit_mib, bool) or not isinstance(memory_limit_mib, int):
                raise TypeError("Modal provider option 'memory_limit_mib' must be an integer")
            _seconds(memory_limit_mib, field_name="memory_limit_mib")

        region_value = options.get("region")
        if region_value is None or isinstance(region_value, str):
            region = region_value
        else:
            region = _string_sequence(region_value, field_name="region")

        setup_steps_value = options.get("image_setup_steps", ())
        if isinstance(setup_steps_value, (str, bytes)) or not isinstance(setup_steps_value, Sequence):
            raise TypeError("Modal provider option 'image_setup_steps' must be a sequence of mappings")
        if not all(isinstance(step, Mapping) for step in setup_steps_value):
            raise TypeError("Modal provider option 'image_setup_steps' must contain only mappings")

        return cls(
            name=name,
            secret_names=_string_sequence(options.get("secret_names", ()), field_name="secret_names"),
            registry_secret_name=registry_secret_name,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
            inbound_cidr_allowlist=inbound_cidr_allowlist,
            cloud=cloud,
            region=region,
            idle_timeout_s=idle_timeout_s,
            cpu_limit=cpu_limit,
            memory_limit_mib=memory_limit_mib,
            image_setup_steps=tuple(ModalImageSetupStep.from_mapping(step) for step in setup_steps_value),
        )


def _require_modal_sdk() -> tuple[Any, Any, Any, Any]:
    try:
        from modal import App, Image, Sandbox, Secret
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "The Modal SDK is required for the modal sandbox provider. "
            "Install nemo-gym[sandbox] before selecting the 'modal' provider."
        ) from e
    return App, Image, Sandbox, Secret


def _modal_authentication_configured() -> bool:
    from modal.config import Config

    config = Config()
    return bool(config.get("token_id")) and bool(config.get("token_secret"))


def _coerce_config(value: Any, config_cls: type[Any]) -> Any:
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, Mapping):
        return config_cls(**value)
    raise TypeError(f"{config_cls.__name__} must be a mapping or {config_cls.__name__} instance")


def _seconds(value: int | float, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Modal {field_name} must be a finite positive number")
    return math.ceil(value)


def _byte_limit(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Modal {field_name} must be a positive integer")
    return value


def _optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"Modal provider option '{field_name}' must be a non-empty string")
    return value


def _string_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"Modal provider option '{field_name}' must be a sequence of strings")
    strings = tuple(value)
    if not all(isinstance(item, str) and item for item in strings):
        raise TypeError(f"Modal provider option '{field_name}' must contain only non-empty strings")
    return strings


def _optional_string_sequence(value: Any, *, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _string_sequence(value, field_name=field_name)


def _gpu_config(resources: SandboxResources) -> str | None:
    if resources.gpu is None and resources.gpu_type is None:
        return None
    count = resources.gpu if resources.gpu is not None else 1
    if count < 1:
        raise ValueError("Modal sandbox GPU count must be at least one")
    gpu_type = resources.gpu_type or "any"
    return gpu_type if count == 1 else f"{gpu_type}:{count}"


async def _modal_call(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a Modal synchronized method through its native async interface."""
    async_method = getattr(method, "aio", None)
    result = async_method(*args, **kwargs) if async_method is not None else method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def _sandbox_id(sandbox: Any) -> str:
    sandbox_id = getattr(sandbox, "object_id", None) or getattr(sandbox, "id", None)
    if not sandbox_id:
        raise ModalCreateError("Modal returned a sandbox without an object id")
    return str(sandbox_id)


def _remote_parent(path: str) -> str | None:
    remote_path = PurePosixPath(path)
    if not remote_path.is_absolute():
        raise ValueError(f"Modal sandbox file path must be absolute: {path!r}")
    parent = str(remote_path.parent)
    return None if parent == "/" else parent


@dataclass(frozen=True)
class _ModalOutputLimitExceeded(RuntimeError):
    stream: str

    @property
    def error_type(self) -> str:
        return f"output_limit_{self.stream}"


class _BoundedExecOutput:
    """Track byte counts and retain only the combined raw output tail."""

    def __init__(self, config: ModalCreateConfig) -> None:
        self._stream_limits = {
            "stdout": config.exec_stdout_limit_bytes,
            "stderr": config.exec_stderr_limit_bytes,
        }
        self._combined_limit = config.exec_combined_output_limit_bytes
        self._stream_totals = {"stdout": 0, "stderr": 0}
        self._combined_total = 0
        self._recent: deque[tuple[str, bytes]] = deque()
        self._recent_bytes = 0

    @staticmethod
    def _to_bytes(chunk: Any) -> bytes:
        if isinstance(chunk, bytes):
            return chunk
        if isinstance(chunk, (bytearray, memoryview)):
            return bytes(chunk)
        if isinstance(chunk, str):
            return chunk.encode("utf-8", errors="replace")
        return str(chunk).encode("utf-8", errors="replace")

    def _append_recent(self, stream: str, data: bytes) -> None:
        if not data:
            return
        if len(data) >= self._combined_limit:
            self._recent.clear()
            self._recent.append((stream, data[-self._combined_limit :]))
            self._recent_bytes = self._combined_limit
            return

        self._recent.append((stream, data))
        self._recent_bytes += len(data)
        excess = self._recent_bytes - self._combined_limit
        while excess > 0:
            oldest_stream, oldest_data = self._recent.popleft()
            if len(oldest_data) <= excess:
                self._recent_bytes -= len(oldest_data)
                excess -= len(oldest_data)
                continue
            self._recent.appendleft((oldest_stream, oldest_data[excess:]))
            self._recent_bytes -= excess
            excess = 0

    def add(self, stream: str, chunk: Any) -> None:
        data = self._to_bytes(chunk)
        self._stream_totals[stream] += len(data)
        self._combined_total += len(data)
        self._append_recent(stream, data)
        if self._stream_totals[stream] > self._stream_limits[stream]:
            raise _ModalOutputLimitExceeded(stream)
        if self._combined_total > self._combined_limit:
            raise _ModalOutputLimitExceeded("combined")

    @staticmethod
    def _decode_tail(data: bytes, limit: int) -> str:
        tail = data[-limit:]
        # Invalid bytes and a leading/trailing partial code point are omitted.
        # Unlike replacement decoding, this guarantees that re-encoding the
        # returned text cannot expand beyond the configured raw-byte limit.
        return tail.decode("utf-8", errors="ignore")

    def text(self, stream: str) -> str:
        data = b"".join(chunk for chunk_stream, chunk in self._recent if chunk_stream == stream)
        return self._decode_tail(data, self._stream_limits[stream])


async def _collect_modal_stream(iterator: AsyncIterator[Any], name: str, output: _BoundedExecOutput) -> None:
    async for chunk in iterator:
        output.add(name, chunk)


async def _copy_bounded_modal_stream(iterator: AsyncIterator[Any], output: Any, max_bytes: int) -> int:
    total = 0
    async for chunk in iterator:
        data = _BoundedExecOutput._to_bytes(chunk)
        if len(data) > max_bytes - total:
            raise SandboxDownloadLimitExceeded(f"Sandbox download exceeded max_bytes={max_bytes}")
        output.write(data)
        total += len(data)
    return total


async def _drain_modal_exec(
    tasks: tuple[asyncio.Task[Any], ...],
    iterators: tuple[AsyncIterator[Any], ...],
) -> None:
    """Drain cancelled process tasks and close their stream iterators."""
    await asyncio.gather(*tasks, return_exceptions=True)
    close_tasks = []
    for iterator in iterators:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            close_tasks.append(asyncio.create_task(_modal_call(close)))
    if close_tasks:
        close_results = await asyncio.gather(*close_tasks, return_exceptions=True)
        if any(isinstance(result, BaseException) for result in close_results):
            raise ModalCleanupError("Modal exec stream draining failed")


class ModalProvider:
    """Sandbox provider backed by Modal Sandboxes."""

    name = "modal"

    def __init__(
        self,
        *,
        connection: ModalConnectionConfig | Mapping[str, Any] | None = None,
        create: ModalCreateConfig | Mapping[str, Any] | None = None,
        probe: ModalProbeConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = _coerce_config(connection, ModalConnectionConfig)
        self._create = _coerce_config(create, ModalCreateConfig)
        self._probe = _coerce_config(probe, ModalProbeConfig)
        self._late_create_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._completed_cleanup_errors: dict[asyncio.Task[None], str] = {}

    @classmethod
    def preflight(cls) -> None:
        """Reject missing Modal authentication before allocating benchmark work."""
        _require_modal_sdk()
        if not _modal_authentication_configured():
            raise ModalCreateError(
                "Modal authentication is not configured; set both MODAL_TOKEN_ID and "
                "MODAL_TOKEN_SECRET or configure a Modal profile"
            )

    async def _terminate_late_create(self, create_task: asyncio.Task[Any]) -> None:
        """Terminate a sandbox whose handle arrives after create was abandoned."""
        try:
            sandbox = await create_task
        except (asyncio.CancelledError, Exception):
            return
        await self._terminate_sandbox(sandbox)

    async def _terminate_sandbox(self, sandbox: Any) -> None:
        for attempt in range(3):
            try:
                await _modal_call(sandbox.terminate, wait=True)
                return
            except Exception as error:
                if type(error).__name__ in {"NotFoundError", "SandboxTerminatedError"}:
                    return
                if attempt == 2:
                    try:
                        sandbox_id = _sandbox_id(sandbox)
                    except ModalCreateError:
                        sandbox_id = "unknown"
                    raise ModalCleanupError(
                        "Modal sandbox termination failed after 3 attempts; "
                        f"sandbox_id={sandbox_id!r}, error_type={type(error).__name__!r}"
                    ) from None
                await asyncio.sleep(0.25 * 2**attempt)

    def _track_cleanup(self, cleanup: Any) -> asyncio.Task[None]:
        cleanup_task = asyncio.create_task(cleanup)
        self._late_create_cleanup_tasks.add(cleanup_task)

        def record_result(task: asyncio.Task[None]) -> None:
            self._late_create_cleanup_tasks.discard(task)
            if task.cancelled():
                return
            error = task.exception()
            if error is not None:
                message = str(error) if isinstance(error, ModalCleanupError) else type(error).__name__
                self._completed_cleanup_errors[task] = message

        cleanup_task.add_done_callback(record_result)
        return cleanup_task

    async def _wait_for_cleanup(self, cleanup_task: asyncio.Task[None]) -> bool:
        try:
            await asyncio.wait_for(
                asyncio.shield(cleanup_task),
                timeout=_seconds(self._create.cleanup_timeout_s, field_name="cleanup_timeout_s"),
            )
            self._completed_cleanup_errors.pop(cleanup_task, None)
            return True
        except asyncio.TimeoutError:
            # Keep the tracked task alive so a stalled provider operation can
            # still finish cleanup after the caller's bounded wait expires.
            return False
        except BaseException:
            self._completed_cleanup_errors.pop(cleanup_task, None)
            raise

    async def _clean_up_late_create(self, create_task: asyncio.Task[Any]) -> bool:
        return await self._wait_for_cleanup(self._track_cleanup(self._terminate_late_create(create_task)))

    async def _clean_up_sandbox(self, sandbox: Any) -> bool:
        return await self._wait_for_cleanup(self._track_cleanup(self._terminate_sandbox(sandbox)))

    async def _clean_up_after_error(self, error: BaseException, sandbox: Any) -> None:
        try:
            completed = await self._clean_up_sandbox(sandbox)
            if not completed:
                error.add_note("Modal sandbox termination exceeded the bounded cleanup wait")
        except Exception as cleanup_error:
            error.add_note(str(cleanup_error))

    async def _clean_up_exec(
        self,
        sandbox: Any,
        tasks: tuple[asyncio.Task[Any], ...],
        iterators: tuple[AsyncIterator[Any], ...],
    ) -> tuple[bool | BaseException, bool | BaseException]:
        """Start termination immediately and bound local process draining."""
        for task in tasks:
            task.cancel()
        termination_task = self._track_cleanup(self._terminate_sandbox(sandbox))
        drain_task = self._track_cleanup(_drain_modal_exec(tasks, iterators))
        results = await asyncio.gather(
            self._wait_for_cleanup(termination_task),
            self._wait_for_cleanup(drain_task),
            return_exceptions=True,
        )
        return results[0], results[1]

    @staticmethod
    def _exec_cleanup_suffix(results: tuple[bool | BaseException, bool | BaseException]) -> str:
        if any(isinstance(result, BaseException) for result in results):
            return "_cleanup_error"
        if not all(result is True for result in results):
            return "_cleanup_timeout"
        return ""

    @staticmethod
    def _note_exec_cleanup(
        error: BaseException,
        results: tuple[bool | BaseException, bool | BaseException],
    ) -> None:
        labels = ("sandbox termination", "Modal exec stream draining")
        for label, result in zip(labels, results, strict=True):
            if result is False:
                error.add_note(f"{label.capitalize()} exceeded the bounded cleanup wait")
            elif isinstance(result, BaseException):
                detail = str(result) if isinstance(result, ModalCleanupError) else type(result).__name__
                error.add_note(f"{label.capitalize()} failed: {detail}")

    async def _app(self, app_cls: Any) -> Any:
        kwargs: dict[str, Any] = {"create_if_missing": self._connection.create_if_missing}
        if self._connection.environment_name is not None:
            kwargs["environment_name"] = self._connection.environment_name
        return await _modal_call(app_cls.lookup, self._connection.app_name, **kwargs)

    def _secret(self, secret_cls: Any, name: str) -> Any:
        kwargs: dict[str, Any] = {}
        if self._connection.environment_name is not None:
            kwargs["environment_name"] = self._connection.environment_name
        return secret_cls.from_name(name, **kwargs)

    def _apply_image_setup(
        self,
        image: Any,
        steps: tuple[ModalImageSetupStep, ...],
        *,
        secret_cls: Any,
    ) -> Any:
        for step in steps:
            secrets = [self._secret(secret_cls, name) for name in step.secret_names]
            for command in step.run:
                if step.user is None or step.user == "root" or step.user == 0:
                    build_command = f"{shlex.quote(step.shell)} -c {shlex.quote(command)}"
                else:
                    build_command = (
                        f"su -s {shlex.quote(step.shell)} -c {shlex.quote(command)} {shlex.quote(str(step.user))}"
                    )
                image = image.run_commands(build_command, env=step.env, secrets=secrets)
        return image

    def _create_kwargs(
        self,
        spec: SandboxSpec,
        options: ModalProviderOptions,
        *,
        app: Any,
        image: Any,
        secrets: list[Any],
    ) -> dict[str, Any]:
        timeout_s = spec.ttl_s if spec.ttl_s is not None else self._create.default_timeout_s
        idle_timeout_s = options.idle_timeout_s
        if idle_timeout_s is None:
            idle_timeout_s = self._create.idle_timeout_s

        kwargs: dict[str, Any] = {
            "app": app,
            "image": image,
            "secrets": secrets,
            "tags": spec.metadata,
            "timeout": _seconds(timeout_s, field_name="ttl_s"),
            "workdir": spec.workdir,
            "block_network": options.block_network,
            "outbound_cidr_allowlist": options.outbound_cidr_allowlist,
            "outbound_domain_allowlist": options.outbound_domain_allowlist,
            "inbound_cidr_allowlist": options.inbound_cidr_allowlist,
        }
        if options.name is not None:
            kwargs["name"] = options.name
        if idle_timeout_s is not None:
            kwargs["idle_timeout"] = _seconds(idle_timeout_s, field_name="idle_timeout_s")
        if options.cpu_limit is not None:
            cpu_request = spec.resources.cpu if spec.resources.cpu is not None else min(0.125, options.cpu_limit)
            if cpu_request > options.cpu_limit:
                raise ValueError("Modal CPU request cannot exceed cpu_limit")
            kwargs["cpu"] = (cpu_request, options.cpu_limit)
        elif spec.resources.cpu is not None:
            kwargs["cpu"] = spec.resources.cpu
        if options.memory_limit_mib is not None:
            memory_request = (
                spec.resources.memory_mib
                if spec.resources.memory_mib is not None
                else min(128, options.memory_limit_mib)
            )
            if memory_request > options.memory_limit_mib:
                raise ValueError("Modal memory request cannot exceed memory_limit_mib")
            kwargs["memory"] = (memory_request, options.memory_limit_mib)
        elif spec.resources.memory_mib is not None:
            kwargs["memory"] = spec.resources.memory_mib
        gpu = _gpu_config(spec.resources)
        if gpu is not None:
            kwargs["gpu"] = gpu
        if options.cloud is not None:
            kwargs["cloud"] = options.cloud
        if options.region is not None:
            kwargs["region"] = options.region
        return kwargs

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a ready Modal sandbox from a registry image."""
        if not spec.image:
            raise ModalCreateError("spec.image is required for the Modal provider")
        if spec.resources.disk_gib is not None:
            raise ModalCreateError(
                "Modal does not expose an ephemeral disk size request; omit SandboxResources.disk_gib"
            )

        App, Image, Sandbox, Secret = _require_modal_sdk()
        options = ModalProviderOptions.from_mapping(spec.provider_options)
        sandbox: Any | None = None
        handle: SandboxHandle | None = None
        create_task: asyncio.Task[Any] | None = None
        try:
            app = await self._app(App)
            registry_secret = (
                self._secret(Secret, options.registry_secret_name)
                if options.registry_secret_name is not None
                else None
            )
            image = Image.from_registry(spec.image, secret=registry_secret)
            image = self._apply_image_setup(image, options.image_setup_steps, secret_cls=Secret)
            secrets = [self._secret(Secret, name) for name in options.secret_names]
            if spec.env:
                # Treat the complete persistent environment as sensitive.
                # Modal's ordinary env field is visible in control-plane inputs.
                secrets.append(Secret.from_dict(spec.env))
            kwargs = self._create_kwargs(spec, options, app=app, image=image, secrets=secrets)
            entrypoint = tuple(spec.entrypoint or ())
            ready_timeout_s = spec.ready_timeout_s
            if ready_timeout_s is None:
                ready_timeout_s = self._create.ready_timeout_s
            try:
                create_task = asyncio.create_task(_modal_call(Sandbox.create, *entrypoint, **kwargs))
                sandbox = await asyncio.wait_for(
                    asyncio.shield(create_task),
                    timeout=_seconds(ready_timeout_s, field_name="ready_timeout_s"),
                )
            except asyncio.TimeoutError as e:
                assert create_task is not None
                timeout_error = ModalCreateTimeoutError(
                    f"Modal sandbox creation timed out after {float(ready_timeout_s):g}s"
                )
                try:
                    completed = await self._clean_up_late_create(create_task)
                    if not completed:
                        timeout_error.add_note("Late Modal sandbox cleanup is still pending")
                except Exception as cleanup_error:
                    timeout_error.add_note(str(cleanup_error))
                raise timeout_error from e

            handle = SandboxHandle(sandbox_id=_sandbox_id(sandbox), provider_name=self.name, raw=sandbox)
            await self._verify_created_handle(handle)
            return handle
        except asyncio.CancelledError as error:
            if handle is not None:
                await self._clean_up_after_error(error, handle.raw)
            elif sandbox is not None:
                await self._clean_up_after_error(error, sandbox)
            elif create_task is not None:
                try:
                    completed = await self._clean_up_late_create(create_task)
                    if not completed:
                        error.add_note("Late Modal sandbox cleanup is still pending")
                except Exception as cleanup_error:
                    error.add_note(str(cleanup_error))
            raise
        except (ModalCreateError, ModalCreateVerificationError) as error:
            if handle is not None:
                await self._clean_up_after_error(error, handle.raw)
            elif sandbox is not None:
                await self._clean_up_after_error(error, sandbox)
            raise
        except Exception as e:
            creation_error = ModalCreateError(f"Modal sandbox creation failed ({type(e).__name__})")
            if sandbox is not None:
                await self._clean_up_after_error(creation_error, sandbox)
            # SDK failures can contain credentials. Do not retain the raw
            # exception as an explicit cause because tracebacks are persisted
            # in Pier job artifacts.
            raise creation_error from None

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        if self._probe.command is None:
            return
        result = await self.exec(handle, self._probe.command, timeout_s=self._probe.timeout_s, user="root")
        stdout = result.stdout or ""
        if result.return_code != 0 or (
            self._probe.expected_stdout is not None and self._probe.expected_stdout not in stdout
        ):
            raise ModalCreateVerificationError(
                "Modal sandbox failed its readiness probe; "
                f"sandbox_id={handle.sandbox_id!r}, return_code={result.return_code}, "
                f"error_type={result.error_type!r}"
            )

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
        """Execute a shell command and collect bounded stdout and stderr."""
        if user is None or user == "root" or user == 0:
            args = ("/bin/sh", "-c", command)
        elif isinstance(user, str) and user:
            args = ("su", "-s", "/bin/sh", "-c", command, user)
        else:
            raise ValueError("Modal exec user must be a username, 'root', 0, or None")

        _, _, _, Secret = _require_modal_sdk()
        kwargs: dict[str, Any] = {
            "workdir": cwd,
            # Modal treats ordinary exec env as control-plane input metadata.
            # Use an ephemeral Secret for the whole map so callers cannot
            # accidentally expose API tokens through provider configuration.
            "secrets": [Secret.from_dict(env)] if env else [],
            # Count provider output as raw bytes. Modal's text decoder is
            # intentionally bypassed so arbitrary command output cannot turn
            # into an SDK decoding failure before Gym applies its limits.
            "text": False,
        }
        if timeout_s is not None:
            kwargs["timeout"] = _seconds(timeout_s, field_name="exec timeout_s")
        tasks: tuple[asyncio.Task[Any], ...] = ()
        iterators: tuple[AsyncIterator[Any], ...] = ()
        output = _BoundedExecOutput(self._create)
        try:
            process = await _modal_call(handle.raw.exec, *args, **kwargs)
            iterators = (process.stdout.__aiter__(), process.stderr.__aiter__())
            stdout_task = asyncio.create_task(_collect_modal_stream(iterators[0], "stdout", output))
            stderr_task = asyncio.create_task(_collect_modal_stream(iterators[1], "stderr", output))
            wait_task = asyncio.create_task(_modal_call(process.wait))
            tasks = (stdout_task, stderr_task, wait_task)
            _, _, return_code = await asyncio.gather(
                stdout_task,
                stderr_task,
                wait_task,
            )
        except _ModalOutputLimitExceeded as error:
            cleanup_results = await self._clean_up_exec(handle.raw, tasks, iterators)
            return SandboxExecResult(
                stdout=output.text("stdout"),
                stderr=output.text("stderr"),
                return_code=-1,
                error_type=error.error_type + self._exec_cleanup_suffix(cleanup_results),
            )
        except asyncio.CancelledError as error:
            cleanup_results = await self._clean_up_exec(handle.raw, tasks, iterators)
            self._note_exec_cleanup(error, cleanup_results)
            raise
        except Exception as error:
            cleanup_results = await self._clean_up_exec(handle.raw, tasks, iterators)
            self._note_exec_cleanup(error, cleanup_results)
            raise
        error_type = "timeout" if return_code == -1 and timeout_s is not None else None
        return SandboxExecResult(
            stdout=output.text("stdout"),
            stderr=output.text("stderr"),
            return_code=int(return_code),
            error_type=error_type,
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Copy one local file into the Modal sandbox."""
        if not source_path.is_file():
            raise FileNotFoundError(f"Modal upload source is not a file: {source_path}")
        parent = _remote_parent(target_path)
        filesystem = handle.raw.filesystem
        if parent is not None:
            await _modal_call(filesystem.make_directory, parent, create_parents=True)
        await _modal_call(filesystem.copy_from_local, source_path, target_path)

    async def download_file(
        self,
        handle: SandboxHandle,
        source_path: str,
        target_path: Path,
        *,
        max_bytes: int | None = None,
    ) -> None:
        """Copy one file from the Modal sandbox to the local filesystem."""
        _remote_parent(source_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if max_bytes is None:
            await _modal_call(handle.raw.filesystem.copy_to_local, source_path, target_path)
            return
        _byte_limit(max_bytes, field_name="download max_bytes")

        descriptor, temporary_name = tempfile.mkstemp(prefix=".nemo-gym-download-", dir=target_path.parent)
        temporary_path = Path(temporary_name)
        tasks: tuple[asyncio.Task[Any], ...] = ()
        iterators: tuple[AsyncIterator[Any], ...] = ()
        try:
            with os.fdopen(descriptor, "wb") as output:
                try:
                    command = f"exec cat -- {shlex.quote(source_path)}"
                    process = await _modal_call(handle.raw.exec, "/bin/sh", "-c", command, text=False)
                    iterators = (process.stdout.__aiter__(), process.stderr.__aiter__())
                    stderr_output = _BoundedExecOutput(self._create)
                    copy_task = asyncio.create_task(_copy_bounded_modal_stream(iterators[0], output, max_bytes))
                    stderr_task = asyncio.create_task(_collect_modal_stream(iterators[1], "stderr", stderr_output))
                    wait_task = asyncio.create_task(_modal_call(process.wait))
                    tasks = (copy_task, stderr_task, wait_task)
                    _, _, return_code = await asyncio.gather(*tasks)
                except asyncio.CancelledError as error:
                    cleanup_results = await self._clean_up_exec(handle.raw, tasks, iterators)
                    self._note_exec_cleanup(error, cleanup_results)
                    raise
                except Exception as error:
                    cleanup_results = await self._clean_up_exec(handle.raw, tasks, iterators)
                    self._note_exec_cleanup(error, cleanup_results)
                    raise
            if return_code != 0:
                raise RuntimeError(
                    f"Modal sandbox download failed; sandbox_id={handle.sandbox_id!r}, return_code={return_code}"
                )
            temporary_path.replace(target_path)
        except BaseException:
            # ``mkstemp`` is already closed by ``fdopen`` in every path after
            # entering the try. If process startup itself fails, close it here.
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary_path.unlink(missing_ok=True)
            raise

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Map Modal's process lifecycle to the provider-neutral status enum."""
        try:
            return_code = await _modal_call(handle.raw.poll)
        except Exception as e:
            if type(e).__name__ in {"NotFoundError", "SandboxTerminatedError"}:
                return SandboxStatus.STOPPED
            return SandboxStatus.UNKNOWN
        return SandboxStatus.RUNNING if return_code is None else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        """Terminate a Modal sandbox, treating an already-gone sandbox as closed."""
        completed = await self._clean_up_sandbox(handle.raw)
        if not completed:
            raise TimeoutError(f"Modal sandbox termination timed out; sandbox_id={handle.sandbox_id!r}") from None

    async def aclose(self) -> None:
        """Drain tracked cleanup and surface delayed termination failures."""
        if self._late_create_cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.gather(*tuple(self._late_create_cleanup_tasks), return_exceptions=True)),
                    timeout=_seconds(self._create.cleanup_timeout_s, field_name="cleanup_timeout_s"),
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Modal cleanup did not finish for {len(self._late_create_cleanup_tasks)} sandbox operation(s)"
                ) from None
        if self._completed_cleanup_errors:
            failures = tuple(self._completed_cleanup_errors.values())
            self._completed_cleanup_errors.clear()
            raise ModalCleanupError("Delayed Modal cleanup failed: " + "; ".join(failures))
