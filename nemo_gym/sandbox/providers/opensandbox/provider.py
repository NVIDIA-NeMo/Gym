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

"""OpenSandbox provider implementation."""

import asyncio
import logging
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

from nemo_gym.sandbox.providers.base import (
    SandboxBatchCreateError,
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
    VolumeMount,
)


LOGGER = logging.getLogger(__name__)


class OpenSandboxBatchCreateError(SandboxBatchCreateError):
    """Raised when a batch sandbox preallocation cannot be completed."""


class OpenSandboxCreateError(SandboxCreateError):
    """Raised when OpenSandbox cannot create a sandbox."""


class OpenSandboxCreateTimeoutError(OpenSandboxCreateError):
    """Raised when OpenSandbox sandbox creation exceeds the client timeout."""


class OpenSandboxCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a newly-created sandbox cannot execute a probe command."""


RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_MARKERS = (
    "all connection attempts failed",
    "connection refused",
    "connection reset",
    "gateway timeout",
    "http 408",
    "http 409",
    "http 425",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "incomplete chunked read",
    "peer closed connection",
    "pod ip is not yet available",
    "pod may still be starting",
    "errimagepull",
    "get endpoint for sandbox",
    "imagepullbackoff",
    "pod failed",
    "podfailed",
    "remote protocol error",
    "service unavailable",
    "server disconnected",
    "status code: 408",
    "status code: 409",
    "status code: 425",
    "status code: 429",
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
    "temporarily unavailable",
    "timed out",
    "timeout",
)
METADATA_VALUE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
DEFAULT_IMAGE_PULL_POLICY = "IfNotPresent"
IMAGE_PULL_POLICY_EXTENSION_KEY = "imagePullPolicy"
IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY = "opensandbox.extensions.image-pull-policy"
PROVIDER_OPTION_PLATFORM = "platform"
PROVIDER_OPTION_SKIP_HEALTH_CHECK = "skip_health_check"
PROVIDER_OPTION_SNAPSHOT_ID = "snapshot_id"
PROVIDER_OPTION_VOLUMES = "volumes"
VALID_IMAGE_PULL_POLICIES = {"Always", "IfNotPresent", "Never"}
STATUS_CODE_RE = re.compile(r"(?:status code|http)\D+(\d{3})", re.IGNORECASE)


def validate_image_pull_policy(image_pull_policy: str) -> str:
    """Validate a Kubernetes-compatible container image pull policy."""
    if image_pull_policy not in VALID_IMAGE_PULL_POLICIES:
        allowed = ", ".join(sorted(VALID_IMAGE_PULL_POLICIES))
        raise ValueError(f"image_pull_policy must be one of: {allowed}")
    return image_pull_policy


def _require_opensandbox_sdk() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from opensandbox import Sandbox
        from opensandbox.config import ConnectionConfig
        from opensandbox.models.execd import RunCommandOpts
        from opensandbox.models.sandboxes import PlatformSpec, Volume
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "OpenSandbox SDK is required for the opensandbox sandbox provider. "
            "Install nemo-gym[sandbox] in the runtime image before using "
            "env.sandbox.provider.name=opensandbox."
        ) from e

    return Sandbox, ConnectionConfig, RunCommandOpts, PlatformSpec, Volume


def _require_opensandbox_sdk_pool() -> tuple[Any, Any, Any, Any]:
    try:
        from opensandbox import (
            AcquirePolicy,
            InMemoryAsyncPoolStateStore,
            PoolCreationSpec,
            SandboxPoolAsync,
        )
    except ImportError as e:
        raise ModuleNotFoundError(
            "OpenSandbox SDK >=0.1.9 is required for native SDK pool batch creation. "
            "Install nemo-gym[sandbox] in the runtime image."
        ) from e

    return AcquirePolicy, InMemoryAsyncPoolStateStore, PoolCreationSpec, SandboxPoolAsync


def _require_tenacity() -> tuple[Any, Any, Any, Any]:
    try:
        from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "tenacity is required for OpenSandbox retry handling. Install nemo-gym[sandbox] before using "
            "env.sandbox.provider.name=opensandbox."
        ) from e

    return AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential


def _httpx_retryable_types() -> tuple[type[BaseException], ...]:
    try:
        import httpx
    except ModuleNotFoundError:
        return tuple()
    return (
        httpx.RemoteProtocolError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.TimeoutException,
        httpx.NetworkError,
    )


def _has_retryable_error_marker(exception: BaseException) -> bool:
    message = str(exception).lower()
    return any(marker in message for marker in RETRYABLE_ERROR_MARKERS)


def _exception_status_code(exception: BaseException) -> int | None:
    status_code = getattr(exception, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    match = STATUS_CODE_RE.search(str(exception))
    if match is None:
        return None
    return int(match.group(1))


def _sdk_error_attributes(
    exception: BaseException,
    *,
    operation: str,
    sandbox_id: str,
    attempt_number: int | None = None,
    max_attempts: int | None = None,
    sleep_s: float | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "provider": OpenSandboxProvider.name,
        "operation": operation,
        "sandbox_id": sandbox_id,
        "error_type": type(exception).__name__,
        "error_message": str(exception)[:500],
    }
    status_code = _exception_status_code(exception)
    if status_code is not None:
        attrs["status_code"] = status_code
    if attempt_number is not None:
        attrs["attempt_number"] = attempt_number
    if max_attempts is not None:
        attrs["max_attempts"] = max_attempts
    if sleep_s is not None:
        attrs["next_sleep_s"] = sleep_s
    return attrs


def _is_retryable_create_error(exception: BaseException) -> bool:
    """Return whether a sandbox create failure is likely transient."""
    if isinstance(exception, SandboxCreateVerificationError):
        return True
    if isinstance(exception, SandboxCreateError):
        return True
    if isinstance(exception, (ConnectionError, OSError, TimeoutError)):
        return True
    httpx_types = _httpx_retryable_types()
    if httpx_types and isinstance(exception, httpx_types):
        return True

    try:
        from opensandbox.exceptions import (
            InvalidArgumentException,
            SandboxApiException,
            SandboxException,
            SandboxInternalException,
            SandboxReadyTimeoutException,
            SandboxUnhealthyException,
        )
    except ModuleNotFoundError:
        return _has_retryable_error_marker(exception)

    if isinstance(exception, InvalidArgumentException):
        return False
    if isinstance(
        exception,
        (
            SandboxInternalException,
            SandboxReadyTimeoutException,
            SandboxUnhealthyException,
        ),
    ):
        return True
    if isinstance(exception, SandboxApiException):
        status_code = getattr(exception, "status_code", None)
        if status_code in RETRYABLE_HTTP_STATUS_CODES:
            return True
        if status_code is not None and status_code < 500:
            return False
    if not isinstance(exception, SandboxException):
        return _has_retryable_error_marker(exception)

    return _has_retryable_error_marker(exception)


def _is_retryable_sdk_operation_error(exception: BaseException) -> bool:
    """Return whether an SDK operation can be retried by Gym.

    The OpenSandbox Python SDK does not retry generated lifecycle, execd, or
    filesystem HTTP calls. It converts network failures into SDK exceptions and
    exposes API status codes, so classify both the wrapper and its original
    cause here.
    """
    if isinstance(exception, TimeoutError):
        return False
    cause = exception.__cause__
    if isinstance(cause, BaseException) and _is_retryable_sdk_operation_error(cause):
        return True
    if isinstance(exception, (ConnectionError, OSError)):
        return True
    httpx_types = _httpx_retryable_types()
    if httpx_types and isinstance(exception, httpx_types):
        return True
    return _is_retryable_create_error(exception)


def _is_missing_sandbox_delete_error(exception: BaseException) -> bool:
    message = str(exception).lower()
    return "sandbox" in message and "not found" in message


def _log_create_retry(retry_state: Any) -> None:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    sleep_s = retry_state.next_action.sleep if retry_state.next_action else None
    LOGGER.warning(
        "Retrying OpenSandbox sandbox create after attempt %s; next_sleep_s=%s; error=%r",
        retry_state.attempt_number,
        sleep_s,
        exception,
    )


def _log_operation_retry(retry_state: Any) -> None:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    sleep_s = retry_state.next_action.sleep if retry_state.next_action else None
    LOGGER.warning(
        "Retrying OpenSandbox SDK operation after attempt %s; next_sleep_s=%s; error=%r",
        retry_state.attempt_number,
        sleep_s,
        exception,
    )


def _string_map(values: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items()}


def _metadata_value(value: Any) -> str:
    normalized = METADATA_VALUE_RE.sub("_", str(value)).strip("._-")
    normalized = normalized[:63].strip("._-")
    return normalized or "metadata"


def _metadata_map(values: dict[str, Any]) -> dict[str, str]:
    return {str(key): _metadata_value(value) for key, value in values.items()}


def _normalize_spec(spec: SandboxSpec) -> SandboxSpec:
    return replace(
        spec,
        env=_string_map(spec.env),
        metadata=_metadata_map(spec.metadata),
        resources=_string_map(spec.resources),
        extensions=_string_map(spec.extensions),
    )


def _to_platform_spec(platform: dict[str, Any]) -> Any:
    _, _, _, PlatformSpec, _ = _require_opensandbox_sdk()
    return PlatformSpec(**platform)


def _volume_mount_name(volume: VolumeMount, index: int) -> str:
    source = volume.container_path or volume.host_path or f"volume-{index}"
    normalized = METADATA_VALUE_RE.sub("-", source.strip("/")).strip("-")
    return (normalized or f"volume-{index}")[:63]


def _volume_to_mapping(volume: VolumeMount | Mapping[str, Any], index: int) -> dict[str, Any]:
    if isinstance(volume, Mapping):
        return dict(volume)
    if not isinstance(volume, VolumeMount):
        raise TypeError(f"OpenSandbox volume entries must be mappings or VolumeMount instances, got {type(volume)!r}")
    if volume.is_efs:
        raise ValueError("OpenSandbox does not support provider-neutral EFS VolumeMount entries")
    if volume.host_path is None:
        raise ValueError("OpenSandbox VolumeMount entries require host_path")
    return {
        "name": _volume_mount_name(volume, index),
        "host": {"path": volume.host_path},
        "mount_path": volume.container_path,
        "read_only": volume.readonly,
    }


def _to_volumes(volumes: list[VolumeMount | Mapping[str, Any]]) -> list[Any]:
    _, _, _, _, Volume = _require_opensandbox_sdk()
    return [Volume(**_volume_to_mapping(volume, index)) for index, volume in enumerate(volumes)]


def _spec_volumes(spec: SandboxSpec) -> list[VolumeMount | Mapping[str, Any]] | None:
    volumes: list[VolumeMount | Mapping[str, Any]] = []
    volumes.extend(spec.volumes)
    provider_volumes = spec.provider_options.get(PROVIDER_OPTION_VOLUMES)
    if provider_volumes is not None:
        volumes.extend(provider_volumes)
    return volumes or None


def _provider_option_bool(provider_options: dict[str, Any], key: str) -> bool | None:
    value = provider_options.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"OpenSandbox provider option {key!r} must be a bool")
    return value


def _seconds_to_timedelta(seconds: int | float | None) -> timedelta | None:
    if seconds is None:
        return None
    return timedelta(seconds=float(seconds))


def _to_sandbox_status(state: Any) -> SandboxStatus:
    normalized = str(state or "").lower()
    if normalized in {"active", "ready", "running"}:
        return SandboxStatus.RUNNING
    if normalized in {"creating", "initializing", "pending", "starting"}:
        return SandboxStatus.STARTING
    if normalized in {"completed", "deleted", "exited", "stopped", "terminated"}:
        return SandboxStatus.STOPPED
    if normalized in {"crashed", "error", "failed", "unhealthy"}:
        return SandboxStatus.ERROR
    return SandboxStatus.UNKNOWN


@dataclass(frozen=True)
class OpenSandboxConnectionConfig:
    """OpenSandbox server connection settings."""

    domain: str | None = None
    api_key: str | None = None
    protocol: str | None = None
    use_server_proxy: bool | None = None
    exec_use_server_proxy: bool | None = None
    request_timeout_s: int | None = None
    connect_timeout_s: int | float | None = None

    def __post_init__(self) -> None:
        if self.connect_timeout_s is not None and self.connect_timeout_s <= 0:
            raise ValueError("connection.connect_timeout_s must be > 0")


@dataclass(frozen=True)
class OpenSandboxCreateConfig:
    """OpenSandbox create/reconnect retry settings."""

    request_timeout_s: int | None = None
    timeout_s: float | None = None
    retries: int = 2
    retry_delay_s: float = 5.0
    retry_max_delay_s: float = 60.0
    image_pull_policy: str | None = DEFAULT_IMAGE_PULL_POLICY
    skip_health_check: bool = False
    connect_attempt_timeout_s: float = 30.0
    connect_poll_s: float = 2.0

    def __post_init__(self) -> None:
        if self.image_pull_policy is not None:
            validate_image_pull_policy(self.image_pull_policy)
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("create.timeout_s must be > 0")
        if self.retries < 0:
            raise ValueError("create.retries must be >= 0")
        if self.retry_delay_s < 0:
            raise ValueError("create.retry_delay_s must be >= 0")
        if self.retry_max_delay_s < 0:
            raise ValueError("create.retry_max_delay_s must be >= 0")
        if self.connect_attempt_timeout_s <= 0:
            raise ValueError("create.connect_attempt_timeout_s must be > 0")
        if self.connect_poll_s <= 0:
            raise ValueError("create.connect_poll_s must be > 0")


@dataclass(frozen=True)
class OpenSandboxProbeConfig:
    """Post-create probe settings."""

    command: str | None = "printf nemo-gym-sandbox-ready"
    expected_stdout: str | None = "nemo-gym-sandbox-ready"
    timeout_s: int = 30
    deadline_s: float | None = None
    sample_count: int | None = None
    stable_count: int = 1
    stable_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.command is not None and self.timeout_s <= 0:
            raise ValueError("probe.timeout_s must be > 0")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("probe.deadline_s must be > 0")
        if self.sample_count is not None and self.sample_count < 1:
            raise ValueError("probe.sample_count must be >= 1")
        if self.stable_count < 1:
            raise ValueError("probe.stable_count must be >= 1")
        if self.stable_delay_s < 0:
            raise ValueError("probe.stable_delay_s must be >= 0")


@dataclass(frozen=True)
class OpenSandboxOperationConfig:
    """Retry and timeout settings for SDK operations after create."""

    retries: int = 3
    retry_delay_s: float = 1.0
    retry_max_delay_s: float = 15.0
    command_retries: int | None = None
    close_timeout_s: float | None = 30.0

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError("operations.retries must be >= 0")
        if self.retry_delay_s < 0:
            raise ValueError("operations.retry_delay_s must be >= 0")
        if self.retry_max_delay_s < 0:
            raise ValueError("operations.retry_max_delay_s must be >= 0")
        if self.command_retries is not None and self.command_retries < 0:
            raise ValueError("operations.command_retries must be >= 0")
        if self.close_timeout_s is not None and self.close_timeout_s <= 0:
            raise ValueError("operations.close_timeout_s must be > 0")


@dataclass(frozen=True)
class OpenSandboxPoolConfig:
    """OpenSandbox SDK pool and batch fanout settings."""

    concurrency: int = 4
    progress_timeout_s: float | None = None
    reconcile_interval_s: float = 0.1
    acquire_poll_interval_s: float = 0.1
    idle_timeout_s: float | None = None
    primary_lock_ttl_s: float | None = None

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("pool.concurrency must be >= 1")
        if self.progress_timeout_s is not None and self.progress_timeout_s <= 0:
            raise ValueError("pool.progress_timeout_s must be > 0")
        if self.reconcile_interval_s <= 0:
            raise ValueError("pool.reconcile_interval_s must be > 0")
        if self.acquire_poll_interval_s <= 0:
            raise ValueError("pool.acquire_poll_interval_s must be > 0")
        if self.idle_timeout_s is not None and self.idle_timeout_s <= 0:
            raise ValueError("pool.idle_timeout_s must be > 0")
        if self.primary_lock_ttl_s is not None and self.primary_lock_ttl_s <= 0:
            raise ValueError("pool.primary_lock_ttl_s must be > 0")


def _coerce_config(value: Any, config_cls: type[Any]) -> Any:
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, Mapping):
        return config_cls(**value)
    raise TypeError(f"{config_cls.__name__} must be a mapping or {config_cls.__name__} instance")


class OpenSandboxProvider:
    """Provider backed by the OpenSandbox SDK/server API.

    Batch allocations use the official OpenSandbox SDK client-side pool.
    """

    name = "opensandbox"

    def __init__(
        self,
        *,
        connection: OpenSandboxConnectionConfig | Mapping[str, Any] | None = None,
        create: OpenSandboxCreateConfig | Mapping[str, Any] | None = None,
        probe: OpenSandboxProbeConfig | Mapping[str, Any] | None = None,
        operations: OpenSandboxOperationConfig | Mapping[str, Any] | None = None,
        pool: OpenSandboxPoolConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = _coerce_config(connection, OpenSandboxConnectionConfig)
        self._create = _coerce_config(create, OpenSandboxCreateConfig)
        self._probe = _coerce_config(probe, OpenSandboxProbeConfig)
        self._operations = _coerce_config(operations, OpenSandboxOperationConfig)
        self._pool = _coerce_config(pool, OpenSandboxPoolConfig)

    def _with_default_image_pull_policy(self, spec: SandboxSpec) -> SandboxSpec:
        """Ensure SDK create requests carry the desired image pull policy."""
        if self._create.image_pull_policy is None:
            return spec

        extensions = dict(spec.extensions)
        image_pull_policy = extensions.get(IMAGE_PULL_POLICY_EXTENSION_KEY) or extensions.get(
            IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY
        )
        if image_pull_policy is None:
            image_pull_policy = self._create.image_pull_policy
        image_pull_policy = validate_image_pull_policy(image_pull_policy)
        extensions.setdefault(IMAGE_PULL_POLICY_EXTENSION_KEY, image_pull_policy)
        extensions.setdefault(IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY, image_pull_policy)
        return replace(spec, extensions=extensions)

    def _connection_config(
        self,
        request_timeout_s: int | float | None = None,
        *,
        use_server_proxy: bool | None = None,
    ) -> Any:
        _, ConnectionConfig, _, _, _ = _require_opensandbox_sdk()
        kwargs: dict[str, Any] = {}
        if self._connection.domain is not None:
            kwargs["domain"] = self._connection.domain
        if self._connection.api_key is not None:
            kwargs["api_key"] = self._connection.api_key
        if self._connection.protocol is not None:
            kwargs["protocol"] = self._connection.protocol
        if use_server_proxy is None:
            use_server_proxy = self._connection.use_server_proxy
        if use_server_proxy is not None:
            kwargs["use_server_proxy"] = use_server_proxy
        if request_timeout_s is None:
            request_timeout_s = self._connection.request_timeout_s
        if request_timeout_s is not None:
            kwargs["request_timeout"] = timedelta(seconds=request_timeout_s)
        return ConnectionConfig(**kwargs)

    def _exec_connection_config(self, request_timeout_s: int | float | None = None) -> Any:
        """Connection config for SDK handles that issue execd/filesystem calls.

        For clustered evaluations, exec traffic should normally use the
        OpenSandbox server proxy. That keeps clients off pod IP routing and lets
        the server resolve the sandbox backend for each request.
        """
        use_server_proxy = self._connection.use_server_proxy
        if self._connection.exec_use_server_proxy is not None:
            use_server_proxy = self._connection.exec_use_server_proxy
        return self._connection_config(
            request_timeout_s=request_timeout_s,
            use_server_proxy=use_server_proxy,
        )

    async def aclose(self) -> None:
        """Close provider-owned resources.

        The provider intentionally does not inject or own OpenSandbox SDK
        network clients. SDK handles are closed per sandbox in ``close``.
        """
        return None

    async def _await_sdk_call(
        self,
        awaitable: Any,
        *,
        operation: str,
        sandbox_id: str,
        timeout_s: float | None,
    ) -> Any:
        if timeout_s is None:
            return await awaitable

        try:
            return await asyncio.wait_for(awaitable, timeout=timeout_s)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Timed out during OpenSandbox {operation} after {timeout_s:g}s; sandbox_id={sandbox_id!r}"
            ) from e

    async def _await_sdk_operation(
        self,
        operation_factory: Callable[[], Awaitable[Any]],
        *,
        operation: str,
        sandbox_id: str,
        timeout_s: float | None,
        retries: int | None = None,
    ) -> Any:
        AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential = _require_tenacity()
        retry_count = self._operations.retries if retries is None else retries
        max_attempts = retry_count + 1

        def _before_sleep(retry_state: Any) -> None:
            _log_operation_retry(retry_state)

        retry_policy = AsyncRetrying(
            retry=retry_if_exception(_is_retryable_sdk_operation_error),
            stop=stop_after_attempt(max_attempts),
            wait=wait_random_exponential(
                multiplier=self._operations.retry_delay_s,
                max=self._operations.retry_max_delay_s,
            ),
            before_sleep=_before_sleep,
            reraise=True,
        )
        async for attempt in retry_policy:
            with attempt:
                return await self._await_sdk_call(
                    operation_factory(),
                    operation=operation,
                    sandbox_id=sandbox_id,
                    timeout_s=timeout_s,
                )

        raise RuntimeError("OpenSandbox SDK operation retry loop did not run")

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        if self._probe.command is None:
            return

        loop = asyncio.get_running_loop()
        deadline_s = self._probe.deadline_s or float(self._probe.timeout_s)
        deadline = loop.time() + deadline_s
        successful_probes = 0
        attempt_number = 0
        last_exception: BaseException | None = None

        while successful_probes < self._probe.stable_count:
            remaining_s = deadline - loop.time()
            if remaining_s <= 0:
                error = OpenSandboxCreateVerificationError(
                    "OpenSandbox sandbox failed create probe command before "
                    "the startup deadline; "
                    f"sandbox_id={handle.sandbox_id!r}, "
                    f"command={self._probe.command!r}, "
                    f"successful_probes={successful_probes}/{self._probe.stable_count}, "
                    f"attempts={attempt_number}, deadline_s={deadline_s:g}"
                )
                raise error from last_exception

            attempt_number += 1
            if self._probe.deadline_s is None:
                command_timeout_s = float(self._probe.timeout_s)
            else:
                command_timeout_s = min(float(self._probe.timeout_s), remaining_s)
            try:
                result = await asyncio.wait_for(
                    self._exec(
                        handle,
                        self._probe.command,
                        timeout_s=command_timeout_s,
                        user="root",
                    ),
                    timeout=command_timeout_s,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exception = e
                successful_probes = 0
                sleep_s = min(self._create.connect_poll_s, max(deadline - loop.time(), 0.0))
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                continue

            stdout = result.stdout or ""
            expected = self._probe.expected_stdout
            if result.return_code != 0 or (expected is not None and expected not in stdout):
                last_exception = OpenSandboxCreateVerificationError(
                    "OpenSandbox sandbox create probe command returned an "
                    f"unexpected result; sandbox_id={handle.sandbox_id!r}, "
                    f"return_code={result.return_code}, expected_stdout={expected!r}, "
                    f"stdout={stdout[:200]!r}, stderr={(result.stderr or '')[:200]!r}, "
                    f"probe={successful_probes + 1}/{self._probe.stable_count}"
                )
                successful_probes = 0
                sleep_s = min(self._create.connect_poll_s, max(deadline - loop.time(), 0.0))
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                continue

            successful_probes += 1
            if successful_probes < self._probe.stable_count and self._probe.stable_delay_s:
                await asyncio.sleep(self._probe.stable_delay_s)

    async def _verify_created_handles(
        self,
        handles: list[SandboxHandle],
    ) -> None:
        """Verify a batch of created handles with bounded probe concurrency."""
        if self._probe.command is None or not handles:
            return

        handles_to_probe = handles
        if self._probe.sample_count is not None and self._probe.sample_count < len(handles):
            sample_count = self._probe.sample_count
            if sample_count == 1:
                sampled_indices = [0]
            else:
                sampled_indices = [
                    round(index * (len(handles) - 1) / (sample_count - 1)) for index in range(sample_count)
                ]
            handles_to_probe = [handles[index] for index in sampled_indices]

        semaphore = asyncio.Semaphore(self._pool.concurrency)

        async def _verify_one(handle: SandboxHandle) -> None:
            async with semaphore:
                await self._verify_created_handle(handle)

        results = await asyncio.gather(
            *(_verify_one(handle) for handle in handles_to_probe),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise OpenSandboxCreateVerificationError(
                "One or more OpenSandbox sandboxes failed create probe "
                f"verification; failed={len(errors)}, total={len(handles)}"
            ) from errors[0]

    async def _cleanup_failed_create_handle(self, handle: SandboxHandle) -> None:
        try:
            await self.close(handle, delete=True)
        except Exception as e:
            LOGGER.warning(
                "Failed to clean up OpenSandbox sandbox after create probe failure; sandbox_id=%s; error=%r",
                handle.sandbox_id,
                e,
            )

    async def _connect_after_create(self, handle: SandboxHandle, spec: SandboxSpec) -> SandboxHandle:
        """Reconnect after SDK create so follow-up calls use a fresh SDK handle."""
        timeout_s = spec.ready_timeout_s
        if timeout_s is None:
            timeout_s = self._create.timeout_s
        if timeout_s is None:
            timeout_s = self._create.connect_attempt_timeout_s

        Sandbox, _, _, _, _ = _require_opensandbox_sdk()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout_s)
        last_exception: BaseException | None = None

        while True:
            remaining_s = deadline - loop.time()
            if remaining_s <= 0:
                error = OpenSandboxCreateTimeoutError(
                    "Timed out connecting to OpenSandbox sandbox after SDK create; "
                    f"sandbox_id={handle.sandbox_id!r}, timeout_s={timeout_s:g}"
                )
                raise error from last_exception

            attempt_timeout_s = min(self._create.connect_attempt_timeout_s, remaining_s)
            try:
                sandbox = await asyncio.wait_for(
                    Sandbox.connect(
                        handle.sandbox_id,
                        connection_config=self._exec_connection_config(request_timeout_s=attempt_timeout_s),
                        connect_timeout=timedelta(seconds=attempt_timeout_s),
                        skip_health_check=True,
                    ),
                    timeout=attempt_timeout_s,
                )
                return SandboxHandle(sandbox_id=str(sandbox.id), provider_name=self.name, raw=sandbox)
            except asyncio.CancelledError:
                raise
            except BaseException as e:
                last_exception = e
                if not _is_retryable_create_error(e):
                    raise
                sleep_s = min(self._create.connect_poll_s, max(deadline - loop.time(), 0.0))
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

    async def _create_once(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a sandbox through ``opensandbox.Sandbox.create``."""
        if spec.extensions.get("poolRef") and self._connection.use_server_proxy is False:
            raise ValueError(
                "OpenSandbox pooled creation requires "
                "use_server_proxy=True so SDK calls are routed through the "
                "server proxy and do not rely on stale cached pod endpoints."
            )

        Sandbox, _, _, _, _ = _require_opensandbox_sdk()

        kwargs: dict[str, Any] = {
            "env": spec.env,
            "metadata": spec.metadata,
            "resource": spec.resources,
            "extensions": spec.extensions,
            "connection_config": self._exec_connection_config(request_timeout_s=self._create.request_timeout_s),
        }
        if spec.image is not None:
            kwargs["image"] = spec.image
        snapshot_id = spec.provider_options.get(PROVIDER_OPTION_SNAPSHOT_ID)
        if snapshot_id is not None:
            kwargs["snapshot_id"] = snapshot_id
        if spec.timeout_s is not None:
            kwargs["timeout"] = timedelta(seconds=spec.timeout_s)
        if spec.ready_timeout_s is not None:
            kwargs["ready_timeout"] = timedelta(seconds=spec.ready_timeout_s)
        if spec.entrypoint is not None:
            kwargs["entrypoint"] = spec.entrypoint
        platform = spec.provider_options.get(PROVIDER_OPTION_PLATFORM)
        volumes = _spec_volumes(spec)
        if platform is not None:
            kwargs["platform"] = _to_platform_spec(platform)
        if volumes is not None:
            kwargs["volumes"] = _to_volumes(volumes)
        if self._create.skip_health_check:
            kwargs["skip_health_check"] = True
        else:
            skip_health_check = _provider_option_bool(spec.provider_options, PROVIDER_OPTION_SKIP_HEALTH_CHECK)
            if skip_health_check is not None:
                kwargs["skip_health_check"] = skip_health_check

        timeout_s = self._create.timeout_s
        if timeout_s is None and self._connection.request_timeout_s is not None:
            timeout_s = float(self._connection.request_timeout_s)

        sandbox_id: str | None = None
        sandbox: Any | None = None
        try:
            if timeout_s is None:
                sandbox = await Sandbox.create(**kwargs)
            else:
                sandbox = await asyncio.wait_for(
                    Sandbox.create(**kwargs),
                    timeout=timeout_s,
                )
            if sandbox is None:
                raise RuntimeError("OpenSandbox SDK create returned no sandbox handle")
            sandbox_id = str(sandbox.id)
        except TimeoutError as e:
            error = OpenSandboxCreateTimeoutError(
                "Timed out creating OpenSandbox sandbox after "
                f"{timeout_s:g}s; image={spec.image!r}, "
                f"poolRef={spec.extensions.get('poolRef')!r}, "
                f"ready_timeout_s={spec.ready_timeout_s!r}"
            )
            raise error from e
        if sandbox_id is None:
            raise RuntimeError("OpenSandbox SDK create returned no sandbox handle")
        created_handle = SandboxHandle(
            sandbox_id=sandbox_id,
            provider_name=self.name,
            raw=sandbox,
        )
        handle = created_handle
        try:
            if self._create.skip_health_check:
                handle = await self._connect_after_create(created_handle, spec)
            await self._verify_created_handle(handle)
        except Exception:
            await self._cleanup_failed_create_handle(created_handle)
            raise
        return handle

    async def _create_with_retries(
        self,
        spec: SandboxSpec,
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> SandboxHandle:
        AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential = _require_tenacity()
        retry_policy = AsyncRetrying(
            retry=retry_if_exception(_is_retryable_create_error),
            stop=stop_after_attempt(self._create.retries + 1),
            wait=wait_random_exponential(
                multiplier=self._create.retry_delay_s,
                max=self._create.retry_max_delay_s,
            ),
            before_sleep=_log_create_retry,
            reraise=True,
        )
        async for attempt in retry_policy:
            with attempt:
                if semaphore is None:
                    return await self._create_once(spec)
                async with semaphore:
                    return await self._create_once(spec)

        raise OpenSandboxBatchCreateError("OpenSandbox create retry loop did not run")

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create one sandbox through the configured OpenSandbox path."""
        spec = self._with_default_image_pull_policy(_normalize_spec(spec))
        return await self._create_with_retries(spec)

    async def _close_many(
        self,
        handles: list[SandboxHandle],
        *,
        delete: bool,
    ) -> list[Any]:
        semaphore = asyncio.Semaphore(self._pool.concurrency)

        async def _close_one(handle: SandboxHandle) -> Any:
            async with semaphore:
                return await self.close(handle, delete=delete)

        return list(
            await asyncio.gather(
                *(_close_one(handle) for handle in handles),
                return_exceptions=True,
            )
        )

    def _validate_sdk_pool_spec(self, spec: SandboxSpec) -> None:
        if spec.image is None:
            raise ValueError("OpenSandbox SDK pool requires SandboxSpec.image")
        if spec.provider_options.get(PROVIDER_OPTION_SNAPSHOT_ID) is not None:
            raise ValueError("OpenSandbox SDK pool does not support snapshot_id")

    def _to_pool_creation_spec(self, spec: SandboxSpec) -> Any:
        self._validate_sdk_pool_spec(spec)
        _, _, PoolCreationSpec, _ = _require_opensandbox_sdk_pool()
        volumes = _spec_volumes(spec)
        return PoolCreationSpec(
            image=spec.image,
            entrypoint=spec.entrypoint,
            resource=spec.resources or None,
            env=spec.env or None,
            metadata=spec.metadata or None,
            extensions=spec.extensions or None,
            platform=_to_platform_spec(spec.provider_options[PROVIDER_OPTION_PLATFORM])
            if PROVIDER_OPTION_PLATFORM in spec.provider_options
            else None,
            volumes=_to_volumes(volumes) if volumes is not None else None,
        )

    async def _wait_sdk_pool_idle(
        self,
        pool: Any,
        *,
        spec: SandboxSpec,
        requested: int,
        timeout_s: float,
        allow_partial: bool,
    ) -> int:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        last_progress_at = loop.time()
        last_idle = 0
        last_snapshot: Any = None

        while True:
            last_snapshot = await pool.snapshot()
            idle_count = int(getattr(last_snapshot, "idle_count", 0) or 0)
            if idle_count >= requested:
                return requested
            if idle_count > last_idle:
                last_idle = idle_count
                last_progress_at = loop.time()

            now = loop.time()
            progress_timeout_s = self._pool.progress_timeout_s
            if progress_timeout_s is not None and now - last_progress_at >= progress_timeout_s:
                if allow_partial and idle_count > 0:
                    return idle_count
                error = OpenSandboxCreateTimeoutError(
                    "Timed out waiting for OpenSandbox SDK pool warmup progress "
                    f"after {progress_timeout_s:g}s; requested={requested}, "
                    f"idle={idle_count}, snapshot={last_snapshot!r}"
                )
                raise error
            if now >= deadline:
                if allow_partial and idle_count > 0:
                    return idle_count
                error = OpenSandboxCreateTimeoutError(
                    "Timed out waiting for OpenSandbox SDK pool warmup after "
                    f"{timeout_s:g}s; requested={requested}, idle={idle_count}, "
                    f"snapshot={last_snapshot!r}"
                )
                raise error
            await asyncio.sleep(self._pool.acquire_poll_interval_s)

    async def _direct_exec_handle_for_acquired_sandbox(self, sandbox: Any, spec: SandboxSpec) -> SandboxHandle:
        handle = SandboxHandle(sandbox_id=str(sandbox.id), provider_name=self.name, raw=sandbox)
        if self._connection.exec_use_server_proxy is None and not self._create.skip_health_check:
            return handle
        return await self._connect_after_create(handle, spec)

    async def _create_batch_sdk_pool(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool,
    ) -> list[SandboxHandle]:
        AcquirePolicy, InMemoryAsyncPoolStateStore, _, SandboxPoolAsync = _require_opensandbox_sdk_pool()
        ready_timeout_s = float(
            spec.ready_timeout_s or self._create.timeout_s or self._connection.request_timeout_s or 300.0
        )
        idle_timeout_s = float(self._pool.idle_timeout_s or spec.timeout_s or max(ready_timeout_s * 2.0, 3600.0))
        primary_lock_ttl_s = float(self._pool.primary_lock_ttl_s or max(ready_timeout_s + 60.0, 60.0))
        pool_name = f"nemo-gym-{uuid4().hex[:12]}"
        skip_health_check = bool(
            self._create.skip_health_check
            or _provider_option_bool(spec.provider_options, PROVIDER_OPTION_SKIP_HEALTH_CHECK)
        )

        async def _warmup_preparer(sandbox: Any) -> None:
            if self._probe.command is None:
                return
            handle = await self._direct_exec_handle_for_acquired_sandbox(sandbox, spec)
            try:
                await self._verify_created_handle(handle)
            finally:
                if handle.raw is not sandbox:
                    try:
                        await self._await_sdk_call(
                            handle.raw.close(),
                            operation="close warmup direct handle",
                            sandbox_id=handle.sandbox_id,
                            timeout_s=self._operations.close_timeout_s,
                        )
                    except Exception as e:
                        LOGGER.warning(
                            "Failed to close temporary OpenSandbox direct exec handle for sandbox %r: %r",
                            handle.sandbox_id,
                            e,
                        )

        pool = SandboxPoolAsync(
            pool_name=pool_name,
            max_idle=count,
            warmup_concurrency=self._pool.concurrency,
            state_store=InMemoryAsyncPoolStateStore(),
            connection_config=self._connection_config(request_timeout_s=self._create.request_timeout_s),
            creation_spec=self._to_pool_creation_spec(spec),
            reconcile_interval=timedelta(seconds=self._pool.reconcile_interval_s),
            primary_lock_ttl=timedelta(seconds=primary_lock_ttl_s),
            acquire_ready_timeout=timedelta(seconds=ready_timeout_s),
            warmup_ready_timeout=timedelta(seconds=ready_timeout_s),
            warmup_sandbox_preparer=_warmup_preparer,
            acquire_skip_health_check=skip_health_check,
            warmup_skip_health_check=skip_health_check,
            idle_timeout=timedelta(seconds=idle_timeout_s),
        )
        handles: list[SandboxHandle] = []
        try:
            await pool.start()
            ready_count = await self._wait_sdk_pool_idle(
                pool,
                spec=spec,
                requested=count,
                timeout_s=ready_timeout_s,
                allow_partial=allow_partial,
            )
            await pool.resize(0)
            sandbox_timeout = _seconds_to_timedelta(spec.timeout_s)
            for index in range(ready_count):
                sandbox = await pool.acquire(
                    sandbox_timeout=sandbox_timeout,
                    policy=AcquirePolicy.FAIL_FAST,
                )
                handle = await self._direct_exec_handle_for_acquired_sandbox(sandbox, spec)
                handles.append(handle)
                LOGGER.info(
                    "Acquired OpenSandbox SDK pool sandbox %s/%s: %s",
                    index + 1,
                    ready_count,
                    sandbox.id,
                )
            return handles
        except Exception:
            await self._close_many(handles, delete=True)
            raise
        finally:
            try:
                await pool.shutdown(graceful=False)
            finally:
                await pool.release_all_idle()

    async def _create_batch_sdk(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        """Create several sandboxes through the OpenSandbox SDK pool."""
        if count < 1:
            raise ValueError("count must be >= 1")
        return await self._create_batch_sdk_pool(
            spec,
            count,
            allow_partial=allow_partial,
        )

    async def create_batch(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        """Create several equivalent OpenSandbox sandboxes."""
        if count < 1:
            raise ValueError("count must be >= 1")
        spec = self._with_default_image_pull_policy(_normalize_spec(spec))
        return await self._create_batch_sdk(
            spec,
            count,
            allow_partial=allow_partial,
        )

    def handle_reference(self, handle: SandboxHandle) -> dict[str, Any]:
        """Build a loop-neutral reference for a sandbox handle.

        OpenSandbox SDK handles are bound to the event loop where they were
        created. Prewarmed handles may cross from a FastAPI prewarm request into
        a thread-pool runner, so only pass a serializable reference across that
        boundary and re-materialize SDK adapters in the consuming event loop.
        """
        return {
            "kind": "sandbox_id",
            "provider": self.name,
            "sandbox_id": handle.sandbox_id,
        }

    async def materialize_handle(self, reference: dict[str, Any]) -> SandboxHandle:
        """Create a loop-local handle from ``handle_reference`` output."""
        kind = reference.get("kind")
        if kind == "sandbox_id":
            return await self.connect(str(reference["sandbox_id"]))
        raise ValueError(f"Unsupported OpenSandbox handle reference kind: {kind!r}")

    async def connect(self, sandbox_id: str) -> SandboxHandle:
        """Connect to an existing OpenSandbox sandbox."""
        Sandbox, _, _, _, _ = _require_opensandbox_sdk()
        kwargs: dict[str, Any] = {
            "connection_config": self._exec_connection_config(),
        }
        if self._connection.connect_timeout_s is not None:
            kwargs["connect_timeout"] = timedelta(seconds=self._connection.connect_timeout_s)
        sandbox = await Sandbox.connect(sandbox_id, **kwargs)
        return SandboxHandle(sandbox_id=str(sandbox.id), provider_name=self.name, raw=sandbox)

    async def attach(self, sandbox_id: str) -> SandboxHandle:
        """Attach to an existing OpenSandbox sandbox."""
        return await self.connect(sandbox_id)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Return the current OpenSandbox lifecycle status."""
        get_info = getattr(handle.raw, "get_info", None)
        if get_info is None:
            return SandboxStatus.UNKNOWN
        info = await self._await_sdk_operation(
            get_info,
            operation="get_info",
            sandbox_id=handle.sandbox_id,
            timeout_s=float(self._connection.request_timeout_s)
            if self._connection.request_timeout_s is not None
            else None,
        )
        raw_status = getattr(info, "status", None)
        return _to_sandbox_status(getattr(raw_status, "state", None) if raw_status is not None else None)

    async def container_ip(self, handle: SandboxHandle) -> str | None:
        """Return the container IP when the OpenSandbox SDK handle exposes it."""
        for attr in ("container_ip", "container_ip_address", "pod_ip", "ip"):
            value = getattr(handle.raw, attr, None)
            if value:
                return str(value)
        return None

    def _command_retry_count(self) -> int:
        return (
            self._operations.retries if self._operations.command_retries is None else self._operations.command_retries
        )

    async def _exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
        retries: int | None = None,
    ) -> SandboxExecResult:
        """Run a command inside an OpenSandbox sandbox."""
        _, _, RunCommandOpts, _, _ = _require_opensandbox_sdk()

        opts_kwargs: dict[str, Any] = {}
        if cwd is not None:
            opts_kwargs["working_directory"] = cwd
        if env is not None:
            opts_kwargs["envs"] = env
        if timeout_s is not None:
            opts_kwargs["timeout"] = timedelta(seconds=timeout_s)

        effective_command = command
        if isinstance(user, int):
            opts_kwargs["uid"] = user
        elif isinstance(user, str) and user != "root":
            effective_command = f"su -s /bin/sh -c {shlex.quote(command)} {shlex.quote(user)}"

        sdk_timeout_s = (
            float(timeout_s) + 60.0
            if timeout_s is not None
            else (
                float(self._connection.request_timeout_s) if self._connection.request_timeout_s is not None else None
            )
        )
        effective_retries = self._command_retry_count() if retries is None else retries
        execution = await self._await_sdk_operation(
            lambda: handle.raw.commands.run(effective_command, opts=RunCommandOpts(**opts_kwargs)),
            operation="command run",
            sandbox_id=handle.sandbox_id,
            timeout_s=sdk_timeout_s,
            retries=effective_retries,
        )
        stdout = "\n".join(msg.text for msg in execution.logs.stdout) or None
        stderr_parts = [msg.text for msg in execution.logs.stderr]
        if execution.error is not None:
            stderr_parts.append(f"{execution.error.name}: {execution.error.value}")
        stderr = "\n".join(stderr_parts) or None
        error_type = None
        if execution.exit_code is not None:
            return_code = execution.exit_code
        elif execution.error is not None:
            return_code = 125
            error_type = "sandbox"
        else:
            return_code = 0

        return SandboxExecResult(stdout=stdout, stderr=stderr, return_code=return_code, error_type=error_type)

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
        """Run a command inside an OpenSandbox sandbox."""
        return await self._exec(
            handle,
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            user=user,
            retries=self._command_retry_count(),
        )

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        """Write one file into an OpenSandbox sandbox."""
        await self._await_sdk_operation(
            lambda: handle.raw.files.write_file(target_path, data),
            operation=f"write_file({target_path})",
            sandbox_id=handle.sandbox_id,
            timeout_s=float(self._connection.request_timeout_s)
            if self._connection.request_timeout_s is not None
            else None,
        )

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        """Read one file from an OpenSandbox sandbox."""
        return await self._await_sdk_operation(
            lambda: handle.raw.files.read_bytes(source_path),
            operation=f"read_file({source_path})",
            sandbox_id=handle.sandbox_id,
            timeout_s=float(self._connection.request_timeout_s)
            if self._connection.request_timeout_s is not None
            else None,
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one local file into an OpenSandbox sandbox."""
        await self.write_file(handle, target_path, source_path.read_bytes())

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one file from an OpenSandbox sandbox."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(await self.read_file(handle, source_path))

    async def close(self, handle: SandboxHandle, *, delete: bool) -> None:
        """Close local SDK resources and optionally terminate the sandbox."""
        kill_error: Exception | None = None
        if delete:
            try:
                await self._await_sdk_operation(
                    lambda: handle.raw.kill(),
                    operation="kill",
                    sandbox_id=handle.sandbox_id,
                    timeout_s=self._operations.close_timeout_s,
                )
            except Exception as e:
                if not _is_missing_sandbox_delete_error(e):
                    kill_error = e
                else:
                    LOGGER.info(
                        "OpenSandbox sandbox %r was already deleted during close",
                        handle.sandbox_id,
                    )

        close_error: Exception | None = None
        try:
            await self._await_sdk_call(
                handle.raw.close(),
                operation="close",
                sandbox_id=handle.sandbox_id,
                timeout_s=self._operations.close_timeout_s,
            )
        except Exception as e:
            close_error = e
            LOGGER.warning(
                "Timed out or failed while closing local OpenSandbox SDK handle for sandbox %r: %r",
                handle.sandbox_id,
                e,
            )

        if kill_error is not None:
            if close_error is not None:
                raise RuntimeError(
                    "Failed to delete and close OpenSandbox sandbox "
                    f"{handle.sandbox_id!r}: delete_error={kill_error!r}, "
                    f"close_error={close_error!r}"
                ) from kill_error
            raise kill_error
        if close_error is not None:
            if delete:
                return
            raise close_error
