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
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)


LOGGER = logging.getLogger(__name__)


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


def _require_tenacity() -> tuple[Any, Any, Any, Any]:
    try:
        from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "tenacity is required for OpenSandbox retry handling. Install nemo-gym[sandbox] before using "
            "env.sandbox.provider.name=opensandbox."
        ) from e

    return AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential


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


def _is_retryable_sdk_operation_error(exception: BaseException, seen: set[int] | None = None) -> bool:
    """Return whether an SDK operation can be retried."""
    if isinstance(exception, TimeoutError):
        return False
    seen = set() if seen is None else seen
    exception_id = id(exception)
    if exception_id in seen:
        return False
    seen.add(exception_id)
    if isinstance(exception, (ConnectionError, OSError)):
        return True
    if _is_retryable_create_error(exception):
        return True
    cause = exception.__cause__
    if isinstance(cause, BaseException):
        return _is_retryable_sdk_operation_error(cause, seen)
    return False


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


def _string_map(values: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items()}


def _resource_quantity(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _resource_map(resources: SandboxResources) -> dict[str, str]:
    values: dict[str, str] = {}
    if resources.cpu is not None:
        values["cpu"] = _resource_quantity(resources.cpu)
    if resources.memory_mib is not None:
        values["memory"] = f"{resources.memory_mib}Mi"
    if resources.disk_gib is not None:
        values["ephemeral-storage"] = f"{resources.disk_gib}Gi"
    if resources.gpu is not None:
        values["gpu"] = str(resources.gpu)
    if resources.gpu_type is not None:
        values["gpu_type"] = resources.gpu_type
    return values


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
    )


def _to_platform_spec(platform: dict[str, Any]) -> Any:
    _, _, _, PlatformSpec, _ = _require_opensandbox_sdk()
    return PlatformSpec(**platform)


def _to_volumes(volumes: list[Mapping[str, Any]]) -> list[Any]:
    _, _, _, _, Volume = _require_opensandbox_sdk()
    return [Volume(**dict(volume)) for volume in volumes]


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
    request_timeout_s: int | None = None
    use_server_proxy: bool = False


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
    stable_count: int = 1
    stable_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.command is not None and self.timeout_s <= 0:
            raise ValueError("probe.timeout_s must be > 0")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("probe.deadline_s must be > 0")
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
    command_retries: int = 0
    close_timeout_s: float | None = 30.0

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError("operations.retries must be >= 0")
        if self.retry_delay_s < 0:
            raise ValueError("operations.retry_delay_s must be >= 0")
        if self.retry_max_delay_s < 0:
            raise ValueError("operations.retry_max_delay_s must be >= 0")
        if self.command_retries < 0:
            raise ValueError("operations.command_retries must be >= 0")
        if self.close_timeout_s is not None and self.close_timeout_s <= 0:
            raise ValueError("operations.close_timeout_s must be > 0")


def _coerce_config(value: Any, config_cls: type[Any]) -> Any:
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, Mapping):
        return config_cls(**value)
    raise TypeError(f"{config_cls.__name__} must be a mapping or {config_cls.__name__} instance")


@dataclass(frozen=True)
class OpenSandboxProviderOptions:
    """Recognized per-sandbox create options read from ``SandboxSpec.provider_options``.

    ``platform`` and ``volumes`` entries are passed through to the OpenSandbox SDK,
    so their inner fields are validated by the SDK rather than here.
    """

    platform: Mapping[str, Any] | None = None
    snapshot_id: str | None = None
    volumes: tuple[Mapping[str, Any], ...] = ()
    skip_health_check: bool | None = None
    extensions: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any] | None) -> "OpenSandboxProviderOptions":
        if options is None:
            return cls()
        if not isinstance(options, Mapping):
            raise TypeError("OpenSandbox provider_options must be a mapping")

        allowed = set(cls.__dataclass_fields__)
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(
                f"Unknown OpenSandbox provider option(s): {', '.join(sorted(unknown))}. "
                f"Supported: {', '.join(sorted(allowed))}"
            )

        platform = options.get("platform")
        if platform is not None and not isinstance(platform, Mapping):
            raise TypeError("OpenSandbox provider option 'platform' must be a mapping")
        snapshot_id = options.get("snapshot_id")
        if snapshot_id is not None and not isinstance(snapshot_id, str):
            raise TypeError("OpenSandbox provider option 'snapshot_id' must be a string")
        volumes = options.get("volumes") or ()
        if not isinstance(volumes, (list, tuple)) or not all(isinstance(volume, Mapping) for volume in volumes):
            raise TypeError("OpenSandbox provider option 'volumes' must be a list of mappings")
        skip_health_check = options.get("skip_health_check")
        if skip_health_check is not None and not isinstance(skip_health_check, bool):
            raise TypeError("OpenSandbox provider option 'skip_health_check' must be a bool")
        extensions = options.get("extensions", {})
        if not isinstance(extensions, Mapping):
            raise TypeError("OpenSandbox provider option 'extensions' must be a mapping")

        return cls(
            platform=dict(platform) if platform is not None else None,
            snapshot_id=snapshot_id,
            volumes=tuple(dict(volume) for volume in volumes),
            skip_health_check=skip_health_check,
            extensions=_string_map(dict(extensions)),
        )


class OpenSandboxProvider:
    """Provider backed by the OpenSandbox SDK/server API."""

    name = "opensandbox"

    def __init__(
        self,
        *,
        connection: OpenSandboxConnectionConfig | Mapping[str, Any] | None = None,
        create: OpenSandboxCreateConfig | Mapping[str, Any] | None = None,
        probe: OpenSandboxProbeConfig | Mapping[str, Any] | None = None,
        operations: OpenSandboxOperationConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = _coerce_config(connection, OpenSandboxConnectionConfig)
        self._create = _coerce_config(create, OpenSandboxCreateConfig)
        self._probe = _coerce_config(probe, OpenSandboxProbeConfig)
        self._operations = _coerce_config(operations, OpenSandboxOperationConfig)

    def _resolve_extensions(self, extensions: Mapping[str, str]) -> dict[str, str]:
        """Add the configured default image pull policy to SDK create extensions."""
        resolved = dict(extensions)
        if self._create.image_pull_policy is None:
            return resolved

        image_pull_policy = (
            resolved.get(IMAGE_PULL_POLICY_EXTENSION_KEY)
            or resolved.get(IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY)
            or self._create.image_pull_policy
        )
        image_pull_policy = validate_image_pull_policy(image_pull_policy)
        resolved.setdefault(IMAGE_PULL_POLICY_EXTENSION_KEY, image_pull_policy)
        resolved.setdefault(IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY, image_pull_policy)
        return resolved

    def _connection_config(
        self,
        request_timeout_s: int | float | None = None,
    ) -> Any:
        _, ConnectionConfig, _, _, _ = _require_opensandbox_sdk()
        kwargs: dict[str, Any] = {}
        if self._connection.domain is not None:
            kwargs["domain"] = self._connection.domain
        if self._connection.api_key is not None:
            kwargs["api_key"] = self._connection.api_key
        if self._connection.protocol is not None:
            kwargs["protocol"] = self._connection.protocol
        if request_timeout_s is None:
            request_timeout_s = self._connection.request_timeout_s
        if request_timeout_s is not None:
            kwargs["request_timeout"] = timedelta(seconds=request_timeout_s)
        if self._connection.use_server_proxy:
            kwargs["use_server_proxy"] = True
        return ConnectionConfig(**kwargs)

    async def aclose(self) -> None:
        """Close provider-owned resources."""
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

    async def _cleanup_failed_create_handle(self, handle: SandboxHandle) -> None:
        try:
            await self.close(handle)
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
                        connection_config=self._connection_config(request_timeout_s=attempt_timeout_s),
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
        Sandbox, _, _, _, _ = _require_opensandbox_sdk()
        options = OpenSandboxProviderOptions.from_mapping(spec.provider_options)

        kwargs: dict[str, Any] = {
            "env": spec.env,
            "metadata": spec.metadata,
            "resource": _resource_map(spec.resources),
            "extensions": self._resolve_extensions(options.extensions),
            "connection_config": self._connection_config(request_timeout_s=self._create.request_timeout_s),
        }
        if spec.image is not None:
            kwargs["image"] = spec.image
        if options.snapshot_id is not None:
            kwargs["snapshot_id"] = options.snapshot_id
        if spec.ttl_s is not None:
            kwargs["timeout"] = timedelta(seconds=spec.ttl_s)
        if spec.ready_timeout_s is not None:
            kwargs["ready_timeout"] = timedelta(seconds=spec.ready_timeout_s)
        if spec.entrypoint is not None:
            kwargs["entrypoint"] = spec.entrypoint
        if options.platform is not None:
            kwargs["platform"] = _to_platform_spec(options.platform)
        if options.volumes:
            kwargs["volumes"] = _to_volumes(list(options.volumes))
        if self._create.skip_health_check:
            kwargs["skip_health_check"] = True
        elif options.skip_health_check is not None:
            kwargs["skip_health_check"] = options.skip_health_check

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
                return await self._create_once(spec)

        raise OpenSandboxCreateError("OpenSandbox create retry loop did not run")

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create one sandbox through the configured OpenSandbox path."""
        return await self._create_with_retries(_normalize_spec(spec))

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

    def _command_retry_count(self) -> int:
        return self._operations.command_retries

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

    async def _write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        """Write one file into an OpenSandbox sandbox."""
        await self._await_sdk_operation(
            lambda: handle.raw.files.write_file(target_path, data),
            operation=f"write_file({target_path})",
            sandbox_id=handle.sandbox_id,
            timeout_s=float(self._connection.request_timeout_s)
            if self._connection.request_timeout_s is not None
            else None,
        )

    async def _read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
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
        await self._write_file(handle, target_path, source_path.read_bytes())

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one file from an OpenSandbox sandbox."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(await self._read_file(handle, source_path))

    async def close(self, handle: SandboxHandle) -> None:
        """Terminate the sandbox and close local SDK resources."""
        stop_error: Exception | None = None
        try:
            await self._await_sdk_operation(
                lambda: handle.raw.kill(),
                operation="kill",
                sandbox_id=handle.sandbox_id,
                timeout_s=self._operations.close_timeout_s,
            )
        except Exception as e:
            if not _is_missing_sandbox_delete_error(e):
                stop_error = e
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

        if stop_error is not None:
            if close_error is not None:
                raise RuntimeError(
                    "Failed to stop and close OpenSandbox sandbox "
                    f"{handle.sandbox_id!r}: stop_error={stop_error!r}, "
                    f"close_error={close_error!r}"
                ) from stop_error
            raise stop_error
        if close_error is not None:
            return
