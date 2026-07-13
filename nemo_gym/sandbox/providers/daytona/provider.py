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

"""Daytona provider implementation."""

import asyncio
import logging
import math
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
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
DAYTONA_COMMAND_ERROR_PREFIX = "Failed to execute command:"
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_MARKERS = (
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
    "rate limit",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
)
STATUS_CODE_RE = re.compile(r"(?:status code|http)\D+(\d{3})", re.IGNORECASE)
DAYTONA_EXTENSION_PREFIX = "daytona."
PROVIDER_OPTION_EXTENSIONS = "extensions"
PROVIDER_OPTION_SNAPSHOT_ID = "snapshot_id"
PROVIDER_OPTION_VOLUMES = "volumes"
SANDBOX_ID_LABEL = "sandbox_id"


class DaytonaCreateError(SandboxCreateError):
    """Raised when Daytona cannot create a sandbox."""


class DaytonaCreateTimeoutError(DaytonaCreateError):
    """Raised when Daytona sandbox creation exceeds the client timeout."""


class DaytonaCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a newly-created Daytona sandbox cannot execute a probe command."""


def _require_daytona_sdk() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from daytona import (
            AsyncDaytona,
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            DaytonaConfig,
            Resources,
            VolumeMount,
        )
    except ImportError as e:
        raise ModuleNotFoundError(
            "Daytona SDK is required for the daytona sandbox provider. "
            "Install nemo-gym[sandbox] before using env.sandbox.provider.name=daytona."
        ) from e

    return (
        AsyncDaytona,
        DaytonaConfig,
        CreateSandboxFromImageParams,
        CreateSandboxFromSnapshotParams,
        Resources,
        VolumeMount,
    )


def _require_tenacity() -> tuple[Any, Any, Any, Any]:
    try:
        from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "tenacity is required for Daytona retry handling. Install nemo-gym[sandbox] before using "
            "env.sandbox.provider.name=daytona."
        ) from e

    return AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential


def _daytona_error_types() -> dict[str, type[BaseException]]:
    try:
        from daytona import (
            DaytonaAuthenticationError,
            DaytonaAuthorizationError,
            DaytonaConflictError,
            DaytonaConnectionError,
            DaytonaError,
            DaytonaNotFoundError,
            DaytonaRateLimitError,
            DaytonaTimeoutError,
            DaytonaValidationError,
        )
    except ImportError:
        return {}

    return {
        "authentication": DaytonaAuthenticationError,
        "authorization": DaytonaAuthorizationError,
        "conflict": DaytonaConflictError,
        "connection": DaytonaConnectionError,
        "error": DaytonaError,
        "not_found": DaytonaNotFoundError,
        "rate_limit": DaytonaRateLimitError,
        "timeout": DaytonaTimeoutError,
        "validation": DaytonaValidationError,
    }


def _exception_status_code(exception: BaseException) -> int | None:
    status_code = getattr(exception, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    match = STATUS_CODE_RE.search(str(exception))
    if match is None:
        return None
    return int(match.group(1))


def _has_retryable_error_marker(exception: BaseException) -> bool:
    message = str(exception).lower()
    return any(marker in message for marker in RETRYABLE_ERROR_MARKERS)


def _is_daytona_command_exec_error(exception: BaseException) -> bool:
    error_types = _daytona_error_types()
    if not error_types or not isinstance(exception, error_types["error"]):
        return False
    return str(exception).strip().startswith(DAYTONA_COMMAND_ERROR_PREFIX)


def _is_empty_daytona_command_exec_error(exception: BaseException) -> bool:
    return _is_daytona_command_exec_error(exception) and str(exception).strip() == DAYTONA_COMMAND_ERROR_PREFIX


def _is_retryable_create_error(exception: BaseException) -> bool:
    if isinstance(exception, SandboxCreateVerificationError):
        return True
    if isinstance(exception, DaytonaCreateTimeoutError):
        return False
    if isinstance(exception, SandboxCreateError):
        return True
    if isinstance(exception, TimeoutError):
        return False
    if isinstance(exception, (ConnectionError, OSError)):
        return True

    error_types = _daytona_error_types()
    if error_types:
        non_retryable_types = (
            error_types["authentication"],
            error_types["authorization"],
            error_types["not_found"],
            error_types["timeout"],
            error_types["validation"],
        )
        retryable_types = (
            error_types["conflict"],
            error_types["connection"],
            error_types["rate_limit"],
        )
        if isinstance(exception, non_retryable_types):
            return False
        if isinstance(exception, retryable_types):
            return True
        if isinstance(exception, error_types["error"]):
            status_code = _exception_status_code(exception)
            if status_code in RETRYABLE_HTTP_STATUS_CODES:
                return True
            if status_code is not None and status_code < 500:
                return False

    status_code = _exception_status_code(exception)
    if status_code in RETRYABLE_HTTP_STATUS_CODES:
        return True
    if status_code is not None and status_code < 500:
        return False
    return _has_retryable_error_marker(exception)


def _is_retryable_operation_error(exception: BaseException) -> bool:
    if isinstance(exception, TimeoutError):
        return False
    if _is_empty_daytona_command_exec_error(exception):
        return True
    cause = exception.__cause__
    if isinstance(cause, BaseException) and _is_retryable_operation_error(cause):
        return True
    if isinstance(exception, (ConnectionError, OSError)):
        return True

    error_types = _daytona_error_types()
    if error_types:
        non_retryable_types = (
            error_types["authentication"],
            error_types["authorization"],
            error_types["not_found"],
            error_types["timeout"],
            error_types["validation"],
        )
        retryable_types = (
            error_types["conflict"],
            error_types["connection"],
            error_types["rate_limit"],
        )
        if isinstance(exception, non_retryable_types):
            return False
        if isinstance(exception, retryable_types):
            return True
    return _is_retryable_create_error(exception)


def _is_missing_sandbox_delete_error(exception: BaseException) -> bool:
    if _exception_status_code(exception) == 404:
        return True
    message = str(exception).lower()
    return "sandbox" in message and "not found" in message


def _daytona_exec_error_message(exception: BaseException) -> str:
    message = str(exception).strip()
    if _is_empty_daytona_command_exec_error(exception):
        message = f"{DAYTONA_COMMAND_ERROR_PREFIX} HTTP request timed out before Daytona returned a response"

    details = [message]
    status_code = _exception_status_code(exception)
    if status_code is not None:
        details.append(f"status_code={status_code}")
    error_code = getattr(exception, "error_code", None)
    if error_code:
        details.append(f"error_code={error_code}")
    return "; ".join(details)


def _daytona_exec_error_result(exception: BaseException) -> SandboxExecResult:
    message = _daytona_exec_error_message(exception)
    is_timeout = "timeout" in message.lower() or "timed out" in message.lower()
    error_type = "timeout" if is_timeout else "sandbox"
    return_code = 124 if is_timeout else 125
    return SandboxExecResult(
        stdout=None,
        stderr=(
            f"Daytona command execution failed before a process exit code was returned: {message}. "
            "The command may not have run."
        ),
        return_code=return_code,
        error_type=error_type,
    )


def _log_create_retry(retry_state: Any) -> None:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    sleep_s = retry_state.next_action.sleep if retry_state.next_action else None
    LOGGER.warning(
        "Retrying Daytona sandbox create after attempt %s; next_sleep_s=%s; error=%r",
        retry_state.attempt_number,
        sleep_s,
        exception,
    )


def _log_operation_retry(retry_state: Any) -> None:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    sleep_s = retry_state.next_action.sleep if retry_state.next_action else None
    LOGGER.warning(
        "Retrying Daytona operation after attempt %s; next_sleep_s=%s; error=%r",
        retry_state.attempt_number,
        sleep_s,
        exception,
    )


class DaytonaConfigBase:
    """Shared dataclass construction for Daytona provider config blocks."""

    @classmethod
    def from_mapping(cls, value: Any) -> Any:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.__name__} must be a mapping or {cls.__name__} instance")
        field_names = {field.name for field in fields(cls) if field.init}
        unsupported_keys = sorted(str(key) for key in value if key not in field_names)
        if unsupported_keys:
            raise ValueError(f"Unsupported Daytona {cls.__name__} settings: {', '.join(unsupported_keys)}")
        return cls(**{key: val for key, val in value.items() if key in field_names})


def _supplied_config_keys(value: Any, config_cls: type[Any]) -> set[str]:
    if not isinstance(value, Mapping):
        return set()
    field_names = {field.name for field in fields(config_cls) if field.init}
    return {str(key) for key in value if key in field_names}


def _string_map(values: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items()}


@dataclass(frozen=True)
class DaytonaProviderOptions:
    """Recognized per-sandbox create options read from ``SandboxSpec.provider_options``."""

    extensions: Mapping[str, str] = field(default_factory=dict)
    snapshot_id: str | None = None
    volumes: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any] | None) -> "DaytonaProviderOptions":
        if options is None:
            return cls()
        if not isinstance(options, Mapping):
            raise TypeError("Daytona provider_options must be a mapping")

        allowed = set(cls.__dataclass_fields__)
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(
                f"Unknown Daytona provider option(s): {', '.join(sorted(unknown))}. "
                f"Supported: {', '.join(sorted(allowed))}"
            )

        extensions = options.get(PROVIDER_OPTION_EXTENSIONS, {})
        if not isinstance(extensions, Mapping):
            raise TypeError("Daytona provider option 'extensions' must be a mapping")
        snapshot_id = options.get(PROVIDER_OPTION_SNAPSHOT_ID)
        if snapshot_id is not None and not isinstance(snapshot_id, str):
            raise TypeError("Daytona provider option 'snapshot_id' must be a string")
        volumes = options.get(PROVIDER_OPTION_VOLUMES)
        if volumes is None:
            volumes = ()
        if not isinstance(volumes, (list, tuple)) or not all(isinstance(volume, Mapping) for volume in volumes):
            raise TypeError("Daytona provider option 'volumes' must be a list of mappings")

        return cls(
            extensions=_string_map(dict(extensions)),
            snapshot_id=snapshot_id,
            volumes=tuple(dict(volume) for volume in volumes),
        )


def _provider_options(spec: SandboxSpec) -> DaytonaProviderOptions:
    return DaytonaProviderOptions.from_mapping(spec.provider_options)


def _normalize_spec(spec: SandboxSpec) -> SandboxSpec:
    _provider_options(spec)
    return replace(
        spec,
        env=_string_map(spec.env),
        metadata=_string_map(spec.metadata),
        provider_options={} if spec.provider_options is None else dict(spec.provider_options),
    )


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _coerce_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return int(str(value).strip())


def _quantity_to_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"resources.{name} must be an integer-like quantity")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"resources.{name} must be an integer-like quantity")
        return int(value)

    text = str(value).strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([A-Za-z]*)", text)
    if match is None:
        raise ValueError(f"resources.{name} must be an integer-like quantity")
    number = float(match.group(1))
    unit = match.group(2).lower()
    if name == "cpu" and unit == "m":
        return max(1, math.ceil(number / 1000))
    if unit in {"mi", "mib"}:
        return math.ceil(number / 1024)
    if unit in {"", "g", "gb", "gi", "gib"}:
        return math.ceil(number)
    raise ValueError(f"resources.{name} uses unsupported unit {unit!r}")


def _spec_extensions(spec: SandboxSpec) -> dict[str, str]:
    return dict(_provider_options(spec).extensions)


def _extension_value(spec: SandboxSpec, key: str) -> str | None:
    return _spec_extensions(spec).get(f"{DAYTONA_EXTENSION_PREFIX}{key}")


def _configured_or_extension(spec: SandboxSpec, key: str, configured: Any) -> Any:
    value = _extension_value(spec, key)
    return configured if value is None else value


def _resources_requested(resources: SandboxResources | Mapping[str, Any]) -> bool:
    if isinstance(resources, SandboxResources):
        return any(getattr(resources, field.name) is not None for field in fields(SandboxResources))
    return bool(resources)


def _mib_to_gib(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"resources.{name} must be an integer-like quantity")
    return math.ceil(float(value) / 1024)


def _to_resources(resources: SandboxResources | Mapping[str, Any]) -> Any | None:
    _, _, _, _, Resources, _ = _require_daytona_sdk()
    kwargs: dict[str, int] = {}
    if isinstance(resources, SandboxResources):
        if resources.cpu is not None:
            kwargs["cpu"] = math.ceil(resources.cpu)
        if resources.memory_mib is not None:
            kwargs["memory"] = _mib_to_gib("memory_mib", resources.memory_mib)
        if resources.disk_gib is not None:
            kwargs["disk"] = resources.disk_gib
        if resources.gpu is not None:
            kwargs["gpu"] = resources.gpu
        if resources.gpu_type is not None:
            raise ValueError("Daytona resource overrides do not support gpu_type")
    else:
        if "cpu" in resources:
            kwargs["cpu"] = _quantity_to_int("cpu", resources["cpu"])
        if "memory" in resources:
            kwargs["memory"] = _quantity_to_int("memory", resources["memory"])
        if "memory_mib" in resources:
            kwargs["memory"] = _mib_to_gib("memory_mib", resources["memory_mib"])
        disk_value = resources.get(
            "disk",
            resources.get("disk_gib", resources.get("ephemeral-storage", resources.get("ephemeral_storage"))),
        )
        if disk_value is not None:
            kwargs["disk"] = _quantity_to_int("disk", disk_value)
        if "gpu" in resources:
            kwargs["gpu"] = _quantity_to_int("gpu", resources["gpu"])
        if resources.get("gpu_type") is not None:
            raise ValueError("Daytona resource overrides do not support gpu_type")
    if not kwargs:
        return None
    return Resources(**kwargs)


def _to_volume_mounts(volumes: list[dict[str, Any]]) -> list[Any]:
    _, _, _, _, _, VolumeMount = _require_daytona_sdk()
    return [VolumeMount(**volume) for volume in volumes]


def _spec_volumes(spec: SandboxSpec) -> list[dict[str, Any]] | None:
    volumes = _provider_options(spec).volumes
    if not volumes:
        return None
    return [dict(volume) for volume in volumes]


def _spec_snapshot_id(spec: SandboxSpec) -> str | None:
    return _provider_options(spec).snapshot_id


def _to_sandbox_status(value: Any) -> SandboxStatus:
    normalized = str(value or "").lower()
    if normalized in {"active", "ready", "running", "started"}:
        return SandboxStatus.RUNNING
    if normalized in {"creating", "initializing", "pending", "starting"}:
        return SandboxStatus.STARTING
    if normalized in {"archived", "completed", "deleted", "exited", "stopped", "terminated"}:
        return SandboxStatus.STOPPED
    if normalized in {"crashed", "error", "failed", "unhealthy"}:
        return SandboxStatus.ERROR
    return SandboxStatus.UNKNOWN


@dataclass(frozen=True)
class DaytonaConnectionConfig(DaytonaConfigBase):
    """Daytona API client connection settings."""

    api_key: str | None = None
    jwt_token: str | None = None
    organization_id: str | None = None
    api_url: str | None = None
    server_url: str | None = None
    target: str | None = None
    connection_pool_maxsize: int | None = None
    otel_enabled: bool | None = None

    def __post_init__(self) -> None:
        if self.connection_pool_maxsize is not None and self.connection_pool_maxsize <= 0:
            raise ValueError("connection.connection_pool_maxsize must be > 0")


@dataclass(frozen=True)
class DaytonaCreateConfig(DaytonaConfigBase):
    """Daytona sandbox creation settings."""

    timeout_s: float = 60.0
    retries: int = 2
    retry_delay_s: float = 1.0
    retry_max_delay_s: float = 30.0
    language: str | None = None
    os_user: str | None = None
    public: bool | None = None
    auto_stop_interval: int | None = None
    auto_archive_interval: int | None = None
    auto_delete_interval: int | None = None
    ephemeral: bool | None = None
    network_block_all: bool | None = None
    network_allow_list: str | None = None

    def __post_init__(self) -> None:
        if self.timeout_s < 0:
            raise ValueError("create.timeout_s must be >= 0")
        if self.retries < 0:
            raise ValueError("create.retries must be >= 0")
        if self.retry_delay_s < 0:
            raise ValueError("create.retry_delay_s must be >= 0")
        if self.retry_max_delay_s < 0:
            raise ValueError("create.retry_max_delay_s must be >= 0")


@dataclass(frozen=True)
class DaytonaProbeConfig(DaytonaConfigBase):
    """Post-create probe settings."""

    command: str | None = "printf nemo-rl-sandbox-ready"
    expected_stdout: str | None = "nemo-rl-sandbox-ready"
    timeout_s: int = 30
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if self.command is not None and self.timeout_s <= 0:
            raise ValueError("probe.timeout_s must be > 0")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("probe.deadline_s must be > 0")


@dataclass(frozen=True)
class DaytonaOperationConfig(DaytonaConfigBase):
    """Retry and timeout settings for Daytona operations after create."""

    retries: int = 3
    retry_delay_s: float = 1.0
    retry_max_delay_s: float = 15.0
    command_retries: int = 0
    command_timeout_margin_s: float = 60.0
    file_timeout_s: int | None = 30 * 60
    close_timeout_s: float | None = 60.0

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError("operations.retries must be >= 0")
        if self.retry_delay_s < 0:
            raise ValueError("operations.retry_delay_s must be >= 0")
        if self.retry_max_delay_s < 0:
            raise ValueError("operations.retry_max_delay_s must be >= 0")
        if self.command_retries < 0:
            raise ValueError("operations.command_retries must be >= 0")
        if self.command_timeout_margin_s < 0:
            raise ValueError("operations.command_timeout_margin_s must be >= 0")
        if self.file_timeout_s is not None and self.file_timeout_s <= 0:
            raise ValueError("operations.file_timeout_s must be > 0")
        if self.close_timeout_s is not None and self.close_timeout_s <= 0:
            raise ValueError("operations.close_timeout_s must be > 0")


@dataclass(frozen=True)
class DaytonaBatchConfig(DaytonaConfigBase):
    """Client-side batch fanout settings."""

    concurrency: int = 4

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("batch.concurrency must be >= 1")


class DaytonaProvider:
    """Provider backed by the Daytona Python SDK."""

    name = "daytona"

    def __init__(
        self,
        *,
        connection: DaytonaConnectionConfig | Mapping[str, Any] | None = None,
        create: DaytonaCreateConfig | Mapping[str, Any] | None = None,
        probe: DaytonaProbeConfig | Mapping[str, Any] | None = None,
        operations: DaytonaOperationConfig | Mapping[str, Any] | None = None,
        batch: DaytonaBatchConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = DaytonaConnectionConfig.from_mapping(connection)
        self._create = DaytonaCreateConfig.from_mapping(create)
        self._probe = DaytonaProbeConfig.from_mapping(probe)
        self._operations = DaytonaOperationConfig.from_mapping(operations)
        self._batch = DaytonaBatchConfig.from_mapping(batch)
        self._connection_supplied_keys = _supplied_config_keys(connection, DaytonaConnectionConfig)
        self._daytona: Any | None = None

    def _client(self) -> Any:
        if self._daytona is not None:
            return self._daytona
        AsyncDaytona, DaytonaConfig, _, _, _, _ = _require_daytona_sdk()
        kwargs: dict[str, Any] = {}
        for key in (
            "api_key",
            "jwt_token",
            "organization_id",
            "api_url",
            "server_url",
            "target",
            "connection_pool_maxsize",
            "otel_enabled",
        ):
            value = getattr(self._connection, key)
            if value is not None or key in self._connection_supplied_keys:
                kwargs[key] = value
        self._daytona = AsyncDaytona() if not kwargs else AsyncDaytona(DaytonaConfig(**kwargs))
        return self._daytona

    async def aclose(self) -> None:
        if self._daytona is None:
            return
        daytona = self._daytona
        self._daytona = None
        await daytona.close()

    async def _await_operation(
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

        retry_policy = AsyncRetrying(
            retry=retry_if_exception(_is_retryable_operation_error),
            stop=stop_after_attempt(max_attempts),
            wait=wait_random_exponential(
                multiplier=self._operations.retry_delay_s,
                max=self._operations.retry_max_delay_s,
            ),
            before_sleep=_log_operation_retry,
            reraise=True,
        )
        async for attempt in retry_policy:
            with attempt:
                awaitable = operation_factory()
                if timeout_s is None or timeout_s == 0:
                    return await awaitable
                try:
                    return await asyncio.wait_for(awaitable, timeout=timeout_s)
                except asyncio.TimeoutError as e:
                    raise TimeoutError(f"Timed out during Daytona {operation}; sandbox_id={sandbox_id!r}") from e

        raise RuntimeError("Daytona operation retry loop did not run")

    def _base_create_kwargs(self, spec: SandboxSpec) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if (name := _extension_value(spec, "name")) is not None:
            kwargs["name"] = name
        if (language := _configured_or_extension(spec, "language", self._create.language)) is not None:
            kwargs["language"] = language
        if (os_user := _configured_or_extension(spec, "os_user", self._create.os_user)) is not None:
            kwargs["os_user"] = os_user
        if spec.env:
            kwargs["env_vars"] = spec.env
        if spec.metadata:
            kwargs["labels"] = spec.metadata
        configured_values = {
            "public": self._create.public,
            "auto_stop_interval": self._create.auto_stop_interval,
            "auto_archive_interval": self._create.auto_archive_interval,
            "auto_delete_interval": self._create.auto_delete_interval,
            "ephemeral": self._create.ephemeral,
            "network_block_all": self._create.network_block_all,
            "network_allow_list": self._create.network_allow_list,
        }
        for key, configured in configured_values.items():
            value = _configured_or_extension(spec, key, configured)
            if value is None:
                continue
            if key in {"public", "ephemeral", "network_block_all"}:
                kwargs[key] = _coerce_bool(f"daytona.{key}", value)
            elif key in {"auto_stop_interval", "auto_archive_interval", "auto_delete_interval"}:
                kwargs[key] = _coerce_int(f"daytona.{key}", value)
            else:
                kwargs[key] = str(value)
        volumes = _spec_volumes(spec)
        if volumes is not None:
            kwargs["volumes"] = _to_volume_mounts(volumes)
        return kwargs

    def _to_create_params(self, spec: SandboxSpec) -> Any | None:
        _, _, ImageParams, SnapshotParams, _, _ = _require_daytona_sdk()
        _provider_options(spec)
        snapshot_id = _spec_snapshot_id(spec)
        if spec.image is not None and snapshot_id is not None:
            raise ValueError("Daytona provider does not support both image and snapshot_id")
        kwargs = self._base_create_kwargs(spec)
        if spec.image is not None:
            kwargs["image"] = spec.image
            resources = _to_resources(spec.resources) if _resources_requested(spec.resources) else None
            if resources is not None:
                kwargs["resources"] = resources
            return ImageParams(**kwargs)
        if snapshot_id is not None:
            if _resources_requested(spec.resources):
                raise ValueError("Daytona snapshot creation does not support resource overrides")
            kwargs["snapshot"] = snapshot_id
            return SnapshotParams(**kwargs)
        return SnapshotParams(**kwargs) if kwargs else None

    def _with_generated_sandbox_id(self, spec: SandboxSpec) -> SandboxSpec:
        if _extension_value(spec, "name") is not None:
            return spec
        sandbox_id = f"nemo-gym-{uuid.uuid4().hex}"
        metadata = dict(spec.metadata)
        metadata.setdefault(SANDBOX_ID_LABEL, sandbox_id)
        provider_options = dict(spec.provider_options)
        extensions = _spec_extensions(spec)
        extensions[f"{DAYTONA_EXTENSION_PREFIX}name"] = sandbox_id
        provider_options[PROVIDER_OPTION_EXTENSIONS] = extensions
        return replace(spec, metadata=metadata, provider_options=provider_options)

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        if self._probe.command is None:
            return
        deadline_s = self._probe.deadline_s or float(self._probe.timeout_s)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + deadline_s
        last_exception: BaseException | None = None
        while loop.time() < deadline:
            try:
                result = await self._exec(handle, self._probe.command, timeout_s=self._probe.timeout_s, retries=0)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exception = e
                await asyncio.sleep(min(1.0, max(deadline - loop.time(), 0.0)))
                continue
            stdout = result.stdout or ""
            if result.return_code == 0 and (
                self._probe.expected_stdout is None or self._probe.expected_stdout in stdout
            ):
                return
            last_exception = DaytonaCreateVerificationError(
                f"Daytona create probe failed; sandbox_id={handle.sandbox_id!r}, "
                f"return_code={result.return_code}, stdout={stdout[:200]!r}, stderr={(result.stderr or '')[:200]!r}"
            )
            await asyncio.sleep(min(1.0, max(deadline - loop.time(), 0.0)))
        raise DaytonaCreateVerificationError(
            f"Daytona sandbox failed create probe before deadline; sandbox_id={handle.sandbox_id!r}"
        ) from last_exception

    async def _create_once(self, spec: SandboxSpec) -> SandboxHandle:
        params = self._to_create_params(spec)
        timeout_s = float(spec.ready_timeout_s if spec.ready_timeout_s is not None else self._create.timeout_s)
        try:
            sandbox = await self._await_operation(
                lambda: self._client().create(params, timeout=timeout_s),
                operation="create",
                sandbox_id="<pending>",
                timeout_s=timeout_s + 5.0 if timeout_s > 0 else None,
                retries=0,
            )
        except TimeoutError as e:
            raise DaytonaCreateTimeoutError(
                f"Timed out creating Daytona sandbox after {timeout_s:g}s; image={spec.image!r}"
            ) from e
        handle = SandboxHandle(sandbox_id=str(sandbox.id), provider_name=self.name, raw=sandbox)
        try:
            await self._verify_created_handle(handle)
        except Exception:
            try:
                await self.close(handle, delete=True)
            except Exception as cleanup_error:
                LOGGER.warning(
                    "Failed to delete Daytona sandbox after create probe failure; sandbox_id=%s; error=%r",
                    handle.sandbox_id,
                    cleanup_error,
                )
            raise
        return handle

    async def _create_with_retries(self, spec: SandboxSpec) -> SandboxHandle:
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
                return await self._create_once(self._with_generated_sandbox_id(spec))

        raise RuntimeError("Daytona create retry loop did not run")

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        return await self._create_with_retries(_normalize_spec(spec))

    async def create_batch(self, spec: SandboxSpec, count: int, *, allow_partial: bool = False) -> list[SandboxHandle]:
        if count < 1:
            raise ValueError("count must be >= 1")
        semaphore = asyncio.Semaphore(self._batch.concurrency)

        async def _create_one() -> SandboxHandle:
            async with semaphore:
                return await self.create(spec)

        results = await asyncio.gather(*(_create_one() for _ in range(count)), return_exceptions=True)
        errors = [result for result in results if isinstance(result, Exception)]
        handles = [result for result in results if isinstance(result, SandboxHandle)]
        if not errors:
            return handles
        if allow_partial:
            return handles
        cleanup_errors: list[tuple[str, BaseException]] = []
        for handle in handles:
            try:
                await self.close(handle, delete=True)
            except Exception as cleanup_error:
                cleanup_errors.append((handle.sandbox_id, cleanup_error))
        cleanup_suffix = ""
        if cleanup_errors:
            cleanup_suffix = "; cleanup_errors=" + ", ".join(
                f"{sandbox_id}: {type(error).__name__}: {error}" for sandbox_id, error in cleanup_errors
            )
        raise DaytonaCreateError(
            "One or more Daytona sandboxes failed during batch create; "
            f"failed={len(errors)}, requested={count}{cleanup_suffix}"
        ) from errors[0]

    async def connect(self, sandbox_id: str) -> SandboxHandle:
        sandbox = await self._client().get(sandbox_id)
        return SandboxHandle(sandbox_id=str(sandbox.id), provider_name=self.name, raw=sandbox)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        refresh_data = getattr(handle.raw, "refresh_data", None)
        if refresh_data is not None:
            await self._await_operation(
                refresh_data,
                operation="refresh_data",
                sandbox_id=handle.sandbox_id,
                timeout_s=self._operations.close_timeout_s,
            )
            return _to_sandbox_status(getattr(handle.raw, "state", None))

        status_value = getattr(handle.raw, "status", None)
        if status_value is not None:
            state = getattr(status_value, "state", status_value)
            return _to_sandbox_status(state)

        get_info = getattr(handle.raw, "get_info", None)
        if get_info is None:
            return SandboxStatus.UNKNOWN
        info = await self._await_operation(
            get_info,
            operation="get_info",
            sandbox_id=handle.sandbox_id,
            timeout_s=self._operations.close_timeout_s,
        )
        info_status = getattr(info, "status", None)
        return _to_sandbox_status(getattr(info_status, "state", info_status))

    def _command_retry_count(self) -> int:
        return self._operations.command_retries

    @staticmethod
    def _effective_command(command: str, user: str | int | None) -> str:
        if user is None or user == "root" or user == 0:
            return command
        raise NotImplementedError(
            "Daytona provider does not support per-command user switching. "
            "Configure create.os_user for sandbox-level user selection until Daytona exposes native per-command users."
        )

    async def _exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        user: str | int | None = None,
        retries: int | None = None,
    ) -> SandboxExecResult:
        try:
            response = await self._await_operation(
                lambda: handle.raw.process.exec(
                    self._effective_command(command, user),
                    cwd=cwd,
                    env=env,
                    timeout=timeout_s,
                ),
                operation="process.exec",
                sandbox_id=handle.sandbox_id,
                timeout_s=float(timeout_s) + self._operations.command_timeout_margin_s
                if timeout_s is not None
                else None,
                retries=self._command_retry_count() if retries is None else retries,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not _is_daytona_command_exec_error(e):
                raise
            LOGGER.warning(
                "Daytona process.exec failed without a process result; sandbox_id=%s; error=%r",
                handle.sandbox_id,
                e,
            )
            return _daytona_exec_error_result(e)
        artifacts = getattr(response, "artifacts", None)
        stdout = getattr(response, "result", None)
        if stdout is None and artifacts is not None:
            stdout = getattr(artifacts, "stdout", None)
        stderr = getattr(response, "stderr", None)
        if stderr is None and artifacts is not None:
            stderr = getattr(artifacts, "stderr", None)
        return_code = getattr(response, "exit_code", None)
        return SandboxExecResult(stdout=stdout, stderr=stderr, return_code=0 if return_code is None else return_code)

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
        return await self._exec(handle, command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        payload = data.encode() if isinstance(data, str) else data
        timeout_s = self._operations.file_timeout_s
        await self._await_operation(
            lambda: handle.raw.fs.upload_file(payload, target_path)
            if timeout_s is None
            else handle.raw.fs.upload_file(payload, target_path, timeout=timeout_s),
            operation=f"upload_file({target_path})",
            sandbox_id=handle.sandbox_id,
            timeout_s=float(timeout_s) if timeout_s is not None else None,
        )

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        timeout_s = self._operations.file_timeout_s
        result = await self._await_operation(
            lambda: handle.raw.fs.download_file(source_path)
            if timeout_s is None
            else handle.raw.fs.download_file(source_path, timeout_s),
            operation=f"download_file({source_path})",
            sandbox_id=handle.sandbox_id,
            timeout_s=float(timeout_s) if timeout_s is not None else None,
        )
        return result.encode() if isinstance(result, str) else result

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        timeout_s = self._operations.file_timeout_s
        await self._await_operation(
            lambda: handle.raw.fs.upload_file(str(source_path), target_path)
            if timeout_s is None
            else handle.raw.fs.upload_file(str(source_path), target_path, timeout=timeout_s),
            operation=f"upload_file({target_path})",
            sandbox_id=handle.sandbox_id,
            timeout_s=float(timeout_s) if timeout_s is not None else None,
        )

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        timeout_s = self._operations.file_timeout_s
        await self._await_operation(
            lambda: handle.raw.fs.download_file(source_path, str(target_path))
            if timeout_s is None
            else handle.raw.fs.download_file(source_path, str(target_path), timeout_s),
            operation=f"download_file({source_path})",
            sandbox_id=handle.sandbox_id,
            timeout_s=float(timeout_s) if timeout_s is not None else None,
        )

    async def close(self, handle: SandboxHandle, *, delete: bool = True) -> None:
        if not delete:
            LOGGER.warning(
                "Retaining Daytona sandbox %r because close(delete=False) was requested; "
                "the remote sandbox remains active until Daytona lifecycle policies stop or delete it.",
                handle.sandbox_id,
            )
            return
        try:
            await self._await_operation(
                lambda: self._client().delete(handle.raw, timeout=self._operations.close_timeout_s or 0),
                operation="delete",
                sandbox_id=handle.sandbox_id,
                timeout_s=self._operations.close_timeout_s,
            )
        except Exception as e:
            if _is_missing_sandbox_delete_error(e):
                LOGGER.info("Daytona sandbox %r was already deleted during close", handle.sandbox_id)
                return
            raise
