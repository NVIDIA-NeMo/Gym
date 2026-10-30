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
import hashlib
import ipaddress
import logging
import math
import random
import re
import shlex
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable, ClassVar, Literal, NoReturn
from urllib.parse import quote, urlsplit

from pydantic import Field, model_validator

from nemo_gym.core.failure_kinds import (
    PROVIDER_COMMAND_OUTCOME_UNKNOWN,
    PROVIDER_COMMAND_TIMEOUT,
    PROVIDER_CREATE_PENDING,
    PROVIDER_CREATE_TIMEOUT,
    PROVIDER_INTERNAL_ERROR,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_NOT_READY,
    PROVIDER_OOM_KILLED,
    PROVIDER_PROTOCOL_ERROR,
    PROVIDER_PTY_BUSY,
    PROVIDER_PTY_UNAVAILABLE,
    PROVIDER_SETUP_FAILED,
    PROVIDER_STATUS_UNAVAILABLE,
    PROVIDER_UNAVAILABLE,
    FailureClassificationConfig,
)
from nemo_gym.core.failure_transport import observe_failure
from nemo_gym.core.outcome_policy import OutcomeAction, OutcomePolicyConfig, failure_outcome_policy
from nemo_gym.core.typed_exceptions import NgFailureEvent

# Re-exported names keep their old import path (see tests/unit_tests/fixtures/import_surface.json).
from nemo_gym.sandbox.attribution import (  # noqa: F401
    ATTRIBUTION_KEY_PREFIX_RE,
    DEFAULT_ATTRIBUTION_KEY_PREFIX,
    METADATA_VALUE_RE,
    RUN_KEY,
    AttributionConfig,
    log_attribution_once,
    resolve_attribution,
    resolve_run_id,
    sanitize_label_value,
)
from nemo_gym.sandbox.providers.base import (
    ORPHAN_REAP_TIMEOUT_S,
    SandboxEndpoint,
    SandboxExecResult,
    SandboxHandle,
    SandboxPtySession,
    SandboxPtySpec,
    SandboxResources,
    SandboxSnapshotNotFoundError,
    SandboxSpec,
    SandboxStatus,
    clamp_resource_requests,
)
from nemo_gym.sandbox.providers.opensandbox.failures import opensandbox_failure
from nemo_gym.sandbox.providers.utils import coerce_config as _coerce_config
from nemo_gym.sandbox.resource_telemetry import ResourceAllocation
from nemo_gym.sandbox.utils import rewrite_image
from nemo_gym.telemetry._fallbacks import is_span_group_enabled
from nemo_gym.telemetry.gym_metrics import record_sandbox_create_retry
from nemo_gym.telemetry.setup import is_metrics_exporter_active
from nemo_gym.telemetry.span_groups import GymSpanGroup
from nemo_gym.utils.exception_inspection import error_to_http_status, explicit_causes


LOGGER = logging.getLogger(__name__)
logging.getLogger("opensandbox").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# Management API listing page size; one listing per release, so a few large pages beat many small ones.
SNAPSHOT_LIST_PAGE_SIZE = 100


@dataclass(eq=False)
class _BackgroundCommandPoll:
    """One background command's poll loop, woken when a checkpoint polling scope of its sandbox opens or closes."""

    wakeup: asyncio.Event = field(default_factory=asyncio.Event)
    # Set when a parked scope opens for the sandbox, cleared before each status request: a request that fails
    # with this set overlapped a checkpoint, while the sandbox may not answer.
    parked: bool = False


class _MissingSandboxTerminateFilter(logging.Filter):
    """Hide the SDK warning emitted when terminate reaches its desired state."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            record.levelno == logging.WARNING
            and message.startswith("Failed to terminate sandbox ")
            and "[KUBERNETES::SANDBOX_NOT_FOUND]" in message
        )


logging.getLogger("opensandbox.adapters.sandboxes_adapter").addFilter(_MissingSandboxTerminateFilter())


@dataclass(frozen=True)
class _SandboxDeathNotice:
    message: str
    failure_kind: str
    provider_status_reason: str | None = None


# The single-pod template names its user container sandbox and its pod <sandbox-id>-0.
# Pool allocations and aggregate multi-pod failures need a separate identity contract.
_USER_CONTAINER = "sandbox"
_LIFECYCLE_SINGLE_POD_SUFFIX = "-0"


def _lifecycle_oom_evidence(info: Any, *, sandbox_id: str) -> _SandboxDeathNotice | None:
    """Read the known controller's persisted current-container report from the control plane.

    Command-status 410 bodies can come from the sandbox's exec daemon, even through
    the server proxy. Only GET /sandboxes/{id} supplies this lifecycle evidence.
    Unknown layouts, mismatched identities and ambiguous container reports fail closed.
    """
    status = getattr(info, "status", None)
    if (
        getattr(info, "id", None) != sandbox_id
        or getattr(status, "state", None) != "Failed"
        or getattr(status, "reason", None) != "FAILED"
    ):
        return None
    report = getattr(status, "message", None)
    if not isinstance(report, str) or len(report) > 65536:
        return None
    pod = re.escape(f"{sandbox_id}{_LIFECYCLE_SINGLE_POD_SUFFIX}")
    container = re.escape(_USER_CONTAINER)
    match = re.fullmatch(
        rf"1/1 observed pods failed; primary reason=ContainerFailed; sample pod={pod}; "
        rf"sample detail=Pod {pod} container {container} exited with code (?P<exit>\d{{1,3}}) \(OOMKilled\)",
        report,
    )
    if match is None or not 1 <= int(match["exit"]) <= 255:
        return None
    return _SandboxDeathNotice(
        f"Sandbox was OOM-killed; lifecycle status confirms {report}",
        failure_kind=PROVIDER_OOM_KILLED,
        provider_status_reason=status.reason,
    )


# execd reports os/exec launch errors through CommandExecError, before a child runs.
# Older servers omit the credential hint; both shapes must retain the fork/exec prefix.
# https://github.com/opensandbox-group/OpenSandbox/blob/c9df3df530464b9ae2a7580856f6b9bd52b507cc/components/execd/pkg/runtime/command.go#L265-L277
_ROOT_LAUNCH_REFUSAL = re.compile(r"fork/exec [^\r\n]+: operation not permitted(?: \(switching to uid=0\b[^\r\n]*\))?")


def _inspect_oom_evidence(inspect: str) -> str | None:
    """Read only the current user container state from OpenSandbox inspect text."""
    lines = inspect.splitlines()
    if sum(line == "Containers:" for line in lines) == 1:
        in_containers = False
        container: str | None = None
        sandbox_count = 0
        sandbox_states: list[str] = []
        for line in lines:
            if line == "Containers:":
                in_containers = True
                container = None
            elif in_containers and line and not line.startswith(" "):
                in_containers = False
            elif in_containers:
                header = re.fullmatch(r"  ([^:\s]+):", line)
                if header is not None:
                    container = header.group(1)
                    if container == _USER_CONTAINER:
                        sandbox_count += 1
                elif container == _USER_CONTAINER and (state := re.fullmatch(r"    State:\s+(.+)", line)):
                    sandbox_states.append(state.group(1))
        if (
            sandbox_count == 1
            and len(sandbox_states) == 1
            and re.fullmatch(r"Terminated \(exit=\d+, reason=OOMKilled\)", sandbox_states[0]) is not None
        ):
            return f"sandbox State: {sandbox_states[0]}"
        return None

    # Docker inspect describes one sandbox container, not a Pod with sidecars.
    oom_lines = [line for line in lines if line.startswith("OOMKilled:")]
    running_lines = [line for line in lines if line.startswith("Running:")]
    if (
        not any(line == "Containers:" for line in lines)
        and sum(line.startswith("Container ID:") for line in lines) == 1
        and len(running_lines) == 1
        and re.fullmatch(r"Running:\s+False", running_lines[0]) is not None
        and len(oom_lines) == 1
        and re.fullmatch(r"OOMKilled:\s+True", oom_lines[0]) is not None
    ):
        return "Docker OOMKilled: True"
    return None


# Kubernetes prefixed-key namespace for auto-injected attribution labels (team/user/workload/run).
# Per-attempt metadata key (under the attribution prefix): the only handle on a sandbox whose create
# request never returned, so a timed-out attempt can find and delete what it started.
CREATE_ATTEMPT_KEY = "create-attempt"
PENDING_SANDBOX_STATES = {"PENDING"}
# The limits the OpenSandbox SDK sends when a create declares none; requests are clamped against them too.
SDK_DEFAULT_RESOURCE_LIMITS = SandboxResources.from_mapping({"cpu": 1, "memory_mib": 2048})
# The SDK's own request timeout when the provider passes none (its ``ConnectionConfig`` default).
SDK_DEFAULT_REQUEST_TIMEOUT_S = 30.0
# Added to the request bound plus the readiness wait when ``create.timeout_s`` is not configured: room for
# the two endpoint lookups between the create request and the health check.
CREATE_BUDGET_SLACK_S = 30.0
# Create budgets already warned about, per process: servers build a provider per sandbox, so an instance
# attribute would repeat the warning on every create of a configuration that is below the sum.
_BUDGET_WARNINGS: set[tuple[float, float, float]] = set()
# Kubernetes label-key prefixes must be DNS-1123 subdomains (max 253 chars).
# Returned as stderr when a finished background command's output file was deleted,
# so a caller that needed the output sees why it is empty.
DELETED_OUTPUT_NOTE = (
    "OpenSandbox could not return this command's output: the output file was deleted, for example "
    "by a command that cleaned the temporary directory. The command's exit status is kept."
)
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


def _require_opensandbox_lifecycle_sdk() -> tuple[Any, Any]:
    """The SDK's management-API pieces that work without a sandbox handle: list and delete."""
    from opensandbox.adapters.factory import AdapterFactory
    from opensandbox.models.sandboxes import SandboxFilter

    return AdapterFactory, SandboxFilter


def _require_tenacity() -> tuple[Any, Any, Any, Any]:
    try:
        from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "tenacity is required for OpenSandbox retry handling. Install nemo-gym[sandbox] before using "
            "env.sandbox.provider.name=opensandbox."
        ) from e

    return AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential


def _exception_status_code(exception: BaseException) -> int | None:
    status = error_to_http_status(exception)
    if status is not None:
        return status
    match = STATUS_CODE_RE.search(str(exception))
    return int(match.group(1)) if match is not None else None


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


def _is_request_timeout(exception: BaseException) -> bool:
    """Whether an SDK create failure is the transport's request timeout on the create request itself.

    The request outlived the SDK's request timeout, so the sandbox it asked for may exist without the
    client ever learning its id, exactly as when the provider's own attempt budget fires. The SDK wraps
    transport errors (``SandboxInternalException`` with the httpx timeout as its cause; newer SDKs raise a
    ``SandboxTimeoutException``), so the cause chain is inspected. A connect or pool timeout sent no request
    and leaves nothing behind; a readiness timeout is raised with the id known and the sandbox already killed.
    """
    try:
        import httpx

        sent_timeouts: tuple[type[BaseException], ...] = (TimeoutError, httpx.TimeoutException)
        unsent_timeouts: tuple[type[BaseException], ...] = (httpx.ConnectTimeout, httpx.PoolTimeout)
    except ModuleNotFoundError:
        sent_timeouts, unsent_timeouts = (TimeoutError,), ()
    seen: set[int] = set()
    current: BaseException | None = exception
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, unsent_timeouts):
            return False
        if isinstance(current, sent_timeouts) or type(current).__name__ == "SandboxTimeoutException":
            return True
        current = current.__cause__
    return False


def _request_was_not_sent(exception: BaseException) -> bool:
    """Whether the transport failed before the request left the client, so sending it again cannot run it twice.

    The innermost transport error in the cause chain decides, as in the SWE-bench Pro patch capture. The SDK wraps
    its client's error, and Gym's aiohttp transport re-raises every aiohttp connection error as
    ``httpx.ConnectError``, a reset after the request was written included; the aiohttp error underneath says
    whether the connection ever opened. httpx raises ``ConnectError`` and ``ConnectTimeout`` only while it opens a
    connection, and ``PoolTimeout`` while it waits for one.
    """
    import socket

    import httpx

    unsent: tuple[type[BaseException], ...] = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
        ConnectionRefusedError,
        socket.gaierror,
    )
    sent: tuple[type[BaseException], ...] = (httpx.TransportError,)
    try:
        import aiohttp

        unsent += (aiohttp.ClientConnectorError, aiohttp.ConnectionTimeoutError)
        sent += (aiohttp.ClientError,)
    except ModuleNotFoundError:
        pass
    not_sent = False
    for link in explicit_causes(exception):
        if isinstance(link, unsent):
            not_sent = True
        elif isinstance(link, sent):
            not_sent = False
    return not_sent


def _is_retryable_create_error(
    exception: BaseException,
    *,
    config: "OpenSandboxCreateOutcomePolicy | None" = None,
    fail_fast_on_pending: bool = False,
) -> bool:
    """Select create recovery from observed facts and explicit boundary policy."""
    failure = (
        exception
        if isinstance(exception, NgFailureEvent)
        else opensandbox_failure(operation="create", cause=exception)
    )
    if failure.failure_kind == PROVIDER_CREATE_PENDING and fail_fast_on_pending:
        return False
    config = config or OpenSandboxCreateOutcomePolicy()
    return failure_outcome_policy(failure, config=config, classification=config.classification) == OutcomeAction.RETRY


def _retry_delay(error: BaseException | None, *, fallback: float, max_delay_s: float) -> float:
    """Bound this request's Retry-After without searching other failure events' causes."""
    if not isinstance(error, NgFailureEvent) or error.http_status != 429:
        return fallback
    import httpx
    from opensandbox.exceptions import SandboxApiException

    from nemo_gym.sandbox.providers._http_transport import _retry_after_seconds

    cause = error.__cause__
    if isinstance(cause, SandboxApiException):
        cause = cause.__cause__
    if isinstance(cause, httpx.HTTPStatusError) and cause.response.status_code == 429:
        # Parsed waits are nonnegative; -1 preserves the polling fallback for an absent or invalid header.
        delay = _retry_after_seconds(cause.response.headers.get("Retry-After"), fallback=-1)
        return min(delay, max_delay_s) if delay >= 0 else fallback
    return fallback


def _is_deleted_output_error(exception: BaseException) -> bool:
    """Return whether a log read failed because execd's output file for the command is gone.

    execd keeps background output under ``os.TempDir()`` (``/tmp`` unless ``TMPDIR`` is set), so
    any command that empties ``/tmp`` deletes it, and the read fails with "error open combined
    output file for command <id>: ... no such file or directory". Rereading cannot bring it back.
    """
    if isinstance(exception, NgFailureEvent) and exception.__cause__ is not None:
        exception = exception.__cause__
    message = str(exception)
    return "combined output file" in message and "no such file or directory" in message


def _is_missing_sandbox_delete_error(exception: BaseException) -> bool:
    """Match kill errors meaning the sandbox is already gone (terminate's goal state).

    Only the terminate path may treat this as success; other operations must
    keep failing loudly on not-found.
    """
    if _exception_status_code(exception) == 404:
        return True
    message = str(exception).lower()
    return "sandbox_not_found" in message or ("sandbox" in message and "not found" in message)


def _snapshot_state(payload: Mapping[str, Any]) -> str:
    """A management API snapshot record's lifecycle state, lowercased; ``""`` when it reports none."""
    status = payload.get("status")
    state = status.get("state") if isinstance(status, dict) else None
    return str(state or payload.get("state") or "").lower()


def _is_terminal_delete_error(error: BaseException) -> bool:
    # The SDK wraps transport failures, including PermissionError, in
    # exceptions whose outer text can misleadingly look transient or missing.
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (asyncio.CancelledError, PermissionError)) or _exception_status_code(current) in (
            401,
            403,
        ):
            return True
        current = current.__cause__
    return False


def _execution_outcome(
    execution: Any, *, operation: str, sandbox_id: str, execution_id: str | None = None
) -> tuple[int, str | None, str | None]:
    """Map a streamed SDK execution to ``(return_code, error_type, stderr_note)``.

    The SDK fills in ``exit_code=0`` itself once execd sends its completion
    event. A stream that ends without a terminal event has an unknown outcome,
    even if it contains partial logs or an error without an exit status.
    """
    if execution.exit_code is not None:
        if execution.exit_code != 0 or execution.error is None:
            return execution.exit_code, None, None
        reason = "OpenSandbox command reported both exit code 0 and an error"
    elif execution.error is not None:
        reason = "OpenSandbox command reported an error without an exit code"
    elif execution.complete is not None:
        return 0, None, None
    else:
        reason = "OpenSandbox command ended without an exit code, error or completion event"
    raise opensandbox_failure(
        operation=operation,
        sandbox_id=sandbox_id,
        execution_id=execution_id,
        failure_kind=PROVIDER_COMMAND_OUTCOME_UNKNOWN,
        reason=reason,
    )


def _root_launch_refusal(execution: Any, *, uid: int | None) -> SandboxExecResult | None:
    """Recognize execd's pre-launch refusal, never user stdout/stderr or exception prose."""
    error = getattr(execution, "error", None)
    message = getattr(error, "value", None)
    name = getattr(error, "name", None)
    # Background launch failures use the controller's SSE RuntimeError envelope.
    prefix = "error running commands failed to start commands: "
    if name == "RuntimeError" and isinstance(message, str) and message.startswith(prefix):
        message = message.removeprefix(prefix)
    elif name != "CommandExecError":
        return None
    if (
        uid == 0
        and getattr(execution, "exit_code", None) is None
        and isinstance(message, str)
        and _ROOT_LAUNCH_REFUSAL.fullmatch(message)
    ):
        # 125 is a non-process sentinel: there was no child to return an exit status.
        return SandboxExecResult(None, message, 125, error_type="user_switch_unavailable")
    return None


def _log_create_retry(retry_state: Any) -> None:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    sleep_s = retry_state.next_action.sleep if retry_state.next_action else None
    LOGGER.warning(
        "Retrying OpenSandbox sandbox create after attempt %s; next_sleep_s=%s; error=%r",
        retry_state.attempt_number,
        sleep_s,
        exception,
    )
    if is_metrics_exporter_active():
        record_sandbox_create_retry(provider="opensandbox")


def _log_operation_retry(retry_state: Any, *, operation: str = "?", sandbox_id: str | None = "?") -> None:
    # operation + sandbox_id make an absorbed create-probe retry distinguishable
    # from a failing agent exec; without them every 502 retry looks identical.
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    sleep_s = retry_state.next_action.sleep if retry_state.next_action else None
    LOGGER.warning(
        "Retrying OpenSandbox SDK operation after attempt %s; next_sleep_s=%s; operation=%s; sandbox_id=%s; error=%r",
        retry_state.attempt_number,
        sleep_s,
        operation,
        sandbox_id,
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
    return sanitize_label_value(value)


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


def _to_network_policy(network_policy: Mapping[str, Any]) -> Any:
    from opensandbox.models.sandboxes import NetworkPolicy

    return NetworkPolicy.model_validate(network_policy)


def _to_volumes(volumes: list[Mapping[str, Any]]) -> list[Any]:
    _, _, _, _, Volume = _require_opensandbox_sdk()
    return [Volume(**dict(volume)) for volume in volumes]


def _to_image_spec(image: str, image_auth: Mapping[str, Any] | None) -> Any:
    if image_auth is None:
        return image
    from opensandbox.models.sandboxes import SandboxImageAuth, SandboxImageSpec

    return SandboxImageSpec(image, auth=SandboxImageAuth(**dict(image_auth)))


def _to_sandbox_status(state: Any) -> SandboxStatus:
    normalized = str(state or "").lower()
    if normalized in {"active", "ready", "running"}:
        return SandboxStatus.RUNNING
    if normalized == "paused":
        return SandboxStatus.PAUSED
    if normalized in {"creating", "initializing", "pausing", "pending", "resuming", "starting"}:
        return SandboxStatus.STARTING
    if normalized in {"completed", "deleted", "exited", "stopped", "terminated"}:
        return SandboxStatus.STOPPED
    if normalized in {"crashed", "error", "failed", "unhealthy"}:
        return SandboxStatus.ERROR
    return SandboxStatus.UNKNOWN


def split_domain_scheme(domain: str) -> tuple[str, str | None]:
    """Split a configured OpenSandbox domain into ``(domain, scheme)``.

    ``"https://sandbox.example:8080/prefix/"`` -> ``("sandbox.example:8080/prefix", "https")``;
    a bare host is returned unchanged with scheme ``None``. Surrounding whitespace
    is stripped either way. Only ``scheme://host[:port][/path-prefix]`` is accepted:
    other schemes, a missing host, or a query string / fragment raise ``ValueError``.
    """
    domain = domain.strip()
    if "://" not in domain:
        return domain, None
    parts = urlsplit(domain)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"connection.domain {domain!r} has unsupported scheme {parts.scheme!r}; use http or https")
    if not parts.netloc:
        raise ValueError(f"connection.domain {domain!r} has no host")
    if parts.query or parts.fragment:
        raise ValueError(
            f"connection.domain {domain!r} must not carry a query string or fragment; "
            "use scheme://host[:port][/path-prefix]"
        )
    # Keep any path prefix (a reverse proxy may mount the server under one); the SDK appends /v1.
    return parts.netloc + parts.path.rstrip("/"), scheme


@dataclass
class OpenSandboxConnectionConfig:
    """OpenSandbox server connection settings.

    With the legacy httpx backend, ``keepalive_expiry_s`` must stay below the
    server's own keep-alive idle timeout (uvicorn defaults to 5s), or sockets are reused after the
    server has closed them; null falls back to the SDK's default transport only
    when certificate verification is enabled and pooling is not disabled.
    ``transport_backend=aiohttp`` uses Gym's global client and connector limits;
    provider-local pooling settings apply only to ``transport_backend=httpx``.
    With the explicit legacy combination ``transport_backend=httpx``,
    ``keepalive_expiry_s=None`` and ``tls_verify=True`` (pooling enabled), the
    SDK owns its default transport. SDK 0.1.15 then discards command-status 410
    bodies; lifecycle evidence remains available independently of that response.
    ``tls_verify`` applies to every connection the provider opens (SDK transport
    and PTY sockets) and is off by
    default; set it for endpoints whose certificate the client can verify.
    ``domain`` may carry its scheme (``https://sandbox.example``). The scheme is
    moved into ``protocol`` and takes precedence over a configured ``protocol``,
    so every URL the provider builds itself (the PTY WebSocket target) agrees
    with the SDK's base URL; the SDK receives the host plus any path prefix
    (``sandbox.example:8080/prefix``). The SDK would accept a scheme in
    ``domain`` on its own, but the provider reads ``protocol`` directly, hence
    the normalization here. Only ``scheme://host[:port][/path-prefix]`` is
    accepted: a query string or fragment is a configuration error.
    """

    domain: str | None = None
    api_key: str | None = None
    protocol: str | None = None
    request_timeout_s: int | None = None
    use_server_proxy: bool = False
    # Open a fresh connection per request. Set this behind a load balancer that
    # silently reaps idle pooled connections, where reusing one hangs the SDK.
    # Costs a handshake per request; otherwise harmless.
    disable_connection_pooling: bool = False
    keepalive_expiry_s: float | None = 3.0
    max_keepalive_connections: int = 20
    max_connections: int | None = 100
    connect_retries: int = 2
    transport_backend: str = "aiohttp"
    tls_verify: bool = False

    def __post_init__(self) -> None:
        if self.domain is None:
            return
        self.domain, scheme = split_domain_scheme(self.domain)
        if scheme is not None:
            self.protocol = scheme


# Shared with the other providers that forward attribution as Kubernetes labels.
OpenSandboxAttributionConfig = AttributionConfig


class OpenSandboxOutcomePolicy(OutcomePolicyConfig):
    """Runtime request policy; execution separately enforces replay safety and budgets."""

    supported_actions: ClassVar[frozenset[OutcomeAction]] = frozenset({OutcomeAction.RETRY, OutcomeAction.FAIL})
    classification: FailureClassificationConfig = Field(default_factory=FailureClassificationConfig)

    by_failure_kind: dict[str, OutcomeAction] = Field(
        default_factory=lambda: {
            PROVIDER_IO_TIMEOUT: OutcomeAction.RETRY,
            PROVIDER_UNAVAILABLE: OutcomeAction.RETRY,
        }
    )
    by_status_code: dict[int, OutcomeAction] = Field(
        default_factory=lambda: {
            **dict.fromkeys((408, 409, 425, 429, 500, 502, 503, 504), OutcomeAction.RETRY),
            410: OutcomeAction.FAIL,
        }
    )
    default_action: OutcomeAction = OutcomeAction.FAIL

    @model_validator(mode="after")
    def validate_status_codes(self) -> "OpenSandboxOutcomePolicy":
        if any(not 400 <= status <= 599 for status in self.by_status_code):
            raise ValueError("Policy HTTP statuses must be in 400..599")
        return self


class OpenSandboxCreateOutcomePolicy(OpenSandboxOutcomePolicy):
    """Create/reconnect defaults; pending fail-fast remains an explicit execution constraint."""

    by_failure_kind: dict[str, OutcomeAction] = Field(
        default_factory=lambda: {
            PROVIDER_IO_TIMEOUT: OutcomeAction.RETRY,
            PROVIDER_UNAVAILABLE: OutcomeAction.RETRY,
            PROVIDER_INTERNAL_ERROR: OutcomeAction.RETRY,
            PROVIDER_CREATE_TIMEOUT: OutcomeAction.RETRY,
            PROVIDER_CREATE_PENDING: OutcomeAction.RETRY,
            PROVIDER_NOT_READY: OutcomeAction.RETRY,
        }
    )


@dataclass(frozen=True)
class OpenSandboxCreateConfig:
    """OpenSandbox create/reconnect retry settings."""

    outcome_policy: OpenSandboxCreateOutcomePolicy = field(
        default_factory=OpenSandboxCreateOutcomePolicy, kw_only=True
    )
    request_timeout_s: int | None = None
    timeout_s: float | None = None
    retries: int = 2
    retry_delay_s: float = 5.0
    retry_max_delay_s: float = 60.0
    # A create whose sandbox was still Pending at the timeout is retried like any other timeout (Pending also
    # covers a full cluster and an image still pulling). Set true to report it once instead, when the specs
    # are known to be placeable and a pod still Pending at the ready timeout means a request above a limit.
    fail_fast_on_pending: bool = False
    image_pull_policy: str | None = DEFAULT_IMAGE_PULL_POLICY
    skip_health_check: bool = False
    connect_attempt_timeout_s: float = 30.0
    connect_poll_s: float = 2.0
    # Refresh a created sandbox's TTL while this provider owns its handle.
    # This does not change task/command timeouts or renew borrowed handles.
    renew_interval_s: float | None = None
    # A lease: each sandbox is created with this TTL, capped by the spec's ``ttl_s``, and every renewal sets it to
    # now plus this TTL again, never past the spec's lifetime. The sandbox lives while the creating process renews
    # it, so the service deletes a crashed process's sandboxes within one TTL instead of at the end of the spec's
    # lifetime. A renewal that fails is logged and tried again at the next interval. Requires ``renew_interval_s``.
    renew_ttl_s: float | None = None
    # Per-call budget of one renewal: a hung renew must not use up the TTL it extends.
    # Unset, the connection's request timeout applies (60 s without one).
    renew_timeout_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome_policy, OpenSandboxCreateOutcomePolicy):
            object.__setattr__(
                self, "outcome_policy", OpenSandboxCreateOutcomePolicy.model_validate(self.outcome_policy)
            )
        if self.image_pull_policy is not None:
            validate_image_pull_policy(self.image_pull_policy)
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("create.timeout_s must be > 0")
        if self.renew_interval_s is not None and (
            not math.isfinite(self.renew_interval_s) or self.renew_interval_s <= 0
        ):
            raise ValueError("create.renew_interval_s must be finite and > 0")
        if self.renew_ttl_s is not None:
            if not math.isfinite(self.renew_ttl_s) or self.renew_ttl_s <= 0:
                raise ValueError("create.renew_ttl_s must be finite and > 0")
            if self.renew_interval_s is None or self.renew_interval_s >= self.renew_ttl_s:
                raise ValueError("create.renew_ttl_s requires a shorter create.renew_interval_s")
        if self.renew_timeout_s is not None and (not math.isfinite(self.renew_timeout_s) or self.renew_timeout_s <= 0):
            raise ValueError("create.renew_timeout_s must be finite and > 0")
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
    user: str | int | None = None

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
    outcome_policy: OpenSandboxOutcomePolicy | Mapping[str, Any] = field(default_factory=OpenSandboxOutcomePolicy)
    command_timeout_headroom_s: float = 60.0
    status_recovery_timeout_s: float = 300.0
    exec_hard_cap_slack_s: float = 30.0
    close_timeout_s: float | None = 30.0
    pause_resume_timeout_s: float = 600.0
    # Poll short status/log requests instead of holding one SSE stream open for
    # the whole command. Set this behind a load balancer that caps stream
    # duration, which would otherwise drop the stream and hang the client.
    background_exec: bool = False
    # Backs off from initial to interval.
    background_poll_initial_s: float = 0.25
    background_poll_interval_s: float = 2.0
    # Inside a checkpoint polling scope only: command polls of that sandbox wake at once and then poll at least
    # this often, so a harness stopped for a checkpoint is seen to end promptly. Other commands keep their backoff.
    checkpoint_background_poll_interval_s: float = 1.0
    # Create-to-ready budget of one snapshot, which pushes the sandbox's layers to the registry. Kept below a
    # controller's usual commit deadline, so a slow push fails the export with a typed error first.
    snapshot_timeout_s: float = 240.0
    # The management API can answer 404 for a snapshot it has only just created. Until a lookup first finds it, a
    # 404 is waited out for this long, within snapshot_timeout_s; 0 fails on the first 404.
    snapshot_lookup_grace_s: float = 60.0
    # Per-request budget for background-command status polls, which are small
    # idempotent GETs: without their own short budget, each poll against an
    # unreachable sandbox hangs for the shared request timeout (tuned for long
    # submits) before failing. None falls back to that shared budget.
    status_poll_timeout_s: float | None = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.outcome_policy, OpenSandboxOutcomePolicy):
            object.__setattr__(self, "outcome_policy", OpenSandboxOutcomePolicy.model_validate(self.outcome_policy))
        for name in ("command_timeout_headroom_s", "status_recovery_timeout_s", "exec_hard_cap_slack_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"operations.{name} must be finite and >= 0")
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
        if self.pause_resume_timeout_s <= 0:
            raise ValueError("operations.pause_resume_timeout_s must be > 0")
        if self.background_poll_interval_s <= 0:
            raise ValueError("operations.background_poll_interval_s must be > 0")
        if self.background_poll_initial_s <= 0:
            raise ValueError("operations.background_poll_initial_s must be > 0")
        if (
            not math.isfinite(self.checkpoint_background_poll_interval_s)
            or self.checkpoint_background_poll_interval_s <= 0
        ):
            raise ValueError("operations.checkpoint_background_poll_interval_s must be finite and > 0")
        if not math.isfinite(self.snapshot_timeout_s) or self.snapshot_timeout_s <= 0:
            raise ValueError("operations.snapshot_timeout_s must be finite and > 0")
        if not math.isfinite(self.snapshot_lookup_grace_s) or self.snapshot_lookup_grace_s < 0:
            raise ValueError("operations.snapshot_lookup_grace_s must be finite and >= 0")
        if self.status_poll_timeout_s is not None and self.status_poll_timeout_s <= 0:
            raise ValueError("operations.status_poll_timeout_s must be > 0")


@dataclass(frozen=True)
class OpenSandboxDiagnosticsConfig:
    """Optional control-plane checks after a command loses backend status.

    Background status recovery shares ``timeout_s`` across diagnostic I/O only;
    status requests and polling delays do not spend that diagnostic budget.
    Every lookup also stays within the original status-recovery deadline.
    """

    inspect_oom: bool = False
    timeout_s: float = 5.0
    request_timeout_s: float = 2.0
    poll_interval_s: float = 0.5

    def __post_init__(self) -> None:
        for name in ("timeout_s", "request_timeout_s", "poll_interval_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"diagnostics.{name} must be finite and > 0")
        if not isinstance(self.inspect_oom, bool):
            raise TypeError("diagnostics.inspect_oom must be a bool")


@dataclass(frozen=True)
class OpenSandboxProviderOptions:
    """Recognized per-sandbox create options read from ``SandboxSpec.provider_options``.

    ``image_auth``, ``network_policy``, ``platform``, and ``volumes`` entries are passed through to the
    OpenSandbox SDK, so their inner fields are validated by the SDK rather than here.
    """

    image_auth: Mapping[str, Any] | None = None
    network_policy: Mapping[str, Any] | None = None
    platform: Mapping[str, Any] | None = None
    snapshot_id: str | None = None
    volumes: tuple[Mapping[str, Any], ...] = ()
    skip_health_check: bool | None = None
    extensions: Mapping[str, str] = field(default_factory=dict)
    # Scheduling requests use SandboxSpec.resources keys. "limits" explicitly
    # mirrors each sandbox's limits; omission leaves defaulting to the server.
    resource_requests: Mapping[str, Any] | Literal["limits"] | None = None

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
        image_auth = options.get("image_auth")
        if image_auth is not None and not isinstance(image_auth, Mapping):
            raise TypeError("OpenSandbox provider option 'image_auth' must be a mapping")
        network_policy = options.get("network_policy")
        if network_policy is not None and not isinstance(network_policy, Mapping):
            raise TypeError("OpenSandbox provider option 'network_policy' must be a mapping")
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
        resource_requests = options.get("resource_requests")
        if (
            resource_requests is not None
            and resource_requests != "limits"
            and not isinstance(resource_requests, Mapping)
        ):
            raise TypeError("OpenSandbox provider option 'resource_requests' must be a mapping or 'limits'")

        return cls(
            image_auth=dict(image_auth) if image_auth is not None else None,
            network_policy=dict(network_policy) if network_policy is not None else None,
            platform=dict(platform) if platform is not None else None,
            snapshot_id=snapshot_id,
            volumes=tuple(dict(volume) for volume in volumes),
            skip_health_check=skip_health_check,
            extensions=_string_map(dict(extensions)),
            resource_requests=dict(resource_requests) if isinstance(resource_requests, Mapping) else resource_requests,
        )


@dataclass
class OpenSandboxNetworkingConfig:
    """Operator assertion that direct sandbox IPs are mutually reachable."""

    enabled: bool = False
    loopback_forwarding: bool = False
    python_executable: str = "python3"
    setup_command: str | None = None


@dataclass
class OpenSandboxImagesConfig:
    """Operator image policy for every sandbox this provider creates.

    ``rewrites`` are ordered ``{from, to}`` prefix rules (a mirror, or a derived image that adds what a
    deployment needs); ``auth`` maps an image prefix to ``{username, password}``, longest prefix wins.
    Tasks and servers keep naming the published image; the deployment decides what is pulled.
    """

    rewrites: list[dict[str, str]] = field(default_factory=list)
    auth: dict[str, dict[str, str]] = field(default_factory=dict)

    def resolve(self, image: str | None) -> tuple[str | None, dict[str, str] | None]:
        resolved = rewrite_image(image, self.rewrites)
        if resolved is None:
            return None, None
        prefix = max((p for p in self.auth if resolved.startswith(p)), key=len, default=None)
        return resolved, dict(self.auth[prefix]) if prefix is not None else None


@dataclass
class OpenSandboxSetupConfig:
    """Commands run as root in every new sandbox before it is handed out.

    For repairs the deployment owns rather than the task: an end-of-life distro pointed at its archive,
    a tool the platform needs in every image. A failing command fails the create.
    """

    commands: list[str] = field(default_factory=list)
    timeout_s: float = 600.0
    # Environment set in every sandbox this provider creates, under the spec's own variables. For
    # platform knobs such as the exec daemon's grace period, never for task or dataset settings.
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class OpenSandboxRuntimeRequirementsConfig:
    """Operator-supplied capability probes and create-time runtime metadata."""

    capability_probes: dict[str, str] = field(default_factory=dict)
    capability_metadata: dict[str, dict[str, str]] = field(default_factory=dict)
    shm_size_metadata_key: str | None = None


@dataclass
class OpenSandboxSharedStorageConfig:
    """Shared host mount and optional deployment-specific placement metadata."""

    host_path: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


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
        diagnostics: OpenSandboxDiagnosticsConfig | Mapping[str, Any] | None = None,
        attribution: OpenSandboxAttributionConfig | Mapping[str, Any] | None = None,
        networking: OpenSandboxNetworkingConfig | Mapping[str, Any] | None = None,
        shared_storage: OpenSandboxSharedStorageConfig | Mapping[str, Any] | None = None,
        runtime_requirements: OpenSandboxRuntimeRequirementsConfig | Mapping[str, Any] | None = None,
        images: OpenSandboxImagesConfig | Mapping[str, Any] | None = None,
        setup: OpenSandboxSetupConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = _coerce_config(connection, OpenSandboxConnectionConfig)
        self._create = _coerce_config(create, OpenSandboxCreateConfig)
        self._probe = _coerce_config(probe, OpenSandboxProbeConfig)
        self._operations = _coerce_config(operations, OpenSandboxOperationConfig)
        self._diagnostics = _coerce_config(diagnostics, OpenSandboxDiagnosticsConfig)
        self._attribution = _coerce_config(attribution, OpenSandboxAttributionConfig)
        self._networking = _coerce_config(networking, OpenSandboxNetworkingConfig)
        self._shared_storage = _coerce_config(shared_storage, OpenSandboxSharedStorageConfig)
        self._runtime_requirements = _coerce_config(runtime_requirements, OpenSandboxRuntimeRequirementsConfig)
        self._images = _coerce_config(images, OpenSandboxImagesConfig)
        self._setup = _coerce_config(setup, OpenSandboxSetupConfig)
        # Reuse the adapter for this provider's SDK clients. The aiohttp adapter
        # borrows Gym's global session; only the legacy httpx adapter owns a pool.
        self._transport: Any | None = None
        # Sessions own aiohttp clients that only close() releases: aclose()
        # sweeps any still open; ended ones are retired on the next create/attach.
        self._pty_sessions: set[Any] = set()
        self._renewals: dict[str, asyncio.Task[None]] = {}
        # Background command polls by sandbox, and the checkpoint polling scopes open on each sandbox.
        self._command_polls: dict[str, set[_BackgroundCommandPoll]] = {}
        self._checkpoint_poll_scopes: dict[str, int] = {}
        self._parked_poll_scopes: dict[str, int] = {}

    def uploads_independent_of_execs(self) -> bool:
        """A foreground command holds one of the shared HTTP pool's connections for as long as it runs, so
        ``max_connections`` running commands leave an upload none; a background command is polled and holds none."""
        return self._operations.background_exec

    def pull_policy_options(self, policy: str) -> dict[str, Any]:
        """Compose's ``pull_policy`` as the per-sandbox image pull policy extension (``_resolve_extensions`` lets a
        per-sandbox value win over the deployment default)."""
        mapped = {"always": "Always", "missing": "IfNotPresent", "never": "Never"}.get(policy)
        if mapped is None:
            raise ValueError(f"pull policy must be always, missing or never, got {policy!r}")
        return {
            "extensions": {IMAGE_PULL_POLICY_EXTENSION_KEY: mapped, IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY: mapped}
        }

    def validate_runtime_requirements(self, *, cap_add: tuple[str, ...], shm_size: int | None) -> dict[str, str]:
        """Reject requirements without an operator-configured implementation."""
        metadata = {}
        for capability in cap_add:
            if not self._runtime_requirements.capability_probes.get(capability, "").strip():
                raise NotImplementedError(f"OpenSandbox requires a capability probe for {capability!r}")
            metadata.update(self._runtime_requirements.capability_metadata.get(capability, {}))
        if shm_size is not None:
            if isinstance(shm_size, bool) or not isinstance(shm_size, int) or shm_size <= 0:
                raise ValueError("shm_size must be a positive number of bytes")
            key = self._runtime_requirements.shm_size_metadata_key
            if not key:
                raise NotImplementedError("OpenSandbox shm_size requires runtime_requirements.shm_size_metadata_key")
            metadata[key] = str(shm_size)
        return metadata

    async def configure_runtime(
        self, handle: SandboxHandle, *, cap_add: tuple[str, ...], shm_size: int | None
    ) -> None:
        """Probe capabilities and verify the shared memory allocated at creation."""
        self.validate_runtime_requirements(cap_add=cap_add, shm_size=shm_size)
        commands = [
            (capability, self._runtime_requirements.capability_probes[capability], "root") for capability in cap_add
        ]
        if shm_size is not None:
            size_check = (
                "set -- $(stat -fc '%S %b' /dev/shm); "
                f"expected=$((({shm_size} + $1 - 1) / $1 * $1)); "
                '[ "$(($1 * $2))" -eq "$expected" ]'
            )
            commands.append(("shm_size", size_check, None))
        for requirement, command, user in commands:
            result = await self.exec(handle, command, user=user, timeout_s=60)
            if result.return_code != 0:
                raise opensandbox_failure(
                    operation="configure_runtime",
                    exit_code=result.return_code if result.error_type is None else None,
                    failure_kind=PROVIDER_SETUP_FAILED,
                    sandbox_id=handle.sandbox_id,
                    reason=f"Sandbox {handle.sandbox_id!r} cannot satisfy {requirement}: "
                    f"stdout={result.stdout!r}; stderr={result.stderr!r}",
                )

    def validate_port_forwarding(self) -> None:
        """Require explicit deployment support for loopback TCP listeners."""
        if not self._networking.loopback_forwarding:
            raise NotImplementedError("OpenSandbox port forwarding requires networking.loopback_forwarding=true")
        if not self._operations.background_exec:
            raise NotImplementedError("OpenSandbox port forwarding requires operations.background_exec=true")

    async def forward_ports(
        self, handle: SandboxHandle, target_address: str, ports: tuple[int, ...], *, ready_file: str
    ) -> None:
        """Run loopback listeners in a foreground command owned by the collection."""
        self.validate_port_forwarding()
        from nemo_gym.sandbox.providers.opensandbox import _port_forward

        address = str(ipaddress.ip_address(target_address))
        validated_ports = SandboxSpec(ports=ports).ports
        if self._networking.setup_command:
            setup = await self.exec(handle, self._networking.setup_command, user="root", timeout_s=180)
            if setup.return_code != 0:
                raise opensandbox_failure(
                    operation="forward_ports",
                    exit_code=setup.return_code if setup.error_type is None else None,
                    failure_kind=PROVIDER_SETUP_FAILED,
                    sandbox_id=handle.sandbox_id,
                    reason=f"Sandbox {handle.sandbox_id!r} port forwarding setup failed: "
                    f"stdout={setup.stdout!r}; stderr={setup.stderr!r}",
                )
        command = shlex.join(
            [
                self._networking.python_executable,
                "-u",
                "-c",
                Path(_port_forward.__file__).read_text(),
                address,
                ready_file,
                *(str(port) for port in validated_ports),
            ]
        )
        result = await self.exec(handle, command, user="root", timeout_s=None)
        raise opensandbox_failure(
            operation="forward_ports",
            exit_code=result.return_code if result.error_type is None else None,
            failure_kind=PROVIDER_SETUP_FAILED,
            sandbox_id=handle.sandbox_id,
            reason=f"Sandbox {handle.sandbox_id!r} port forwarding exited ({result.return_code}): "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}",
        )

    def validate_networking(self) -> None:
        """Require explicit deployment support for inter-sandbox networking."""
        if not self._networking.enabled:
            raise NotImplementedError("OpenSandbox networking requires networking.enabled=true")

    async def network_address(self, handle: SandboxHandle) -> str:
        """Resolve a direct container IP, independently of client proxy mode."""
        from opensandbox.constants import DEFAULT_EXECD_PORT

        self.validate_networking()
        # Sandbox.get_endpoint follows use_server_proxy; its service adapter
        # exposes the explicit direct lookup needed for peer connections.
        resolved = await self._await_sdk_operation(
            lambda: handle.raw._sandbox_service.get_sandbox_endpoint(handle.sandbox_id, DEFAULT_EXECD_PORT, False),
            operation="get_network_address",
            sandbox_id=handle.sandbox_id,
            timeout_s=self._connection.request_timeout_s,
            replay_safe=True,
        )
        endpoint = str(resolved.endpoint or "")
        parsed = urlsplit(endpoint if "://" in endpoint else f"http://{endpoint}")
        if (
            resolved.headers
            or parsed.port != DEFAULT_EXECD_PORT
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
            or parsed.scheme not in ("http", "https")
        ):
            raise ValueError("OpenSandbox networking requires a direct, unmapped IP endpoint without routing headers")
        address = ipaddress.ip_address(parsed.hostname or "")
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            raise ValueError("OpenSandbox networking requires a peer-reachable IP address")
        return str(address)

    async def set_hosts(self, handle: SandboxHandle, hosts: Mapping[str, str]) -> None:
        """Append validated peer aliases to the sandbox hosts file."""
        self.validate_networking()
        entries = []
        for name, address in hosts.items():
            if len(name) > 253 or not all(
                re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9])?", label) for label in name.split(".")
            ):
                raise ValueError(f"Invalid sandbox hostname: {name!r}")
            entries.append(shlex.quote(f"{ipaddress.ip_address(address)} {name}\n"))
        if entries:
            result = await self.exec(handle, "printf '%s' " + " ".join(entries) + " >> /etc/hosts", user="root")
            if result.return_code != 0:
                raise opensandbox_failure(
                    operation="set_hosts",
                    exit_code=result.return_code if result.error_type is None else None,
                    failure_kind=PROVIDER_SETUP_FAILED,
                    sandbox_id=handle.sandbox_id,
                    reason=f"Could not configure sandbox hosts for {handle.sandbox_id!r}: "
                    f"stdout={result.stdout!r}; stderr={result.stderr!r}",
                )

    def shared_volume_metadata(self) -> dict[str, str]:
        """Return placement metadata required by the configured shared storage."""
        if not self._shared_storage.host_path:
            raise NotImplementedError("OpenSandbox shared storage requires shared_storage.host_path")
        return dict(self._shared_storage.metadata)

    def shared_volume_options(self, source: str | None, target: str, *, read_only: bool = False) -> dict[str, Any]:
        """Mount a project directory, or the shared root for bootstrap when source is None."""
        self.shared_volume_metadata()
        base = self._shared_storage.host_path
        if not PurePosixPath(base).is_absolute():
            raise ValueError("Shared storage host_path must be absolute")
        if source is not None and (
            not source
            or not PurePosixPath(source).parts
            or PurePosixPath(source).is_absolute()
            or ".." in source.split("/")
            or "\\" in source
            or "\x00" in source
        ):
            raise ValueError("Shared storage source must be a safe project-relative path")
        if not PurePosixPath(target).is_absolute() or ".." in target.split("/") or "\x00" in target:
            raise ValueError("Shared storage target must be an absolute container path without traversal")
        name = "shared-" + hashlib.sha256(f"{source}:{target}".encode()).hexdigest()[:16]
        return {
            "volumes": [
                {
                    "name": name,
                    "host": {"path": base},
                    **({"subPath": str(PurePosixPath(source))} if source is not None else {}),
                    "mountPath": target,
                    "readOnly": read_only,
                }
            ]
        }

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
            # OpenSandbox SDK 0.1.15 appends ``/v1`` directly. Normalizing here
            # prevents a configured trailing slash from producing ``//v1``.
            kwargs["domain"] = self._connection.domain.rstrip("/")
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
            # The SDK's execd-facing clients (health ping, commands, files)
            # send only ConnectionConfig.headers — api_key alone never reaches
            # proxied /proxy/* routes, so servers that enforce auth there 401
            # every health ping and create times out at ready_timeout. Inject
            # the key only in proxy mode: a direct sandbox endpoint runs
            # untrusted code and must never see it.
            if self._connection.api_key is not None:
                kwargs["headers"] = {"OPEN-SANDBOX-API-KEY": self._connection.api_key}
        if (
            self._connection.transport_backend == "aiohttp"
            or self._connection.keepalive_expiry_s is not None
            or self._connection.disable_connection_pooling
            or not self._connection.tls_verify
        ):
            kwargs["transport"] = self._get_transport()
        return ConnectionConfig(**kwargs)

    def _get_transport(self) -> Any:
        """Return the provider-owned shared transport, building it on first use."""
        if self._transport is None:
            from nemo_gym.sandbox.providers.opensandbox._http_transport import OpenSandboxHttpTransport

            self._transport = OpenSandboxHttpTransport(self._build_transport())
        return self._transport

    def _build_transport(self) -> Any:
        """Use Gym's global HTTP pool, or the explicitly selected legacy backend."""
        if self._connection.transport_backend == "aiohttp":
            from nemo_gym.sandbox.providers._http_transport import GymAiohttpTransport

            return GymAiohttpTransport(verify=self._connection.tls_verify, raise_on_rate_limit=True)

        import httpx

        max_keepalive = (
            0 if self._connection.disable_connection_pooling else self._connection.max_keepalive_connections
        )
        limits = httpx.Limits(
            max_connections=self._connection.max_connections,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=self._connection.keepalive_expiry_s,
        )
        verify = self._connection.tls_verify
        return httpx.AsyncHTTPTransport(verify=verify, limits=limits, retries=self._connection.connect_retries)

    async def _retire_closed_pty_sessions(self) -> None:
        """Release sessions that ended on their own; their aiohttp client is
        only freed by ``close()``. Called from create/attach so the tracking
        set cannot grow without bound."""
        for stale in [s for s in self._pty_sessions if s.closed]:
            try:
                # Release only: a pump can end because another client took the
                # session over, and an owned close() would DELETE the session
                # that client is still using. Ended-by-exit sessions lose their
                # server-side record with the sandbox instead.
                stale._owned = False
                await stale.close()
            except Exception:
                LOGGER.warning(
                    "Failed to close ended PTY session %r", getattr(stale, "session_id", "?"), exc_info=True
                )
            self._pty_sessions.discard(stale)

    async def aclose(self) -> None:
        """Close provider-owned resources."""
        for sandbox_id in list(self._renewals):
            await self._stop_renewal(sandbox_id)
        # PTY sessions hold their own aiohttp clients, which the shared httpx
        # transport below does not cover.
        for session in list(self._pty_sessions):
            try:
                await session.close()
            except Exception:
                LOGGER.warning(
                    "Failed to close PTY session %r during aclose", getattr(session, "session_id", "?"), exc_info=True
                )
        self._pty_sessions.clear()
        transport, self._transport = self._transport, None
        if transport is not None:
            await transport.aclose()

    # -- checkpoint polling scopes ------------------------------------------------------------------------------

    @contextmanager
    def checkpoint_exec_polling(self, handle: SandboxHandle) -> Iterator[None]:
        """Wake the background command polls of this sandbox and poll at least every
        ``operations.checkpoint_background_poll_interval_s`` until the scope exits.

        For a harness being stopped or paused for a checkpoint, so its command is seen to end promptly.
        """
        sandbox_id = handle.sandbox_id
        self._checkpoint_poll_scopes[sandbox_id] = self._checkpoint_poll_scopes.get(sandbox_id, 0) + 1
        self._wake_command_polls(sandbox_id)
        try:
            yield
        finally:
            self._leave_scope(self._checkpoint_poll_scopes, sandbox_id)
            self._wake_command_polls(sandbox_id)

    @contextmanager
    def parked_exec_polling(self, handle: SandboxHandle) -> Iterator[None]:
        """Hold the background command polls of this sandbox, without asking it, until the scope exits.

        For a harness launch paused for a checkpoint: the sandbox is being snapshotted and may not answer for a
        while. A status request that fails after the scope opened is asked again once it exits, with a fresh retry
        allowance. Do not wait inside the scope for a command of this sandbox: its polls are held too.
        """
        sandbox_id = handle.sandbox_id
        self._parked_poll_scopes[sandbox_id] = self._parked_poll_scopes.get(sandbox_id, 0) + 1
        for poll in self._command_polls.get(sandbox_id, ()):
            poll.parked = True
        try:
            yield
        finally:
            self._leave_scope(self._parked_poll_scopes, sandbox_id)
            self._wake_command_polls(sandbox_id)

    @staticmethod
    def _leave_scope(scopes: dict[str, int], sandbox_id: str) -> None:
        remaining = scopes[sandbox_id] - 1
        if remaining:
            scopes[sandbox_id] = remaining
        else:
            del scopes[sandbox_id]

    def _wake_command_polls(self, sandbox_id: str) -> None:
        for poll in self._command_polls.get(sandbox_id, ()):
            poll.wakeup.set()

    async def _wait_command_poll(self, poll: _BackgroundCommandPoll, *, interval_s: float) -> None:
        """Wait for the next poll or a scope change, including a wakeup delivered before this call."""
        if poll.wakeup.is_set():
            poll.wakeup.clear()
            return
        sleep_task = asyncio.create_task(asyncio.sleep(interval_s))
        wake_task = asyncio.create_task(poll.wakeup.wait())
        try:
            done, _ = await asyncio.wait({sleep_task, wake_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            sleep_task.cancel()
            wake_task.cancel()
            await asyncio.gather(sleep_task, wake_task, return_exceptions=True)
            poll.wakeup.clear()

    async def _wait_while_parked(self, sandbox_id: str, poll: _BackgroundCommandPoll) -> None:
        while sandbox_id in self._parked_poll_scopes:
            await self._wait_command_poll(poll, interval_s=self._operations.background_poll_interval_s)

    # -- snapshots ----------------------------------------------------------------------------------------------

    async def snapshot(self, handle: SandboxHandle, *, name: str | None = None) -> str:
        """Start a snapshot of a running sandbox and return its id at once.

        Creation is asynchronous on the server: the snapshot can create sandboxes only once
        :meth:`wait_snapshot_ready` returns. The sandbox keeps running either way. Not retried: a repeated request
        could leave a second snapshot nobody knows the id of.
        """
        info = await self._await_sdk_call(
            handle.raw.create_snapshot(name),
            operation="create_snapshot",
            sandbox_id=handle.sandbox_id,
            timeout_s=self._connection.request_timeout_s,
        )
        snapshot_id = getattr(info, "id", None)
        if not snapshot_id:
            raise RuntimeError(
                f"OpenSandbox create_snapshot returned no snapshot id for sandbox {handle.sandbox_id!r}"
            )
        return str(snapshot_id)

    async def wait_snapshot_ready(self, snapshot_id: str) -> None:
        """Poll the management API until the snapshot is ``Ready``, within ``operations.snapshot_timeout_s``.

        A snapshot that fails, or is not ready in time, raises. Transient answers are waited out within the
        timeout: 429, 5xx and connection errors at any time, and a 404 before the first successful lookup for
        ``operations.snapshot_lookup_grace_s`` (the API can answer 404 for a snapshot it has only just created).
        Any other answer, or a 404 once the snapshot was found, raises at once.
        """
        timeout_s = self._operations.snapshot_timeout_s
        grace_s = self._operations.snapshot_lookup_grace_s
        poll_s = self._create.connect_poll_s
        loop = asyncio.get_running_loop()
        started = loop.time()
        found = False
        not_found = 0
        deadline = asyncio.timeout(timeout_s)
        path = f"/v1/snapshots/{quote(snapshot_id, safe='')}"
        try:
            async with deadline:
                while True:
                    try:
                        status, payload = await self._management_request("GET", path)
                    except (OSError, asyncio.TimeoutError, ConnectionError) as error:
                        LOGGER.warning("OpenSandbox snapshot %r lookup failed; retrying: %r", snapshot_id, error)
                        await asyncio.sleep(poll_s)
                        continue
                    if status == 404 and not found and loop.time() - started < grace_s:
                        not_found += 1
                        await asyncio.sleep(poll_s)
                        continue
                    if status == 429 or status >= 500:
                        LOGGER.warning("OpenSandbox snapshot %r lookup -> HTTP %d; retrying", snapshot_id, status)
                        await asyncio.sleep(poll_s)
                        continue
                    if status >= 400:
                        detail = ""
                        if status == 404 and not_found and not found:
                            detail = f" ({not_found + 1} lookups over {loop.time() - started:.1f}s)"
                        raise RuntimeError(
                            f"OpenSandbox snapshot {snapshot_id!r} lookup failed -> HTTP {status}{detail}"
                        )
                    if not found and not_found:
                        LOGGER.warning(
                            "OpenSandbox snapshot %r appeared %.1fs into its wait; %d earlier lookups got 404",
                            snapshot_id,
                            loop.time() - started,
                            not_found,
                        )
                    found = True
                    state = _snapshot_state(payload)
                    if state == "ready":
                        return
                    if state in {"failed", "error"}:
                        reason = (payload.get("status") or {}).get("message") or state
                        raise RuntimeError(f"OpenSandbox snapshot {snapshot_id!r} failed: {reason}")
                    if not state:
                        # Waiting on a record that reports no state would only end in the timeout.
                        raise RuntimeError(
                            f"OpenSandbox snapshot {snapshot_id!r} lookup returned no state: {payload!r}"
                        )
                    await asyncio.sleep(poll_s)
        except TimeoutError as error:
            if not deadline.expired():
                raise
            raise TimeoutError(
                f"Timed out waiting for OpenSandbox snapshot {snapshot_id!r} to be ready after {timeout_s:g}s"
            ) from error

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a snapshot over the management API; one that is already gone (404) is not an error.

        429, 5xx and connection errors are retried ``operations.retries`` times with the operation backoff.
        """
        path = f"/v1/snapshots/{quote(snapshot_id, safe='')}"
        attempts = self._operations.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                status, _ = await self._management_request("DELETE", path)
            except (OSError, asyncio.TimeoutError, ConnectionError) as error:
                if attempt == attempts:
                    raise RuntimeError(f"OpenSandbox snapshot {snapshot_id!r} delete failed: {error!r}") from error
            else:
                if status < 400 or status == 404:
                    return
                if not (status == 429 or status >= 500) or attempt == attempts:
                    raise RuntimeError(f"OpenSandbox snapshot {snapshot_id!r} delete failed -> HTTP {status}")
            await asyncio.sleep(
                min(self._operations.retry_delay_s * 2 ** (attempt - 1), self._operations.retry_max_delay_s)
            )

    async def list_snapshots(self, prefix: str) -> list[dict[str, Any]]:
        """Every snapshot whose name starts with ``prefix``, as ``{"id", "name", "state", "sandbox_id"}``.

        The management API's ``name`` query parameter matches a whole name, not a prefix, so this reads every page
        of the listing and filters here. Pages are collected before anything is deleted, so deletes do not shift
        them. A failed page request (after the operation retries for 429 and 5xx) raises.
        """
        found: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = await self._management_get_page(page)
            items = payload.get("items")
            pagination = payload.get("pagination")
            if not isinstance(items, list) or not isinstance(pagination, dict):
                raise RuntimeError("OpenSandbox snapshot listing is missing items or pagination")
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    raise RuntimeError("OpenSandbox snapshot listing contains an item without an id")
                name = item.get("name")
                if isinstance(name, str) and name.startswith(prefix):
                    found.append(
                        {
                            "id": item["id"],
                            "name": name,
                            "state": _snapshot_state(item),
                            "sandbox_id": item.get("sandboxId"),
                        }
                    )
            if pagination.get("hasNextPage") is not True:
                return found
            page += 1

    async def create_from_snapshot(self, snapshot_id: str, spec: SandboxSpec) -> SandboxHandle:
        """Create a sandbox from a snapshot: OpenSandbox takes ``snapshot_id`` in place of the image."""
        fork = replace(spec, image=None, provider_options={**spec.provider_options, "snapshot_id": snapshot_id})
        try:
            return await self.create(fork)
        except Exception as error:
            if "SNAPSHOT::NOT_FOUND" in f"{error!s} {error!r}":
                raise SandboxSnapshotNotFoundError(f"OpenSandbox has no snapshot {snapshot_id!r}: {error}") from error
            raise

    async def _management_get_page(self, page: int) -> dict[str, Any]:
        attempts = self._operations.retries + 1
        params = {"page": str(page), "pageSize": str(SNAPSHOT_LIST_PAGE_SIZE)}
        for attempt in range(1, attempts + 1):
            try:
                status, payload = await self._management_request("GET", "/v1/snapshots", params=params)
            except (OSError, asyncio.TimeoutError, ConnectionError) as error:
                if attempt == attempts:
                    raise RuntimeError(f"OpenSandbox snapshot listing failed: {error!r}") from error
            else:
                if status < 400:
                    return payload
                if not (status == 429 or status >= 500) or attempt == attempts:
                    raise RuntimeError(f"OpenSandbox snapshot listing failed -> HTTP {status}")
            await asyncio.sleep(
                min(self._operations.retry_delay_s * 2 ** (attempt - 1), self._operations.retry_max_delay_s)
            )
        raise RuntimeError("OpenSandbox snapshot listing retry loop did not run")

    async def _management_request(
        self, method: str, path: str, *, params: Mapping[str, str] | None = None
    ) -> tuple[int, dict[str, Any]]:
        """One call to the server's management API over Gym's shared aiohttp client, which the SDK handle does not
        expose; returns the HTTP status and the JSON object body (``{}`` for an error status or an empty body)."""
        import aiohttp

        from nemo_gym.client.http_client import request

        if self._connection.domain is None:
            raise RuntimeError("OpenSandbox snapshot operations require connection.domain")
        # The connection config keeps the scheme in ``protocol`` (the SDK's default is http) and the host, with any
        # path prefix, in ``domain``.
        base_url = f"{self._connection.protocol or 'http'}://{self._connection.domain.rstrip('/')}"
        headers = {"OPEN-SANDBOX-API-KEY": self._connection.api_key} if self._connection.api_key else {}
        timeout_s = float(self._connection.request_timeout_s) if self._connection.request_timeout_s else 30.0
        # Same certificate policy as the SDK transport and the PTY sockets (connection.tls_verify).
        tls: dict[str, Any] = {} if self._connection.tls_verify else {"ssl": False}
        extra: dict[str, Any] = {"params": dict(params)} if params else {}
        response = await request(
            method,
            f"{base_url}{path}",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
            _control=True,
            **tls,
            **extra,
        )
        async with response:
            if response.status >= 400:
                return response.status, {}
            try:
                payload = await response.json(content_type=None)
            except (aiohttp.ContentTypeError, ValueError) as error:
                raise RuntimeError(
                    f"OpenSandbox management API answered {method} {path} with HTTP {response.status} and a body "
                    "that is not JSON"
                ) from error
            if payload is not None and not isinstance(payload, dict):
                raise RuntimeError(
                    f"OpenSandbox management API answered {method} {path} with HTTP {response.status} and a body "
                    f"that is not a JSON object: {payload!r}"
                )
            return response.status, payload or {}

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        """Return a descriptor for reattaching to this sandbox by id.

        OpenSandbox sandboxes are reachable by id from any process that has the
        connection config, so the id alone is enough to reconnect and no sandbox
        server is needed to share one. ``scope`` is ignored: OpenSandbox has no
        lease concept of its own.
        """
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        """Rebuild a live handle from an OpenSandbox sandbox id via the SDK.

        Running sandboxes are health-checked unless the caller opts out. A
        paused sandbox has no exec daemon to check; resume rebuilds its
        endpoints and performs the health check instead.
        """
        Sandbox, _, _, _, _ = _require_opensandbox_sdk()
        sandbox_id = str(descriptor["sandbox_id"])
        timeout_s = self._create.connect_attempt_timeout_s
        # A cancelled SDK call skips its own transport cleanup; see resume().
        config = self._connection_config(request_timeout_s=timeout_s).with_transport_if_missing()
        sandbox = None
        try:
            async with asyncio.timeout(timeout_s):
                sandbox = await Sandbox.connect(
                    sandbox_id,
                    connection_config=config,
                    connect_timeout=timedelta(seconds=timeout_s),
                    skip_health_check=True,
                )
                handle = SandboxHandle(sandbox_id=str(sandbox.id), provider_name=self.name, raw=sandbox)
                if not self._create.skip_health_check and await self.status(handle) != SandboxStatus.PAUSED:
                    await sandbox.check_ready(
                        timedelta(seconds=timeout_s),
                        timedelta(seconds=self._create.connect_poll_s),
                    )
                return handle
        except BaseException as error:
            if sandbox is None:
                await config.close_transport_if_owned()
            else:
                try:
                    await self._await_sdk_call(
                        sandbox.close(),
                        operation="close_after_connect_failure",
                        sandbox_id=sandbox_id,
                        timeout_s=self._operations.close_timeout_s,
                    )
                except Exception as e:
                    LOGGER.warning("Failed to close OpenSandbox handle after connect failure %r: %r", sandbox_id, e)
            if isinstance(error, Exception) and not isinstance(error, NgFailureEvent):
                raise opensandbox_failure(operation="connect", sandbox_id=sandbox_id, cause=error) from error
            raise

    async def pause(self, handle: SandboxHandle) -> None:
        """Pause a sandbox and wait until it reports paused.

        Local PTY clients are detached first, while execd can still answer the
        close handshake; server sessions are never deleted, so they remain
        attachable if the pause request fails. After resume, the Kubernetes
        backend has replaced the runtime (open a new PTY); the Docker backend
        thawed it (re-attach by id).
        """
        # Bounded per session and outside the pause deadline.
        for session in [s for s in self._pty_sessions if s._sandbox_id == handle.sandbox_id]:
            try:
                session._owned = False
                await self._await_sdk_call(
                    session.close(),
                    operation="detach_pty",
                    sandbox_id=handle.sandbox_id,
                    timeout_s=self._operations.close_timeout_s,
                )
            except Exception:
                LOGGER.warning(
                    "Failed to detach PTY session %r before pausing sandbox %r",
                    getattr(session, "session_id", "?"),
                    handle.sandbox_id,
                    exc_info=True,
                )
            self._pty_sessions.discard(session)

        timeout_s = self._operations.pause_resume_timeout_s
        lifecycle_timeout = asyncio.timeout(timeout_s)
        try:
            async with lifecycle_timeout:
                await self._await_sdk_call(
                    handle.raw.pause(),
                    operation="pause",
                    sandbox_id=handle.sandbox_id,
                    timeout_s=self._connection.request_timeout_s,
                )
                while True:
                    status = await self.status(handle)
                    if status == SandboxStatus.PAUSED:
                        return
                    if status in {SandboxStatus.ERROR, SandboxStatus.STOPPED}:
                        raise opensandbox_failure(
                            operation="pause",
                            sandbox_id=handle.sandbox_id,
                            failure_kind=PROVIDER_UNAVAILABLE,
                            reason=f"OpenSandbox sandbox {handle.sandbox_id!r} entered {status.value} while pausing",
                        )
                    await asyncio.sleep(self._create.connect_poll_s)
        except TimeoutError as e:
            if not lifecycle_timeout.expired():
                raise opensandbox_failure(operation="pause", sandbox_id=handle.sandbox_id, cause=e) from e
            raise opensandbox_failure(
                operation="pause",
                sandbox_id=handle.sandbox_id,
                failure_kind=PROVIDER_IO_TIMEOUT,
                reason=f"Timed out waiting for OpenSandbox sandbox {handle.sandbox_id!r} to pause after {timeout_s:g}s",
            ) from e

        except Exception as error:
            if isinstance(error, NgFailureEvent):
                raise
            raise opensandbox_failure(operation="pause", sandbox_id=handle.sandbox_id, cause=error) from error

    async def resume(self, handle: SandboxHandle) -> None:
        """Resume a paused sandbox and rebuild its SDK clients and endpoints.

        One ``pause_resume_timeout_s`` deadline covers the request, endpoint
        rebuild and readiness check. On timeout the server-side state is
        unknown: reconnect and check ``status()`` before retrying. See
        ``pause()`` for what happens to PTY sessions.
        """
        Sandbox, _, _, _, _ = _require_opensandbox_sdk()
        timeout_s = self._operations.pause_resume_timeout_s
        # Hold the config: cancellation skips the SDK's own cleanup, which would
        # leak an SDK-owned default transport. Closing our shared one is a no-op.
        config = self._connection_config().with_transport_if_missing()
        lifecycle_timeout = asyncio.timeout(timeout_s)
        try:
            async with lifecycle_timeout:
                resumed = await Sandbox.resume(
                    handle.sandbox_id,
                    connection_config=config,
                    resume_timeout=timedelta(seconds=timeout_s),
                    health_check_polling_interval=timedelta(seconds=self._create.connect_poll_s),
                    skip_health_check=self._create.skip_health_check,
                )
        except BaseException as e:
            await config.close_transport_if_owned()
            if isinstance(e, TimeoutError) and lifecycle_timeout.expired():
                raise opensandbox_failure(
                    operation="resume",
                    sandbox_id=handle.sandbox_id,
                    failure_kind=PROVIDER_IO_TIMEOUT,
                    reason=f"Timed out waiting for OpenSandbox sandbox {handle.sandbox_id!r} to resume after {timeout_s:g}s; "
                    "reconnect and check status() before retrying",
                ) from e
            if isinstance(e, Exception) and not isinstance(e, NgFailureEvent):
                raise opensandbox_failure(operation="resume", sandbox_id=handle.sandbox_id, cause=e) from e
            raise

        # New handle first, so a cancellation during the best-effort close
        # cannot lose a completed resume.
        old_raw, handle.raw = handle.raw, resumed
        try:
            await self._await_sdk_call(
                old_raw.close(),
                operation="close_pre_resume_handle",
                sandbox_id=handle.sandbox_id,
                timeout_s=self._operations.close_timeout_s,
            )
        except Exception as e:
            LOGGER.warning("Failed to close pre-resume OpenSandbox handle %r: %r", handle.sandbox_id, e)

    async def _await_sdk_call(
        self,
        awaitable: Any,
        *,
        operation: str,
        sandbox_id: str | None,
        timeout_s: float | None,
    ) -> Any:
        if timeout_s is None:
            return await awaitable

        timer = asyncio.timeout(timeout_s)
        try:
            async with timer:
                return await awaitable
        except TimeoutError as e:
            if not timer.expired():
                raise
            raise TimeoutError(
                f"Timed out during OpenSandbox {operation} after {timeout_s:g}s; sandbox_id={sandbox_id!r}"
            ) from e

    async def _await_sdk_operation(
        self,
        operation_factory: Callable[[], Awaitable[Any]],
        *,
        operation: str,
        sandbox_id: str | None,
        timeout_s: float | None,
        retries: int | None = None,
        # The default classifier treats per-call timeouts as terminal, which is
        # right for mutating calls but wrong for short idempotent polls; those
        # callers pass their own predicate.
        is_retryable: Callable[[BaseException], bool] | None = None,
        replay_safe: bool = False,
    ) -> Any:
        renewal = self._renewals.get(sandbox_id) if sandbox_id is not None else None
        if renewal is not None and renewal.done() and not renewal.cancelled():
            error = renewal.exception()
            if error is not None:
                raise opensandbox_failure(
                    operation="lifetime renewal",
                    sandbox_id=sandbox_id,
                    cause=error,
                    failure_kind=PROVIDER_UNAVAILABLE,
                    reason="Sandbox lifetime renewal failed before provider operation",
                ) from error
        AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential = _require_tenacity()
        retry_count = self._operations.retries if retries is None else retries
        max_attempts = retry_count + 1

        def _before_sleep(retry_state: Any) -> None:
            _log_operation_retry(retry_state, operation=operation, sandbox_id=sandbox_id)

        def retry_allowed(error: BaseException) -> bool:
            return (
                isinstance(error, NgFailureEvent)
                and (replay_safe or error.failure_kind != PROVIDER_IO_TIMEOUT)
                and (is_retryable is None or is_retryable(error.__cause__ or error))
                and failure_outcome_policy(
                    error,
                    config=self._operations.outcome_policy,
                    classification=self._operations.outcome_policy.classification,
                )
                == OutcomeAction.RETRY
            )

        backoff = wait_random_exponential(
            multiplier=self._operations.retry_delay_s, max=self._operations.retry_max_delay_s
        )
        retry_policy = AsyncRetrying(
            retry=retry_if_exception(retry_allowed),
            stop=stop_after_attempt(max_attempts),
            wait=lambda state: _retry_delay(
                state.outcome.exception(), fallback=backoff(state), max_delay_s=self._operations.retry_max_delay_s
            ),
            before_sleep=_before_sleep,
            reraise=True,
        )
        async for attempt in retry_policy:
            with attempt:
                try:
                    return await self._await_sdk_call(
                        operation_factory(),
                        operation=operation,
                        sandbox_id=sandbox_id,
                        timeout_s=timeout_s,
                    )
                except Exception as error:
                    if isinstance(error, NgFailureEvent):
                        failure = error
                    else:
                        failure = opensandbox_failure(operation=operation, sandbox_id=sandbox_id, cause=error)
                    observe_failure(failure)
                    if failure is error:
                        raise
                    raise failure from error

        raise RuntimeError("OpenSandbox SDK operation retry loop did not run")

    async def _submit_command(
        self,
        operation_factory: Callable[[], Awaitable[Any]],
        *,
        operation: str,
        sandbox_id: str,
        timeout_s: float | None,
        retries: int,
    ) -> Any:
        """Submit once unless the request never left the client.

        Neither a 502 nor a timeout proves non-delivery, and command retry settings cannot
        override that: the provider has no idempotency key or authoritative non-delivery
        receipt. A connection that never opened is that receipt, so such a failure, and only
        such a failure, is sent again within ``operations.retries``.
        """
        renewal = self._renewals.get(sandbox_id)
        if renewal is not None and renewal.done() and not renewal.cancelled():
            error = renewal.exception()
            if error is not None:
                raise opensandbox_failure(
                    operation="lifetime renewal",
                    sandbox_id=sandbox_id,
                    cause=error,
                    failure_kind=PROVIDER_UNAVAILABLE,
                    reason="Sandbox lifetime renewal failed before command submission",
                ) from error
        return await self._await_sdk_operation(
            operation_factory,
            operation=operation,
            sandbox_id=sandbox_id,
            timeout_s=timeout_s,
            retries=self._operations.retries,
            is_retryable=_request_was_not_sent,
            replay_safe=True,
        )

    async def _run_setup_commands(self, handle: SandboxHandle) -> None:
        for command in self._setup.commands:
            result = await self._exec_setup_command(handle, command)
            if result.return_code != 0:
                raise opensandbox_failure(
                    operation="setup",
                    exit_code=result.return_code if result.error_type is None else None,
                    sandbox_id=handle.sandbox_id,
                    failure_kind=PROVIDER_SETUP_FAILED,
                    reason=f"Sandbox {handle.sandbox_id!r} setup command failed (exit {result.return_code}): "
                    f"{command}\n{(result.stderr or result.stdout or '')[-500:]}",
                )

    async def _exec_setup_command(self, handle: SandboxHandle, command: str) -> SandboxExecResult:
        """One ``setup.commands`` entry as root, or as the image's own user when the sandbox cannot switch
        identity.

        An image with a non-root ``USER`` on a cell that drops ``CAP_SETUID`` refuses the switch before the command
        runs (``nemo_gym.sandbox.transfer.exec_as_root`` degrades the same way), so a command written to no-op where
        it does not apply gets to no-op, and one that needs root there fails on its own result, as before.
        """
        result = await self.exec(handle, command, user="root", timeout_s=self._setup.timeout_s)
        if result.error_type != "user_switch_unavailable":
            return result
        LOGGER.debug(
            "Sandbox %r cannot switch to root; running setup command as the image's user: %s",
            handle.sandbox_id,
            command.strip().splitlines()[0][:80] if command.strip() else command,
        )
        return await self.exec(handle, command, timeout_s=self._setup.timeout_s)

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
                error = opensandbox_failure(
                    operation="create_probe",
                    sandbox_id=handle.sandbox_id,
                    failure_kind=PROVIDER_NOT_READY,
                    reason="OpenSandbox sandbox failed create probe command before "
                    "the startup deadline; "
                    f"sandbox_id={handle.sandbox_id!r}, "
                    f"command={self._probe.command!r}, "
                    f"successful_probes={successful_probes}/{self._probe.stable_count}, "
                    f"attempts={attempt_number}, deadline_s={deadline_s:g}",
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
                        user=self._probe.user,
                    ),
                    timeout=command_timeout_s,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exception = (
                    e
                    if isinstance(e, NgFailureEvent)
                    else opensandbox_failure(operation="create_probe", sandbox_id=handle.sandbox_id, cause=e)
                )
                observe_failure(last_exception)
                successful_probes = 0
                sleep_s = min(self._create.connect_poll_s, max(deadline - loop.time(), 0.0))
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                continue

            stdout = result.stdout or ""
            expected = self._probe.expected_stdout
            if result.return_code != 0 or (expected is not None and expected not in stdout):
                last_exception = opensandbox_failure(
                    operation="create_probe",
                    sandbox_id=handle.sandbox_id,
                    exit_code=result.return_code if result.error_type is None else None,
                    failure_kind=PROVIDER_NOT_READY,
                    reason="OpenSandbox sandbox create probe command returned an "
                    f"unexpected result; sandbox_id={handle.sandbox_id!r}, "
                    f"return_code={result.return_code}, expected_stdout={expected!r}, "
                    f"stdout={stdout[:200]!r}, stderr={(result.stderr or '')[:200]!r}, "
                    f"probe={successful_probes + 1}/{self._probe.stable_count}",
                )
                observe_failure(last_exception)
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
                error = opensandbox_failure(
                    operation="connect_after_create",
                    sandbox_id=handle.sandbox_id,
                    failure_kind=PROVIDER_CREATE_TIMEOUT,
                    reason="Timed out connecting to OpenSandbox sandbox after SDK create; "
                    f"sandbox_id={handle.sandbox_id!r}, timeout_s={timeout_s:g}",
                )
                raise error from last_exception

            attempt_timeout_s = min(self._create.connect_attempt_timeout_s, remaining_s)
            try:
                sandbox = await asyncio.wait_for(
                    Sandbox.connect(
                        handle.sandbox_id,
                        connection_config=self._connection_config(),
                        connect_timeout=timedelta(seconds=attempt_timeout_s),
                        skip_health_check=self._create.skip_health_check,
                    ),
                    timeout=attempt_timeout_s,
                )
                return SandboxHandle(sandbox_id=str(sandbox.id), provider_name=self.name, raw=sandbox)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exception = (
                    e
                    if isinstance(e, NgFailureEvent)
                    else opensandbox_failure(operation="connect_after_create", sandbox_id=handle.sandbox_id, cause=e)
                )
                observe_failure(last_exception)
                if not _is_retryable_create_error(last_exception, config=self._create.outcome_policy):
                    if last_exception is e:
                        raise
                    raise last_exception from e
                sleep_s = min(
                    max(
                        self._create.connect_poll_s,
                        _retry_delay(
                            last_exception,
                            fallback=self._create.connect_poll_s,
                            max_delay_s=self._create.retry_max_delay_s,
                        ),
                    ),
                    max(deadline - loop.time(), 0.0),
                )
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

    async def endpoint(
        self,
        handle: SandboxHandle,
        port: int,
    ) -> SandboxEndpoint:
        """Resolve one client-reachable direct or server-proxied service URL."""

        get_endpoint = getattr(handle.raw, "get_endpoint", None)
        if get_endpoint is None:
            raise NotImplementedError(
                "The installed opensandbox SDK does not expose Sandbox.get_endpoint; "
                "sandbox service endpoints require opensandbox>=0.1.15"
            )
        resolved = await self._await_sdk_operation(
            lambda: get_endpoint(port),
            operation="get_endpoint",
            sandbox_id=handle.sandbox_id,
            timeout_s=(
                float(self._connection.request_timeout_s) if self._connection.request_timeout_s is not None else None
            ),
            replay_safe=True,
        )
        endpoint_url = str(getattr(resolved, "endpoint", "") or "")
        if not endpoint_url:
            raise opensandbox_failure(
                operation="endpoint",
                failure_kind=PROVIDER_PROTOCOL_ERROR,
                sandbox_id=handle.sandbox_id,
                reason=f"OpenSandbox returned an empty endpoint for sandbox {handle.sandbox_id!r} port {port}",
            )
        if "://" not in endpoint_url:
            # Use the SDK handle's effective configuration so environment-
            # resolved domains and protocols match the lifecycle request.
            scheme = urlsplit(handle.raw.connection_config.get_base_url()).scheme or "http"
            endpoint_url = f"{scheme}://{endpoint_url.lstrip('/')}"
        connection_config = getattr(handle.raw, "connection_config", None)
        headers = dict(getattr(connection_config, "headers", None) or {})
        # Match the SDK's service adapters: connection-wide headers apply to
        # every request, while endpoint-specific routing or auth headers win.
        # The upstream proxy-auth fix adds the management API key to
        # ConnectionConfig.headers only in server-proxy mode, so direct
        # sandbox endpoints never receive it.
        headers.update(getattr(resolved, "headers", None) or {})
        return SandboxEndpoint(endpoint=endpoint_url, headers=headers)

    def _start_renewal(self, handle: SandboxHandle, ttl_s: int | float, *, lifetime_s: int | float | None) -> None:
        """Renew the sandbox's TTL every ``renew_interval_s`` (jittered by 10%) until cancelled or the sandbox is gone.

        Without a lease every renewal sets the TTL to ``ttl_s`` again, and one that fails after its retries ends the
        loop and fails the sandbox's later operations. With a lease (``renew_ttl_s``) every renewal sets it to the
        lease, never past ``lifetime_s`` (the spec's ``ttl_s``) from creation; a renewal that fails is logged and the
        next one tries again, since the sandbox has the lease minus one interval of slack. Either way, a renewal
        that finds the sandbox gone (stopped or deleted elsewhere) ends the loop quietly.
        """
        interval_s = self._create.renew_interval_s
        lease = self._create.renew_ttl_s is not None
        loop = asyncio.get_running_loop()
        lifetime_end = None if lifetime_s is None else loop.time() + lifetime_s
        timeout_s = self._create.renew_timeout_s or self._connection.request_timeout_s or 60

        async def renew() -> None:
            while True:
                await asyncio.sleep(interval_s * random.uniform(0.9, 1.1))
                duration = ttl_s
                if lifetime_end is not None:
                    duration = min(duration, lifetime_end - loop.time())
                    if duration <= 0:
                        return  # the spec's lifetime is over; the service ends the sandbox at its current TTL
                try:
                    await self._await_sdk_operation(
                        lambda: handle.raw.renew(timedelta(seconds=duration)),
                        operation="renew",
                        sandbox_id=handle.sandbox_id,
                        timeout_s=timeout_s,
                    )
                except Exception as error:
                    if _is_missing_sandbox_delete_error(error):
                        LOGGER.info("OpenSandbox sandbox %r is gone; stopping its renewal", handle.sandbox_id)
                        return
                    if not lease:
                        raise
                    LOGGER.warning(
                        "OpenSandbox lifetime renewal failed for sandbox %r; retrying in about %gs: %r",
                        handle.sandbox_id,
                        interval_s,
                        error,
                    )

        def report_failure(task: asyncio.Task[None]) -> None:
            if not task.cancelled() and (error := task.exception()) is not None:
                LOGGER.error("OpenSandbox lifetime renewal failed for sandbox %r: %r", handle.sandbox_id, error)

        task = asyncio.create_task(renew(), name=f"opensandbox-renew-{handle.sandbox_id}")
        task.add_done_callback(report_failure)
        self._renewals[handle.sandbox_id] = task

    async def _stop_renewal(self, sandbox_id: str) -> BaseException | None:
        task = self._renewals.pop(sandbox_id, None)
        if task is None:
            return None
        task.cancel()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
        return result if isinstance(result, Exception) else None

    async def _create_once(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a sandbox through ``opensandbox.Sandbox.create``."""
        # A lease caps the spec's TTL; the renewals then keep the sandbox alive up to the spec's lifetime.
        ttl_s = spec.ttl_s
        if self._create.renew_ttl_s is not None:
            ttl_s = self._create.renew_ttl_s if spec.ttl_s is None else min(spec.ttl_s, self._create.renew_ttl_s)
        elif self._create.renew_interval_s is not None and (
            spec.ttl_s is None or self._create.renew_interval_s >= spec.ttl_s
        ):
            raise ValueError("create.renew_interval_s requires a longer, explicit sandbox ttl_s")
        Sandbox, _, _, _, _ = _require_opensandbox_sdk()
        options = OpenSandboxProviderOptions.from_mapping(spec.provider_options)

        timeout_s, request_timeout_s = self._create_budget(spec)
        # One key per attempt: a request that times out never learns its sandbox id, and this is how the
        # orphan it may have left pending is found and deleted.
        attempt_key = f"{self._attribution.key_prefix}{CREATE_ATTEMPT_KEY}"
        metadata = {**spec.metadata, attempt_key: uuid.uuid4().hex}

        kwargs: dict[str, Any] = {
            "env": {**self._setup.env, **(spec.env or {})} or None,
            "metadata": metadata,
            "resource": _resource_map(spec.resources),
            "extensions": self._resolve_extensions(options.extensions),
            "connection_config": self._connection_config(request_timeout_s=request_timeout_s),
        }
        if options.resource_requests == "limits":
            # Match the SDK's defaults when a service has no explicit resources.
            kwargs["resource"] = kwargs["resource"] or _resource_map(SDK_DEFAULT_RESOURCE_LIMITS)
            kwargs["resource_requests"] = dict(kwargs["resource"])
        elif options.resource_requests is not None:
            # A spec without resources gets the SDK's default limits, so the requests are held to those.
            limits = spec.resources if kwargs["resource"] else SDK_DEFAULT_RESOURCE_LIMITS
            requests, clamped = clamp_resource_requests(options.resource_requests, limits)
            for key, (request, limit) in clamped.items():
                LOGGER.warning(
                    "OpenSandbox resource request %s=%s exceeds the sandbox's limit %s%s; clamping the request to "
                    "the limit (Kubernetes never schedules a pod whose request exceeds its limit); image=%r",
                    key,
                    request,
                    limit,
                    "" if kwargs["resource"] else " (the SDK's default, as the sandbox declares no resources)",
                    spec.image,
                )
            kwargs["resource_requests"] = _resource_map(SandboxResources.from_mapping(requests))
        if spec.image is not None:
            image, auth = self._images.resolve(spec.image)
            kwargs["image"] = _to_image_spec(image, options.image_auth or auth)
        if options.snapshot_id is not None:
            kwargs["snapshot_id"] = options.snapshot_id
        if ttl_s is not None:
            kwargs["timeout"] = timedelta(seconds=ttl_s)
        if spec.ready_timeout_s is not None:
            kwargs["ready_timeout"] = timedelta(seconds=spec.ready_timeout_s)
        if spec.entrypoint is not None:
            kwargs["entrypoint"] = spec.entrypoint
        if options.platform is not None:
            kwargs["platform"] = _to_platform_spec(options.platform)
        if options.network_policy is not None:
            kwargs["network_policy"] = _to_network_policy(options.network_policy)
        if options.volumes:
            kwargs["volumes"] = _to_volumes(list(options.volumes))
        if self._create.skip_health_check:
            kwargs["skip_health_check"] = True
        elif options.skip_health_check is not None:
            kwargs["skip_health_check"] = options.skip_health_check

        if is_span_group_enabled(GymSpanGroup.SANDBOX):
            from opentelemetry.trace import get_current_span

            from nemo_gym.sandbox.providers.opensandbox.resource_telemetry import resource_amounts
            from nemo_gym.sandbox.resource_telemetry import amount_attributes
            from nemo_gym.telemetry._fallbacks import safe_set_span_attributes

            submitted = {}
            for kind, resource_map in (
                ("limits", kwargs["resource"]),
                ("requests", kwargs.get("resource_requests", {})),
            ):
                submitted.update(amount_attributes(resource_amounts(resource_map), f"submitted.{kind}."))
            safe_set_span_attributes(get_current_span(), submitted)

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
                raise opensandbox_failure(
                    operation="create_once",
                    failure_kind=PROVIDER_PROTOCOL_ERROR,
                    reason="OpenSandbox SDK create returned no sandbox handle",
                )
            sandbox_id = str(sandbox.id)
        except TimeoutError as e:
            await self._raise_timed_out_create(e, spec, metadata, attempt_key, f"after {timeout_s:g}s")
        except asyncio.CancelledError:
            # A caller's deadline (the environment server's seed timeout) or shutdown while the create
            # request was still open: the SDK never learned the id of the sandbox it may have left behind.
            if sandbox is None:
                await asyncio.shield(self._reap_orphaned_create(metadata, attempt_key))
            raise
        except Exception as e:
            if sandbox is not None or not _is_request_timeout(e):
                raise
            # The SDK's own request timeout fired before the attempt budget: the same open request, the same
            # sandbox it may have left behind without telling the client its id.
            request_bound_s = SDK_DEFAULT_REQUEST_TIMEOUT_S if request_timeout_s is None else request_timeout_s
            await self._raise_timed_out_create(
                e, spec, metadata, attempt_key, f"at the SDK's request timeout of {request_bound_s:g}s ({e})"
            )
        if sandbox_id is None:
            raise opensandbox_failure(
                operation="create_once",
                failure_kind=PROVIDER_PROTOCOL_ERROR,
                reason="OpenSandbox SDK create returned no sandbox handle",
            )
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
            await self._run_setup_commands(handle)
        except BaseException:
            # Also on cancellation (a caller's timeout, shutdown): otherwise the pod lives to its TTL.
            # The shield lets the cleanup finish even if this task is cancelled again meanwhile.
            await asyncio.shield(self._cleanup_failed_create_handle(created_handle))
            raise
        if self._create.renew_ttl_s is not None:
            # A spec lifetime within the lease needs no renewal: the sandbox ends at its own TTL.
            if spec.ttl_s is None or spec.ttl_s > self._create.renew_ttl_s:
                self._start_renewal(handle, self._create.renew_ttl_s, lifetime_s=spec.ttl_s)
        elif self._create.renew_interval_s is not None:
            self._start_renewal(handle, spec.ttl_s, lifetime_s=None)
        return handle

    def _create_budget(self, spec: SandboxSpec) -> tuple[float | None, int | float | None]:
        """The per-attempt budget and the SDK request timeout for one create of ``spec``.

        The create request blocks until the pod is allocated, so a configured request timeout
        (``create.request_timeout_s``, else ``connection.request_timeout_s``) is capped at the spec's ready
        timeout: an unschedulable pod then fails within one ready timeout instead of the full request
        timeout. When none is configured the SDK's own default stays, as the connection config the create
        is given lives on the handle for the sandbox's life.

        The SDK's readiness wait starts only once the request returns, so the attempt budget must hold the
        request bound plus ``ready_timeout_s``. A configured ``create.timeout_s`` is respected as is (a
        warning names it when it is below that sum); otherwise the budget is that sum plus a slack.
        """
        request_timeout_s: int | float | None = self._create.request_timeout_s
        if request_timeout_s is None:
            request_timeout_s = self._connection.request_timeout_s
        if request_timeout_s is not None and spec.ready_timeout_s is not None:
            request_timeout_s = min(request_timeout_s, spec.ready_timeout_s)
        timeout_s = self._create.timeout_s
        if spec.ready_timeout_s is None:
            if timeout_s is None and self._connection.request_timeout_s is not None:
                timeout_s = float(self._connection.request_timeout_s)
            return timeout_s, request_timeout_s
        request_bound_s = SDK_DEFAULT_REQUEST_TIMEOUT_S if request_timeout_s is None else float(request_timeout_s)
        needed_s = request_bound_s + float(spec.ready_timeout_s)
        if timeout_s is None:
            return needed_s + CREATE_BUDGET_SLACK_S, request_timeout_s
        key = (float(timeout_s), request_bound_s, float(spec.ready_timeout_s))
        if timeout_s < needed_s and key not in _BUDGET_WARNINGS:
            _BUDGET_WARNINGS.add(key)
            LOGGER.warning(
                "create.timeout_s=%g is below the create request bound of %gs plus the spec's ready_timeout_s of "
                "%gs: a sandbox the SDK would still have reported ready within its budgets is given up on as "
                "timed out. Raise create.timeout_s to at least %g or leave it unset; image=%r",
                timeout_s,
                request_bound_s,
                spec.ready_timeout_s,
                needed_s,
                spec.image,
            )
        return timeout_s, request_timeout_s

    async def _raise_timed_out_create(
        self, cause: BaseException, spec: SandboxSpec, metadata: dict[str, str], attempt_key: str, how: str
    ) -> NoReturn:
        """Reap what a timed-out create left behind and raise the error that describes it.

        Report provider_create_pending only for an observed Pending sandbox; otherwise report
        provider_create_timeout. The create policy chooses retry, with fail_fast_on_pending
        preventing another attempt for a Pending sandbox when configured.
        """
        orphans = await asyncio.shield(self._reap_orphaned_create(metadata, attempt_key))
        pending = [o for o in orphans if str(o.get("state") or "").upper() in PENDING_SANDBOX_STATES]
        if pending:
            orphan = pending[0]
            raise opensandbox_failure(
                operation="create",
                sandbox_id=orphan["id"],
                failure_kind=PROVIDER_CREATE_PENDING,
                reason=f"OpenSandbox sandbox {orphan['id']} was still {orphan['state']} when the create timed out {how} "
                f"(reason={orphan.get('reason')!r}, message={orphan.get('message')!r}); image={spec.image!r}, "
                f"ready_timeout_s={spec.ready_timeout_s!r}. The sandbox was deleted. "
                "Pending alone does not distinguish resource limits, capacity shortage, or an image still starting.",
            ) from cause
        raise opensandbox_failure(
            operation="create",
            failure_kind=PROVIDER_CREATE_TIMEOUT,
            reason=f"Timed out creating OpenSandbox sandbox {how}; image={spec.image!r}, ready_timeout_s={spec.ready_timeout_s!r}",
        ) from cause

    async def _reap_orphaned_create(self, metadata: dict[str, str], attempt_key: str) -> list[dict[str, Any]]:
        """Find and delete the sandbox a timed-out create request left behind; return what was found.

        The request never returned, so the client holds no sandbox id: the sandbox is found through the
        run's attribution metadata and this attempt's unique key. Each match is deleted and described
        (``id``, ``state``, ``reason``, ``message``) so the caller can tell a pod that never left
        ``PENDING`` from one that was merely slow. Failures are logged, never raised: the create's own
        timeout error is the one the caller must see.
        """
        scope = {
            key: metadata[key] for key in (f"{self._attribution.key_prefix}{RUN_KEY}", attempt_key) if key in metadata
        }
        try:
            return await asyncio.wait_for(
                self._delete_sandboxes_by_metadata(scope, attempt_key), timeout=ORPHAN_REAP_TIMEOUT_S
            )
        except Exception as e:
            if not isinstance(e, NgFailureEvent):
                observe_failure(opensandbox_failure(operation="reap_orphaned_create", cause=e))
            LOGGER.warning(
                "Could not reap the sandbox left by a timed-out OpenSandbox create; scope=%s; error=%r", scope, e
            )
            return []

    async def _delete_sandboxes_by_metadata(self, scope: dict[str, str], attempt_key: str) -> list[dict[str, Any]]:
        AdapterFactory, SandboxFilter = _require_opensandbox_lifecycle_sdk()
        service = AdapterFactory(self._connection_config()).create_sandbox_service()
        found: list[dict[str, Any]] = []
        page = 1
        while True:
            listed = await self._await_sdk_operation(
                lambda: service.list_sandboxes(SandboxFilter(metadata=scope, page=page, page_size=100)),
                operation="list_sandboxes",
                sandbox_id=None,
                timeout_s=self._connection.request_timeout_s,
                replay_safe=True,
            )
            for info in listed.sandbox_infos:
                # The server-side metadata filter is advisory; never delete a sandbox this attempt did not make.
                if (info.metadata or {}).get(attempt_key) != scope.get(attempt_key):
                    continue
                status = getattr(info, "status", None)
                found.append(
                    {
                        "id": str(info.id),
                        "state": getattr(status, "state", None),
                        "reason": getattr(status, "reason", None),
                        "message": getattr(status, "message", None),
                    }
                )
            if not listed.pagination.has_next_page:
                break
            page += 1
        for orphan in found:
            await self._kill_with_retry(lambda: service.kill_sandbox(orphan["id"]), sandbox_id=orphan["id"])
            LOGGER.warning(
                "Deleted OpenSandbox sandbox %s left by a timed-out create; state=%s, reason=%s, message=%s",
                orphan["id"],
                orphan["state"],
                orphan["reason"],
                orphan["message"],
            )
        return found

    async def _create_with_retries(
        self,
        spec: SandboxSpec,
    ) -> SandboxHandle:
        AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential = _require_tenacity()
        backoff = wait_random_exponential(multiplier=self._create.retry_delay_s, max=self._create.retry_max_delay_s)
        retry_policy = AsyncRetrying(
            retry=retry_if_exception(
                lambda error: _is_retryable_create_error(
                    error, config=self._create.outcome_policy, fail_fast_on_pending=self._create.fail_fast_on_pending
                )
            ),
            stop=stop_after_attempt(self._create.retries + 1),
            wait=lambda state: _retry_delay(
                state.outcome.exception(), fallback=backoff(state), max_delay_s=self._create.retry_max_delay_s
            ),
            before_sleep=_log_create_retry,
            reraise=True,
        )
        async for attempt in retry_policy:
            with attempt:
                try:
                    return await self._create_once(spec)
                except (ValueError, NotImplementedError, ImportError):
                    # Invalid local configuration is rejected before a provider operation.
                    raise
                except Exception as error:
                    failure = (
                        error
                        if isinstance(error, NgFailureEvent)
                        else opensandbox_failure(operation="create", cause=error)
                    )
                    observe_failure(failure)
                    if failure is error:
                        raise
                    raise failure from error

        raise RuntimeError("OpenSandbox create retry loop did not run")

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create one sandbox through the configured OpenSandbox path.

        Job attribution keys (``team`` / ``user`` / ``workload`` / ``run``) are merged into the
        spec's metadata (explicit spec keys win) so every sandbox is attributable via its labels.
        """
        spec = replace(spec, metadata={**self._attribution.metadata(), **spec.metadata})
        return await self._create_with_retries(_normalize_spec(spec))

    async def resource_sample(self, handle: SandboxHandle) -> dict[str, str | int | float | bool]:
        """Read execd's existing snapshot; preserve its timestamp and measurement names."""
        from nemo_gym.sandbox.resource_telemetry import PREFIX

        sample = await handle.raw.get_metrics()
        return {
            PREFIX + "sample.source": "execd/metrics",
            PREFIX + "sample.timestamp_ms": sample.timestamp,
            PREFIX + "sample.cpu_count": sample.cpu_count,
            PREFIX + "sample.cpu_used_percentage": sample.cpu_used_percentage,
            PREFIX + "sample.memory_total_mib": sample.memory_total_in_mib,
            PREFIX + "sample.memory_used_mib": sample.memory_used_in_mib,
        }

    async def resource_allocation(self, handle: SandboxHandle) -> ResourceAllocation:
        """Read effective requests and limits from the deployment's inspection endpoint."""
        from urllib.parse import quote

        import aiohttp

        from nemo_gym.client.http_client import request
        from nemo_gym.sandbox.providers.opensandbox.resource_telemetry import parse_allocation

        connection = self._connection_config()
        headers = dict(connection.headers)
        if connection.get_api_key():
            headers["OPEN-SANDBOX-API-KEY"] = connection.get_api_key()
        response = await request(
            "GET",
            f"{connection.get_base_url()}/sandboxes/{quote(handle.sandbox_id, safe='')}/diagnostics/inspect",
            headers=headers,
            ssl=self._connection.tls_verify,
            timeout=aiohttp.ClientTimeout(total=2),
            _max_num_tries=1,
        )
        try:
            response.raise_for_status()
            try:
                await response.content.readexactly(65537)
            except asyncio.IncompleteReadError as exc:
                body = exc.partial
            else:
                raise ValueError("Inspection response exceeded 64 KiB")
            # Keep credentials and query parameters out of exported identity.
            url = urlsplit(connection.get_base_url())
            authority = url.netloc.rsplit("@", 1)[-1]
            scope = f"{url.scheme}://{authority}{url.path.rstrip('/')}"
            return parse_allocation(body.decode("utf-8"), scope=scope)
        finally:
            response.release()

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
            replay_safe=True,
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
        elif user == "root":
            opts_kwargs["uid"] = 0
        elif isinstance(user, str):
            effective_command = f"su -s /bin/sh -c {shlex.quote(command)} {shlex.quote(user)}"

        sdk_timeout_s = (
            float(timeout_s) + self._operations.command_timeout_headroom_s
            if timeout_s is not None
            else (
                float(self._connection.request_timeout_s) if self._connection.request_timeout_s is not None else None
            )
        )
        effective_retries = self._command_retry_count() if retries is None else retries

        async def _dispatch() -> SandboxExecResult:
            if self._operations.background_exec:
                return await self._exec_background(
                    handle,
                    effective_command,
                    opts_kwargs,
                    sdk_timeout_s=sdk_timeout_s,
                    total_timeout_s=timeout_s,
                    retries=effective_retries,
                )

            execution = await self._submit_command(
                lambda: handle.raw.commands.run(effective_command, opts=RunCommandOpts(**opts_kwargs)),
                operation="command run",
                sandbox_id=handle.sandbox_id,
                timeout_s=sdk_timeout_s,
                retries=effective_retries,
            )
            if refusal := _root_launch_refusal(execution, uid=opts_kwargs.get("uid")):
                return refusal
            stdout = "\n".join(msg.text for msg in execution.logs.stdout) or None
            stderr_parts = [msg.text for msg in execution.logs.stderr]
            if execution.error is not None:
                stderr_parts.append(f"{execution.error.name}: {execution.error.value}")
            return_code, error_type, note = _execution_outcome(
                execution, operation="exec", sandbox_id=handle.sandbox_id, execution_id=getattr(execution, "id", None)
            )
            if note is not None:
                stderr_parts.append(note)
            stderr = "\n".join(stderr_parts) or None

            return SandboxExecResult(stdout=stdout, stderr=stderr, return_code=return_code, error_type=error_type)

        # Backstop for wedges the inner deadlines miss. Background exec polls, so
        # sdk_timeout_s bounds a single request rather than the command: without
        # timeout_s there is nothing to cap against.
        operation_deadline: float | None = None
        try:
            if sdk_timeout_s is None or (self._operations.background_exec and timeout_s is None):
                return await _dispatch()
            hard_cap_s = 2.0 * float(sdk_timeout_s) + self._operations.exec_hard_cap_slack_s
            # asyncio.timeout instead of wait_for: since Python 3.11
            # asyncio.TimeoutError IS builtin TimeoutError, so a wait_for-based
            # cap would also catch timeouts raised INSIDE the dispatch (e.g. an
            # SDK request timeout) and relabel them as a hard-cap trip.
            hard_cap = asyncio.timeout(hard_cap_s)
            operation_deadline = hard_cap.when()
            try:
                async with hard_cap:
                    return await _dispatch()
            except TimeoutError as e:
                if not hard_cap.expired():
                    raise
                raise opensandbox_failure(
                    operation="exec",
                    sandbox_id=handle.sandbox_id,
                    cause=e,
                    failure_kind=PROVIDER_IO_TIMEOUT,
                    reason=f"OpenSandbox exec exceeded hard cap of {hard_cap_s:g}s; the command outcome is unknown "
                    f"(sandbox_id={handle.sandbox_id!r})",
                ) from e
        except Exception as error:
            if isinstance(error, NgFailureEvent) and (
                error.operation == "command status"
                or error.failure_kind in (PROVIDER_OOM_KILLED, PROVIDER_STATUS_UNAVAILABLE)
            ):
                raise
            deadline = asyncio.get_running_loop().time() + self._diagnostics.timeout_s
            if operation_deadline is not None:
                deadline = min(deadline, operation_deadline)
            evidence = await self._sandbox_death_evidence(handle, deadline=deadline)
            if evidence is not None:
                raise opensandbox_failure(
                    operation="exec",
                    sandbox_id=handle.sandbox_id,
                    cause=error,
                    failure_kind=evidence.failure_kind,
                    reason=evidence.message,
                    provider_status_reason=evidence.provider_status_reason,
                ) from error
            if isinstance(error, NgFailureEvent):
                raise
            raise opensandbox_failure(
                operation="exec",
                sandbox_id=handle.sandbox_id,
                cause=error,
            ) from error

    async def _sandbox_death_evidence(
        self, handle: SandboxHandle, *, deadline: float, poll: bool = True
    ) -> _SandboxDeathNotice | None:
        """One evidence path for exec and PTY, bounded by the caller's existing deadline.

        Lifecycle reasons do not identify the affected container, so even OOMKilled
        establishes only unavailability. A recognized control-plane container report
        or optional inspect can confirm OOM for the current user container.
        """
        loop = asyncio.get_running_loop()
        terminal_notice = None
        inspected = False
        while (remaining := deadline - loop.time()) > 0:
            get_info = getattr(handle.raw, "get_info", None)
            if get_info is not None:
                try:
                    async with asyncio.timeout(min(self._diagnostics.request_timeout_s, remaining)):
                        info = await get_info()
                    if evidence := _lifecycle_oom_evidence(info, sandbox_id=handle.sandbox_id):
                        return evidence
                    status = getattr(info, "status", None)
                    state, reason = getattr(status, "state", None), getattr(status, "reason", None)
                    if _to_sandbox_status(state) in {SandboxStatus.ERROR, SandboxStatus.STOPPED}:
                        # The lifecycle API exposes a workload reason without a container identity.
                        # Keep OOMKilled as evidence, but do not attribute a possible sidecar OOM.
                        # Keep the control plane's own report as bounded evidence: it is what tells a reader
                        # whether the workload hit its limit or was evicted. It never changes the failure kind.
                        report = getattr(status, "message", None)
                        detail = f", message={report[:256]!r}" if isinstance(report, str) and report else ""
                        terminal_notice = _SandboxDeathNotice(
                            f"Sandbox is unavailable; lifecycle status: state={state!r}, "
                            f"reason={str(reason)[:256]!r}{detail}",
                            failure_kind=PROVIDER_UNAVAILABLE,
                            provider_status_reason=reason if isinstance(reason, str) else None,
                        )
                except Exception:
                    # Optional diagnosis must not replace the original failure.
                    pass
            # An early inspect can precede termination. Recheck once when lifecycle
            # first becomes terminal, using only the original deadline's remainder.
            if (
                self._diagnostics.inspect_oom
                and (not inspected or terminal_notice is not None)
                and deadline > loop.time()
            ):
                inspected = True
                try:
                    evidence = await self._control_plane_inspect_oom_evidence(handle, deadline=deadline)
                except Exception:
                    evidence = None
                if evidence is not None:
                    return _SandboxDeathNotice(
                        f"Sandbox was OOM-killed: inspect confirms {evidence}",
                        failure_kind=PROVIDER_OOM_KILLED,
                        provider_status_reason=terminal_notice.provider_status_reason if terminal_notice else None,
                    )
            if terminal_notice is not None or get_info is None or not poll:
                return terminal_notice
            await asyncio.sleep(min(self._diagnostics.poll_interval_s, max(0, deadline - loop.time())))
        return terminal_notice

    async def _control_plane_inspect_oom_evidence(self, handle: SandboxHandle, *, deadline: float) -> str | None:
        """Opt-in OpenSandbox 1.1.0 text fallback; no supported structured equivalent is assumed."""
        import aiohttp

        from nemo_gym.client import http_client

        remaining = min(self._diagnostics.request_timeout_s, deadline - asyncio.get_running_loop().time())
        if remaining <= 0:
            return None
        config = self._connection_config(request_timeout_s=remaining)
        url = f"{config.get_base_url().rstrip('/')}/sandboxes/{quote(handle.sandbox_id, safe='')}/diagnostics/inspect"
        api_key = config.get_api_key()
        headers = {"OPEN-SANDBOX-API-KEY": api_key} if api_key else {}
        try:
            async with asyncio.timeout(remaining):
                response = await http_client.request(
                    "GET",
                    url,
                    _control=True,
                    _max_num_tries=1,
                    _server_name="opensandbox_control_plane",
                    headers=headers,
                    ssl=self._connection.tls_verify,
                    timeout=aiohttp.ClientTimeout(total=remaining),
                )
                async with response:
                    if response.status != 200:
                        return None
                    body = await response.content.read(65537)
        except (aiohttp.ClientError, TimeoutError):
            return None
        if len(body) > 65536:
            return None
        return _inspect_oom_evidence(body.decode("utf-8", errors="replace"))

    async def _pty_failure(self, handle: SandboxHandle, cause: BaseException) -> NgFailureEvent:
        deadline = asyncio.get_running_loop().time() + self._diagnostics.timeout_s
        evidence = await self._sandbox_death_evidence(handle, deadline=deadline)
        return opensandbox_failure(
            operation="pty",
            sandbox_id=handle.sandbox_id,
            cause=cause,
            failure_kind=evidence.failure_kind if evidence is not None else PROVIDER_COMMAND_OUTCOME_UNKNOWN,
            reason=evidence.message if evidence is not None else "PTY connection lost; command outcome is unknown",
            provider_status_reason=evidence.provider_status_reason if evidence is not None else None,
        )

    async def _exec_background(
        self,
        handle: SandboxHandle,
        command: str,
        opts_kwargs: dict[str, Any],
        *,
        sdk_timeout_s: float | None,
        total_timeout_s: int | float | None,
        retries: int,
    ) -> SandboxExecResult:
        """Register this command's poll for checkpoint polling scopes, and release it on every exit."""
        poll = _BackgroundCommandPoll(parked=handle.sandbox_id in self._parked_poll_scopes)
        self._command_polls.setdefault(handle.sandbox_id, set()).add(poll)
        try:
            return await self._exec_background_polled(
                handle,
                command,
                opts_kwargs,
                sdk_timeout_s=sdk_timeout_s,
                total_timeout_s=total_timeout_s,
                retries=retries,
                poll=poll,
            )
        finally:
            polls = self._command_polls[handle.sandbox_id]
            polls.discard(poll)
            if not polls:
                del self._command_polls[handle.sandbox_id]

    async def _exec_background_polled(
        self,
        handle: SandboxHandle,
        command: str,
        opts_kwargs: dict[str, Any],
        *,
        sdk_timeout_s: float | None,
        total_timeout_s: int | float | None,
        retries: int,
        poll: _BackgroundCommandPoll,
    ) -> SandboxExecResult:
        """Run a command as a background execution polled via short requests.

        The logs endpoint returns one combined stream, so unlike the foreground
        path ``stdout`` carries both streams and ``stderr`` is set only when the
        sandbox itself reports an error.

        Gym enforces ``total_timeout_s`` itself: once its own deadline passes
        while the command is still running, it interrupts the command and
        returns the timeout result, as the session path does. execd reports its
        own timeout kill only as a signal death, so its timer is moved past
        Gym's deadline and serves as a backstop.
        """
        _, _, RunCommandOpts, _, _ = _require_opensandbox_sdk()
        background_opts = dict(opts_kwargs)
        background_opts["background"] = True
        command_timeout_s = float(total_timeout_s) if total_timeout_s else None
        if command_timeout_s is not None:
            background_opts["timeout"] = timedelta(
                seconds=command_timeout_s + self._operations.command_timeout_headroom_s / 2
            )

        loop = asyncio.get_running_loop()

        def run_command() -> Awaitable[Any]:
            return handle.raw.commands.run(command, opts=RunCommandOpts(**background_opts))

        execution = await self._submit_command(
            run_command,
            operation="command run (background submit)",
            sandbox_id=handle.sandbox_id,
            timeout_s=sdk_timeout_s,
            retries=retries,
        )
        if refusal := _root_launch_refusal(execution, uid=opts_kwargs.get("uid")):
            return refusal
        execution_id = getattr(execution, "id", None)
        if not execution_id:
            raise opensandbox_failure(
                operation="exec",
                sandbox_id=handle.sandbox_id,
                failure_kind=PROVIDER_COMMAND_OUTCOME_UNKNOWN,
                reason="OpenSandbox background command did not return an execution id",
            )
        command_deadline = loop.time() + command_timeout_s if command_timeout_s is not None else None
        timed_out = False

        # The server enforces the command timeout; leave the client headroom.
        deadline = (
            loop.time() + float(total_timeout_s) + self._operations.command_timeout_headroom_s
            if total_timeout_s is not None
            else None
        )
        poll_timeout_s = (
            float(self._connection.request_timeout_s) if self._connection.request_timeout_s is not None else 60.0
        )
        # Status polls are sub-second GETs; against an unreachable sandbox each
        # one would otherwise hang for the shared budget above (tuned for long
        # submits) per retry before the typed failure fires.
        status_timeout_s = self._operations.status_poll_timeout_s
        if status_timeout_s is None:
            status_timeout_s = poll_timeout_s

        status_failure_deadline: float | None = None
        diagnostic_remaining_s = 0.0
        status_failure_error: NgFailureEvent | None = None
        seen_running = False
        poll_interval = min(self._operations.background_poll_initial_s, self._operations.background_poll_interval_s)
        while True:
            # A parked sandbox is being snapshotted and may not answer: ask once the park ends,
            # with a fresh status-recovery window, since the park, not the sandbox, kept the status unanswered.
            if handle.sandbox_id in self._parked_poll_scopes or poll.parked:
                await self._wait_while_parked(handle.sandbox_id, poll)
                status_failure_deadline = None
                status_failure_error = None
            poll.parked = False
            limits = [value for value in (deadline, status_failure_deadline) if value is not None]
            remaining = min(limits) - loop.time() if limits else None
            if remaining is not None and remaining <= 0:
                if status_failure_deadline is not None and (deadline is None or status_failure_deadline < deadline):
                    failure_kind = PROVIDER_STATUS_UNAVAILABLE
                    reason = (
                        "OpenSandbox command status unavailable beyond recovery deadline; command outcome is unknown"
                    )
                else:
                    failure_kind = PROVIDER_COMMAND_OUTCOME_UNKNOWN
                    reason = "OpenSandbox client wait expired; command completion is unknown"
                    if seen_running:
                        reason += "; last observed running=True"
                raise opensandbox_failure(
                    operation="command status",
                    sandbox_id=handle.sandbox_id,
                    execution_id=execution_id,
                    failure_kind=failure_kind,
                    cause=status_failure_error,
                    reason=reason,
                ) from status_failure_error
            try:
                status = await self._await_sdk_call(
                    handle.raw.commands.get_command_status(execution_id),
                    operation="command status",
                    sandbox_id=handle.sandbox_id,
                    timeout_s=min(status_timeout_s, remaining) if remaining is not None else status_timeout_s,
                )
            except Exception as error:
                if poll.parked:
                    # A park began while this request was in flight, so the sandbox may have been unresponsive for
                    # the checkpoint rather than lost: ask again once the park ends.
                    LOGGER.warning(
                        "Forgave a failed OpenSandbox command status during a checkpoint park; "
                        "sandbox_id=%r, execution_id=%r: %s",
                        handle.sandbox_id,
                        execution_id,
                        error,
                    )
                    continue
                failure = (
                    error
                    if isinstance(error, NgFailureEvent)
                    else opensandbox_failure(
                        operation="command status",
                        sandbox_id=handle.sandbox_id,
                        execution_id=execution_id,
                        cause=error,
                    )
                )
                observe_failure(failure)
                if status_failure_deadline is None:
                    status_failure_deadline = loop.time() + self._operations.status_recovery_timeout_s
                    if deadline is not None:
                        status_failure_deadline = min(status_failure_deadline, deadline)
                    diagnostic_remaining_s = self._diagnostics.timeout_s
                status_failure_error = failure
                # Keep the unused diagnostic budget for later status failures:
                # lifecycle/inspect evidence may arrive after the first lookup.
                diagnostic_started = loop.time()
                diagnostic_deadline = min(status_failure_deadline, diagnostic_started + diagnostic_remaining_s)
                action = failure_outcome_policy(
                    failure,
                    config=self._operations.outcome_policy,
                    classification=self._operations.outcome_policy.classification,
                )
                # A final 410 has no later status retry to discover a delayed lifecycle
                # update. Spend only the remaining diagnostic budget awaiting that report.
                evidence = await self._sandbox_death_evidence(
                    handle,
                    deadline=diagnostic_deadline,
                    poll=failure.http_status == 410 and action != OutcomeAction.RETRY,
                )
                diagnostic_remaining_s = max(0.0, diagnostic_remaining_s - (loop.time() - diagnostic_started))
                if evidence is not None:
                    raise opensandbox_failure(
                        operation="command status",
                        sandbox_id=handle.sandbox_id,
                        execution_id=execution_id,
                        failure_kind=evidence.failure_kind,
                        cause=failure,
                        reason=evidence.message,
                        provider_status_reason=evidence.provider_status_reason,
                    ) from failure
                if action != OutcomeAction.RETRY:
                    if failure is error:
                        raise
                    raise failure from error
            else:
                status_failure_deadline = None
                status_failure_error = None
                if (
                    not all(hasattr(status, name) for name in ("running", "exit_code"))
                    or type(status.running) is not bool
                ):
                    raise opensandbox_failure(
                        operation="command status",
                        sandbox_id=handle.sandbox_id,
                        execution_id=execution_id,
                        failure_kind=PROVIDER_COMMAND_OUTCOME_UNKNOWN,
                        reason="OpenSandbox command status has missing or invalid running/exit_code fields",
                    )
                if status.running is False:
                    break
                seen_running = True
                if command_deadline is not None and not timed_out and loop.time() >= command_deadline:
                    timed_out = await self._interrupt_at_deadline(
                        handle, execution_id, timeout_s=command_timeout_s, request_timeout_s=status_timeout_s
                    )
                    # One attempt: if the interrupt fails, execd's backstop timer still stops the command.
                    command_deadline = None
            limits = [value for value in (deadline, status_failure_deadline) if value is not None]
            # Wake at Gym's command deadline, but once it has passed (e.g. while status reads fail) it must not
            # zero the polling and backoff delay.
            if command_deadline is not None and command_deadline > loop.time():
                limits.append(command_deadline)
            # Retry caps must not shorten the configured polling cadence, even when the cap is zero.
            delay = max(
                poll_interval,
                _retry_delay(
                    status_failure_error, fallback=poll_interval, max_delay_s=self._operations.retry_max_delay_s
                ),
            )
            if handle.sandbox_id in self._checkpoint_poll_scopes:
                delay = min(delay, self._operations.checkpoint_background_poll_interval_s)
            if limits:
                delay = min(delay, max(0, min(limits) - loop.time()))
            # A park or a checkpoint polling scope that starts meanwhile wakes the wait.
            await self._wait_command_poll(poll, interval_s=delay)
            poll_interval = min(poll_interval * 1.5, self._operations.background_poll_interval_s)

        # A background launch refusal may be available only in command status.
        # execd sets 255 and its fork/exec error before a child exists; a child
        # exiting 255 instead has status.error="exit status 255".
        if (
            opts_kwargs.get("uid") == 0
            and status.exit_code == 255
            and isinstance(message := getattr(status, "error", None), str)
            and _ROOT_LAUNCH_REFUSAL.fullmatch(message)
        ):
            return SandboxExecResult(None, message, 125, error_type="user_switch_unavailable")

        # The execution has finished, so one call returns its whole buffer; the
        # cursor this endpoint reports back is the end offset rather than a
        # more-data flag, so there is no tail to follow.
        output_deleted = False
        try:
            logs = await self._await_sdk_operation(
                lambda: handle.raw.commands.get_background_command_logs(execution_id),
                operation="command logs",
                sandbox_id=handle.sandbox_id,
                timeout_s=poll_timeout_s,
                retries=self._operations.retries,
                is_retryable=lambda error: not _is_deleted_output_error(error),
                replay_safe=True,
            )
        except TimeoutError as error:
            raise opensandbox_failure(operation="provider I/O", sandbox_id=handle.sandbox_id, cause=error) from error
        except Exception as error:
            # A command that cleaned the temporary directory (e.g. a verifier resetting its
            # scratch roots), or another one in the same sandbox, deleted execd's output file.
            # The status read above holds the authoritative exit code; without one there is no
            # evidence of how the command ended, so that case still raises, as does any other
            # error, including a reworded execd message.
            if not _is_deleted_output_error(error) or status.exit_code is None:
                raise
            LOGGER.warning(
                "OpenSandbox command output file was deleted; keeping the command's exit status %r. "
                "sandbox_id=%r execution_id=%r",
                status.exit_code,
                handle.sandbox_id,
                execution_id,
            )
            logs = None
            output_deleted = True
        stdout = getattr(logs, "content", None) or None
        status_error = getattr(status, "error", None)
        stderr = status_error or None
        if output_deleted:
            stderr = f"{stderr}\n{DELETED_OUTPUT_NOTE}" if stderr else DELETED_OUTPUT_NOTE
        if timed_out:
            note = f"Command timed out after {command_timeout_s:g}s; Gym interrupted it"
            return SandboxExecResult(stdout, f"{stderr}\n{note}" if stderr else note, 124, error_type="timeout")
        error_type = None
        if status.exit_code is not None:
            if status.exit_code == 0 and status_error is not None:
                raise opensandbox_failure(
                    operation="exec",
                    sandbox_id=handle.sandbox_id,
                    execution_id=execution_id,
                    failure_kind=PROVIDER_COMMAND_OUTCOME_UNKNOWN,
                    reason="OpenSandbox background command reported both exit code 0 and an error",
                )
            return_code = status.exit_code
        else:
            raise opensandbox_failure(
                operation="exec",
                sandbox_id=handle.sandbox_id,
                execution_id=execution_id,
                failure_kind=PROVIDER_COMMAND_OUTCOME_UNKNOWN,
                reason="OpenSandbox background command finished without an exit code",
            )

        return SandboxExecResult(stdout=stdout, stderr=stderr, return_code=return_code, error_type=error_type)

    async def _interrupt_at_deadline(
        self, handle: SandboxHandle, execution_id: str, *, timeout_s: float, request_timeout_s: float
    ) -> bool:
        """Interrupt a background command still running at Gym's deadline; whether the interrupt was accepted."""
        observe_failure(
            opensandbox_failure(
                operation="command status",
                sandbox_id=handle.sandbox_id,
                execution_id=execution_id,
                failure_kind=PROVIDER_COMMAND_TIMEOUT,
                reason=f"Gym command deadline expired after {timeout_s:g}s",
            )
        )
        try:
            await self._await_sdk_call(
                handle.raw.commands.interrupt(execution_id),
                operation="command interrupt",
                sandbox_id=handle.sandbox_id,
                timeout_s=request_timeout_s,
            )
        except Exception as error:
            # The command may have finished on its own just before the interrupt; its status says how it ended.
            observe_failure(
                error
                if isinstance(error, NgFailureEvent)
                else opensandbox_failure(
                    operation="command interrupt", sandbox_id=handle.sandbox_id, execution_id=execution_id, cause=error
                )
            )
            return False
        return True

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

    async def exec_with_background_services(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        """Preserve background services that redirect their stdout and stderr."""
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be nonnegative")
        # Native bash sessions expose only a working directory. Apply overrides
        # inside the session, with env after su so account defaults cannot replace
        # task-provided values. Quote argv rather than interpolating shell assignments.
        if env:
            command = shlex.join(
                ["env", "--", *(f"{key}={value}" for key, value in env.items()), "bash", "-c", command]
            )
        if user is not None:
            # su requires an account name; resolve numeric UIDs inside the sandbox.
            account = f'"$(id -nu -- {user})"' if isinstance(user, int) else shlex.quote(user)
            command = f"su -s /bin/sh -c {shlex.quote(command)} -- {account}"
        commands = handle.raw.commands
        try:
            session_id = await self._await_sdk_call(
                commands.create_session(working_directory=cwd),
                operation="create bash session",
                sandbox_id=handle.sandbox_id,
                timeout_s=self._connection.request_timeout_s,
            )
        except Exception as error:
            raise opensandbox_failure(
                operation="create bash session", sandbox_id=handle.sandbox_id, cause=error
            ) from error

        # A nested shell keeps exit/exec in the command from replacing the SDK's
        # session wrapper. The client owns the deadline: the SDK's own timeout can
        # kill only the shell and then hang waiting for a child's output pipe.
        async def run_in_session() -> Any:
            try:
                return await commands.run_in_session(
                    session_id, f"bash -c {shlex.quote(command)}", timeout=timedelta(0)
                )
            except TimeoutError as error:
                raise opensandbox_failure(
                    operation="run_in_session", sandbox_id=handle.sandbox_id, cause=error
                ) from error

        task = asyncio.create_task(run_in_session())

        async def release_session() -> None:
            try:
                # Delete while the request is still active: execd then knows the
                # process group to kill. After normal completion it only drops
                # session state; background services live until sandbox teardown.
                try:
                    await self._await_sdk_call(
                        commands.delete_session(session_id),
                        operation="delete bash session",
                        sandbox_id=handle.sandbox_id,
                        timeout_s=10,
                    )
                except Exception as error:
                    raise opensandbox_failure(
                        operation="delete bash session", sandbox_id=handle.sandbox_id, cause=error
                    ) from error
                if not task.done():
                    await asyncio.wait({task}, timeout=10)
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        primary_error: BaseException | None = None
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout_s)
            timed_out = not done
            if done and not task.cancelled():
                primary_error = task.exception()
            if timed_out:
                observe_failure(
                    opensandbox_failure(
                        operation="command session",
                        sandbox_id=handle.sandbox_id,
                        execution_id=session_id,
                        failure_kind=PROVIDER_COMMAND_TIMEOUT,
                        reason=f"Gym command session deadline expired after {timeout_s:g}s",
                    )
                )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup = asyncio.create_task(release_session())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                try:
                    await cleanup
                except Exception as error:
                    observe_failure(error)
                raise
            except Exception as error:
                if primary_error is None:
                    raise
                # Cleanup is a separate failed operation, not the cause of the command failure.
                observe_failure(error)

        if task.done() and not task.cancelled():
            task_error = task.exception()
            if isinstance(task_error, NgFailureEvent):
                raise task_error
        if timed_out and (task.cancelled() or task.exception() is not None):
            return SandboxExecResult(
                None,
                f"Command timed out after {timeout_s:g}s; OpenSandbox command session was deleted",
                124,
                error_type="timeout",
            )
        try:
            execution = task.result()
        except Exception as error:
            raise opensandbox_failure(operation="run_in_session", sandbox_id=handle.sandbox_id, cause=error) from error
        stdout = "\n".join(msg.text for msg in execution.logs.stdout) or None
        stderr_parts = [msg.text for msg in execution.logs.stderr]
        if timed_out:
            stderr_parts.append(f"Command timed out after {timeout_s:g}s; OpenSandbox command session was deleted")
            return SandboxExecResult(stdout, "\n".join(stderr_parts), 124, error_type="timeout")
        if execution.error is not None:
            stderr_parts.append(f"{execution.error.name}: {execution.error.value}")
        return_code, error_type, note = _execution_outcome(
            execution,
            operation="run_in_session",
            sandbox_id=handle.sandbox_id,
            execution_id=getattr(execution, "id", None),
        )
        if note is not None:
            stderr_parts.append(note)
        return SandboxExecResult(stdout, "\n".join(stderr_parts) or None, return_code, error_type)

    def _pty_http_client(self) -> Any:
        """Return the aiohttp client for one PTY session (same ``tls_verify`` as the SDK transport)."""
        import aiohttp

        if not self._connection.tls_verify:
            return aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False))
        return aiohttp.ClientSession()

    async def _pty_session_missing(
        self, base_url: str, headers: dict[str, str], session_id: str, request_timeout_s: float | None
    ) -> bool:
        """True only when execd itself reports the PTY session does not exist.

        A proxy 404 (route not registered yet) lacks execd's error code, and a
        failed check is treated as unknown so the attach proceeds as before.
        """
        import aiohttp

        try:
            async with self._pty_http_client() as client:
                async with client.get(
                    f"{base_url}/pty/{session_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=request_timeout_s),
                ) as response:
                    return response.status == 404 and "CONTEXT_NOT_FOUND" in await response.text()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False  # unknown: keep the takeover retries

    async def _pty_target(self, handle: SandboxHandle) -> tuple[str, dict[str, str], float | None]:
        """Resolve the sandbox's execd base URL, headers and request timeout."""
        from opensandbox.constants import DEFAULT_EXECD_PORT

        # A None connection timeout would also disable aiohttp's own 300s
        # default, leaving create/attach unbounded against a stalled proxy.
        request_timeout_s = (
            float(self._connection.request_timeout_s) if self._connection.request_timeout_s is not None else 300.0
        )
        endpoint = await self._await_sdk_operation(
            lambda: handle.raw.get_endpoint(DEFAULT_EXECD_PORT),
            operation="get_pty_endpoint",
            replay_safe=True,
            sandbox_id=handle.sandbox_id,
            timeout_s=request_timeout_s,
        )
        headers = dict(endpoint.headers)
        if self._connection.api_key:
            headers["OPEN-SANDBOX-API-KEY"] = self._connection.api_key
        return f"{self._connection.protocol}://{endpoint.endpoint}", headers, request_timeout_s

    async def create_pty(self, handle: SandboxHandle, spec: SandboxPtySpec) -> SandboxPtySession:
        """Open an interactive execd PTY session inside a sandbox."""
        from nemo_gym.sandbox.providers.opensandbox.pty import open_pty_session

        base_url, headers, request_timeout_s = await self._pty_target(handle)
        session = await open_pty_session(
            client=self._pty_http_client(),
            base_url=base_url,
            headers=headers,
            spec=spec,
            request_timeout_s=request_timeout_s,
            sandbox_id=handle.sandbox_id,
            diagnose=lambda cause: self._pty_failure(handle, cause),
        )
        session._sandbox_id = handle.sandbox_id
        await self._retire_closed_pty_sessions()
        self._pty_sessions.add(session)
        return session

    async def attach_pty(
        self,
        handle: SandboxHandle,
        session_id: str,
        *,
        takeover: bool = True,
        since: int | None = None,
    ) -> SandboxPtySession:
        """Re-attach to an existing execd PTY session by id."""
        from nemo_gym.sandbox.providers.opensandbox.pty import _PTY_TAKEOVER_RETRY_DELAYS, attach_pty_session

        base_url, headers, request_timeout_s = await self._pty_target(handle)
        # execd refuses a missing session with the same close as a held one (for
        # example after a pause replaced the runtime), which the takeover retries
        # below would ride out for tens of seconds. Ask first.
        if await self._pty_session_missing(base_url, headers, session_id, request_timeout_s):
            raise opensandbox_failure(
                operation="attach_pty",
                sandbox_id=handle.sandbox_id,
                failure_kind=PROVIDER_PTY_UNAVAILABLE,
                reason=f"PTY session {session_id} not found",
            )
        if takeover:
            # Release our own live attachment first, so the takeover below has
            # nothing to evict and cannot be refused as "already attached".
            for stale in [s for s in self._pty_sessions if s.session_id == session_id and not s.closed]:
                try:
                    await stale.detach()
                except NgFailureEvent:
                    pass  # already dead; the retry loop below rides out eviction
        # A takeover evicts the attached client, and execd waits for that
        # client to acknowledge. A half-open peer (silently dropped along the
        # proxy path) cannot answer, so execd's eviction times out and the
        # attach comes back as a policy-violation close — reported as "already
        # has an attached client" — while the stale client is torn down in the
        # background. A fresh attempt then lands, so retry exactly that case.
        for delay in (*_PTY_TAKEOVER_RETRY_DELAYS, None):
            try:
                session = await attach_pty_session(
                    client=self._pty_http_client(),
                    base_url=base_url,
                    headers=headers,
                    session_id=session_id,
                    takeover=takeover,
                    since=since,
                    request_timeout_s=request_timeout_s,
                    sandbox_id=handle.sandbox_id,
                    diagnose=lambda cause: self._pty_failure(handle, cause),
                )
                break
            except NgFailureEvent as e:
                if not takeover or e.failure_kind != PROVIDER_PTY_BUSY:
                    raise
                if delay is None:
                    # The eviction never completed across every retry. When the
                    # sandbox itself is gone (node loss, OOM kill) the stale
                    # attachment can never acknowledge its eviction, so the
                    # takeover is refused forever — report the sandbox's death
                    # instead of the refusal. Keep bounded polling because the
                    # terminal lifecycle status can lag the final refusal.
                    deadline = asyncio.get_running_loop().time() + self._diagnostics.timeout_s
                    evidence = await self._sandbox_death_evidence(handle, deadline=deadline)
                    if evidence is None:
                        raise
                    raise opensandbox_failure(
                        operation="pty",
                        sandbox_id=handle.sandbox_id,
                        cause=e,
                        failure_kind=evidence.failure_kind,
                        reason=evidence.message,
                        provider_status_reason=evidence.provider_status_reason,
                    ) from e
            await asyncio.sleep(delay)
        session._sandbox_id = handle.sandbox_id
        await self._retire_closed_pty_sessions()
        self._pty_sessions.add(session)
        return session

    async def _write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        """Write one file into an OpenSandbox sandbox."""
        try:
            await self._await_sdk_operation(
                lambda: handle.raw.files.write_file(target_path, data),
                operation="write_file",
                sandbox_id=handle.sandbox_id,
                timeout_s=float(self._connection.request_timeout_s)
                if self._connection.request_timeout_s is not None
                else None,
            )
        except TimeoutError as error:
            raise opensandbox_failure(operation="provider I/O", sandbox_id=handle.sandbox_id, cause=error) from error

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one local file into an OpenSandbox sandbox."""
        await self._write_file(handle, target_path, source_path.read_bytes())

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one file from an OpenSandbox sandbox, streamed to disk in chunks."""
        target_path.parent.mkdir(parents=True, exist_ok=True)

        async def stream() -> None:
            # A retry starts the file again, so every attempt truncates it. The writes run in a thread so a
            # large file does not block the event loop.
            with target_path.open("wb") as out:
                async for chunk in await handle.raw.files.read_bytes_stream(source_path):
                    await asyncio.to_thread(out.write, chunk)

        try:
            await self._await_sdk_operation(
                stream,
                operation="download_file",
                sandbox_id=handle.sandbox_id,
                replay_safe=True,
                # No overall deadline: the transfer time grows with the file, and the SDK client's request timeout
                # still applies to every read, so a stalled stream fails.
                timeout_s=None,
            )
        except TimeoutError as error:
            raise opensandbox_failure(operation="provider I/O", sandbox_id=handle.sandbox_id, cause=error) from error

    async def _kill_with_retry(self, kill: Callable[[], Awaitable[None]], *, sandbox_id: str) -> None:
        """Apply the same idempotent deletion and permission guards to handles and orphan cleanup."""

        async def kill_ignore_missing() -> None:
            # Terminate is idempotent: not-found means the sandbox is already
            # gone (double-termination race, or a previous run's leftover).
            # Swallow it before the retry wrapper so the 404 is never retried.
            try:
                await kill()
            except Exception as e:
                if _is_terminal_delete_error(e) or not _is_missing_sandbox_delete_error(e):
                    raise
                LOGGER.debug("OpenSandbox sandbox %r already gone; treating terminate as success", sandbox_id)

        def retry_delete(error: BaseException) -> bool:
            # This predicate is a safety constraint; the shared outcome policy
            # still selects whether the observed failure should be retried.
            return not _is_terminal_delete_error(error)

        await self._await_sdk_operation(
            kill_ignore_missing,
            operation="kill",
            sandbox_id=sandbox_id,
            timeout_s=self._operations.close_timeout_s,
            is_retryable=retry_delete,
            replay_safe=True,
        )

    async def close(self, handle: SandboxHandle) -> None:
        """Terminate the sandbox and close local SDK resources."""
        renewal_error = await self._stop_renewal(handle.sandbox_id)

        # DELETE is idempotent, so a lost reply can be reconciled. Attempts and
        # backoff share one budget; failure keeps the SDK handle usable for retry.
        try:
            async with asyncio.timeout(self._operations.close_timeout_s):
                await self._kill_with_retry(handle.raw.kill, sandbox_id=handle.sandbox_id)
        except TimeoutError as error:
            failure = opensandbox_failure(
                operation="kill",
                sandbox_id=handle.sandbox_id,
                cause=error,
                reason="OpenSandbox deletion recovery deadline expired; deletion is unconfirmed",
            )
            observe_failure(failure)
            raise failure from error

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

        if renewal_error is not None:
            raise opensandbox_failure(
                operation="lifetime renewal",
                sandbox_id=handle.sandbox_id,
                failure_kind=PROVIDER_UNAVAILABLE,
                cause=renewal_error,
                reason=f"OpenSandbox lifetime renewal failed for sandbox {handle.sandbox_id!r}",
            ) from renewal_error
        if close_error is not None:
            return
