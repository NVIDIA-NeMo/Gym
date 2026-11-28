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

"""Configuration objects for the IdeGYM sandbox provider.

The provider constructor takes one section per concern — ``connection``, ``create``,
``exec``, ``verify``, ``files``, ``operations``, ``attribution`` — mirroring the shipped
``configs/idegym.yaml`` block. Every section validates in ``__post_init__`` so a bad
Hydra value fails at server start rather than mid-rollout, and every section is frozen
and hashable so the shared orchestrator session can be keyed on its connection config.

Comments here cover only what a reader of the code needs: sentinel values, cross-field
constraints, and the reason behind a default. Knob-by-knob operator guidance lives once,
in ``configs/idegym.yaml``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from nemo_gym.sandbox.attribution import AttributionConfig
from nemo_gym.sandbox.providers.utils import coerce_config


# Each transfer chunk crosses the orchestrator base64-encoded inside one JSON forward,
# which the orchestrator stores per async operation, so a chunk stays a few MiB at most.
MAX_FILE_CHUNK_BYTES = 8 * 1024 * 1024

# The orchestrator rejects a keepalive longer than 24 hours.
MAX_KEEPALIVE_MINUTES = 24 * 60.0

# RFC-1035 caps Kubernetes resource names at 63 characters, and the orchestrator derives
# the pod name as `<server_name>-<server_id>`, so the name the provider sends leaves room
# for that suffix.
MAX_SERVER_NAME_LENGTH = 63
SERVER_ID_SUFFIX_RESERVE = 8


class TransportBackend(StrEnum):
    """HTTP transport used underneath the IdeGYM client's httpx session."""

    AIOHTTP = "aiohttp"
    HTTPX = "httpx"


def _require_positive(value: float | int | None, name: str, *, allow_none: bool = False) -> None:
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{name} must be set")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def _require_non_negative(value: float | int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")


def _require_keepalive_minutes(value: float | None, name: str) -> None:
    _require_positive(value, name, allow_none=True)
    if value is not None and value > MAX_KEEPALIVE_MINUTES:
        raise ValueError(f"{name} must be <= {MAX_KEEPALIVE_MINUTES:g}, got {value!r}")


@dataclass(frozen=True)
class IdeGymConnectionConfig:
    """How to reach the IdeGYM orchestrator and how to identify this process to it.

    One registered IdeGYM client is shared by every sandbox created from an identical
    connection config (see :mod:`nemo_gym.sandbox.providers.idegym.session`), so this
    section doubles as that session's cache key and must stay hashable.
    """

    orchestrator_url: str = "idegym.test"
    namespace: str = "idegym"
    # Quotas match the client *name*, so every sandbox of one job shares it. Left unset,
    # the session derives it from job attribution.
    client_name: str | None = None
    username: str | None = None
    password: str | None = None
    # 0 uses the shared node pool.
    nodes_count: int = 0
    heartbeat_interval_s: int = 60
    request_timeout_s: int = 60
    transport_backend: str = TransportBackend.AIOHTTP
    # Caps in-flight requests rather than whole operations: a create holds a connection
    # only while polling, so provisioning cannot block execs on running sandboxes. None
    # means no cap on either backend.
    max_connections: int | None = 64
    # httpx backend only: aiohttp has no cap on idle connections, just their expiry.
    max_keepalive_connections: int = 20
    keepalive_expiry_s: float | None = 5.0
    # Retries of a request whose connection never opened, so it is safe for any method.
    connect_retries: int = 2
    # Retries of an idempotent GET (operation-status polls above all) on a transport error
    # or a 429/502/503/504. The SDK sends each poll once, so one flaky poll would otherwise
    # abort a long exec or start.
    poll_retries: int = 3
    retry_delay_s: float = 0.5
    # None leaves tracing to the SDK, which is opt-in: it traces only when
    # IDEGYM_OTEL_TRACING_ENDPOINT is set. Once an endpoint is set here, the SDK reads none
    # of its IDEGYM_OTEL_TRACING_* variables, so the timeout and credentials below apply.
    tracing_endpoint: str | None = None
    tracing_timeout_s: float = 10.0
    tracing_username: str | None = None
    tracing_password: str | None = None

    def __post_init__(self) -> None:
        if not self.orchestrator_url:
            raise ValueError("connection.orchestrator_url must be a non-empty URL or host")
        if not self.namespace:
            raise ValueError("connection.namespace must be a non-empty Kubernetes namespace")
        if self.client_name is not None and not self.client_name.strip():
            raise ValueError("connection.client_name must be non-empty when set")
        if self.transport_backend not in TransportBackend:
            raise ValueError(
                f"connection.transport_backend must be one of {[m.value for m in TransportBackend]}, "
                f"got {self.transport_backend!r}"
            )
        _require_non_negative(self.nodes_count, "connection.nodes_count")
        _require_positive(self.heartbeat_interval_s, "connection.heartbeat_interval_s")
        _require_positive(self.request_timeout_s, "connection.request_timeout_s")
        _require_positive(self.max_connections, "connection.max_connections", allow_none=True)
        _require_non_negative(self.max_keepalive_connections, "connection.max_keepalive_connections")
        _require_positive(self.keepalive_expiry_s, "connection.keepalive_expiry_s", allow_none=True)
        _require_non_negative(self.connect_retries, "connection.connect_retries")
        _require_non_negative(self.poll_retries, "connection.poll_retries")
        _require_non_negative(self.retry_delay_s, "connection.retry_delay_s")
        _require_positive(self.tracing_timeout_s, "connection.tracing_timeout_s")

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.tracing_endpoint)


@dataclass(frozen=True)
class IdeGymPollingConfig:
    """Backoff for the orchestrator's async-operation polling.

    Start-server, stop-server and every forwarded call (exec, each file chunk) return an
    operation id that the SDK polls until it reaches a terminal state, so these knobs
    decide how hard a large fan-out of sandboxes hammers the orchestrator. Keepalive and
    status are single requests. How long to poll is not configured here: the SDK takes it
    from the call's own budget (``ready_timeout_s``, the exec or chunk timeout,
    ``close_timeout_s``).
    """

    initial_delay_s: float = 0.25
    # Fixed interval; 0 selects the exponential schedule below.
    interval_s: float = 0.0
    backoff_factor: float = 1.5
    max_delay_s: float = 30.0

    def __post_init__(self) -> None:
        if self.backoff_factor < 1:
            raise ValueError("polling.backoff_factor must be >= 1")
        _require_positive(self.initial_delay_s, "polling.initial_delay_s")
        _require_non_negative(self.interval_s, "polling.interval_s")
        _require_positive(self.max_delay_s, "polling.max_delay_s")


@dataclass(frozen=True)
class IdeGymCreateConfig:
    """Server provisioning: readiness budget, retries, and pod shape defaults."""

    # Bounds the whole start_server call: pod scheduling, image pull, and the
    # orchestrator's own 429 back-pressure retries.
    ready_timeout_s: float = 1200.0
    retries: int = 3
    retry_delay_s: float = 5.0
    retry_max_delay_s: float = 60.0
    # The SDK's own in-call retry delay for a 429, bounded by `ready_timeout_s`.
    busy_retry_delay_s: float = 15.0
    server_name_prefix: str = "nemo-gym"
    # Folded into the generated server name, in order. Cosmetic now that metadata also
    # becomes pod labels, but it keeps `kubectl get pods` readable.
    server_name_metadata_keys: tuple[str, ...] = ("instance_id",)
    # The only strategy that can apply here: the reuse lookup considers servers left
    # FINISHED, and this provider always stops the ones it creates.
    reuse_strategy: str = "NONE"
    # The SDK's default. The shipped YAML turns it on for SWE-bench style images; code
    # building the provider directly has to ask for a root container.
    run_as_root: bool = False
    service_port: int = 80
    container_port: int = 8000
    # 0 surfaces the first crash instead of looping restarts.
    max_restarts: int = 0
    # Minutes to hold each server against IdeGYM's inactivity reaper, taken right after
    # start and extended before an exec that would outlive it. None leaves servers to the
    # cluster's inactivity timeout.
    keepalive_minutes: float | None = None
    polling: IdeGymPollingConfig | Mapping[str, Any] = field(default_factory=IdeGymPollingConfig)

    def __post_init__(self) -> None:
        # Frozen dataclass, so normalizing a field goes through object.__setattr__.
        object.__setattr__(self, "polling", coerce_config(self.polling, IdeGymPollingConfig))
        object.__setattr__(self, "server_name_metadata_keys", tuple(self.server_name_metadata_keys or ()))
        if not self.server_name_prefix:
            raise ValueError("create.server_name_prefix must be non-empty")
        for name, port in (("service_port", self.service_port), ("container_port", self.container_port)):
            if not 0 <= port <= 65535:
                raise ValueError(f"create.{name} must be between 0 and 65535, got {port}")
        _require_positive(self.ready_timeout_s, "create.ready_timeout_s")
        _require_non_negative(self.retries, "create.retries")
        _require_non_negative(self.retry_delay_s, "create.retry_delay_s")
        _require_non_negative(self.retry_max_delay_s, "create.retry_max_delay_s")
        _require_non_negative(self.busy_retry_delay_s, "create.busy_retry_delay_s")
        _require_non_negative(self.max_restarts, "create.max_restarts")
        _require_keepalive_minutes(self.keepalive_minutes, "create.keepalive_minutes")


@dataclass(frozen=True)
class IdeGymExecConfig:
    """Command execution through the IdeGYM bash tool."""

    default_timeout_s: float | None = 180.0
    # Grace period a timed-out process group gets on SIGTERM before it is SIGKILLed.
    graceful_termination_timeout_s: float = 2.0
    # Client-side budget on top of the command timeout, covering the orchestrator round
    # trip. Keeping it non-zero means the sandbox's own timeout fires first, so the
    # caller gets the partial output instead of a bare transport error.
    request_overhead_s: float = 60.0
    # Bytes kept per output stream, head and tail around an IdeGYM truncation marker. The
    # default is the SDK's own; None keeps complete output.
    max_output_bytes: int | None = 1024 * 1024

    def __post_init__(self) -> None:
        _require_positive(self.default_timeout_s, "exec.default_timeout_s", allow_none=True)
        _require_non_negative(self.graceful_termination_timeout_s, "exec.graceful_termination_timeout_s")
        _require_non_negative(self.request_overhead_s, "exec.request_overhead_s")
        _require_positive(self.max_output_bytes, "exec.max_output_bytes", allow_none=True)


@dataclass(frozen=True)
class IdeGymVerifyConfig:
    """Post-start verification, on top of the orchestrator's own readiness wait.

    IdeGYM reports a server only once its pod passes the Kubernetes readiness probe
    against the server's health endpoint, so what is left worth checking is that
    ``spec.workdir`` exists in the image.
    """

    timeout_s: float = 30.0
    # Fail create on a missing `spec.workdir` instead of letting every later exec fail
    # long after the cause is visible.
    check_workdir: bool = True

    def __post_init__(self) -> None:
        _require_positive(self.timeout_s, "verify.timeout_s")


@dataclass(frozen=True)
class IdeGymFilesConfig:
    """File transfer through the SDK's chunked binary upload and download."""

    # The SDK's default chunk size.
    chunk_bytes: int = 1024 * 1024
    # Bounds each chunk request, not the whole transfer.
    timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if not 1 <= self.chunk_bytes <= MAX_FILE_CHUNK_BYTES:
            raise ValueError(f"files.chunk_bytes must be between 1 and {MAX_FILE_CHUNK_BYTES}, got {self.chunk_bytes}")
        _require_positive(self.timeout_s, "files.timeout_s")


@dataclass(frozen=True)
class IdeGymOperationsConfig:
    """Post-create lifecycle calls: status, keepalive and teardown."""

    close_timeout_s: float = 180.0
    # Bounds a status read and a keepalive extension alike.
    status_timeout_s: float = 60.0
    # Rounds of stop_server after a transient failure; status() is never retried. Each SDK
    # stop_server already makes up to 3 attempts, but all of them share close_timeout_s.
    retries: int = 1
    retry_delay_s: float = 1.0
    retry_max_delay_s: float = 15.0
    # Ceiling for a single keepalive extension before a long exec.
    keepalive_minutes_max: float = MAX_KEEPALIVE_MINUTES

    def __post_init__(self) -> None:
        _require_positive(self.close_timeout_s, "operations.close_timeout_s")
        _require_positive(self.status_timeout_s, "operations.status_timeout_s")
        _require_non_negative(self.retries, "operations.retries")
        _require_non_negative(self.retry_delay_s, "operations.retry_delay_s")
        _require_non_negative(self.retry_max_delay_s, "operations.retry_max_delay_s")
        _require_keepalive_minutes(self.keepalive_minutes_max, "operations.keepalive_minutes_max")


@dataclass(frozen=True)
class IdeGymAttributionConfig(AttributionConfig):
    """Job attribution: Kubernetes labels on every server pod, plus the client name.

    The labels work as on OpenSandbox (``key_prefix``-namespaced team, user, workload and
    run). The registered client is also named ``<prefix>-<team>-<user>-<workload>``,
    because that name is what the IdeGYM dashboard groups by and what per-name resource
    quotas key on. ``run`` stays out of the name, which has to stay stable across
    launches for quota matching to work.
    """

    client_name_prefix: str = "nemo-gym"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.client_name_prefix:
            raise ValueError("attribution.client_name_prefix must be non-empty")
