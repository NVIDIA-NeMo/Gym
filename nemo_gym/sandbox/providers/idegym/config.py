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
``exec``, ``probe``, ``files``, ``operations``, ``attribution`` — mirroring the shipped
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

from nemo_gym.sandbox.providers.utils import coerce_config


# The sandbox runs `/bin/sh -c "bash -c <script>"`, so the script travels as one
# execve() argument. Linux caps that at MAX_ARG_STRLEN (128 KiB) and rejects longer
# ones with E2BIG, so generated scripts stay comfortably below it.
MAX_COMMAND_BYTES = 100 * 1024

# Ceiling for one upload chunk. base64 inflates by 4/3, and the encoded payload shares
# the command budget with the `cd`, the `mkdir -p`, and the caller's env exports, so the
# limit leaves real headroom rather than filling the budget with payload alone.
MAX_UPLOAD_CHUNK_BYTES = 64 * 1024

# The probe asserts stdout equality, so these two only make sense as a pair.
DEFAULT_PROBE_COMMAND = "printf idegym-sandbox-ready"
DEFAULT_PROBE_EXPECTED = "idegym-sandbox-ready"

# RFC-1035 caps Kubernetes resource names at 63 characters, and the orchestrator derives
# the pod name as `<server_name>-<server_id>`, so the name the provider sends leaves room
# for that suffix.
MAX_SERVER_NAME_LENGTH = 63
SERVER_ID_SUFFIX_RESERVE = 8


class UserMode(StrEnum):
    """How ``exec(user=...)`` is honored.

    The IdeGYM bash tool has no user field: commands run as whatever user the server
    container runs as, which ``provider_options.run_as_root`` decides at pod level.
    ``IGNORE`` therefore drops the request with one warning, while the two switch modes
    wrap the script in a privilege-dropping command for images that ship one.
    """

    IGNORE = "ignore"
    RUNUSER = "runuser"
    SU = "su"


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
    # only while polling, so provisioning cannot block execs on running sandboxes.
    max_connections: int | None = 64
    max_keepalive_connections: int = 20
    keepalive_expiry_s: float | None = 5.0
    connect_retries: int = 2
    # The SDK ships a default OTLP endpoint, so tracing stays off until one is set here:
    # NeMo Gym does not send telemetry to third parties implicitly.
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
        _require_positive(self.tracing_timeout_s, "connection.tracing_timeout_s")

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.tracing_endpoint)


@dataclass(frozen=True)
class IdeGymPollingConfig:
    """Backoff for the orchestrator's async-operation polling.

    Every mutating IdeGYM call returns an operation id that the SDK polls until it
    reaches a terminal state, so these knobs decide how hard a large fan-out of
    sandboxes hammers the orchestrator while it provisions pods.
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
    ready_timeout_s: float = 900.0
    retries: int = 3
    retry_delay_s: float = 5.0
    retry_max_delay_s: float = 60.0
    # The SDK's own in-call retry delay for a 429, bounded by `ready_timeout_s`.
    busy_retry_delay_s: float = 15.0
    server_name_prefix: str = "nemo-gym"
    # Folded into the generated server name, in order, so a pod traces back to its task.
    server_name_metadata_keys: tuple[str, ...] = ("instance_id",)
    # The only strategy that can apply here: the reuse lookup considers servers left
    # FINISHED, and this provider always stops the ones it creates.
    reuse_strategy: str = "NONE"
    run_as_root: bool = False
    service_port: int = 80
    container_port: int = 8000
    # 0 surfaces the first crash instead of looping restarts.
    max_restarts: int = 0
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
    user_mode: str = UserMode.IGNORE

    def __post_init__(self) -> None:
        if self.user_mode not in UserMode:
            raise ValueError(f"exec.user_mode must be one of {[m.value for m in UserMode]}, got {self.user_mode!r}")
        _require_positive(self.default_timeout_s, "exec.default_timeout_s", allow_none=True)
        _require_non_negative(self.graceful_termination_timeout_s, "exec.graceful_termination_timeout_s")
        _require_non_negative(self.request_overhead_s, "exec.request_overhead_s")


@dataclass(frozen=True)
class IdeGymProbeConfig:
    """Readiness verification run after the orchestrator reports a started server.

    A started pod is not the same as a sandbox that can run commands, so ``create()``
    only returns once this probe has passed ``stable_count`` times.
    """

    command: str | None = DEFAULT_PROBE_COMMAND
    expected_stdout: str | None = DEFAULT_PROBE_EXPECTED
    timeout_s: float = 30.0
    deadline_s: float | None = 180.0
    stable_count: int = 1
    # Non-zero: the probe polls a remote orchestrator, so back-to-back retries would
    # hammer it for the whole deadline while a pod is still warming up.
    stable_delay_s: float = 2.0
    # Fail create on a missing `spec.workdir` instead of letting every later exec fail
    # on `cd`, long after the cause is visible.
    verify_workdir: bool = True

    def __post_init__(self) -> None:
        if self.stable_count < 1:
            raise ValueError("probe.stable_count must be >= 1")
        # Always validated: the workdir check uses this timeout even when the probe
        # command is disabled.
        _require_positive(self.timeout_s, "probe.timeout_s")
        _require_positive(self.deadline_s, "probe.deadline_s", allow_none=True)
        _require_non_negative(self.stable_delay_s, "probe.stable_delay_s")


@dataclass(frozen=True)
class IdeGymFilesConfig:
    """File transfer, which rides base64 over the bash tool.

    IdeGYM exposes no binary transfer endpoint through the orchestrator's JSON request
    forwarding, so bytes are chunked and shell-encoded. Upload chunks share the command
    budget with the caller's ``env`` exports, so a large ``env`` can still push a legal
    chunk over it; download chunks are bounded by how much text the orchestrator is
    willing to store per operation.
    """

    upload_chunk_bytes: int = 48 * 1024
    download_chunk_bytes: int = 192 * 1024
    # Refuse oversized downloads rather than filling the caller's memory with a
    # base64-inflated copy of, say, a multi-gigabyte build tree.
    max_download_bytes: int | None = 64 * 1024 * 1024
    timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if self.upload_chunk_bytes < 1:
            raise ValueError("files.upload_chunk_bytes must be >= 1")
        if self.upload_chunk_bytes > MAX_UPLOAD_CHUNK_BYTES:
            raise ValueError(
                f"files.upload_chunk_bytes must be <= {MAX_UPLOAD_CHUNK_BYTES} so the base64-encoded chunk "
                f"still leaves room for the rest of the command, got {self.upload_chunk_bytes}"
            )
        if self.download_chunk_bytes < 1:
            raise ValueError("files.download_chunk_bytes must be >= 1")
        _require_positive(self.max_download_bytes, "files.max_download_bytes", allow_none=True)
        _require_positive(self.timeout_s, "files.timeout_s")


@dataclass(frozen=True)
class IdeGymOperationsConfig:
    """Post-create lifecycle calls: status polling and teardown."""

    close_timeout_s: float = 180.0
    status_timeout_s: float = 60.0
    # Retries for control-plane calls that are safe to repeat (status, delete).
    retries: int = 2
    retry_delay_s: float = 1.0
    retry_max_delay_s: float = 15.0

    def __post_init__(self) -> None:
        _require_positive(self.close_timeout_s, "operations.close_timeout_s")
        _require_positive(self.status_timeout_s, "operations.status_timeout_s")
        _require_non_negative(self.retries, "operations.retries")
        _require_non_negative(self.retry_delay_s, "operations.retry_delay_s")
        _require_non_negative(self.retry_max_delay_s, "operations.retry_max_delay_s")


@dataclass(frozen=True)
class IdeGymAttributionConfig:
    """Job attribution, which names the registered IdeGYM client.

    IdeGYM has no per-server label API, so attribution cannot become Kubernetes labels
    the way it does on OpenSandbox. Naming the client instead is what the dashboard
    groups by and what per-name resource quotas key on, so a team's sandboxes stay
    attributable and quota-bounded together as ``<prefix>-<team>-<user>-<workload>``.
    """

    enabled: bool = True
    team: str | None = None
    user: str | None = None
    workload: str | None = None
    # Resolved and logged, but deliberately kept out of the client name, which has to
    # stay stable across launches for quota matching to work.
    run: str | None = None
    client_name_prefix: str = "nemo-gym"

    def __post_init__(self) -> None:
        if not self.client_name_prefix:
            raise ValueError("attribution.client_name_prefix must be non-empty")
