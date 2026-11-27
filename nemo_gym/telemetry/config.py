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
"""Schema for the ``telemetry:`` block of a NeMo Gym config.

``nemo_gym.telemetry.setup`` translates this block into ``NEMO_GYM_OTEL_*`` environment
variables in the orchestrating CLI process *before* it spawns the server processes, so
every server, and every Ray worker under them, inherits one consistent configuration.
Raw ``NEMO_GYM_OTEL_*`` / ``NEMO_LENS_*`` / ``OTEL_*`` env vars always win over these
YAML values: the translation uses ``os.environ.setdefault``.

This module imports only pydantic. It never requires nemo-lens, so it is safe to import
unconditionally from Gym's config machinery.
"""

import re
from typing import Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator


# The node_exporter metrics the collector's local `metrics.jsonl` keeps by default: CPU time, load, memory, pressure
# stall and network/InfiniBand byte totals, plus `up` to tell a filtered exporter from an unreachable one. The rest of
# node_exporter's output (per-port InfiniBand error counters, systemd unit states, NFS mount statistics, hardware
# sensors and the like) is node health rather than benchmark performance, and was about 90% of the file's bytes.
LOCAL_NODE_METRICS = (
    "up",
    "node_cpu_seconds_total",
    "node_load(1|5|15)",
    "node_memory_(MemTotal|MemAvailable|MemFree|Cached|Buffers|Shmem|SwapTotal|SwapFree)_bytes",
    "node_pressure_.*",
    "node_network_(receive|transmit)_bytes_total",
    "node_infiniband_port_data_(received|transmitted)_bytes_total",
)

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SCRAPE_TARGET_RE = re.compile(r"^[^\s:/]+:\d{1,5}$")
# Resource attributes the collector sets itself, which `resource_attributes` may not name.
COLLECTOR_RESOURCE_ATTRIBUTES = frozenset({"service.name", "user", "run_id", "host", "slurm_job_id", "Authorization"})


class CollectorConfig(BaseModel, extra="forbid"):
    """An OpenTelemetry Collector that the process starting Gym's servers runs beside them.

    That is ``gym env start`` and a ``gym eval run`` that starts its own servers; ``gym eval submit`` has its own
    collector (the ``otel`` block). Setting this block is what turns it on: without it no collector starts, so a
    machine without ``otelcol-contrib`` runs as before. It needs ``telemetry.enabled``.
    """

    binary: str = "otelcol-contrib"
    """The collector executable: a path, or a name looked up on ``PATH``."""

    output_dir: Optional[str] = None
    """Where ``otel/<run id>/`` (the JSONL, the rendered config and the collector's log) is written.
    Defaults to ``nemo_gym_log_dir`` when set, else the working directory."""

    remote_export: bool = False
    """Also export to ``endpoint`` over OTLP/HTTP. Off by default: local JSONL only."""

    endpoint: Optional[str] = None
    """OTLP/HTTP ingest base URL, for remote export."""

    service_name: Optional[str] = None
    """The ``service.name`` the backend routes the token by, for remote export."""

    token_env: str = "OTEL_TOKEN"
    """The environment variable holding the ingest token. Read by the collector; never written to a file."""

    scrape_targets: dict[str, str] = Field(default_factory=dict)
    """Prometheus ``/metrics`` endpoints to scrape, as ``host:port`` by name, e.g. ``{policy: "localhost:8000"}``
    for a vLLM server. Each is scraped as job ``<component>/<name>`` with the label ``gym_service: <name>``."""

    component: str = "gym-vllm"
    """Display identity prefix of the scraped metrics (``service.name.override``)."""

    gpu_metrics_port: Optional[int] = Field(default=None, ge=1, le=65535)
    """dcgm-exporter on this host, scraped as job ``dcgm`` when set."""

    node_metrics_port: Optional[int] = Field(default=None, ge=1, le=65535)
    """node_exporter on this host, scraped as job ``node`` when set."""

    local_node_metrics: Optional[list[str]] = Field(default_factory=lambda: list(LOCAL_NODE_METRICS))
    """node_exporter metrics kept in the local ``metrics.jsonl``, as regular expressions matched against the whole
    name. ``null`` keeps all of them."""

    local_node_metrics_extra: list[str] = Field(default_factory=list)
    """More patterns kept on top of ``local_node_metrics``."""

    resource_attributes: dict[str, str] = Field(default_factory=dict)
    """Added to everything the collector exports, e.g. ``benchmark``, ``cluster`` or ``model``, so a run can be
    filtered like a submitted one. A producer's own value of the same attribute is kept. ``user``, ``run_id``,
    ``host`` and, under Slurm, ``slurm_job_id`` are set already."""

    scrape_interval_seconds: int = Field(default=15, ge=1)

    # The ports the collector listens on, on localhost only. Two runs on one host each need their own.
    otlp_grpc_port: int = Field(default=4317, ge=1, le=65535)
    otlp_http_port: int = Field(default=4318, ge=1, le=65535)
    health_port: int = Field(default=4319, ge=1, le=65535)

    health_check_timeout_seconds: float = Field(default=60.0, gt=0)
    """How long a collector that was started has to answer its health check, as ``otel.health_check_timeout_seconds``
    for a submitted job. At the run's start, failing it fails the start; at a restart, it counts as one."""

    shutdown_timeout_seconds: float = Field(default=30.0, gt=0)
    """How long the collector has to flush and exit after the servers have stopped, before it is killed."""

    max_restarts: int = Field(default=5, ge=0)
    """How many times a collector that exits during the run is started again. After that the run goes on without one."""

    @field_validator("token_env")
    @classmethod
    def _validate_token_env(cls, v: str) -> str:
        if not _ENV_VAR_NAME_RE.match(v):
            raise ValueError(f"{v!r} is not a valid environment variable name")
        return v

    @field_validator("scrape_targets")
    @classmethod
    def _validate_scrape_targets(cls, v: dict[str, str]) -> dict[str, str]:
        for name, target in v.items():
            if not _SCRAPE_TARGET_RE.match(target):
                raise ValueError(f"scrape target {name!r}: {target!r} is not host:port")
        return v

    @field_validator("local_node_metrics", "local_node_metrics_extra")
    @classmethod
    def _validate_metric_patterns(cls, v: list[str] | None) -> list[str] | None:
        for pattern in v or ():
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"{pattern!r} is not a valid regular expression: {e}") from e
        return v

    @field_validator("resource_attributes")
    @classmethod
    def _validate_resource_attributes(cls, v: dict[str, str]) -> dict[str, str]:
        reserved = sorted(COLLECTOR_RESOURCE_ATTRIBUTES & v.keys())
        if reserved:
            raise ValueError(f"{', '.join(reserved)} are set by Gym and cannot be given here")
        return v

    @model_validator(mode="after")
    def _validate_collector(self) -> "CollectorConfig":
        ports = (self.otlp_grpc_port, self.otlp_http_port, self.health_port)
        if len(set(ports)) != len(ports):
            raise ValueError(f"otlp_grpc_port, otlp_http_port and health_port must differ, got {ports}")
        if not self.remote_export:
            return self
        # Blank counts as missing: an unset `${oc.env:VAR,""}` resolves to "".
        missing = [field for field in ("endpoint", "service_name") if not (getattr(self, field) or "").strip()]
        if missing:
            raise ValueError(f"remote_export needs {' and '.join(missing)}; set them, or remote_export: false")
        parts = urlsplit(self.endpoint.strip())
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"endpoint must be an http:// or https:// URL with a host, got {self.endpoint!r}")
        return self


class MemoryProfilingConfig(BaseModel):
    """Periodic host and logical-server memory sampling."""

    enabled: bool = False
    """Collect host memory and every server process tree."""

    interval_seconds: float = Field(default=1.0, gt=0)
    """Seconds between samples."""


class TelemetryConfig(BaseModel, extra="allow"):
    """OpenTelemetry / nemo-lens configuration for NeMo Gym.

    Telemetry is optional twice over: it activates only when ``enabled`` is true *and*
    nemo-lens is installed (``uv sync --extra telemetry``). When either is missing every
    instrumentation site in ``nemo_gym`` degrades to a ~0-cost no-op and Gym behaves
    exactly as it does today.
    """

    enabled: bool = False
    """Master switch. When false, all instrumentation is a ~0-cost no-op."""

    service_name: str = "nemo-gym"
    """``service.name`` reported to the backend.

    Passed through the standard ``OTEL_SERVICE_NAME`` env var, not through
    ``instrument_fastapi(service_name=...)`` — that parameter is accepted and ignored by
    nemo-lens at the pinned commit (``contrib/fastapi.py``).

    Each server process appends its own Gym server name, so a run yields
    ``nemo-gym/example_single_tool_call``, ``nemo-gym/policy_model``, and so on rather
    than three indistinguishable ``nemo-gym`` services. Set ``service_name_per_server``
    to false to opt out.
    """

    service_name_per_server: bool = True
    """Suffix ``service_name`` with each server's Gym config name.

    On by default because Gym runs several processes that would otherwise share one
    ``service.name``, which makes a backend's service map useless. The unsuffixed name
    stays available as the ``nemo.gym.service_group`` resource attribute."""

    span_groups: str = "default"
    """Span-group spec: a preset (``default`` | ``per_rollout`` | ``all``) or a
    comma-separated list of group names (e.g. ``"default,sandbox"``). See
    :class:`~nemo_gym.telemetry.span_groups.GymSpanGroup`. An unknown name is logged as a
    warning and ignored rather than rejected."""

    traces_enabled: bool = True
    """Emit trace spans."""

    metrics_enabled: bool = True
    """Emit metric instruments (the ``gym.*`` histograms/gauges plus the FastAPI
    instrumentor's ``http.server.*``)."""

    memory_profiling: MemoryProfilingConfig = Field(default_factory=MemoryProfilingConfig)
    """Sample each Gym-managed server process tree and emit memory metrics."""

    sandbox_resource_sample_interval_seconds: float = Field(default=10.0, ge=0)
    """Seconds between existing provider resource snapshots; zero disables sampling.

    Requires sandbox spans and an active trace exporter. Records are correlated with the
    creating rollout, including when cleanup runs outside its request context.
    """

    event_loop_sample_interval_seconds: float = Field(default=1.0, gt=0)
    """Seconds between samples of each process's event-loop lag, when metrics are on."""

    event_loop_task_count: bool = False
    """Also report how many asyncio tasks exist on each event loop, at every sample.

    Off by default because counting tasks walks every task on the loop while holding it.
    That takes about a millisecond per sample at 16,000 tasks, which a driver at high concurrency can reach.
    """

    logs_enabled: bool = False
    """Bridge Python logging to OTel logs, exported with trace correlation."""

    exporter: str = "otlp"
    """Exporter backend: ``otlp`` | ``console``. ``console`` writes one JSON object per
    line to stdout, which is what the local validation flow greps."""

    otlp_endpoint: Optional[str] = None
    """``OTEL_EXPORTER_OTLP_ENDPOINT`` for the ``otlp`` exporter, e.g.
    ``https://api.honeycomb.io``. Optional: the standard env var still works and always
    wins (``setdefault``), so any OTLP-compatible backend or collector that expects the
    raw env var keeps working unchanged. Setting it here just means a run's destination is
    committed with the rest of the config instead of exported by hand every time."""

    otlp_protocol: Optional[str] = None
    """``OTEL_EXPORTER_OTLP_PROTOCOL``: ``grpc`` (the OTel SDK default) or
    ``http/protobuf``."""

    otlp_headers: Optional[str] = None
    """``OTEL_EXPORTER_OTLP_HEADERS``, e.g. ``x-honeycomb-team=...``. Prefer
    ``${oc.env:MY_API_KEY}`` interpolation for the secret portion so the raw credential
    does not live in a committed YAML file."""

    run_id: Optional[str] = None
    """Correlates every process of one ``gym env start`` / ``gym env test`` invocation.
    Generated in the supervisor and inherited by the servers when unset."""

    collector: Optional[CollectorConfig] = None
    """Start an OpenTelemetry Collector beside the servers (see :class:`CollectorConfig`). Unless an
    OTLP endpoint is set, the process that starts it points the servers at it."""
