# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiohttp connection-pool capacity diagnostics and queue-wait metrics."""

import asyncio
import logging
import resource
import time
from asyncio.exceptions import CancelledError
from contextvars import ContextVar, Token
from math import ceil
from pathlib import Path
from typing import Any, NamedTuple, Optional, Protocol

from aiohttp import TraceConfig

from nemo_gym.telemetry.gym_metrics import record_http_connection_pool_queue_duration
from nemo_gym.telemetry.setup import is_metrics_exporting


logger = logging.getLogger(__name__)


class ConnectionPoolConfig(Protocol):
    global_aiohttp_connector_limit: int
    global_aiohttp_connector_limit_per_host: int
    global_aiohttp_intended_concurrency: Optional[int]
    global_aiohttp_intended_concurrency_per_host: Optional[int]


class ConnectionPoolCapacity(NamedTuple):
    workers: int
    total: int
    per_host: int
    intended: Optional[int]
    intended_per_host: Optional[int]


_REPORTED_CAPACITIES: set[tuple[object, ...]] = set()
_SERVER_NAME: ContextVar[str] = ContextVar("nemo_gym_http_server_name", default="external")


def set_server_name(server_name: str) -> Token[str]:
    """Set the bounded destination label while one logical request and its retries run."""
    return _SERVER_NAME.set(server_name)


def reset_server_name(token: Token[str]) -> None:
    """Restore the caller's destination label."""
    _SERVER_NAME.reset(token)


def _ephemeral_port_capacity() -> Optional[int]:
    """Return Linux's approximate per-destination ephemeral-port budget when available."""
    try:
        low, high = (int(value) for value in Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split())
    except (OSError, ValueError):
        return None
    return high - low + 1


def connection_pool_capacity(cfg: ConnectionPoolConfig, workers: int) -> ConnectionPoolCapacity:
    """Calculate aiohttp limits for one process, preserving explicit unlimited values."""
    if workers < 1:
        raise ValueError(f"FastAPI worker count must be at least 1, got {workers}.")

    configured_total = cfg.global_aiohttp_connector_limit
    configured_per_host = cfg.global_aiohttp_connector_limit_per_host
    total = configured_total // workers if configured_total else 0
    per_host = configured_per_host // workers if configured_per_host else 0
    if (configured_total > 0 and total == 0) or (configured_per_host > 0 and per_host == 0):
        raise ValueError(
            "positive aiohttp connector limits must remain at least 1 after division by FastAPI workers: "
            f"workers={workers}, aggregate_total={configured_total}, aggregate_per_host={configured_per_host}, "
            f"effective_total={total}, effective_per_host={per_host}. Increase the aggregate limits, reduce workers, "
            "or set a limit explicitly to 0 for unlimited."
        )

    intended = (
        ceil(cfg.global_aiohttp_intended_concurrency / workers)
        if cfg.global_aiohttp_intended_concurrency is not None
        else None
    )
    intended_per_host = (
        ceil(cfg.global_aiohttp_intended_concurrency_per_host / workers)
        if cfg.global_aiohttp_intended_concurrency_per_host is not None
        else None
    )
    return ConnectionPoolCapacity(workers, total, per_host, intended, intended_per_host)


def _effective_per_host_limit(total: int, per_host: int) -> int:
    if total == 0:
        return per_host
    if per_host == 0:
        return total
    return min(total, per_host)


def _display_limit(limit: int) -> str:
    return "unlimited" if limit == 0 else str(limit)


def report_connection_pool_capacity(
    cfg: ConnectionPoolConfig,
    capacity: ConnectionPoolCapacity,
    *,
    visible: bool = False,
) -> None:
    """Report one server/CLI process group's pool sizing and unsafe capacity mismatches."""
    workers, total, per_host, intended, intended_per_host = capacity
    report_key = (
        workers,
        cfg.global_aiohttp_connector_limit,
        cfg.global_aiohttp_connector_limit_per_host,
        cfg.global_aiohttp_intended_concurrency,
        cfg.global_aiohttp_intended_concurrency_per_host,
    )
    if report_key in _REPORTED_CAPACITIES:
        return
    _REPORTED_CAPACITIES.add(report_key)

    file_descriptors = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    ephemeral_ports = _ephemeral_port_capacity()
    enforced_per_host = _effective_per_host_limit(total, per_host)
    capacity_message = (
        f"aiohttp connection pool capacity for this server/CLI process group: workers={workers} "
        f"aggregate_total={_display_limit(cfg.global_aiohttp_connector_limit)} "
        f"aggregate_per_host={_display_limit(cfg.global_aiohttp_connector_limit_per_host)} "
        f"effective_total={_display_limit(total)} effective_per_host={_display_limit(enforced_per_host)} "
        f"configured_per_host={_display_limit(per_host)} "
        f"intended_per_worker={intended} intended_per_host_per_worker={intended_per_host} "
        f"file_descriptor_soft_limit={file_descriptors} ephemeral_ports_per_destination={ephemeral_ports}. "
        "The aiohttp total limit is a scheduling limit, not a strict open-socket cap across multiple hosts."
    )
    if visible:
        print(capacity_message, flush=True)
    else:
        logger.info(capacity_message)

    warnings = []
    if intended is not None and total and intended > total:
        warnings.append(f"intended per-worker concurrency {intended} exceeds effective total limit {total}")
    if intended_per_host is not None and enforced_per_host and intended_per_host > enforced_per_host:
        warnings.append(
            f"intended per-host concurrency {intended_per_host} exceeds effective per-host limit {enforced_per_host}"
        )
    if file_descriptors != resource.RLIM_INFINITY and (total == 0 or total >= file_descriptors):
        warnings.append(
            f"effective total limit {_display_limit(total)} can exhaust the file-descriptor soft limit "
            f"{file_descriptors} before accounting for non-HTTP descriptors"
        )
    if intended is not None and file_descriptors != resource.RLIM_INFINITY and intended >= file_descriptors:
        warnings.append(
            f"intended per-worker concurrency {intended} can exhaust the file-descriptor soft limit "
            f"{file_descriptors} before accounting for non-HTTP descriptors"
        )
    aggregate_intended_per_host = cfg.global_aiohttp_intended_concurrency_per_host
    if (
        aggregate_intended_per_host is not None
        and ephemeral_ports is not None
        and aggregate_intended_per_host > ephemeral_ports
    ):
        warnings.append(
            f"aggregate intended per-host concurrency {aggregate_intended_per_host} exceeds the approximate "
            f"per-destination ephemeral-port budget {ephemeral_ports}"
        )
    if warnings:
        logger.warning(
            "aiohttp connection pool may queue requests or exhaust host resources: %s. Adjust connector limits or "
            "concurrency while accounting for file-descriptor, ephemeral-port, and backend connection budgets.",
            "; ".join(warnings),
        )


class ConnectionQueueContext:
    """Queue state for one aiohttp request attempt."""

    def __init__(self, server_name: str) -> None:
        self.server_name = server_name
        self.started_at: Optional[float] = None
        self.duration_ms = 0.0
        self.queue_constraints: set[str] = set()
        self.recorded = False

    def queued(self, queue_constraint: str) -> None:
        self.queue_constraints.add(queue_constraint)
        if self.started_at is None:
            self.started_at = time.perf_counter()

    def released(self) -> None:
        if self.started_at is not None:
            self.duration_ms += (time.perf_counter() - self.started_at) * 1000.0
            self.started_at = None

    def queue_constraint(self) -> str:
        if not self.queue_constraints:
            return "none"
        known = self.queue_constraints - {"unknown"}
        if len(known) > 1:
            return "mixed"
        return next(iter(known)) if known else "unknown"

    def record(self, attempt_outcome: str) -> None:
        if self.recorded:
            return
        self.recorded = True
        self.released()
        record_http_connection_pool_queue_duration(
            self.duration_ms,
            queue_constraint=self.queue_constraint(),
            attempt_outcome=attempt_outcome,
            server_name=self.server_name,
        )


def _connector_queue_constraint(session: Any) -> str:
    """Classify the binding connector limit when aiohttp reports a queue wait."""
    connector = getattr(session, "connector", None)
    limit = getattr(connector, "limit", None)
    acquired = getattr(connector, "_acquired", None)
    if limit is None or acquired is None:
        return "unknown"
    if limit == 0:
        return "per_host"
    try:
        return "total" if limit - len(acquired) <= 0 else "per_host"
    except Exception:
        return "unknown"


async def _on_connection_queued_start(session: Any, context: ConnectionQueueContext, _params: Any) -> None:
    try:
        context.queued(_connector_queue_constraint(session))
    except Exception:
        logger.debug("Failed to start aiohttp connection-queue telemetry", exc_info=True)


async def _on_connection_queued_end(_session: Any, context: ConnectionQueueContext, _params: Any) -> None:
    try:
        context.released()
    except Exception:
        logger.debug("Failed to finish aiohttp connection-queue telemetry", exc_info=True)


async def _on_request_end(_session: Any, context: ConnectionQueueContext, _params: Any) -> None:
    try:
        context.record("ok")
    except Exception:
        logger.debug("Failed to record aiohttp connection-queue telemetry", exc_info=True)


async def _on_request_exception(_session: Any, context: ConnectionQueueContext, params: Any) -> None:
    try:
        if isinstance(params.exception, CancelledError):
            outcome = "cancelled"
        elif isinstance(params.exception, asyncio.TimeoutError):
            outcome = "timeout"
        else:
            outcome = "error"
        context.record(outcome)
    except Exception:
        logger.debug("Failed to record aiohttp connection-queue telemetry", exc_info=True)


def _trace_context_factory(*, trace_request_ctx: Any = None) -> ConnectionQueueContext:
    return ConnectionQueueContext(server_name=_SERVER_NAME.get())


def build_connection_pool_trace_configs() -> list[TraceConfig]:
    """Build queue instrumentation only when this process exports metrics."""
    if not is_metrics_exporting():
        return []
    trace_config = TraceConfig(trace_config_ctx_factory=_trace_context_factory)
    trace_config.on_connection_queued_start.append(_on_connection_queued_start)
    trace_config.on_connection_queued_end.append(_on_connection_queued_end)
    trace_config.on_request_end.append(_on_request_end)
    trace_config.on_request_exception.append(_on_request_exception)
    return [trace_config]
