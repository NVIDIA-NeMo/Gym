# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiohttp connection-pool capacity diagnostics and queue-wait metrics."""

import logging
import resource
import time
from contextvars import ContextVar, Token
from math import ceil
from pathlib import Path
from threading import Lock
from typing import Any, NamedTuple, Optional, Protocol

from aiohttp import TCPConnector
from aiohttp.client_reqrep import ClientRequest, ConnectionKey
from aiohttp.connector import Connection
from aiohttp.tracing import Trace

from nemo_gym.telemetry.gym_metrics import (
    record_http_connection_pool_queue_duration,
    register_http_connection_pool_connect_counter,
)
from nemo_gym.telemetry.setup import is_metrics_exporter_active


logger = logging.getLogger(__name__)


class ConnectionPoolConfig(Protocol):
    """Connector budgets and optional demand estimates for one server or rollout CLI.

    Configured limits apply to one server process group before division across its
    FastAPI workers, not the whole deployment. An explicit limit of zero is unlimited.
    Intended concurrency is optional expected outbound demand for that same group;
    ``None`` disables the corresponding sizing check.
    """

    global_aiohttp_connector_limit: int
    global_aiohttp_connector_limit_per_host: int
    global_aiohttp_intended_concurrency: Optional[int]
    global_aiohttp_intended_concurrency_per_host: Optional[int]


class ConnectionPoolCapacity(NamedTuple):
    """Per-process connector limits and demand after division across ``workers``.

    ``workers`` is the size of one server's FastAPI process group, or one for the CLI.
    Positive ``total`` and ``per_host`` limits are rounded down; zero means explicitly
    unlimited. ``per_host`` is the divided configured value, before a finite ``total``
    constrains effective per-host capacity.
    ``intended`` and ``intended_per_host`` are expected demand rounded up per worker,
    or ``None`` when the corresponding estimate is not configured.
    """

    workers: int
    total: int
    per_host: int
    intended: Optional[int]
    intended_per_host: Optional[int]


_REPORTED_CAPACITIES: set[tuple[object, ...]] = set()
_SERVER_NAME: ContextVar[str] = ContextVar("nemo_gym_http_server_name", default="external")
_CONNECT_COUNTS: dict[str, int] = {}
_CONNECT_COUNTS_LOCK = Lock()


def _connect_count_snapshot() -> dict[str, int]:
    with _CONNECT_COUNTS_LOCK:
        return _CONNECT_COUNTS.copy()


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
        f"per_worker_per_host={_display_limit(per_host)} "
        f"intended_per_worker={intended} intended_per_host_per_worker={intended_per_host} "
        f"file_descriptor_soft_limit={file_descriptors} ephemeral_ports_per_destination={ephemeral_ports}. "
        "The aiohttp total limit is a scheduling limit, not a strict open-socket cap across multiple hosts."
    )
    if visible:
        print(capacity_message, flush=True)
    else:
        logger.info(capacity_message)

    pool_warnings = []
    file_descriptor_warnings = []
    if intended is not None and total and intended > total:
        pool_warnings.append(f"intended per-worker concurrency {intended} exceeds effective total limit {total}")
    if intended_per_host is not None and enforced_per_host and intended_per_host > enforced_per_host:
        pool_warnings.append(
            f"intended per-host concurrency {intended_per_host} exceeds effective per-host limit {enforced_per_host}"
        )
    if intended is not None and file_descriptors != resource.RLIM_INFINITY and intended >= file_descriptors:
        file_descriptor_warnings.append(
            f"intended per-worker concurrency {intended} can exhaust the file-descriptor soft limit "
            f"{file_descriptors} before accounting for non-HTTP descriptors"
        )
    aggregate_intended_per_host = cfg.global_aiohttp_intended_concurrency_per_host
    if (
        aggregate_intended_per_host is not None
        and ephemeral_ports is not None
        and aggregate_intended_per_host > ephemeral_ports
    ):
        pool_warnings.append(
            f"aggregate intended per-host concurrency {aggregate_intended_per_host} exceeds the approximate "
            f"per-destination ephemeral-port budget {ephemeral_ports}"
        )
    if pool_warnings:
        logger.warning(
            "aiohttp connection pool may queue requests or exhaust host resources: %s. Adjust connector limits or "
            "concurrency while accounting for file-descriptor, ephemeral-port, and backend connection budgets.",
            "; ".join(pool_warnings),
        )
    if file_descriptor_warnings:
        logger.warning(
            "aiohttp file-descriptor capacity may be exhausted: %s. Reduce intended concurrency or raise the "
            "file-descriptor soft limit while accounting for non-HTTP descriptors.",
            "; ".join(file_descriptor_warnings),
        )


def _connector_queue_constraint(connector: Any) -> str:
    """Classify the binding connector limit when aiohttp reports a queue wait."""
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


class QueueTimedTCPConnector(TCPConnector):
    """TCPConnector that measures how long connections wait for a free pool slot.

    Every `connect()` call increments `connect_total` for the current destination.
    Each queued acquisition records one `queue_duration_ms` sample.
    The sample is labelled with the limit that was binding when the wait started.
    The wait is timed by overriding aiohttp's private `_wait_for_available_connection`.
    `test_queue_wait_override_matches_aiohttp_signature` fails if aiohttp changes that method.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        register_http_connection_pool_connect_counter(_connect_count_snapshot)

    async def connect(self, req: ClientRequest, *args: Any, **kwargs: Any) -> Connection:
        server_name = _SERVER_NAME.get()
        with _CONNECT_COUNTS_LOCK:
            _CONNECT_COUNTS[server_name] = _CONNECT_COUNTS.get(server_name, 0) + 1
        return await super().connect(req, *args, **kwargs)

    async def _wait_for_available_connection(self, key: ConnectionKey, traces: list[Trace]) -> None:
        queue_constraint = _connector_queue_constraint(self)
        started_at = time.perf_counter()
        outcome = "abandoned"
        try:
            await super()._wait_for_available_connection(key, traces)
            outcome = "ok"
        finally:
            try:
                record_http_connection_pool_queue_duration(
                    (time.perf_counter() - started_at) * 1000.0,
                    queue_constraint=queue_constraint,
                    queue_outcome=outcome,
                    server_name=_SERVER_NAME.get(),
                )
            except Exception:
                # Diagnostics must never alter request, timeout, or cancellation behavior.
                logger.debug("Failed to record aiohttp connection-queue telemetry", exc_info=True)


def build_connection_pool_connector(**kwargs: Any) -> TCPConnector:
    """Build the timed connector only while this process exports metrics."""
    connector_cls = QueueTimedTCPConnector if is_metrics_exporter_active() else TCPConnector
    return connector_cls(**kwargs)
