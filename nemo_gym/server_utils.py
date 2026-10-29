# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ast
import asyncio
import atexit
import json
import logging
import re
import resource
import socket
import sys
import time
import warnings
from abc import abstractmethod
from asyncio.exceptions import CancelledError
from contextlib import asynccontextmanager
from functools import partial
from importlib import import_module
from importlib.util import find_spec
from ipaddress import ip_network
from os import environ, getenv
from pathlib import Path
from threading import Thread
from traceback import format_exc, print_exc
from typing import Any, ClassVar, List, Literal, NamedTuple, Optional, TextIO, Tuple, Type, Union, Unpack
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import orjson
import requests
import uvicorn
from aiohttp import (
    ClientConnectorError,
    ClientOSError,
    ClientResponse,
    ClientResponseError,
    ClientSession,
    ClientTimeout,
    DummyCookieJar,
    ServerDisconnectedError,
)
from aiohttp.client import _RequestOptions
from anyio import create_task_group
from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from multidict import CIMultiDict
from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, PrivateAttr, model_validator
from requests.exceptions import ConnectionError
from starlette.middleware.sessions import SessionMiddleware
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol

from nemo_gym import WORKING_DIR
from nemo_gym._checkpoint.settings import checkpoint_settings
from nemo_gym.config_types import (
    DEFAULT_UVICORN_BACKLOG,
    ROLLOUT_PATH_PREFIX,
    TOKEN_CAPTURE_PATH_SEGMENT,
    TRACE_CONTEXT_PATH_SEGMENT,
    BaseRunServerInstanceConfig,
    BaseServerConfig,
    HeadServerUnreachableError,
)
from nemo_gym.global_config import (
    DRY_RUN_KEY_NAME,
    HEAD_SERVER_KEY_NAME,
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    OBSERVABILITY_ENABLED_KEY_NAME,
    RAY_HEAD_NODE_ADDRESS_KEY_NAME,
    UVICORN_TIMEOUT_WORKER_HEALTHCHECK,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
    get_global_config_dict,
    maybe_get_global_config_dict,
)
from nemo_gym.profiling import Profiler
from nemo_gym.rollout_correlation import (
    ROUTED_PATH_STATE_KEY,
    VERIFIER_INVOCATION_HEADER,
    current_rollout_id,
    current_verifier_invocation_id,
    maybe_rollout_id_from_run_body,
    rollout_context,
    scope_with_routed_path,
)
from nemo_gym.session_routing import (
    SESSION_CLAIM_SCOPE_KEY,
    SESSION_ID_SCOPE_KEY,
    SESSION_SOCKET_DIR_ENV,
    create_session_socket_dir,
    install_session_routing,
)
from nemo_gym.telemetry._fallbacks import is_span_group_enabled, safe_set_span_attributes
from nemo_gym.telemetry.connection_pool import (
    QueueTimedTCPConnector,
    build_connection_pool_connector,
    connection_pool_capacity,
    report_connection_pool_capacity,
    reset_server_name,
    set_server_name,
)
from nemo_gym.telemetry.observer import observe_app_event_loop, request_counts
from nemo_gym.telemetry.setup import is_trace_exporter_active
from nemo_gym.telemetry.span_groups import GymSpanGroup
from nemo_gym.telemetry.startup import ServerStartupTimeline


try:
    from opentelemetry import context as _otel_context
except ImportError:  # the telemetry extra is optional; without the SDK there is no span context to isolate
    _otel_context = None


logger = logging.getLogger(__name__)

_GLOBAL_AIOHTTP_CLIENT: Union[None, ClientSession] = None
_GLOBAL_AIOHTTP_CLIENT_LOOP: Optional[asyncio.AbstractEventLoop] = None
_GLOBAL_AIOHTTP_CLIENT_CONFIG: Optional["GlobalAIOHTTPAsyncClientConfig"] = None
_GLOBAL_AIOHTTP_CONTROL_CLIENT: Union[None, ClientSession] = None
_GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG: bool = False
_GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY: bool = False
_UPSTREAM_ERROR_LOG_BODY_CHARS = 2000
# Bound both the raw request prefix and its escaped representation to 4 KiB.
_VALIDATION_ERROR_LOG_BODY_CHARS = 4096
_VALIDATION_ERROR_LOG_MAX_ERRORS = 20
_VALIDATION_ERROR_LOG_FIELD_CHARS = 256
_VALIDATION_ERROR_LOG_LOC_ITEMS = 8
# Bound the rendered error summaries in the log message to 4 KiB.
_VALIDATION_ERROR_LOG_ERRORS_CHARS = 4096


def _escaped_log_text(value: str, max_chars: int) -> tuple[str, bool]:
    """Return bounded, control-safe text with a visible marker when truncated."""
    raw_prefix = value[:max_chars]
    rendered = json.dumps(raw_prefix, ensure_ascii=True)[1:-1]
    truncated = len(value) > len(raw_prefix) or len(rendered) > max_chars
    suffix = "...[truncated]" if truncated else ""
    # The marker shares the existing budget; preserve complete JSON escapes.
    content_limit = max_chars - len(suffix)
    safe_end = 0
    index = 0
    while index < len(rendered) and index < content_limit:
        escape_chars = 1
        if rendered[index] == "\\":
            escape_chars = 6 if index + 1 < len(rendered) and rendered[index + 1] == "u" else 2
        if index + escape_chars > content_limit:
            break
        index += escape_chars
        safe_end = index

    return rendered[:safe_end] + suffix, truncated


def _escaped_log_prefix(value: str, max_chars: int) -> tuple[str, bool]:
    """Return a quoted, control-safe prefix bounded by ``max_chars``."""
    escaped, truncated = _escaped_log_text(value, max_chars - 2)
    return f'"{escaped}"', truncated


def _bounded_validation_error_value(value: Any) -> tuple[Any, bool]:
    if isinstance(value, str):
        return _escaped_log_text(value, _VALIDATION_ERROR_LOG_FIELD_CHARS)
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    return f"<{type(value).__name__}>", True


def _validation_error_summaries(errors: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    summaries = []
    truncated = len(errors) > _VALIDATION_ERROR_LOG_MAX_ERRORS
    for error in errors[:_VALIDATION_ERROR_LOG_MAX_ERRORS]:
        error_type, type_truncated = _bounded_validation_error_value(error.get("type"))
        message, message_truncated = _bounded_validation_error_value(error.get("msg"))
        location = error.get("loc")
        if isinstance(location, (list, tuple)):
            location_items = []
            location_truncated = len(location) > _VALIDATION_ERROR_LOG_LOC_ITEMS
            for item in location[:_VALIDATION_ERROR_LOG_LOC_ITEMS]:
                bounded_item, item_truncated = _bounded_validation_error_value(item)
                location_items.append(bounded_item)
                location_truncated = location_truncated or item_truncated
        else:
            bounded_location, location_truncated = _bounded_validation_error_value(location)
            location_items = [bounded_location]

        summaries.append({"type": error_type, "loc": location_items, "msg": message})
        truncated = truncated or type_truncated or location_truncated or message_truncated

    return summaries, truncated


def _validation_errors_log_text(error_summaries: list[dict[str, Any]]) -> tuple[str, bool]:
    """Render the error summaries for the log message, bounded by ``_VALIDATION_ERROR_LOG_ERRORS_CHARS``."""
    # Custom validator messages can echo credentials from headers, queries, paths, or cookies.
    # Keep their type and location visible without copying their messages into the default log.
    message_summaries = [
        summary if summary["loc"] and summary["loc"][0] == "body" else {key: summary[key] for key in ("type", "loc")}
        for summary in error_summaries
    ]
    rendered = json.dumps(message_summaries, ensure_ascii=True, separators=(",", ":"))
    if len(rendered) <= _VALIDATION_ERROR_LOG_ERRORS_CHARS:
        return rendered, False
    suffix = "...[truncated]"
    return rendered[: _VALIDATION_ERROR_LOG_ERRORS_CHARS - len(suffix)] + suffix, True


async def _log_validation_exception(request: Request, exc: RequestValidationError) -> None:
    errors = exc.errors()
    error_summaries, errors_truncated = _validation_error_summaries(errors)
    # The summaries are also attached as ``extra``, but the default formatter drops extras,
    # so render them into the message too: a 422 log line should say which field failed.
    errors_text, errors_text_truncated = _validation_errors_log_text(error_summaries)
    errors_log_truncated = errors_truncated or errors_text_truncated
    errors_suffix = " ...[truncated]" if errors_log_truncated else ""
    extra = {
        "validation_error_count": len(errors),
        "validation_errors": error_summaries,
        "validation_errors_truncated": errors_truncated,
    }

    has_body_error = any(
        isinstance(error.get("loc"), (list, tuple)) and error["loc"] and error["loc"][0] == "body" for error in errors
    )
    if not has_body_error:
        logger.warning(
            "Request validation failed; validation_error_count=%d validation_errors=%s "
            "validation_errors_truncated=%s%s",
            len(errors),
            errors_text,
            errors_log_truncated,
            errors_suffix,
            extra=extra,
        )
        return

    try:
        body = await request.body()
    except Exception:
        logger.warning(
            "Request validation failed; request body unavailable; "
            "validation_error_count=%d validation_errors=%s validation_errors_truncated=%s%s",
            len(errors),
            errors_text,
            errors_log_truncated,
            errors_suffix,
            extra=extra,
        )
        return

    raw_prefix = body[:_VALIDATION_ERROR_LOG_BODY_CHARS]
    escaped_prefix, prefix_truncated = _escaped_log_prefix(
        raw_prefix.decode("utf-8", errors="replace"), _VALIDATION_ERROR_LOG_BODY_CHARS
    )
    body_truncated = len(body) > len(raw_prefix) or prefix_truncated
    extra.update(
        {
            "request_body_size_bytes": len(body),
            "request_body_prefix": escaped_prefix,
            "request_body_truncated": body_truncated,
        }
    )
    logger.warning(
        "Request validation failed; request_body_size_bytes=%d request_body_truncated=%s request_body_prefix=%s "
        "validation_error_count=%d validation_errors=%s validation_errors_truncated=%s%s",
        len(body),
        body_truncated,
        escaped_prefix,
        len(errors),
        errors_text,
        errors_log_truncated,
        errors_suffix,
        extra=extra,
    )


async def _validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
    try:
        await _log_validation_exception(request, exc)
    except Exception:
        # Diagnostics must not alter FastAPI's response contract.
        pass
    return await request_validation_exception_handler(request, exc)


NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME = "NEMO_GYM_MODEL_SERVER_NAME"
NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME = "NEMO_GYM_MODEL_SERVER_BASE_URL"


class _PickleSafeRequestInfo(NamedTuple):
    url: str
    method: str
    headers: CIMultiDict[str]
    real_url: str


class GlobalAIOHTTPAsyncClientConfig(BaseModel):
    global_aiohttp_connector_limit: int = Field(
        default=100 * 1024,
        ge=0,
        description="Per-server connection budget divided across FastAPI workers; 0 is unlimited.",
    )
    global_aiohttp_connector_limit_per_host: int = Field(
        default=1024,
        ge=0,
        description="Per-server, per-host connection budget divided across FastAPI workers; 0 is unlimited.",
    )
    global_aiohttp_intended_concurrency: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional per-server expected concurrent HTTP requests used for capacity warnings.",
    )
    global_aiohttp_intended_concurrency_per_host: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional per-server expected concurrent HTTP requests to one host used for capacity warnings.",
    )
    # Control calls (checkpoint coordination, generation cuts) use their own pool,
    # so long-running data calls that hold every data connection to a host cannot starve them.
    global_aiohttp_control_connector_limit: int = Field(default=256, ge=0)
    global_aiohttp_control_connector_limit_per_host: int = Field(default=64, ge=0)

    global_aiohttp_client_request_debug: bool = False

    # Bounds match the Linux kernel limits; values outside them make setsockopt fail with EINVAL.
    global_aiohttp_tcp_keepalive_idle_seconds: int = Field(
        default=60,
        ge=1,
        le=32767,
        description=(
            "TCP_KEEPIDLE: seconds a socket must be idle before the kernel starts sending keepalive probes. "
            "Applies to outgoing aiohttp connections and to connections accepted by Gym servers."
        ),
    )
    global_aiohttp_tcp_keepalive_interval_seconds: int = Field(
        default=10,
        ge=1,
        le=32767,
        description=(
            "TCP_KEEPINTVL: seconds between successive keepalive probes. "
            "Applies to outgoing aiohttp connections and to connections accepted by Gym servers."
        ),
    )
    global_aiohttp_tcp_keepalive_probes: int = Field(
        default=3,
        ge=1,
        le=127,
        description=(
            "TCP_KEEPCNT: number of unanswered probes before the kernel drops the connection. "
            "Applies to outgoing aiohttp connections and to connections accepted by Gym servers."
        ),
    )


def get_global_aiohttp_client(
    global_config_dict_parser_config: Optional[GlobalConfigDictParserConfig] = None,
    global_config_dict_parser_cls: Type[GlobalConfigDictParser] = GlobalConfigDictParser,
) -> ClientSession:
    global _GLOBAL_AIOHTTP_CLIENT

    if _GLOBAL_AIOHTTP_CLIENT is not None:
        return _GLOBAL_AIOHTTP_CLIENT

    # Read only a config this process already has, cached or injected through NEMO_GYM_CONFIG_DICT; never parse
    # its argv here, which in a Ray task worker exits the process. A process with neither gets the default client
    # limits: env.yaml and +global_aiohttp_* do not reach it. An explicit parser config or class still parses.
    global_config_dict = maybe_get_global_config_dict()
    if global_config_dict is None and (
        global_config_dict_parser_config is not None or global_config_dict_parser_cls is not GlobalConfigDictParser
    ):
        global_config_dict = get_global_config_dict(
            global_config_dict_parser_config=global_config_dict_parser_config,
            global_config_dict_parser_cls=global_config_dict_parser_cls,
        )
    if global_config_dict is None:
        cfg = GlobalAIOHTTPAsyncClientConfig()
    else:
        cfg = GlobalAIOHTTPAsyncClientConfig.model_validate(global_config_dict)

    return set_global_aiohttp_client(cfg)


def _set_tcp_keepalive(sock: socket.socket, idle_seconds: int, interval_seconds: int, probes: int) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for opt, opt_value in (
        # macOS has no TCP_KEEPIDLE; CPython exposes its equivalent as TCP_KEEPALIVE.
        (getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None)), idle_seconds),
        (getattr(socket, "TCP_KEEPINTVL", None), interval_seconds),
        (getattr(socket, "TCP_KEEPCNT", None), probes),
    ):
        if opt is not None:
            sock.setsockopt(socket.IPPROTO_TCP, opt, opt_value)


def _make_keepalive_socket_factory(
    idle_seconds: int,
    interval_seconds: int,
    probes: int,
):
    def factory(addr_info) -> socket.socket:
        family, type_, proto, _canonname, _sockaddr = addr_info
        sock = socket.socket(family=family, type=type_, proto=proto)
        _set_tcp_keepalive(sock, idle_seconds, interval_seconds, probes)
        return sock

    return factory


class KeepaliveHttpToolsProtocol(HttpToolsProtocol):
    """Uvicorn's httptools protocol with TCP keepalive on every accepted connection.

    A model server answers only after generation finishes, so a client connection can carry no bytes for many
    minutes. Stateful network hops on some paths evict flows that stay idle that long, which silently drops the
    eventual reply and leaves the client waiting on a half-open connection. Keepalive probes keep the flow alive and
    let the kernel reap peers that are really gone. Uses the same ``global_aiohttp_tcp_keepalive_*`` settings as the
    outgoing client connections.

    Handed to uvicorn as ``partial(KeepaliveHttpToolsProtocol, keepalive=(idle, interval, probes))``: multi-worker
    uvicorn pickles its config into worker processes, so this class must stay importable at module level.
    """

    def __init__(self, *args: Any, keepalive: Tuple[int, int, int], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._keepalive = keepalive

    def connection_made(self, transport: asyncio.Transport) -> None:
        sock = transport.get_extra_info("socket")
        if sock is not None and sock.family in (socket.AF_INET, socket.AF_INET6):
            _set_tcp_keepalive(sock, *self._keepalive)
        super().connection_made(transport)


def set_global_aiohttp_client(cfg: GlobalAIOHTTPAsyncClientConfig) -> ClientSession:  # pragma: no cover
    assert not is_global_aiohttp_client_setup(), (
        "There is already a global aiohttp client setup. Please refactor your code or call `global_aiohttp_client_exit` if you want to explicitly re-make the client!"
    )

    num_workers = get_nemo_gym_fastapi_num_workers()
    capacity = connection_pool_capacity(cfg, num_workers)
    if not is_nemo_gym_fastapi_worker():
        report_connection_pool_capacity(cfg, capacity)
    connector = build_connection_pool_connector(
        limit=capacity.total,
        limit_per_host=capacity.per_host,
        keepalive_timeout=15.0,
        socket_factory=_make_keepalive_socket_factory(
            idle_seconds=cfg.global_aiohttp_tcp_keepalive_idle_seconds,
            interval_seconds=cfg.global_aiohttp_tcp_keepalive_interval_seconds,
            probes=cfg.global_aiohttp_tcp_keepalive_probes,
        ),
    )
    client_session = ClientSession(
        connector=connector,
        timeout=ClientTimeout(),
        cookie_jar=DummyCookieJar(),
    )

    global _GLOBAL_AIOHTTP_CLIENT, _GLOBAL_AIOHTTP_CLIENT_CONFIG, _GLOBAL_AIOHTTP_CLIENT_LOOP
    _GLOBAL_AIOHTTP_CLIENT = client_session
    _GLOBAL_AIOHTTP_CLIENT_CONFIG = cfg
    _GLOBAL_AIOHTTP_CLIENT_LOOP = asyncio.get_running_loop()

    global _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG
    _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG = cfg.global_aiohttp_client_request_debug

    global _GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY
    _GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY = isinstance(connector, QueueTimedTCPConnector)

    return _GLOBAL_AIOHTTP_CLIENT


def get_global_aiohttp_control_client() -> ClientSession:  # pragma: no cover
    """The reserved pool for control calls, created on first use next to the data client."""
    global _GLOBAL_AIOHTTP_CONTROL_CLIENT
    if _GLOBAL_AIOHTTP_CONTROL_CLIENT is None or _GLOBAL_AIOHTTP_CONTROL_CLIENT.closed:
        get_global_aiohttp_client()
        cfg = _GLOBAL_AIOHTTP_CLIENT_CONFIG or GlobalAIOHTTPAsyncClientConfig()
        _GLOBAL_AIOHTTP_CONTROL_CLIENT = ClientSession(
            connector=build_connection_pool_connector(
                limit=cfg.global_aiohttp_control_connector_limit,
                limit_per_host=cfg.global_aiohttp_control_connector_limit_per_host,
                keepalive_timeout=15.0,
                socket_factory=_make_keepalive_socket_factory(
                    idle_seconds=cfg.global_aiohttp_tcp_keepalive_idle_seconds,
                    interval_seconds=cfg.global_aiohttp_tcp_keepalive_interval_seconds,
                    probes=cfg.global_aiohttp_tcp_keepalive_probes,
                ),
            ),
            timeout=ClientTimeout(),
            cookie_jar=DummyCookieJar(),
        )
    return _GLOBAL_AIOHTTP_CONTROL_CLIENT


def is_global_aiohttp_client_setup() -> bool:  # pragma: no cover
    return _GLOBAL_AIOHTTP_CLIENT is not None


def is_global_aiohttp_client_request_debug_enabled() -> bool:
    return _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG


def _detach_global_aiohttp_clients() -> list[ClientSession]:
    """Clear the process-wide data and control clients and return the ones still open."""

    global _GLOBAL_AIOHTTP_CLIENT, _GLOBAL_AIOHTTP_CLIENT_LOOP, _GLOBAL_AIOHTTP_CONTROL_CLIENT
    global _GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY
    clients = [
        client
        for client in (_GLOBAL_AIOHTTP_CLIENT, _GLOBAL_AIOHTTP_CONTROL_CLIENT)
        if client is not None and not client.closed
    ]
    _GLOBAL_AIOHTTP_CLIENT = None
    _GLOBAL_AIOHTTP_CLIENT_LOOP = None
    _GLOBAL_AIOHTTP_CONTROL_CLIENT = None
    _GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY = False
    return clients


async def close_global_aiohttp_client() -> None:
    """Close the process-wide data and control clients on their owning event loop."""

    for client in _detach_global_aiohttp_clients():
        await client.close()


def global_aiohttp_client_exit():  # pragma: no cover
    # The control pool is created on demand from requests running on the data client's loop.
    owner_loop = _GLOBAL_AIOHTTP_CLIENT_LOOP
    clients = _detach_global_aiohttp_clients()
    if not clients:
        return

    async def close_client() -> None:
        for client in clients:
            await client.close()

    if owner_loop is None or owner_loop.is_closed():
        # aiohttp has no owner-loop work left once the loop is closed, but the
        # coroutine must still run so the session transitions to closed.
        asyncio.run(close_client())
        return

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is owner_loop:
        owner_loop.create_task(close_client())
    elif owner_loop.is_running():
        asyncio.run_coroutine_threadsafe(close_client(), owner_loop).result(timeout=5)
    else:
        owner_loop.run_until_complete(close_client())


atexit.register(global_aiohttp_client_exit)


# This is not intended to be changed. If you want to increase this, we should probably figure out how to improve server-side robustness.
MAX_NUM_TRIES = 3

_NUM_SERVER_DISCONNECTED_ERROR: int = 0
_NUM_CLIENT_OS_ERROR: int = 0
DISCONNECTED_CLIENT_OS_PRINT_INTERVAL: int = 100
DISCONNECTED_CLIENT_OS_HELP_TEXT = """We've run into this issue in two different scenarios previously:
1. Too many open connections and not enough sockets due to the file descriptor limit being hit.
    - Increase ulimit.
    - Bash example: https://github.com/NVIDIA-NeMo/RL/blob/de55be7777bbf034c04e41c40382c44725e8aa4b/ray.sub#L81
    - Python example: https://github.com/NVIDIA-NeMo/Gym/blob/c74ffddb3d8190cd717508b0830916b19a26e6cd/nemo_gym/server_utils.py#L495
2. Depending on the serving framework and config, the server may be overloaded and is dropping connections.
    - Increase adapter server replicas."""


async def request(
    method: str,
    url: str,
    _internal: bool = False,
    _max_connection_retries: Optional[int] = None,
    _server_name: Optional[str] = None,
    _max_num_tries: Optional[int] = None,
    _control: bool = False,
    _resend_after_disconnect: bool = True,
    _read_body: bool = False,
    **kwargs: Unpack[_RequestOptions],
) -> ClientResponse:  # pragma: no cover
    """Make an outbound HTTP call through Gym's shared aiohttp client.

    This is the only place Gym talks to another server, so it is also the only place
    trace context has to be injected: every agent -> model and agent -> resources hop goes
    through here. `CLAUDE.md` bans httpx precisely to keep it that way.

    ``_server_name`` is the bounded logical destination label for pool metrics: a configured
    ``ServerClient`` server name, ``remote_agent_service`` for the remote agent's external
    service, or ``None`` for the fallback label ``external``. It is retained across retries
    and redirects and is not forwarded to aiohttp.

    ``_max_num_tries`` caps this call's total attempts on every exception path. It replaces the
    default generic-error limit (``MAX_NUM_TRIES`` attempts for external calls, unbounded for
    internal ones). A ``_max_connection_retries`` limit still applies as well.

    ``_resend_after_disconnect=False`` raises a dropped connection instead of sending the request again, because the
    server may already have received it. A connection that was never made is still retried.

    ``_read_body=True`` reads a success reply's body before returning, but only within a ``_max_num_tries`` or
    ``_max_connection_retries`` bound and only when resending is allowed, so a reply cut off mid-body is sent again
    within that bound. Otherwise, and for an error reply, the response is returned unread: the caller's own read gets
    the complete body or raises the original error, so the caller's failure handling (for collection, its rollout
    attempt budget) decides whether the request runs again.
    """
    if _max_num_tries is not None and _max_num_tries < 1:
        raise ValueError("_max_num_tries must be at least 1")

    # Faster JSON dumps than the default aiohttp json
    if kwargs.get("json"):
        kwargs["data"] = orjson.dumps(kwargs.pop("json"))
        kwargs.setdefault("headers", dict())
        kwargs["headers"]["Content-Type"] = "application/json"

    # Gate first: a disabled site costs one frozenset lookup and nothing else. Gym runs at
    # 16k+ concurrency, so this is a hot path (kb/knowledge/conventions/hot-path-overhead.md).
    # The span and its `traceparent` are the only outputs, so they are skipped while traces are off.
    traced = is_span_group_enabled(GymSpanGroup.HTTP_CLIENT) and is_trace_exporter_active()
    send = _traced_request if traced else _request_with_retries
    counts = request_counts()
    if counts is None:
        return await send(
            method,
            url,
            _internal=_internal,
            _max_num_tries=_max_num_tries,
            _max_connection_retries=_max_connection_retries,
            _server_name=_server_name,
            _control=_control,
            _resend_after_disconnect=_resend_after_disconnect,
            _read_body=_read_body,
            **kwargs,
        )
    # With metrics exporting, count calls until this helper returns or raises, including eligible body reads.
    counts.outbound_in_flight += 1
    try:
        return await send(
            method,
            url,
            _internal=_internal,
            _max_num_tries=_max_num_tries,
            _max_connection_retries=_max_connection_retries,
            _server_name=_server_name,
            _control=_control,
            _resend_after_disconnect=_resend_after_disconnect,
            _read_body=_read_body,
            **kwargs,
        )
    finally:
        counts.outbound_in_flight -= 1


async def _traced_request(
    method: str,
    url: str,
    _internal: bool = False,
    _max_connection_retries: Optional[int] = None,
    _server_name: Optional[str] = None,
    _max_num_tries: Optional[int] = None,
    _control: bool = False,
    _resend_after_disconnect: bool = True,
    _read_body: bool = False,
    **kwargs: Unpack[_RequestOptions],
) -> ClientResponse:  # pragma: no cover
    """`_request_with_retries` wrapped in a CLIENT span, with `traceparent` injected.

    Uses `nemo_gym.telemetry.spans.client_span` rather than nemo-lens's `managed_span`
    because the latter cannot set `SpanKind` — see that module for why an INTERNAL span
    is wrong on a cross-service hop.

    Injection happens inside the span so the header carries *this* span as the parent —
    the receiving server's FastAPI instrumentation extracts it and its SERVER span becomes
    our child. That edge is what makes one rollout a single trace across three processes.

    Retries reuse the same span rather than starting one per attempt: the caller asked for
    one logical request, and the retry count is recorded as an attribute instead.
    """
    from nemo_gym.telemetry.contrib import inject_trace_context
    from nemo_gym.telemetry.spans import client_span

    with client_span(f"HTTP {method.upper()}") as span:
        headers = kwargs.get("headers")
        # aiohttp accepts several mapping types here; normalise to a dict we can inject into
        # without mutating a caller's object.
        headers = dict(headers) if headers else {}
        inject_trace_context(headers)
        kwargs["headers"] = headers

        if span is not None:
            attributes = {
                "http.request.method": method.upper(),
                "url.full": _redacted_url(url),
                "nemo.gym.rollout.id": current_rollout_id(),
            }
            safe_set_span_attributes(span, attributes)

        response = await _request_with_retries(
            method,
            url,
            _internal=_internal,
            _max_num_tries=_max_num_tries,
            _max_connection_retries=_max_connection_retries,
            _server_name=_server_name,
            _control=_control,
            _resend_after_disconnect=_resend_after_disconnect,
            _read_body=_read_body,
            **kwargs,
        )

        if span is not None:
            safe_set_span_attributes(span, {"http.response.status_code": response.status})
        return response


def _redacted_url(url: str) -> str:
    """Strip the query string from a URL before it becomes a span attribute.

    Query strings in Gym carry API keys and, on rollout-prefixed routes, task content.
    The path is the useful part for a trace; the query is a leak waiting to happen.
    """
    return url.split("?", 1)[0]


def _format_upstream_error_body(content: Any) -> str:
    """Render a bounded upstream error body without hiding its useful traceback."""
    if isinstance(content, bytes):
        prefix = content[: _UPSTREAM_ERROR_LOG_BODY_CHARS * 4].decode("utf-8", errors="replace")
        truncated = len(content) > _UPSTREAM_ERROR_LOG_BODY_CHARS * 4
    else:
        prefix = str(content)
        truncated = False

    truncated = truncated or len(prefix) > _UPSTREAM_ERROR_LOG_BODY_CHARS
    return prefix[:_UPSTREAM_ERROR_LOG_BODY_CHARS] + ("…" if truncated else "")


def format_exception_with_response_body(error: BaseException) -> str:
    """Format an exception, appending its bounded HTTP response body when available.

    ``str(ClientResponseError)`` names only the status, reason phrase, and URL. The upstream server's own reason,
    such as an agent's ``AgentSeedRejectedError`` detail, is in the body that ``raise_for_status`` keeps on
    ``response_content``.
    """
    content = getattr(error, "response_content", None) if isinstance(error, ClientResponseError) else None
    if not content:
        return str(error)
    return f"{error}: {_format_upstream_error_body(content)}"


def _format_upstream_error_log(server_name: str, error: ClientResponseError) -> str:
    request_info = error.request_info
    upstream_url = getattr(request_info, "real_url", None)
    if upstream_url is None:
        upstream_url = getattr(request_info, "url", None)
    upstream_url = _redacted_url(str(upstream_url)) if upstream_url is not None else "unknown"
    return (
        "🚨 [upstream_request_failed] "
        f"server={server_name} method={getattr(request_info, 'method', 'unknown')} "
        f"url={upstream_url} status={error.status}\n"
        f"Response content: {_format_upstream_error_body(error.response_content)}"
    )


async def _request_with_retries(
    method: str,
    url: str,
    _internal: bool = False,
    _max_connection_retries: Optional[int] = None,
    _server_name: Optional[str] = None,
    _max_num_tries: Optional[int] = None,
    _control: bool = False,
    _resend_after_disconnect: bool = True,
    _read_body: bool = False,
    **kwargs: Unpack[_RequestOptions],
) -> ClientResponse:  # pragma: no cover
    client = get_global_aiohttp_control_client() if _control else get_global_aiohttp_client()
    # Initialization stays inside the client span and sets the metrics flag before it is read.
    token = set_server_name(_server_name or "external") if _GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY else None
    try:
        num_tries = 1
        explicit_tries = 0
        retries = 0
        retry_start = time.monotonic()
        while True:
            if _max_num_tries is not None:
                explicit_tries += 1
            try:
                response = await client.request(method=method, url=url, **kwargs)
                if (
                    _read_body
                    and response.ok
                    and _resend_after_disconnect
                    and (_max_num_tries is not None or _max_connection_retries is not None)
                ):
                    # A success body cut off mid-read is retried below within the caller's bound. Without one, each
                    # resend would start another episode, and an error reply is the caller's to read in full: such a
                    # reply goes back unread, so the caller's own read raises the original error, with its status.
                    await response.read()
                return response
            except ServerDisconnectedError:
                if not _resend_after_disconnect:
                    raise
                global _NUM_SERVER_DISCONNECTED_ERROR
                _NUM_SERVER_DISCONNECTED_ERROR += 1
                retries += 1
                if _max_num_tries is not None and explicit_tries >= _max_num_tries:
                    raise
                if _NUM_SERVER_DISCONNECTED_ERROR % DISCONNECTED_CLIENT_OS_PRINT_INTERVAL == 0:
                    print(
                        f"[request_retry url={url} error=ServerDisconnectedError retry={retries} elapsed_s={time.monotonic() - retry_start:.1f}] "
                        f"Hit {_NUM_SERVER_DISCONNECTED_ERROR} global `ServerDisconnectedError` while querying {url}.\n{DISCONNECTED_CLIENT_OS_HELP_TEXT}",
                        flush=True,
                    )

                # Retrying forever is wrong if the endpoint is expected to sometimes die and move.
                if _max_connection_retries is not None and retries >= _max_connection_retries:
                    raise

                await asyncio.sleep(0.5)
            except ClientOSError as error:
                if not _resend_after_disconnect and not isinstance(error, ClientConnectorError):
                    raise
                global _NUM_CLIENT_OS_ERROR
                _NUM_CLIENT_OS_ERROR += 1
                retries += 1
                if _max_num_tries is not None and explicit_tries >= _max_num_tries:
                    raise
                if _NUM_CLIENT_OS_ERROR % DISCONNECTED_CLIENT_OS_PRINT_INTERVAL == 0:
                    print(
                        f"[request_retry url={url} error=ClientOSError retry={retries} elapsed_s={time.monotonic() - retry_start:.1f}] "
                        f"Hit {_NUM_CLIENT_OS_ERROR} global `ClientOSError` while querying {url}.\n{DISCONNECTED_CLIENT_OS_HELP_TEXT}",
                        flush=True,
                    )

                if _max_connection_retries is not None and retries >= _max_connection_retries:
                    raise

                await asyncio.sleep(0.5)
            except Exception as e:
                if _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG:
                    print_exc()
                if not _resend_after_disconnect:
                    # A connection that was never made is a ClientOSError, handled above;
                    # anything else may have sent it.
                    raise

                # num_tries only advances on the default path, so count explicit attempts when capped.
                attempts = explicit_tries if _max_num_tries is not None else num_tries
                if _max_connection_retries is not None and attempts >= _max_connection_retries:
                    raise

                if _max_num_tries is not None:
                    if explicit_tries >= _max_num_tries:
                        raise
                else:
                    # Internal calls skip this print and the MAX_NUM_TRIES limit, since the head server shuts
                    # everything down if a Gym server fails. They still advance `num_tries`, so the
                    # `_max_connection_retries` check above stops them. Connection errors count in `retries`.
                    if not _internal:
                        print(
                            f"""Hit an exception while making a request (try {num_tries}): {type(e)}: {e}
Sleeping 0.5s and retrying...
"""
                        )
                        if num_tries >= MAX_NUM_TRIES:
                            raise e

                    num_tries += 1

                await asyncio.sleep(0.5)
    finally:
        if token is not None:
            reset_server_name(token)


async def raise_for_status(response: ClientResponse, content: Optional[bytes] = None) -> None:  # pragma: no cover
    if not response.ok:
        if content is None:
            content = await response.content.read()
        if _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG:
            # Not the full `request_info`: its headers carry `Authorization: Bearer <api key>`.
            request_info = response.request_info
            print(f"""Request info: {request_info.method} {_redacted_url(str(request_info.real_url))}
Response content: {content}""")

        try:
            response.raise_for_status()
        except ClientResponseError as e:
            # Set the response content here so we have access to it down the line.
            e.response_content = content
            # aiohttp stores unpicklable multidict proxies in this exception.
            # Preserve request details as pickle-safe values for cross-process propagation.
            request_info = e.request_info
            e.request_info = _PickleSafeRequestInfo(
                url=str(request_info.url),
                method=request_info.method,
                headers=CIMultiDict(request_info.headers),
                real_url=str(request_info.real_url),
            )
            e.history = ()
            e.headers = CIMultiDict(e.headers) if e.headers is not None else None
            e.args = (e.request_info, e.history)
            raise e


async def get_response_json(response: ClientResponse) -> Any:
    return orjson.loads(await response.read())


DEFAULT_HEAD_SERVER_PORT = 11000

ServerStatus = Union[Literal["success"], Literal["connection_error"], Literal["timeout"], Literal["unknown_error"]]


def _connectable_client_host(host: str) -> str:
    """Return a connectable client host for a server bind address."""

    if host == "0.0.0.0":
        return "127.0.0.1"
    if host in {"::", "[::]"}:
        return "[::1]"
    return host


class ServerClient(BaseModel):
    head_server_config: BaseServerConfig

    global_config_dict: DictConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Resolved base URLs, cached by server name.
    _server_base_urls: dict[str, str] = PrivateAttr(default_factory=dict)
    _checkpoint_enabled: Optional[bool] = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        # Validate the ``checkpoint:`` block once, when the client is built at startup, rather than on every request.
        self.checkpoint_enabled()

    def checkpoint_enabled(self) -> bool:
        """Return whether the global ``checkpoint:`` block enables partial-rollout checkpointing."""
        if self._checkpoint_enabled is None:
            self._checkpoint_enabled = checkpoint_settings(self.global_config_dict) is not None
        return self._checkpoint_enabled

    @staticmethod
    def _client_host(host: str) -> str:
        """Return a connectable host for a server bind address.

        Wildcard addresses are valid for binding a server but are not valid
        advertised client targets. In particular, proxy-aware HTTP clients can
        route ``0.0.0.0`` through an external proxy instead of reaching the
        local server.
        """
        return _connectable_client_host(host)

    def assistant_message_header(self, model_server_name: str) -> bytes | None:
        """Read the optional header property of harnesses using this model server.

        The property lives on the harness package so model workers need not import
        an agent's app or install its runtime dependencies.
        """
        headers = set()
        for instance in self.global_config_dict.values():
            if not isinstance(instance, (dict, DictConfig)):
                continue
            for harness, config in instance.get("responses_api_agents", {}).items():
                model = config.get("model_server") or {}
                if model.get("name") != model_server_name or model.get("type") != "responses_api_models":
                    continue
                package = import_module(f"responses_api_agents.{harness}")
                header = getattr(package, "_assistant_message_header", None)
                if header is not None:
                    headers.add(header.lower())
        if len(headers) > 1:
            raise ValueError(f"Harnesses using model server {model_server_name!r} declare different assistant headers")
        return next(iter(headers), None)

    @classmethod
    def load_head_server_config(cls) -> BaseServerConfig:
        global_config_dict = get_global_config_dict()
        server_config_dict = global_config_dict[HEAD_SERVER_KEY_NAME]
        config = BaseServerConfig.model_validate(server_config_dict)
        return config

    @classmethod
    def load_from_global_config(cls, head_server_config: Optional[BaseServerConfig] = None) -> "ServerClient":
        """Build a client from the fully resolved global config.

        Gym-launched server processes reuse the config injected by their parent.
        Other processes fetch the config from the head server.
        """
        if head_server_config is None:
            head_server_config = cls.load_head_server_config()

        if _has_injected_global_config_env():
            return cls(
                head_server_config=head_server_config,
                global_config_dict=get_global_config_dict(),
            )

        # It's critical we use requests here instead of the global httpx client since a FastAPI server may be run downstream of this function call.
        head_server_host = _connectable_client_host(head_server_config.host)
        head_server_url = f"http://{head_server_host}:{head_server_config.port}"
        try:
            response = requests.get(
                f"{head_server_url}/global_config_dict_yaml",
            )
        except ConnectionError as e:
            # requests' ConnectionError also covers proxy and name-resolution failures; keep the real reason
            # (with its traceback) for --verbose, since the ConfigError below is printed without its cause.
            logger.debug(
                "Could not fetch the global config from the head server at %s", head_server_url, exc_info=True
            )
            # A ConfigError so the CLI prints just this message (no traceback); the cause stays chained.
            raise HeadServerUnreachableError(
                f"Could not connect to the head server at {head_server_url}. Is the head server running? "
                "Start it with: `gym env start`. If it is already running on a different host or port, pass "
                "`++head_server.host=<host>` / `++head_server.port=<port>` so this command can find it."
            ) from e

        global_config_dict_yaml = response.content.decode()
        global_config_dict = OmegaConf.create(json.loads(global_config_dict_yaml))

        return cls(head_server_config=head_server_config, global_config_dict=global_config_dict)

    async def request(
        self,
        server_name: str,
        url_path: str,
        method: str,
        *,
        _control: bool = False,
        _resend_after_disconnect: bool = True,
        **kwargs: Unpack[_RequestOptions],
    ) -> ClientResponse:
        """Call another server. ``_control`` sends the call over the reserved control pool."""
        model_server_name = getenv(NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME)
        model_server_base_url = getenv(NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME)
        if model_server_base_url and server_name == model_server_name:
            # Subprocess agents do not inherit the current rollout context.
            # The launcher provides a model URL that already contains the rollout prefix.
            # Use that URL instead of rebuilding it from global server configuration.
            base_url = model_server_base_url.rstrip("/")
        else:
            base_url = self._resolve_base_url(server_name)

        json_obj = kwargs.get("json")
        if "json" in kwargs:
            if isinstance(json_obj, BaseModel):
                json_obj = json_obj.model_dump(exclude_unset=True)
                kwargs["json"] = json_obj

        # Checkpointing needs every model and resources call attributed to its rollout attempt,
        # so it carries the rollout prefix even when observability is off.
        attribute_rollout = (
            self.global_config_dict.get(OBSERVABILITY_ENABLED_KEY_NAME, False) or self.checkpoint_enabled()
        )
        server_entry = self.global_config_dict.get(server_name)
        rollout_id = current_rollout_id()
        if attribute_rollout and server_entry is not None and "resources_servers" in server_entry:
            if url_path == "/verify":
                rollout_id = rollout_id or maybe_rollout_id_from_run_body(json_obj)
            if rollout_id is not None and not url_path.startswith(f"/{ROLLOUT_PATH_PREFIX}/"):
                url_path = f"{rollout_path_prefix(rollout_id)}{url_path}"

        if (
            rollout_id is not None
            and attribute_rollout
            and server_entry is not None
            and "responses_api_models" in server_entry
            and url_path.partition("?")[0] in {"/v1/responses", "/v1/chat/completions", "/v1/messages"}
            and not url_path.startswith(f"/{ROLLOUT_PATH_PREFIX}/")
        ):
            url_path = f"{rollout_path_prefix(rollout_id)}{url_path}"
            verifier_invocation_id = current_verifier_invocation_id()
            if verifier_invocation_id is not None:
                kwargs["headers"] = {
                    **(kwargs.get("headers") or {}),
                    VERIFIER_INVOCATION_HEADER: verifier_invocation_id,
                }

        return await request(
            method=method,
            url=f"{base_url}{url_path}",
            _internal=True,
            _server_name=server_name,
            _control=_control,
            # A /run starts an episode attempt, so with checkpointing it is never sent twice.
            # A dropped connection usually means the server's worker died mid-episode.
            # Sending it again would run the attempt a second time,
            # and the first run's sessions on other servers would never be freed: the second run replies,
            # so the controller never retires the attempt.
            _resend_after_disconnect=_resend_after_disconnect
            and not (self.checkpoint_enabled() and url_path == "/run"),
            # Collection reads the whole /run reply, so a reply cut off mid-body counts as a failed attempt.
            _read_body=url_path == "/run",
            **kwargs,
        )

    async def get(
        self,
        server_name: str,
        url_path: str,
        **kwargs: Unpack[_RequestOptions],
    ) -> ClientResponse:
        """
        Args:
            server_name: str
                The name of the server you are trying to call.
            url_path: str
                The URL path in the server you are trying to call e.g. "/v1/responses".

        """
        return await self.request(
            server_name=server_name,
            url_path=url_path,
            method="GET",
            **kwargs,
        )

    async def post(
        self,
        server_name: str,
        url_path: str,
        **kwargs: Unpack[_RequestOptions],
    ) -> ClientResponse:
        """
        Args:
            server_name: str
                The name of the server you are trying to call.
            url_path: str
                The URL path in the server you are trying to call e.g. "/v1/responses".

        """
        return await self.request(
            server_name=server_name,
            url_path=url_path,
            method="POST",
            **kwargs,
        )

    def poll_for_status(self, server_name: str, *, timeout_seconds: float = 5) -> ServerStatus:  # pragma: no cover
        if server_name == HEAD_SERVER_KEY_NAME:
            server_config_dict = self.global_config_dict[HEAD_SERVER_KEY_NAME]
        else:
            server_config_dict = get_first_server_config_dict(self.global_config_dict, server_name)

        try:
            requests.get(self._build_server_base_url(server_config_dict), timeout=timeout_seconds)
            # We don't check the status code since there may not be a route at /
            return "success"
        except requests.exceptions.ConnectionError:
            return "connection_error"
        except requests.exceptions.Timeout:
            return "timeout"
        except Exception:
            return "unknown_error"

    def _resolve_base_url(self, server_name: str) -> str:
        cached = self._server_base_urls.get(server_name)
        if cached is not None:
            return cached
        server_config_dict = get_first_server_config_dict(self.global_config_dict, server_name)
        base_url = self._build_server_base_url(server_config_dict)
        self._server_base_urls[server_name] = base_url
        return base_url

    def _build_server_base_url(self, server_config_dict: OmegaConf) -> str:
        # Use the module helper so this method remains safe when borrowed by a
        # duck-typed or mocked ServerClient.
        host = _connectable_client_host(server_config_dict.host)
        return f"http://{host}:{server_config_dict.port}"


def _has_injected_global_config_env() -> bool:
    return getenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME) is not None


SESSION_ID_KEY = "session_id"


class BaseServer(BaseModel):
    """
    All instances of BaseServer are queryable using ServerClient.
    """

    config: BaseRunServerInstanceConfig

    @classmethod
    def load_config_from_global_config(cls) -> "BaseRunServerInstanceConfig":
        config_path_str = getenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME)
        global_config_dict = get_global_config_dict()
        server_config_dict = get_first_server_config_dict(global_config_dict, config_path_str)

        server_config_cls: Type[BaseRunServerInstanceConfig] = cls.model_fields["config"].annotation
        server_config = server_config_cls.model_validate(
            OmegaConf.to_container(server_config_dict, resolve=True) | {"name": config_path_str}
        )

        return server_config

    def setup_liveness(self, app: FastAPI) -> None:
        @app.get("/readyz", include_in_schema=False)
        @app.get("/livez", include_in_schema=False)
        @app.get("/healthz", include_in_schema=False)
        @app.get("/health", include_in_schema=False)
        @app.get("/", include_in_schema=False)
        async def _liveness():
            return {"status": "ok"}


class ProfilingMiddlewareInputConfig(BaseModel):
    # Relative to the Gym root dir.
    profiling_results_dirpath: Optional[str] = None


class ProfilingMiddlewareConfig(ProfilingMiddlewareInputConfig):
    profiling_enabled: bool = False


class UvicornLoggingConfig(BaseModel):
    # Default to False for regular use cases.
    uvicorn_logging_show_200_ok: bool = False


class UvicornBacklogConfig(BaseModel):
    # Connections the kernel queues for a server before it accepts them.
    # The kernel caps it at net.core.somaxconn.
    uvicorn_backlog: PositiveInt = DEFAULT_UVICORN_BACKLOG


# Every IPv4 address as it appears on a dual-stack socket.
_ALL_V4_MAPPED = ip_network("::ffff:0:0/96")


class UvicornProxyHeadersConfig(BaseModel):
    # Gym servers call each other directly, so proxy headers stay off: uvicorn would otherwise let
    # any caller rewrite its own client host and URL scheme through X-Forwarded-*.
    uvicorn_proxy_headers: bool = False
    # Trusted proxy addresses. Required when uvicorn_proxy_headers is enabled.
    uvicorn_forwarded_allow_ips: Optional[List[str]] = None

    @model_validator(mode="after")
    def _require_trusted_proxy_allowlist(self) -> "UvicornProxyHeadersConfig":
        if not self.uvicorn_proxy_headers:
            return self

        allow_ips = [ip.strip() for ip in (self.uvicorn_forwarded_allow_ips or []) if ip.strip()]
        if not allow_ips:
            raise ValueError("uvicorn_proxy_headers=True requires a non-empty uvicorn_forwarded_allow_ips allowlist.")
        if "*" in allow_ips:
            raise ValueError("uvicorn_forwarded_allow_ips must not be '*': it trusts forwarded headers from any peer.")
        for address in allow_ips:
            try:
                network = ip_network(address)
            except ValueError as exc:
                # Anything uvicorn cannot parse as an address is kept as a literal it will never
                # match against a TCP peer, so the entry would silently trust nothing.
                raise ValueError(
                    f"uvicorn_forwarded_allow_ips entry {address!r} is not a valid IP address or CIDR range: {exc}"
                ) from exc
            # prefixlen 0 covers a whole family; an IPv6 supernet of ::ffff:0:0/96 covers all of
            # IPv4 once peers arrive IPv4-mapped on a dual-stack socket.
            covers_all_v4_mapped = network.version == 6 and network.supernet_of(_ALL_V4_MAPPED)
            if network.prefixlen == 0 or covers_all_v4_mapped:
                raise ValueError(
                    f"uvicorn_forwarded_allow_ips entry {address!r} covers every address: "
                    "it trusts forwarded headers from any peer."
                )

        self.uvicorn_forwarded_allow_ips = allow_ips
        return self


_NEMO_GYM_STARTED_RAY_CLUSTER: bool = False


def ray_is_installed() -> bool:
    """Whether Ray can be imported in this process, checked without importing it."""
    return find_spec("ray") is not None


def _get_ray():
    """Import Ray only for processes configured to use it."""
    try:
        import ray
    except ImportError as e:
        raise ImportError(
            "This process needs Ray, but Ray isn't installed in its environment. Install the `nemo-gym[ray]` extra."
        ) from e

    return ray


def initialize_ray() -> None:
    """
    Initialize ray cluster in a process.
    We store the Ray address in the global config dict so that child processes can connect to it.
    This avoids the need to start a new Ray cluster in each child process.
    Note: This function will modify the global config dict - update `ray_head_node_address`
    """

    ray = _get_ray()
    if ray.is_initialized():
        print("Ray already initialized")
        return

    global_config_dict = get_global_config_dict()
    ray_head_node_address = global_config_dict.get(RAY_HEAD_NODE_ADDRESS_KEY_NAME)
    ray_init_kwargs = dict(ignore_reinit_error=True)

    if ray_head_node_address:
        print(f"Connecting to Ray cluster at specified address: {ray_head_node_address}")
        ray_init_kwargs["address"] = ray_head_node_address
    else:
        print("NeMo Gym is starting a new Ray cluster...")

    ray.init(**ray_init_kwargs)

    if not ray_head_node_address:
        global _NEMO_GYM_STARTED_RAY_CLUSTER
        _NEMO_GYM_STARTED_RAY_CLUSTER = True
        with open_dict(global_config_dict):
            global_config_dict["ray_head_node_address"] = ray.get_runtime_context().gcs_address
        print(f"Started Ray cluster at {global_config_dict['ray_head_node_address']}")


def maybe_ray_cluster_exit():  # pragma: no cover
    global _NEMO_GYM_STARTED_RAY_CLUSTER

    if not _NEMO_GYM_STARTED_RAY_CLUSTER:
        return

    print("Shutting down Ray cluster spun up by NeMo Gym...")
    ray = sys.modules.get("ray")
    if ray is not None:
        ray.shutdown()

    _NEMO_GYM_STARTED_RAY_CLUSTER = False


atexit.register(maybe_ray_cluster_exit)

# These environment variables are the ONLY environment variables that Gym uses. Please do not set these, they are only used here to pass information
# from main proc to child procs under FastAPI/uvicorn parallelism
IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME = "IS_NEMO_GYM_FASTAPI_WORKER"
IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME = "IS_NEMO_GYM_FASTAPI_ENTRYPOINT"
NEMO_GYM_FASTAPI_NUM_WORKERS = "NEMO_GYM_FASTAPI_NUM_WORKERS"


def is_nemo_gym_fastapi_worker() -> bool:  # pragma: no cover
    return getenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME) == "1"


def set_is_nemo_gym_fastapi_worker() -> None:  # pragma: no cover
    environ[IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME] = "1"


def is_nemo_gym_fastapi_entrypoint(file: str) -> bool:  # pragma: no cover
    return is_nemo_gym_fastapi_worker() and file.endswith(getenv(IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME))


def set_is_nemo_gym_fastapi_entrypoint(file: str) -> None:  # pragma: no cover
    environ[IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME] = file


def get_nemo_gym_fastapi_num_workers() -> int:  # pragma: no cover
    return int(getenv(NEMO_GYM_FASTAPI_NUM_WORKERS, "1"))


def set_nemo_gym_fastapi_num_workers(num_workers: int) -> None:  # pragma: no cover
    environ[NEMO_GYM_FASTAPI_NUM_WORKERS] = str(num_workers)


#: Base class name -> the Gym server-type directory it lives in. Matched against the MRO
#: rather than imported, because base_resources_server and friends import *this* module.
_TELEMETRY_SERVER_TYPE_BY_BASE = {
    "SimpleResourcesServer": "resources_servers",
    "SimpleResponsesAPIAgent": "responses_api_agents",
    "SimpleResponsesAPIModel": "responses_api_models",
    "BaseEnvironmentServer": "environment_servers",
}


def _telemetry_server_type(server_cls: Type) -> Optional[str]:
    """Return the Gym server-type name for *server_cls*, or None if it is not one."""
    for base in server_cls.__mro__:
        server_type = _TELEMETRY_SERVER_TYPE_BY_BASE.get(base.__name__)
        if server_type is not None:
            return server_type
    return None


def _cancelled_request_route(scope: Scope) -> str:
    """Return a registered route template, never a path containing request or rollout IDs."""
    route = scope.get("route")
    if route is not None:
        return getattr(route, "path_format", "<unmatched>")

    # Prefix middleware copies the scope, but shares its actual routed path in state.
    # Respect each server's prefix rules and resolve only registered templates.
    path = scope.get("state", {}).get(ROUTED_PATH_STATE_KEY, scope["path"])
    routed_scope = {**scope, "path": path}
    for candidate in getattr(scope.get("app"), "routes", ()):
        match, _ = candidate.matches(routed_scope)
        if match == Match.FULL:
            return getattr(candidate, "path_format", "<unmatched>")
    return "<unmatched>"


class RequestContextIsolationMiddleware:
    """Start every request from a clean context, whatever task uvicorn started it from.

    uvicorn starts a *pipelined* request's task from inside the previous request's task
    (``on_response_complete`` → ``loop.create_task``), so that task inherits the previous
    request's contextvars. The OTel ASGI instrumentation, which runs outermost, has the previous
    request's server span current at that point; for the new request it sees a current span,
    skips ``traceparent`` extraction, and files the request as an INTERNAL child of the previous
    one, under the previous rollout's trace. Pipelining happens whenever a client reuses a
    kept-alive connection before this server has sent the terminating body message, which
    Starlette's ``BaseHTTPMiddleware`` sends late.

    Installed outside the instrumentor (see ``install_request_context_isolation``), this resets
    the OTel context for the request's duration, so the instrumentor extracts the inbound
    ``traceparent``. It also clears the rollout id. Today that is a guard only: Gym's
    ``BaseHTTPMiddleware`` layers run the rollout middleware, ``/run`` and the handlers in a
    child task, so a rollout id set there does not reach the task uvicorn copies; the guard
    keeps that true if those layers are ever replaced by plain ASGI middleware.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        otel_token = _otel_context.attach(_otel_context.Context()) if _otel_context is not None else None
        try:
            with rollout_context(None):
                await self.app(scope, receive, send)
        finally:
            if otel_token is not None:
                _otel_context.detach(otel_token)


def install_request_context_isolation(app: FastAPI) -> None:
    """Wrap the complete middleware stack, instrumentation included, in context isolation.

    `add_middleware` is not enough: the FastAPI instrumentor installs its middleware by
    overriding `build_middleware_stack`, so it always ends up outside anything added that way.
    Wrapping the build the same way, after instrumentation, puts isolation outside the
    instrumentor, which is the only place it works: the instrumentor must see an empty context
    so it extracts `traceparent`, and the handlers must still see the server span the
    instrumentor then makes current. Call it after `instrument_app_for_telemetry`.
    """
    if getattr(app.state, "nemo_gym_request_context_isolation_installed", False):
        return
    app.state.nemo_gym_request_context_isolation_installed = True
    build_inner = app.build_middleware_stack

    def build_middleware_stack() -> ASGIApp:
        return RequestContextIsolationMiddleware(build_inner())

    app.build_middleware_stack = build_middleware_stack  # type: ignore[method-assign]
    app.middleware_stack = None


# A W3C ``traceparent``, version 00, the only form the propagator writes.
_TRACEPARENT_PATTERN = r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}"
_ROLLOUT_PREFIX_PATTERN = rf"/{re.escape(ROLLOUT_PATH_PREFIX)}/[^/]+(?:/{re.escape(TOKEN_CAPTURE_PATH_SEGMENT)})?"
_ROLLOUT_PREFIX_RE = re.compile(rf"^{_ROLLOUT_PREFIX_PATTERN}(?=/|$)")
_TRACE_CONTEXT_PATH_RE = re.compile(
    rf"^(?P<head>(?:{_ROLLOUT_PREFIX_PATTERN})?)"
    rf"/{re.escape(TRACE_CONTEXT_PATH_SEGMENT)}/(?P<traceparent>{_TRACEPARENT_PATTERN})(?P<rest>/.*)$"
)


class TraceContextPathMiddleware:
    """Strip ``/ng-trace/<traceparent>`` from the path and supply it as the inbound ``traceparent``.

    The segment comes from ``apply_rollout_prefix``: a model URL an agent hands to a client that
    sends no ``traceparent`` header, such as a harness inside a sandbox. A header the client did
    send wins. The segment is stripped whether or not traces are on, so such a URL always routes.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "") if scope["type"] == "http" else ""
        # Substring test first: this runs on every request of every server, almost none of which carry the segment.
        match = _TRACE_CONTEXT_PATH_RE.match(path) if f"/{TRACE_CONTEXT_PATH_SEGMENT}/" in path else None
        if match is None:
            await self.app(scope, receive, send)
            return
        headers = list(scope.get("headers") or [])
        if not any(name.lower() == b"traceparent" for name, _ in headers):
            headers.append((b"traceparent", match.group("traceparent").encode("ascii")))
        scope = scope_with_routed_path(scope, match.group("head") + match.group("rest"))
        scope["headers"] = headers
        segment = f"/{TRACE_CONTEXT_PATH_SEGMENT}/{match.group('traceparent')}"

        async def send_with_segment_on_redirect(message: Message) -> None:
            # The client only ever had the URL, so a redirect it follows must carry the segment again.
            if message["type"] == "http.response.start":
                message = _restore_trace_segment_on_redirect(message, segment=segment, request_headers=headers)
            await send(message)

        await self.app(scope, receive, send_with_segment_on_redirect)


def _restore_trace_segment_on_redirect(
    message: Message, *, segment: str, request_headers: list[tuple[bytes, bytes]]
) -> Message:
    """Put ``/ng-trace/<traceparent>`` back into root-relative and same-origin redirect targets.

    Inserted after a rollout or token-capture prefix (which the capture middleware has already restored)
    or at the root; cross-origin targets and targets that already carry a segment are left alone.
    """
    status = int(message.get("status") or 0)
    if not 300 <= status < 400:
        return message
    request_host = next((value.decode("latin-1") for key, value in request_headers if key.lower() == b"host"), "")
    headers = list(message.get("headers") or [])
    changed = False
    for index, (key, value) in enumerate(headers):
        if key.lower() != b"location":
            continue
        parts = urlsplit(value.decode("latin-1"))
        if (parts.netloc and parts.netloc != request_host) or not parts.path.startswith("/"):
            continue
        if f"/{TRACE_CONTEXT_PATH_SEGMENT}/" in parts.path:
            continue
        prefix = _ROLLOUT_PREFIX_RE.match(parts.path)
        head = prefix.group(0) if prefix else ""
        path = f"{head}{segment}{parts.path[len(head) :]}"
        headers[index] = (key, urlunsplit(parts._replace(path=path)).encode("latin-1"))
        changed = True
    return {**message, "headers": headers} if changed else message


def install_trace_context_path(app: FastAPI) -> None:
    """Install ``TraceContextPathMiddleware`` outside the instrumentor, which reads headers and path when it
    opens the server span. Same mechanism as ``install_request_context_isolation``; call it after
    ``instrument_app_for_telemetry`` and before ``install_request_context_isolation``, which stays outermost.
    """
    if getattr(app.state, "nemo_gym_trace_context_path_installed", False):
        return
    app.state.nemo_gym_trace_context_path_installed = True
    build_inner = app.build_middleware_stack

    def build_middleware_stack() -> ASGIApp:
        return TraceContextPathMiddleware(build_inner())

    app.build_middleware_stack = build_middleware_stack  # type: ignore[method-assign]
    app.middleware_stack = None


class ClientDisconnectCancellationMiddleware:
    """Cancel an in-flight HTTP request when its client disconnects."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.num_cancelled = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        counts = request_counts()
        if counts is None:
            await self._handle(scope, receive, send, None)
            return
        counts.in_flight += 1
        try:
            await self._handle(scope, receive, send, counts)
        except Exception:
            # A handled error, such as an HTTPException, became a response inside and is not counted here.
            counts.unhandled_exceptions += 1
            raise
        finally:
            counts.in_flight -= 1

    async def _handle(self, scope: Scope, receive: Receive, send: Send, counts: Any) -> None:
        received_messages: asyncio.Queue[Message] = asyncio.Queue()
        client_disconnected = asyncio.Event()
        response_complete = asyncio.Event()
        debug = is_global_aiohttp_client_request_debug_enabled()
        started_at = time.monotonic() if debug else None
        response_started = False

        async def receive_message() -> Message:
            return await received_messages.get()

        async def send_message(message: Message) -> None:
            nonlocal response_started
            if client_disconnected.is_set():
                return

            if counts is not None and message["type"] == "http.response.start":
                counts.count_response(message["status"])
            await send(message)
            if message["type"] == "http.response.start":
                response_started = True
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                response_complete.set()

        # The listener is the sole reader of the original ASGI receive channel.
        # Forwarding request messages keeps the body available to the app while
        # also allowing disconnects to cancel handlers that no longer call receive.
        async with create_task_group() as task_group:

            async def run_app() -> None:
                try:
                    await self.app(scope, receive_message, send_message)
                finally:
                    task_group.cancel_scope.cancel()

            async def listen_for_disconnect() -> None:
                while True:
                    message = await receive()
                    if message["type"] != "http.disconnect":
                        await received_messages.put(message)
                        continue

                    # Uvicorn returns http.disconnect from receive() once the response is complete,
                    # even if the peer did not disconnect early. Only cancel requests whose response
                    # has not finished being sent.
                    if response_complete.is_set():
                        return

                    await received_messages.put(message)
                    client_disconnected.set()
                    self.num_cancelled += 1
                    if counts is not None:
                        counts.count_cancellation(_cancelled_request_route(scope), response_started=response_started)
                    if debug or self.num_cancelled % 100 == 0:
                        client = scope.get("client")
                        client_address = f"{client[0]}:{client[1]}" if client else "-:-"
                        detail = ""
                        if started_at is not None:
                            elapsed_s = time.monotonic() - started_at
                            phase = "response_started" if response_started else "before_response_start"
                            detail = f" after {elapsed_s:.1f}s {phase}"
                        print(
                            f'{client_address} - "{scope["method"]} {scope["path"]}" '
                            f"499 CLIENT DISCONNECTED{detail} ({self.num_cancelled} total for this server worker)"
                        )
                    task_group.cancel_scope.cancel()
                    return

            task_group.start_soon(run_app)
            task_group.start_soon(listen_for_disconnect)


def _connect_server_to_ray(server_class: type, global_config_dict: DictConfig) -> None:
    """Join the run's Ray cluster if this server declares that it uses Ray."""
    if not server_class.ray_enabled:
        return
    if _has_injected_global_config_env() and not global_config_dict.get(RAY_HEAD_NODE_ADDRESS_KEY_NAME):
        # Gym only launches servers without a cluster address when it decided no server needs Ray. Fail instead
        # of letting initialize_ray start a private cluster in this process and in each of its workers.
        raise RuntimeError(
            f"{server_class.__module__}.{server_class.__name__} uses Ray, but this run has no Ray cluster. "
            f"Declare `ray_enabled = True` on {server_class.__name__} itself so `gym` starts Ray for it, and "
            "install the `nemo-gym[ray]` extra where `gym` runs."
        )
    initialize_ray()


def _declared_class_flag(class_node: ast.ClassDef, name: str) -> bool | None:
    """The literal boolean assigned to `name` in a class body, annotated or not, or None when there is none."""
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            targets = item.targets
        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        if isinstance(item.value, ast.Constant) and isinstance(item.value.value, bool):
            return item.value.value
    return None


def _declared_ray_enabled(class_node: ast.ClassDef) -> bool | None:
    """The literal `ray_enabled` assigned in a class body, or None when there is none."""
    return _declared_class_flag(class_node, "ray_enabled")


def entrypoint_class_flag(entrypoint_fpath: Path, name: str) -> bool | None:
    """The boolean class attribute `name` declared by the server an entrypoint starts, read from source.

    Gym settles some things before any server exists, and each server runs in its own venv, so this reads the
    source instead of importing it. It finds the classes the entrypoint calls `run_webserver()` on and checks
    their literal `name` declarations (`name = True` or `name: ClassVar[bool] = True` in the class body).

    Returns True when any of them declares True, False when all of them declare False, and None when that can't
    be settled from the source alone, such as an inherited declaration, a class imported from another module, or
    an unreadable entrypoint.
    """
    try:
        source = entrypoint_fpath.read_text()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None

    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    invoked_class_names = {
        node.func.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_webserver"
        and isinstance(node.func.value, ast.Name)
    }
    if not invoked_class_names:
        return None
    declarations = [
        _declared_class_flag(classes[class_name], name) if class_name in classes else None
        for class_name in invoked_class_names
    ]
    if True in declarations:
        return True
    if all(declaration is False for declaration in declarations):
        return False
    return None


def entrypoint_ray_enabled(entrypoint_fpath: Path) -> bool | None:
    """The `ray_enabled` declaration of the server an entrypoint starts, read from source.

    The supervisor must decide whether to start Ray before any server exists; see `entrypoint_class_flag` for
    what the source alone can settle.
    """
    return entrypoint_class_flag(entrypoint_fpath, "ray_enabled")


def refuse_unsupported_num_workers(server: "SimpleServer") -> None:
    """Raise when the server asks for several workers but cannot run with them."""
    if not server.config.num_workers or server.config.num_workers <= 1:
        return
    reason = server.multiple_workers_unsupported_reason()
    if reason is not None:
        raise ValueError(f"{server.config.name} needs num_workers=1: {reason}")


class SimpleServer(BaseServer):
    server_client: ServerClient
    # Servers that use Ray must declare `ray_enabled = True` on the class their entrypoint runs. Gym reads it
    # before launch to decide whether to start Ray, and connects only those servers to it.
    ray_enabled: ClassVar[bool] = False
    # Whether this server keeps state per session,
    # so that with several uvicorn workers a session's requests must reach the worker that created it.
    # See nemo_gym.session_routing.
    routes_sessions_to_owner: ClassVar[bool] = False

    def claimed_session_routing(self) -> dict[str, Any]:
        """Arguments to ``install_session_routing``: where requests name their session, and how long claims last.

        A server that sets ``routes_sessions_to_owner`` implements it.
        """
        raise NotImplementedError(f"{type(self).__name__} routes sessions but does not say how they are named")

    @abstractmethod
    def setup_webserver(self) -> FastAPI:
        pass

    def multiple_workers_unsupported_reason(self) -> Optional[str]:
        """Why this server cannot run with `num_workers > 1`, or None when it can.

        With several workers, uvicorn imports the entrypoint in each worker and serves its module-level `app`.
        A server whose state or resources several workers cannot share overrides this and returns the reason,
        and `run_webserver` refuses `num_workers > 1` with it before uvicorn starts.
        The reason reads as the end of the sentence "<server> needs num_workers=1: <reason>".
        It is a method rather than a class attribute so that a server can decide from its config.
        """
        return None

    def setup_checkpoint_participant(self, app: FastAPI) -> None:
        """Take part in partial-rollout checkpoints on the app ``setup_webserver`` built, if it does not yet.

        Called for every server, so a server whose ``setup_webserver`` builds its own app takes part too.
        """

    def setup_telemetry(self) -> None:
        """Initialise this process's nemo-lens telemetry. Idempotent, once per process.

        Every Gym server is its own process with its own providers — there is no parent
        handle to inherit, only the `NEMO_GYM_OTEL_*` environment the supervisor
        exported before spawning it. A failure here is logged and swallowed: telemetry
        must never stop a server from serving.
        """
        from nemo_gym.telemetry.setup import init_telemetry

        init_telemetry(
            server_name=self.config.name,
            server_type=_telemetry_server_type(type(self)),
        )

    def instrument_app_for_telemetry(self, app: FastAPI) -> None:
        """Apply OTel FastAPI auto-instrumentation to *app*, if telemetry is exporting traces.

        Gives every server the inbound half of context propagation plus dimensioned
        `http.server.*` metrics, which is why Gym does not use nemo-lens's undimensioned
        `gym.server.request_duration_ms` (see `nemo_gym/telemetry/metrics.py`).

        The instrumentor costs tens of microseconds per request even when no span is exported, so a
        metrics-only run skips it and does without the `http.server.*` metrics.

        A server whose instrumentation could not be applied keeps serving; it just does not
        extract inbound trace context.
        """
        from nemo_gym.telemetry.contrib import instrument_fastapi_app
        from nemo_gym.telemetry.setup import get_telemetry

        telemetry = get_telemetry()
        if telemetry is None or not telemetry.is_exporting or not is_trace_exporter_active():
            return
        # Gated on the span group like every other site, so `server` actually controls
        # something. Checked here rather than per request because the instrumentor is
        # installed once at startup, by which point init_telemetry has resolved the groups.
        if not is_span_group_enabled(GymSpanGroup.SERVER):
            return
        instrument_fastapi_app(app)

    def get_session_middleware_key(self) -> str:
        # This method is here to override in case we want to ever use an actual session middleware secret key.
        # e.g. for an actual product.
        return f"{self.__class__.__name__}___{self.config.name}"

    def setup_session_middleware(self, app: FastAPI) -> None:
        if getattr(app.state, "nemo_gym_session_middleware_installed", False):
            return
        app.state.nemo_gym_session_middleware_installed = True

        # The multiple middleware execution order described in https://fastapi.tiangolo.com/tutorial/middleware/#multiple-middleware-execution-order
        # Says that if you register middlewares A and then B,
        # - at request time: They execute B first then A
        # - at response time: They return to A first and then B
        # So for adding session IDs, that middleware must run after SessionMiddleware, so it must be registered before it.

        @app.middleware("http")
        async def add_session_id(request: Request, call_next):  # pragma: no cover
            # Always assign so Starlette 1.0+ marks session.modified=True and re-sends Set-Cookie.
            # A new session takes the ID the session router chose, if any (see nemo_gym.session_routing):
            # a typed resources seed's resources_session_id, or an ID the router minted.
            new_session_id = request.scope.get(SESSION_ID_SCOPE_KEY) or str(uuid4())
            claim = request.scope.get(SESSION_CLAIM_SCOPE_KEY)
            session = request.session
            minted = SESSION_ID_KEY not in session
            session[SESSION_ID_KEY] = session.get(SESSION_ID_KEY, new_session_id)
            if claim is None:
                response = await call_next(request)
                if minted and response.status_code >= 400:
                    # A failed request keeps nothing under a new session, so its reply names none: a seed refused
                    # for another episode's resources_session_id must not hand this client that session's cookie.
                    del session[SESSION_ID_KEY]
                return response

            # With several workers, a session nobody claims, new or named by a cookie whose claim ended,
            # is claimed only if the handler uses it, before the reply leaves this worker,
            # so the client's next request finds the claim.
            # Starlette's session records each access through request.session; only the handler's count from here.
            tracks_access = hasattr(session, "mark_accessed")
            if tracks_access:
                session.accessed = False
            response: Response = await call_next(request)
            used = session.accessed if tracks_access else True
            if tracks_access:
                # The reply keeps its Vary: Cookie header, as for any request that reads the session.
                session.mark_accessed()
            if used and not claim():
                # Another worker claimed the same ID meanwhile; serving the request here would split the session.
                return JSONResponse({"detail": "This session's ID is already claimed. Retry it."}, status_code=409)
            if minted and (not used or response.status_code >= 400):
                # No cookie names an ID that nothing holds, so a later request cannot name an unclaimed session,
                # and a failed request hands out no cookie, as with one worker.
                del session[SESSION_ID_KEY]
            return response

        session_middleware_key = self.get_session_middleware_key()
        app.add_middleware(SessionMiddleware, secret_key=session_middleware_key, session_cookie=session_middleware_key)

    def setup_exception_middleware(self, app: FastAPI) -> None:  # pragma: no cover
        @app.middleware("http")
        async def exception_handling_middleware(request: Request, call_next):
            try:
                return await call_next(request)
            except ClientResponseError as e:
                assert hasattr(e, "response_content"), (
                    "Please use `nemo_gym.server_utils.raise_for_status` for HTTP exceptions!"
                )

                response_content = f"Hit an exception in {self.get_session_middleware_key()} calling an inner server: {e.response_content}"
                print(_format_upstream_error_log(self.get_session_middleware_key(), e), flush=True)

                return JSONResponse(content=response_content, status_code=500)
            except CancelledError:
                return JSONResponse(content="Request was cancelled", status_code=500)
            except Exception as e:
                print(
                    f"""🚨 Caught an exception printed above in {self.config.name} ({self.__class__.__name__}). If you expect this to be fed back into this model, the exception repr i.e. `repr(e)` is returned to the model. However, please make sure this exception is caught in your server and returned to the model as appropriate. See https://fastapi.tiangolo.com/tutorial/handling-errors/#use-httpexception
Formatted exception: {format_exc()}
repr(e): {repr(e)}"""
                )
                return JSONResponse(content=repr(e), status_code=500)
            except:
                print_exc()
                print(
                    f"""🚨 Caught an unknown exception printed above in {self.config.name} ({self.__class__.__name__}). If you expect this to be fed back into this model, nothing meaningful is returned to the model. Please make sure this exception is caught in your server and returned to the model as appropriate. See https://fastapi.tiangolo.com/tutorial/handling-errors/#use-httpexception"""
                )
                return JSONResponse(content="An unknown error occurred", status_code=500)

    def setup_cancellation_middleware(self, app: FastAPI) -> None:
        app.add_middleware(ClientDisconnectCancellationMiddleware)

    def setup_profiling(self, app: FastAPI, profiling_config: ProfilingMiddlewareConfig) -> None:  # pragma: no cover
        base_profile_dir = WORKING_DIR / profiling_config.profiling_results_dirpath / self.get_session_middleware_key()
        profiler = Profiler(name=self.config.name, base_profile_dir=base_profile_dir)

        main_app_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_wrapper(app):
            profiler.start()

            async with main_app_lifespan(app) as maybe_state:
                yield maybe_state

            profiler.stop()

        app.router.lifespan_context = lifespan_wrapper

        @app.get("/stats")
        def stats():
            profiler.dump()
            return Response()

    def set_ulimit(self, target_soft_limit: int = 65535):  # pragma: no cover
        # From https://github.com/vllm-project/vllm/blob/fed8a9b107df3e27d57728c6911c7d308b871477/vllm/utils/__init__.py#L2790
        resource_type = resource.RLIMIT_NOFILE
        current_soft, current_hard = resource.getrlimit(resource_type)

        if current_soft < target_soft_limit:
            try:
                resource.setrlimit(resource_type, (target_soft_limit, current_hard))
            except ValueError as e:
                print(
                    "Found ulimit of %s and failed to automatically increase "
                    "with error %s. This can cause fd limit errors like "
                    "`OSError: [Errno 24] Too many open files`. Consider "
                    "increasing with ulimit -n",
                    current_soft,
                    e,
                )

    def prefix_server_logs(self) -> None:  # pragma: no cover
        # Adapted from https://github.com/vllm-project/vllm/blob/ab74b2a27a4eb88b90356bfb4b452d29edf05574/vllm/utils/system_utils.py#L205

        def _add_prefix(file: TextIO) -> None:
            prefix = f"({self.config.name}) "
            file_write = file.write

            def write_with_prefix(s: str):
                if not s:
                    return

                if file.start_new_line:
                    file_write(prefix)

                idx = 0
                while (next_idx := s.find("\n", idx)) != -1:
                    next_idx += 1
                    file_write(s[idx:next_idx])
                    if next_idx == len(s):
                        file.start_new_line = True
                        return

                    file_write(prefix)
                    idx = next_idx

                file_write(s[idx:])
                file.start_new_line = False

            file.start_new_line = True
            file.write = write_with_prefix

        is_main_fastapi_proc = not is_nemo_gym_fastapi_worker()
        if is_main_fastapi_proc:
            _add_prefix(sys.stdout)
            _add_prefix(sys.stderr)

    @classmethod
    def run_webserver(cls) -> Optional[FastAPI]:  # pragma: no cover
        startup = ServerStartupTimeline(
            getenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, cls.__name__),
            _telemetry_server_type(cls),
            worker=is_nemo_gym_fastapi_worker(),
        )
        global_config_dict = get_global_config_dict()

        is_main_fastapi_proc = not is_nemo_gym_fastapi_worker()

        server_config = cls.load_config_from_global_config()
        startup.mark("load_config")
        _connect_server_to_ray(cls, global_config_dict)
        if cls.ray_enabled:
            startup.mark("init_ray")

        server_client = ServerClient(
            head_server_config=ServerClient.load_head_server_config(),
            global_config_dict=global_config_dict,
        )
        server = cls(config=server_config, server_client=server_client)
        startup.mark("init_server")
        # Before the dry-run return, so a dry run reports the misconfiguration too.
        refuse_unsupported_num_workers(server)

        if global_config_dict[DRY_RUN_KEY_NAME]:
            return

        # Before setup_webserver so the OTel providers exist by the time the app is
        # instrumented below: FastAPIInstrumentor captures the tracer provider at
        # instrument time, and a no-op provider captured here would stay no-op forever.
        # Once per process — uvicorn's multi-worker path re-enters run_webserver in each
        # worker via the module import, and init_telemetry is idempotent.
        server.setup_telemetry()
        startup.mark("init_telemetry")

        app = server.setup_webserver()
        startup.mark("setup_webserver")
        observe_app_event_loop(app, "server")
        server.setup_checkpoint_participant(app)
        # After the app is fully built so subclass routes are present. Only resources servers expose tools over MCP,
        # so gating the lazy import on their config keeps the MCP SDK out of agent/model processes that never need it.
        mcp_token_routing = {}
        if getattr(getattr(server, "config", None), "expose_tools_over_mcp", False):
            from nemo_gym.mcp_auto_exposure import TOKEN_HEADER, maybe_auto_expose, session_token_serializer

            maybe_auto_expose(server, app)
            startup.mark("expose_tools_over_mcp")
            mcp_token_routing = dict(
                mcp_token_header=TOKEN_HEADER, mcp_token_serializer=session_token_serializer(server)
            )
        server.setup_liveness(app)
        server.set_ulimit()
        server.prefix_server_logs()
        connection_pool_config = GlobalAIOHTTPAsyncClientConfig.model_validate(global_config_dict)
        pool_capacity = connection_pool_capacity(
            connection_pool_config,
            server.config.num_workers or 1,
        )
        if is_main_fastapi_proc:
            report_connection_pool_capacity(connection_pool_config, pool_capacity, visible=True)
        server.setup_exception_middleware(app)
        backlog = UvicornBacklogConfig.model_validate(global_config_dict).uvicorn_backlog
        # A worker of a multi-worker server serves a private socket in this directory and claims sessions in it.
        # The main process of a multi-worker server never serves the app.
        session_socket_dir = getenv(SESSION_SOCKET_DIR_ENV) if not is_main_fastapi_proc else None
        is_multi_worker_main_proc = is_main_fastapi_proc and (server.config.num_workers or 1) > 1
        if server.routes_sessions_to_owner and not is_multi_worker_main_proc:
            # Outside SessionMiddleware and the exception middleware:
            # a forwarded request is handled only by the worker that claims its session.
            session_cookie = server.get_session_middleware_key()
            install_session_routing(
                app,
                socket_dir=session_socket_dir,
                session_cookie=session_cookie,
                secret_key=session_cookie,
                backlog=backlog,
                **mcp_token_routing,
                **server.claimed_session_routing(),
            )
        # Register last so cancellation wraps the complete request stack.
        server.setup_cancellation_middleware(app)
        # Must precede uvicorn.run: Starlette refuses add_middleware once the app has
        # started. This is the ingress half of cross-process propagation — the instrumentor
        # extracts an inbound `traceparent` and parents this server's SERVER span to the
        # caller's CLIENT span.
        server.instrument_app_for_telemetry(app)
        # Outside the instrumentor so it sees the `traceparent` a client could only put in the URL.
        install_trace_context_path(app)
        # After the instrumentor and outermost of all, so the instrumentor runs inside a clean
        # context even when uvicorn started this request's task from the previous request's.
        install_request_context_isolation(app)

        app.exception_handler(RequestValidationError)(_validation_exception_handler)

        profiling_config = ProfilingMiddlewareConfig.model_validate(global_config_dict)
        if profiling_config.profiling_enabled:
            server.setup_profiling(app, profiling_config)
        startup.mark("configure_app")

        uvicorn_logging_cfg = UvicornLoggingConfig.model_validate(global_config_dict)
        uvicorn_proxy_cfg = UvicornProxyHeadersConfig.model_validate(global_config_dict)
        keepalive_cfg = GlobalAIOHTTPAsyncClientConfig.model_validate(global_config_dict)
        if not uvicorn_logging_cfg.uvicorn_logging_show_200_ok and is_main_fastapi_proc:
            print(
                "Disabling a uvicorn access logging so that the logs aren't spammed with 200 OK messages. This is to help errors pop up better and filter out noise."
            )

        uvicorn_kwargs = dict(
            host=server.config.host,
            port=server.config.port,
            # We add a very small graceful shutdown timeout so when we shutdown we cancel all inflight requests and there are no lingering requests (requests are cancelled)
            timeout_graceful_shutdown=0.5,
            # Some workers may take a while for imports and setup_webserver.
            timeout_worker_healthcheck=global_config_dict.get(UVICORN_TIMEOUT_WORKER_HEALTHCHECK, 30),
            # Ensure server keepalive > client keepalive
            timeout_keep_alive=30,
            backlog=backlog,
            # Parse HTTP with httptools instead of pure-Python h11, and enable TCP keepalive on every accepted
            # connection. Explicit selection prevents Uvicorn from silently falling back to h11; a missing or
            # incompatible httptools wheel fails at import.
            http=partial(
                KeepaliveHttpToolsProtocol,
                keepalive=(
                    keepalive_cfg.global_aiohttp_tcp_keepalive_idle_seconds,
                    keepalive_cfg.global_aiohttp_tcp_keepalive_interval_seconds,
                    keepalive_cfg.global_aiohttp_tcp_keepalive_probes,
                ),
            ),
            access_log=uvicorn_logging_cfg.uvicorn_logging_show_200_ok,
            # Internal-only by default. Enabling this requires an explicit trusted-proxy allowlist,
            # so forwarded headers are never honored from an arbitrary peer.
            proxy_headers=uvicorn_proxy_cfg.uvicorn_proxy_headers,
            forwarded_allow_ips=uvicorn_proxy_cfg.uvicorn_forwarded_allow_ips or [],
        )

        if server.config.num_workers and server.config.num_workers > 1:
            # TODO this is very dirty. We need a cleaner way to populate this information in the configs data structures.
            server_instance_config_dict = global_config_dict[server.config.name]
            first_level_key = list(server_instance_config_dict.keys())[0]
            second_level_key = list(server_instance_config_dict[first_level_key].keys())[0]
            relative_fpath = f"{first_level_key}/{second_level_key}/{server.config.entrypoint}"
            module_import_str = relative_fpath.replace(".py", "").replace("/", ".")

            set_is_nemo_gym_fastapi_worker()
            set_is_nemo_gym_fastapi_entrypoint(str(relative_fpath))
            set_nemo_gym_fastapi_num_workers(server.config.num_workers)
            if is_main_fastapi_proc and server.routes_sessions_to_owner:
                # The workers uvicorn spawns inherit this, and each serves a private socket in it.
                environ[SESSION_SOCKET_DIR_ENV] = create_session_socket_dir()

            uvicorn_kwargs["app"] = f"{module_import_str}:app"
            uvicorn_kwargs["workers"] = server.config.num_workers
        else:
            uvicorn_kwargs["app"] = app

        # The main process of a multi-worker server hands the app to Uvicorn workers and never serves it, so it reports now.
        # Every process that serves the app reports when the app's lifespan startup finishes.
        if is_main_fastapi_proc and uvicorn_kwargs.get("workers"):
            startup.report()
        else:
            startup.report_when_serving(app)

        if is_main_fastapi_proc:
            startup.announce_serve()
            try:
                uvicorn.run(**uvicorn_kwargs)
            finally:
                # BatchSpanProcessor only exports on a timer, so without an explicit flush
                # the last seconds of a run — including the spans of whatever was in
                # flight at shutdown — are silently dropped.
                from nemo_gym.telemetry.setup import shutdown_telemetry

                shutdown_telemetry()

        return app


class HeadServer(BaseServer):
    config: BaseServerConfig
    _server_instances: List[dict] = []
    _ready: bool = PrivateAttr(default=False)
    # Serialized global config returned to clients.
    _cached_yaml: Optional[str] = None

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        @app.get("/livez", include_in_schema=False)
        @app.get("/", include_in_schema=False)
        async def _liveness():
            return {"status": "ok"}

        @app.get("/readyz", include_in_schema=False)
        @app.get("/healthz", include_in_schema=False)
        @app.get("/health", include_in_schema=False)
        async def _readiness(response: Response):
            if not self._ready:
                response.status_code = 503
                return {"status": "starting"}
            return {"status": "ok"}

        app.get("/global_config_dict_yaml")(self.global_config_dict_yaml)
        app.get("/server_instances")(self.get_server_instances)

        return app

    def mark_ready(self) -> None:
        self._ready = True

    def get_server_instances(self) -> List[dict]:
        return self._server_instances

    def set_server_instances(self, instances: List) -> None:
        self._server_instances = instances

    def invalidate_global_config_dict_yaml_cache(self) -> None:
        """Clear the serialized global config cache."""
        self._cached_yaml = None

    @classmethod
    def run_webserver(cls) -> Tuple[uvicorn.Server, Thread, "HeadServer"]:
        config = ServerClient.load_head_server_config()
        server = cls(config=config)
        global_config_dict = get_global_config_dict()
        uvicorn_proxy_cfg = UvicornProxyHeadersConfig.model_validate(global_config_dict)

        app = server.setup_webserver()
        # The head server runs on its own thread and event loop inside the supervisor process.
        observe_app_event_loop(app, "head_server")

        config = uvicorn.Config(
            app,
            host=server.config.host,
            port=server.config.port,
            proxy_headers=uvicorn_proxy_cfg.uvicorn_proxy_headers,
            forwarded_allow_ips=uvicorn_proxy_cfg.uvicorn_forwarded_allow_ips or [],
            backlog=UvicornBacklogConfig.model_validate(global_config_dict).uvicorn_backlog,
        )
        uvicorn_server = uvicorn.Server(config=config)

        thread = Thread(target=uvicorn_server.run, daemon=True)
        thread.start()

        return uvicorn_server, thread, server

    async def global_config_dict_yaml(self) -> str:
        if self._cached_yaml is None:
            self._cached_yaml = OmegaConf.to_yaml(get_global_config_dict())
        return self._cached_yaml


class ServerInstanceDisplayConfig(BaseModel):
    config_path: Optional[str] = None
    dir_path: Optional[Path] = None
    entrypoint: Optional[str] = None
    host: Optional[str] = None
    name: Optional[str] = None
    pid: Optional[int] = None
    port: Optional[int] = None
    process_name: Optional[str] = None
    server_type: Optional[str] = None
    start_time: Optional[float] = None
    status: Optional[ServerStatus] = None
    uptime_seconds: Optional[float] = None
    url: Optional[str] = None


def get_server_url(server_name: str) -> str:
    global_config_dict = get_global_config_dict()

    model_server_config = get_first_server_config_dict(
        global_config_dict,
        server_name,
    )

    return f"http://{model_server_config['host']}:{model_server_config['port']}"


def rollout_path_prefix(rollout_id: Optional[str], *, token_capture: bool = False) -> str:
    """Return the leading model-server path prefix for a rollout, if available."""
    if not rollout_id:
        return ""
    capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if token_capture else ""
    return f"/{ROLLOUT_PATH_PREFIX}/{rollout_id}{capture_segment}"


def trace_context_path_segment() -> str:
    """Return ``/ng-trace/<traceparent>`` for the current span, or ``""`` while traces are off or no span is current.

    ``TraceContextPathMiddleware`` on the receiving Gym server turns it back into the header.
    """
    if not is_trace_exporter_active():
        return ""
    from nemo_gym.telemetry.contrib import inject_trace_context

    carrier: dict[str, str] = {}
    inject_trace_context(carrier)
    traceparent = carrier.get("traceparent", "")
    if not re.fullmatch(_TRACEPARENT_PATTERN, traceparent):
        return ""
    return f"/{TRACE_CONTEXT_PATH_SEGMENT}/{traceparent}"


def apply_rollout_prefix(
    base_url: str, rollout_id: Optional[str], *, token_capture: bool = False, trace_context: bool = True
) -> str:
    """Append the rollout prefix and, while traces export, the current span's trace context to a model-server root URL.

    Every caller hands the result to a client Gym does not instrument (a harness in a sandbox, a
    third-party client library), which is why the trace context travels in the URL here; Gym's own
    clients send the ``traceparent`` header instead and use ``url_path_for_run``.

    The span named is the one current when the URL is built, so pass ``trace_context=False`` for a
    URL that is built once and reused across rollouts (a cached client): otherwise every later
    rollout's calls would be parented to the first one's span.
    """
    prefix = rollout_path_prefix(rollout_id, token_capture=token_capture) if rollout_id else ""
    if trace_context:
        prefix += trace_context_path_segment()
    if not prefix:
        return base_url
    return base_url.rstrip("/") + prefix


def setup_server_client(head_server_config: Optional[BaseServerConfig] = None) -> ServerClient:  # pragma: no cover
    server_client = ServerClient.load_from_global_config(head_server_config)

    # We set this rollout global aiohttp client to use the same max connections as the underlying head server global config.
    if not is_global_aiohttp_client_setup():
        set_global_aiohttp_client(cfg=GlobalAIOHTTPAsyncClientConfig.model_validate(server_client.global_config_dict))

    return server_client
