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
from importlib.util import find_spec
from ipaddress import ip_network
from os import environ, getenv
from pathlib import Path
from threading import Thread
from traceback import format_exc, print_exc
from typing import Any, ClassVar, List, Optional, TextIO, Tuple, Type
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import uvicorn
from aiohttp import (
    ClientResponseError,
)
from anyio import create_task_group
from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import BaseModel, PositiveInt, PrivateAttr, model_validator
from starlette.middleware.sessions import SessionMiddleware
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol

from nemo_gym import WORKING_DIR
from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.client.http_client import (
    GlobalAIOHTTPAsyncClientConfig,
    _detach_global_aiohttp_clients,
    _format_upstream_error_log,
    _set_tcp_keepalive,
    global_aiohttp_client_exit,
    is_global_aiohttp_client_request_debug_enabled,
)
from nemo_gym.client.server_client import (
    _TRACEPARENT_PATTERN,
    SESSION_ID_KEY,
    ServerClient,
    ServerStatus,
)
from nemo_gym.config.config_types import (
    DEFAULT_UVICORN_BACKLOG,
    ROLLOUT_PATH_PREFIX,
    TOKEN_CAPTURE_PATH_SEGMENT,
    TRACE_CONTEXT_PATH_SEGMENT,
    BaseRunServerInstanceConfig,
    BaseServerConfig,
)
from nemo_gym.config.global_config import (
    DRY_RUN_KEY_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    RAY_HEAD_NODE_ADDRESS_KEY_NAME,
    UVICORN_TIMEOUT_WORKER_HEALTHCHECK,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.core.episode import rollout_context
from nemo_gym.server_runtime.request_handling.rollout_context import ROUTED_PATH_STATE_KEY, scope_with_routed_path
from nemo_gym.server_runtime.request_handling.session_routing import (
    SESSION_CLAIM_SCOPE_KEY,
    SESSION_ID_SCOPE_KEY,
    SESSION_SOCKET_DIR_ENV,
    create_session_socket_dir,
    install_session_routing,
)
from nemo_gym.telemetry._fallbacks import is_span_group_enabled
from nemo_gym.telemetry.connection_pool import (
    connection_pool_capacity,
    report_connection_pool_capacity,
)
from nemo_gym.telemetry.observer import observe_app_event_loop, request_counts
from nemo_gym.telemetry.profiling import (
    Profiler,
    ProfilingMiddlewareConfig,
)
from nemo_gym.telemetry.setup import is_trace_exporter_active
from nemo_gym.telemetry.span_groups import GymSpanGroup
from nemo_gym.telemetry.startup import ServerStartupTimeline
from nemo_gym.utils.server_process import (
    _has_injected_global_config_env,
    is_nemo_gym_fastapi_worker,
    set_is_nemo_gym_fastapi_entrypoint,
    set_is_nemo_gym_fastapi_worker,
    set_nemo_gym_fastapi_num_workers,
)


try:
    from opentelemetry import context as _otel_context
except ImportError:  # the telemetry extra is optional; without the SDK there is no span context to isolate
    _otel_context = None


logger = logging.getLogger(__name__)

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


async def close_global_aiohttp_client() -> None:
    """Close the process-wide data and control clients on their owning event loop."""

    for client in _detach_global_aiohttp_clients():
        await client.close()


atexit.register(global_aiohttp_client_exit)


DEFAULT_HEAD_SERVER_PORT = 11000


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

    # Direct MCP dispatch has no client connection to monitor.
    skipped_on_direct_mcp_dispatch = True

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
    # See nemo_gym.server_runtime.request_handling.session_routing.
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
            # A new session takes the ID the session router chose, if any (see nemo_gym.server_runtime.request_handling.session_routing):
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

        # Direct MCP dispatch materializes the session itself.
        add_session_id.skipped_on_direct_mcp_dispatch = True

        session_middleware_key = self.get_session_middleware_key()
        app.add_middleware(SessionMiddleware, secret_key=session_middleware_key, session_cookie=session_middleware_key)

    def setup_exception_middleware(self, app: FastAPI) -> None:  # pragma: no cover
        @app.middleware("http")
        async def exception_handling_middleware(request: Request, call_next):
            try:
                return await call_next(request)
            except ClientResponseError as e:
                assert hasattr(e, "response_content"), (
                    "Please use `nemo_gym.client.http_client.raise_for_status` for HTTP exceptions!"
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

        # Direct MCP dispatch handles exceptions in the /mcp request itself.
        exception_handling_middleware.skipped_on_direct_mcp_dispatch = True

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
            from nemo_gym.server_runtime.mcp_auto_exposure import (
                TOKEN_HEADER,
                maybe_auto_expose,
                session_token_serializer,
            )

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


# Names that moved to other modules stay importable from here until their removal release; see nemo_gym/_moved_paths.py.
__getattr__ = moved_names_getattr(__name__)
