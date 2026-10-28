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
"""The shared aiohttp clients every Gym process uses to call HTTP endpoints, and requests with retries."""

import asyncio
import socket
import time
from traceback import print_exc
from typing import Any, NamedTuple, Optional, Type, Union, Unpack

import orjson
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
from multidict import CIMultiDict
from pydantic import BaseModel, Field

from nemo_gym.config.global_config import (
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
    maybe_get_global_config_dict,
)
from nemo_gym.core.episode import (
    current_rollout_id,
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
from nemo_gym.telemetry.observer import request_counts
from nemo_gym.telemetry.setup import is_trace_exporter_active
from nemo_gym.telemetry.span_groups import GymSpanGroup
from nemo_gym.utils.server_process import (
    get_nemo_gym_fastapi_num_workers,
    is_nemo_gym_fastapi_worker,
)


_GLOBAL_AIOHTTP_CLIENT: Union[None, ClientSession] = None


_GLOBAL_AIOHTTP_CLIENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


_GLOBAL_AIOHTTP_CLIENT_CONFIG: Optional["GlobalAIOHTTPAsyncClientConfig"] = None


_GLOBAL_AIOHTTP_CONTROL_CLIENT: Union[None, ClientSession] = None


_GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG: bool = False


_GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY: bool = False


_UPSTREAM_ERROR_LOG_BODY_CHARS = 2000


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
            **kwargs,
        )
    # While metrics are exporting, count calls still waiting for their response, which ends when the headers arrive.
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
                return await client.request(method=method, url=url, **kwargs)
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
