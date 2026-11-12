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
import asyncio
import atexit
import json
import logging
import resource
import signal
import socket
import sys
from abc import abstractmethod
from asyncio.exceptions import CancelledError
from contextlib import asynccontextmanager
from functools import partial
from ipaddress import ip_network
from os import environ, getenv
from traceback import format_exception, print_exception
from typing import Any, Awaitable, Callable, ClassVar, List, Optional, Protocol, Sequence, TextIO, Tuple, Type
from urllib.parse import urlsplit, urlunsplit

import uvicorn
from aiohttp import (
    ClientResponseError,
)
from fastapi import FastAPI, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf
from pydantic import BaseModel, PositiveInt, ValidationError, model_validator
from starlette.middleware.sessions import SessionMiddleware
from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol

from nemo_gym import WORKING_DIR
from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.client.http_client import (
    GlobalAIOHTTPAsyncClientConfig,
    _format_upstream_error_log,
    _set_tcp_keepalive,
)
from nemo_gym.client.server_client import (
    ServerClient,
    rollout_path_prefix,
)
from nemo_gym.config.config_types import (
    DEFAULT_UVICORN_BACKLOG,
    BaseRunServerInstanceConfig,
)
from nemo_gym.config.global_config import (
    DRY_RUN_KEY_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    UVICORN_TIMEOUT_WORKER_HEALTHCHECK,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.core.correlation import CorrelationPathError, parse_path
from nemo_gym.core.episode_context import current_rollout_id
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.server_runtime.ray_support import (
    _connect_server_to_ray,
    maybe_ray_cluster_exit,
)
from nemo_gym.server_runtime.request_handling.disconnect import (
    ClientDisconnectCancellationMiddleware,
)
from nemo_gym.server_runtime.request_handling.errors import (
    _validation_exception_handler,
)
from nemo_gym.server_runtime.request_handling.exception_handling import ExceptionHandlingMiddleware
from nemo_gym.server_runtime.request_handling.request_context import (
    install_request_context_isolation,
)
from nemo_gym.server_runtime.request_handling.session_id import SessionIDMiddleware
from nemo_gym.server_runtime.request_handling.session_routing import (
    SESSION_SOCKET_DIR_ENV,
    create_session_socket_dir,
    install_session_routing,
)
from nemo_gym.server_runtime.request_handling.trace_context import install_trace_context_path
from nemo_gym.server_runtime.uvicorn_reuse_port import reuse_port_supported, run_with_reuse_port
from nemo_gym.telemetry._fallbacks import is_span_group_enabled
from nemo_gym.telemetry.connection_pool import (
    connection_pool_capacity,
    report_connection_pool_capacity,
)
from nemo_gym.telemetry.observer import observe_app_event_loop
from nemo_gym.telemetry.profiling import (
    Profiler,
    ProfilingMiddlewareConfig,
)
from nemo_gym.telemetry.setup import is_trace_exporter_active
from nemo_gym.telemetry.span_groups import GymSpanGroup
from nemo_gym.telemetry.startup import ServerStartupTimeline
from nemo_gym.utils.server_process import (
    is_nemo_gym_fastapi_worker,
    set_is_nemo_gym_fastapi_entrypoint,
    set_is_nemo_gym_fastapi_worker,
    set_nemo_gym_fastapi_num_workers,
)


logger = logging.getLogger(__name__)


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
        try:
            server_config = server_config_cls.model_validate(
                OmegaConf.to_container(server_config_dict, resolve=True) | {"name": config_path_str}
            )
        except ValidationError as exc:
            # Name the source setting when a required field interpolates an unset value.
            unresolved = OmegaConf.to_container(server_config_dict, resolve=False)
            missing = set()
            for error in exc.errors():
                if error.get("input") is None and error["loc"]:
                    source = unresolved.get(error["loc"][0])
                    if isinstance(source, str) and source.startswith("${") and source.endswith("}"):
                        key = source[2:-1]
                        if key in global_config_dict and global_config_dict[key] is None:
                            missing.add(key)
            if missing:
                settings = ", ".join(sorted(missing))
                overrides = " ".join(f"++{key}=<value>" for key in sorted(missing))
                raise ValueError(
                    f"Required settings for {config_path_str} are unset: {settings}. "
                    f"Provide them in a config file or via CLI overrides: {overrides}"
                ) from None
            raise

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


class UvicornReusePortConfig(BaseModel):
    # Multi-worker servers only, Linux only: each worker listens on its own SO_REUSEPORT socket, so the
    # kernel spreads connections evenly across workers instead of whichever worker accepts first.
    # A server's own `uvicorn_reuse_port` overrides this.
    uvicorn_reuse_port: bool = False


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


def refuse_unsupported_num_workers(server: "SimpleServer") -> None:
    """Raise when the server asks for several workers but cannot run with them."""
    if not server.config.num_workers or server.config.num_workers <= 1:
        return
    reason = server.multiple_workers_unsupported_reason()
    if reason is not None:
        raise ValueError(f"{server.config.name} needs num_workers=1: {reason}")


def _unblock_shutdown_signals() -> None:
    """Let SIGINT and SIGTERM, on which uvicorn shuts a server down, reach this process. A process starts with the
    signal mask of the thread that launched it, so a server launched from a thread that blocks these signals, such as
    any thread of a Ray worker, would otherwise leave RunHelper's interrupt pending and never run its shutdown.
    Unblocking a signal that is not blocked changes nothing."""
    signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))


class SimpleServer(BaseServer):
    server_client: ServerClient
    # Servers that use Ray must declare `ray_enabled = True` on the class their entrypoint runs. Gym reads it
    # before launch to decide whether to start Ray, and connects only those servers to it.
    ray_enabled: ClassVar[bool] = False
    # Whether this server keeps state per session,
    # so that with several uvicorn workers a session's requests must reach the worker that created it.
    # See nemo_gym.server_runtime.request_handling.session_routing.
    routes_sessions_to_owner: ClassVar[bool] = False

    def owned_session_routing(self) -> dict[str, Any]:
        """Arguments to ``install_session_routing``: where requests name their session, and how long ownership lasts.

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
        app.add_middleware(SessionIDMiddleware)

        session_middleware_key = self.get_session_middleware_key()
        app.add_middleware(SessionMiddleware, secret_key=session_middleware_key, session_cookie=session_middleware_key)

    def setup_exception_middleware(self, app: FastAPI) -> None:
        app.add_middleware(ExceptionHandlingMiddleware, failure_reply=partial(SimpleServer._failure_reply, self))

    def _failure_reply(self, e: BaseException) -> JSONResponse:  # pragma: no cover
        """The reply for an exception the route did not handle."""
        if isinstance(e, NgFailureEvent) and e.http_error is not None:
            e = e.http_error
        if isinstance(e, ClientResponseError):
            assert hasattr(e, "response_content"), (
                "Please use `nemo_gym.client.http_client.raise_for_status` for HTTP exceptions!"
            )

            response_content = f"Hit an exception in {self.get_session_middleware_key()} calling an inner server: {e.response_content}"
            print(_format_upstream_error_log(self.get_session_middleware_key(), e), flush=True)

            from nemo_gym.trajectory.trajectory_record import failure_observations

            try:
                observations = failure_observations(json.loads(e.response_content))
            except (TypeError, ValueError):
                observations = {}
            content = {"detail": response_content, **observations} if observations else response_content
            return JSONResponse(content=content, status_code=500)
        if isinstance(e, CancelledError):
            return JSONResponse(content="Request was cancelled", status_code=500)
        if isinstance(e, Exception):
            print(
                f"""🚨 Caught an exception printed above in {self.config.name} ({self.__class__.__name__}). If you expect this to be fed back into this model, the exception repr i.e. `repr(e)` is returned to the model. However, please make sure this exception is caught in your server and returned to the model as appropriate. See https://fastapi.tiangolo.com/tutorial/handling-errors/#use-httpexception
Formatted exception: {"".join(format_exception(e))}
repr(e): {repr(e)}"""
            )
            return JSONResponse(content=repr(e), status_code=500)
        print_exception(e)
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
        _unblock_shutdown_signals()
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
        reuse_port = server.config.uvicorn_reuse_port
        if reuse_port is None:
            reuse_port = UvicornReusePortConfig.model_validate(global_config_dict).uvicorn_reuse_port
        if reuse_port and (server.config.num_workers or 1) > 1 and not reuse_port_supported():
            raise RuntimeError(
                f"{server.config.name} sets uvicorn_reuse_port, which needs Linux, where SO_REUSEPORT spreads "
                "connections across sockets"
            )

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
        # A worker of a multi-worker server serves a private socket in this directory and records session owners in it.
        # The main process of a multi-worker server never serves the app.
        session_socket_dir = getenv(SESSION_SOCKET_DIR_ENV) if not is_main_fastapi_proc else None
        is_multi_worker_main_proc = is_main_fastapi_proc and (server.config.num_workers or 1) > 1
        if server.routes_sessions_to_owner and not is_multi_worker_main_proc:
            # Outside SessionMiddleware and the exception middleware:
            # a forwarded request is handled only by the worker that owns its session.
            session_cookie = server.get_session_middleware_key()
            install_session_routing(
                app,
                socket_dir=session_socket_dir,
                session_cookie=session_cookie,
                secret_key=session_cookie,
                backlog=backlog,
                **mcp_token_routing,
                **server.owned_session_routing(),
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
                if uvicorn_kwargs.get("workers") and reuse_port:
                    run_with_reuse_port(**uvicorn_kwargs)
                else:
                    uvicorn.run(**uvicorn_kwargs)
            finally:
                # BatchSpanProcessor only exports on a timer, so without an explicit flush
                # the last seconds of a run — including the spans of whatever was in
                # flight at shutdown — are silently dropped.
                from nemo_gym.telemetry.setup import shutdown_telemetry

                shutdown_telemetry()

        return app


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _url_root(url: str) -> tuple[str, Optional[str], Optional[int], list[str]]:
    """Scheme, lowercased host, effective port and path segments, as HTTP clients normalize them."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    port = parts.port if parts.port is not None else _DEFAULT_PORTS.get(scheme)
    return scheme, parts.hostname, port, [s for s in parts.path.split("/") if s]


def _end_of_segments(path: str, count: int) -> int:
    """The index in ``path`` just past its first ``count`` non-empty segments; repeated slashes are kept as sent."""
    seen = 0
    index = 0
    while seen < count:
        while index < len(path) and path[index] == "/":
            index += 1
        while index < len(path) and path[index] != "/":
            index += 1
        seen += 1
    return index


def with_rollout_prefix(url: str, base_urls: Sequence[str]) -> str:
    """Insert ``/ng-rollout/<current rollout>`` after the matching base URL, for clients with a fixed base URL.

    A long-lived client (for example one a resources server calls from ``verify()``) keeps one base URL; this
    rewrites each request URL instead, so the call reaches the Gym model server exactly like a prefixed one.
    Only URLs under one of ``base_urls`` are changed, compared by scheme, host (case-insensitive), effective port
    and whole path segments. A URL whose path below the base already carries a rollout prefix is left as it is.
    Without a current rollout the URL is unchanged, and the model server forwards the call without capture.

    The current rollout is a ``ContextVar``: a synchronous client run through ``asyncio.to_thread`` or
    ``contextvars.copy_context().run`` sees it, one run through ``loop.run_in_executor`` or a thread pool does not.
    """
    if isinstance(base_urls, str):
        raise TypeError("base_urls must be a sequence of base URLs, not a single string")
    scheme, host, port, segments = _url_root(url)
    for base_url in base_urls:
        base_scheme, base_host, base_port, base_segments = _url_root(base_url)
        if (scheme, host, port) != (base_scheme, base_host, base_port) or segments[
            : len(base_segments)
        ] != base_segments:
            continue
        rest = "/" + "/".join(segments[len(base_segments) :])
        try:
            resolved = parse_path(rest, allow_empty_rest=True)
        except CorrelationPathError:
            # The receiving server answers a malformed prefix; adding a second one would only hide it.
            return url
        if resolved is not None and resolved[0].capture_key is not None:
            return url
        rollout_id = current_rollout_id()
        if not rollout_id:
            logger.debug(f"No current rollout, so {url} is sent without a rollout prefix and is not captured")
            return url
        parts = urlsplit(url)
        cut = _end_of_segments(parts.path, len(base_segments))
        return urlunsplit(parts._replace(path=parts.path[:cut] + rollout_path_prefix(rollout_id) + parts.path[cut:]))
    return url


_ORIGINAL_URL_EXTENSION = "nemo_gym.original_url"


class RewritableRequest(Protocol):
    """The part of an ``httpx.Request`` the rollout-prefix hooks use: a URL whose type parses a string."""

    url: object


def rollout_prefix_hook(base_urls: Sequence[str]) -> Callable[[RewritableRequest], None]:
    """An ``httpx.Client`` request event hook that applies :func:`with_rollout_prefix` to every request."""
    if isinstance(base_urls, str):
        raise TypeError("base_urls must be a sequence of base URLs, not a single string")

    def hook(request: RewritableRequest) -> None:
        # A request object sent again (under another rollout, or none) is rewritten from the URL it was built with,
        # so an earlier send's prefix is never mistaken for an explicit one. Any other URL, such as a redirect
        # target (httpx copies extensions onto it), starts afresh.
        current = str(request.url)
        extensions = getattr(request, "extensions", None)
        if extensions is None:
            request.url = type(request.url)(with_rollout_prefix(current, base_urls))
            return
        original, last_written = extensions.get(_ORIGINAL_URL_EXTENSION, (current, None))
        if current != last_written:
            original = current
        rewritten = with_rollout_prefix(original, base_urls)
        extensions[_ORIGINAL_URL_EXTENSION] = (original, rewritten)
        request.url = type(request.url)(rewritten)

    return hook


def async_rollout_prefix_hook(base_urls: Sequence[str]) -> Callable[[RewritableRequest], Awaitable[None]]:
    """An ``httpx.AsyncClient`` request event hook that applies :func:`with_rollout_prefix` to every request."""
    hook = rollout_prefix_hook(base_urls)

    async def async_hook(request: RewritableRequest) -> None:
        hook(request)

    return async_hook


# Names that moved to other modules stay importable from here until their removal release; see nemo_gym/_moved_paths.py.
__getattr__ = moved_names_getattr(__name__)
