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
import errno as errno_module
import json
import resource
import socket
import sys
import time
from abc import abstractmethod
from contextlib import asynccontextmanager
from enum import Enum
from logging import Filter as LoggingFilter
from logging import LogRecord, getLogger
from os import environ, getenv
from pathlib import Path
from threading import Thread
from traceback import format_exc, print_exc
from typing import Any, Dict, List, Literal, Optional, TextIO, Tuple, Type, Union, Unpack
from urllib.parse import urlsplit
from uuid import uuid4

import orjson
import ray
import requests
import uvicorn
from aiohttp import (
    ClientConnectionResetError,
    ClientConnectorError,
    ClientError,
    ClientOSError,
    ClientResponse,
    ClientResponseError,
    ClientSession,
    ClientTimeout,
    DummyCookieJar,
    InvalidURL,
    ServerDisconnectedError,
    ServerTimeoutError,
    TCPConnector,
)
from aiohttp.client import _RequestOptions
from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import BaseModel, ConfigDict, Field
from requests.exceptions import ConnectionError
from starlette.middleware.sessions import SessionMiddleware

from nemo_gym import WORKING_DIR
from nemo_gym.config_types import (
    ROLLOUT_PATH_PREFIX,
    BaseRunServerInstanceConfig,
    BaseServerConfig,
)
from nemo_gym.global_config import (
    DRY_RUN_KEY_NAME,
    HEAD_SERVER_KEY_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    OBSERVABILITY_ENABLED_KEY_NAME,
    RAY_HEAD_NODE_ADDRESS_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.profiling import Profiler
from nemo_gym.rollout_correlation import current_rollout_id, maybe_rollout_id_from_run_body


_GLOBAL_AIOHTTP_CLIENT: Union[None, ClientSession] = None
_GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG: bool = False
# `ClientTimeout` is per session, so the session carries the external deadline and
# `ServerClient.request` passes this one explicitly for server-to-server calls.
_GLOBAL_AIOHTTP_INTERNAL_SOCK_CONNECT_TIMEOUT_SECONDS: float = 10.0


class GlobalAIOHTTPAsyncClientConfig(BaseModel):
    global_aiohttp_connector_limit: int = 100 * 1024
    global_aiohttp_connector_limit_per_host: int = 1024

    global_aiohttp_client_request_debug: bool = False

    global_aiohttp_tcp_keepalive_idle_seconds: int = Field(
        default=60,
        description=("TCP_KEEPIDLE: seconds a socket must be idle before the kernel starts sending keepalive probes."),
    )
    global_aiohttp_tcp_keepalive_interval_seconds: int = Field(
        default=10,
        description=("TCP_KEEPINTVL: seconds between successive keepalive probes."),
    )
    global_aiohttp_tcp_keepalive_probes: int = Field(
        default=3,
        description=("TCP_KEEPCNT: number of unanswered probes before the kernel drops the connection."),
    )

    global_aiohttp_sock_connect_timeout_seconds: float = Field(
        default=30.0,
        description=(
            "Deadline for opening a new socket to an external endpoint, in seconds. Bounds a host "
            "that silently drops connection attempts rather than refusing them, which the kernel "
            "otherwise abandons only after roughly two minutes. This is aiohttp's own default for "
            "`sock_connect`. It covers the socket connect alone, not waiting for a pooled "
            "connection, so a saturated pool is never failed by it."
        ),
    )
    global_aiohttp_internal_sock_connect_timeout_seconds: float = Field(
        default=10.0,
        description=(
            "The same deadline for calls to other NeMo Gym servers, which are near or on the same "
            "machine. Not lower than this: under load a localhost accept queue can fill, the kernel "
            "drops the SYN, and the retransmit does not arrive for a full second."
        ),
    )


def get_global_aiohttp_client(
    global_config_dict_parser_config: Optional[GlobalConfigDictParserConfig] = None,
    global_config_dict_parser_cls: Type[GlobalConfigDictParser] = GlobalConfigDictParser,
) -> ClientSession:  # pragma: no cover
    global _GLOBAL_AIOHTTP_CLIENT

    if _GLOBAL_AIOHTTP_CLIENT is not None:
        return _GLOBAL_AIOHTTP_CLIENT

    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=global_config_dict_parser_config,
        global_config_dict_parser_cls=global_config_dict_parser_cls,
    )
    cfg = GlobalAIOHTTPAsyncClientConfig.model_validate(global_config_dict)

    return set_global_aiohttp_client(cfg)


def _make_keepalive_socket_factory(
    idle_seconds: int,
    interval_seconds: int,
    probes: int,
):
    def factory(addr_info) -> socket.socket:
        family, type_, proto, _canonname, _sockaddr = addr_info
        sock = socket.socket(family=family, type=type_, proto=proto)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt_name, opt_value in (
            ("TCP_KEEPIDLE", idle_seconds),
            ("TCP_KEEPINTVL", interval_seconds),
            ("TCP_KEEPCNT", probes),
        ):
            opt = getattr(socket, opt_name, None)
            if opt is not None:
                sock.setsockopt(socket.IPPROTO_TCP, opt, opt_value)
        return sock

    return factory


def set_global_aiohttp_client(cfg: GlobalAIOHTTPAsyncClientConfig) -> ClientSession:  # pragma: no cover
    assert not is_global_aiohttp_client_setup(), (
        "There is already a global aiohttp client setup. Please refactor your code or call `global_aiohttp_client_exit` if you want to explicitly re-make the client!"
    )

    num_workers = get_nemo_gym_fastapi_num_workers()
    client_session = ClientSession(
        connector=TCPConnector(
            limit=cfg.global_aiohttp_connector_limit // num_workers,
            limit_per_host=cfg.global_aiohttp_connector_limit_per_host // num_workers,
            keepalive_timeout=15.0,
            socket_factory=_make_keepalive_socket_factory(
                idle_seconds=cfg.global_aiohttp_tcp_keepalive_idle_seconds,
                interval_seconds=cfg.global_aiohttp_tcp_keepalive_interval_seconds,
                probes=cfg.global_aiohttp_tcp_keepalive_probes,
            ),
        ),
        # `total` stays unset: generations legitimately run for minutes and must not be cut off.
        # Only `sock_connect` is set, so a blackholed connect fails on our deadline rather than the
        # kernel's SYN retry budget. Deliberately not `connect`, which also covers waiting for a
        # free pooled connection; at Gym's concurrency that wait is normal and not a failure.
        timeout=ClientTimeout(sock_connect=cfg.global_aiohttp_sock_connect_timeout_seconds),
        cookie_jar=DummyCookieJar(),
    )

    global _GLOBAL_AIOHTTP_CLIENT
    _GLOBAL_AIOHTTP_CLIENT = client_session

    global _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG
    _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG = cfg.global_aiohttp_client_request_debug

    global _GLOBAL_AIOHTTP_INTERNAL_SOCK_CONNECT_TIMEOUT_SECONDS
    _GLOBAL_AIOHTTP_INTERNAL_SOCK_CONNECT_TIMEOUT_SECONDS = cfg.global_aiohttp_internal_sock_connect_timeout_seconds

    return _GLOBAL_AIOHTTP_CLIENT


def is_global_aiohttp_client_setup() -> bool:  # pragma: no cover
    return _GLOBAL_AIOHTTP_CLIENT is not None


def is_global_aiohttp_client_request_debug_enabled() -> bool:
    return _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG


def internal_client_timeout() -> ClientTimeout:
    """The per-call timeout for a request to another NeMo Gym server.

    Only `sock_connect` is set. Everything else stays unset, so a `/run` that legitimately takes an
    hour is not cut off, and a request queued behind a saturated connection pool is not failed.
    """
    return ClientTimeout(sock_connect=_GLOBAL_AIOHTTP_INTERNAL_SOCK_CONNECT_TIMEOUT_SECONDS)


def global_aiohttp_client_exit():  # pragma: no cover
    if not is_global_aiohttp_client_setup():
        return

    global _GLOBAL_AIOHTTP_CLIENT
    asyncio.run(_GLOBAL_AIOHTTP_CLIENT.close())

    _GLOBAL_AIOHTTP_CLIENT = None


atexit.register(global_aiohttp_client_exit)


# This is not intended to be changed. If you want to increase this, we should probably figure out how to improve server-side robustness.
MAX_NUM_TRIES = 3


class _RequestFailureKind(str, Enum):
    """The categories `_classify_request_failure` sorts a failed `request()` call into.

    Private to this module and specific to the `request()` path. This is not a general error
    taxonomy for NeMo Gym.
    """

    # The connection was never established: `ClientConnectorError` and its DNS and proxy
    # subclasses, for a refused connect, an unreachable host or network, or a failed lookup.
    UNREACHABLE = "unreachable"
    # A local resource ran out while connecting: EMFILE, ENFILE, ENOBUFS or ENOMEM, read from
    # `ClientConnectorError.os_error`.
    RESOURCE = "resource"
    # A connection was established and then lost: `ServerDisconnectedError`,
    # `ClientConnectionResetError`, a bare `ClientOSError`, or an unrecognised `ClientError`.
    PEER_DROP = "peer_drop"
    # A deadline expired: `ServerTimeoutError` and its connect and read subclasses, or
    # `asyncio.TimeoutError`.
    TIMEOUT = "timeout"
    # A malformed URL (`InvalidURL`) or anything that is not a `ClientError` at all, which means a
    # programming error rather than a network one.
    FATAL = "fatal"


_RESOURCE_ERRNOS = frozenset({errno_module.EMFILE, errno_module.ENFILE, errno_module.ENOBUFS, errno_module.ENOMEM})


def _classify_request_failure(exc: BaseException) -> _RequestFailureKind:
    """Sort an exception from `client.request()` into a `_RequestFailureKind`.

    Every exception maps to a kind; there is no "unknown". Two fallbacks make that true. An
    `aiohttp.ClientError` this function does not recognise is treated as `PEER_DROP`, so a new
    exception type in a future aiohttp is retried and reported rather than raised immediately.
    Anything that is not a `ClientError` at all is `FATAL`, on the grounds that it is a programming
    error rather than a network one.

    Order matters and breaking it is silent. `ClientConnectorError` subclasses `ClientOSError`, so
    testing the parent first would swallow every refused connection into `PEER_DROP`, which is the
    bug this classification exists to fix. `InvalidURL` is both a `ClientError` and a `ValueError`,
    so it has to be tested before either.
    """
    if isinstance(exc, InvalidURL):
        return _RequestFailureKind.FATAL
    if isinstance(exc, ClientConnectorError):
        # Includes the DNS and proxy subclasses. Read `.os_error` rather than `.errno`: a bare
        # `ClientOSError` has neither, and this runs inside an `except` block where an
        # `AttributeError` would turn a retryable failure into a crash.
        os_error = getattr(exc, "os_error", None)
        if os_error is not None and getattr(os_error, "errno", None) in _RESOURCE_ERRNOS:
            return _RequestFailureKind.RESOURCE
        return _RequestFailureKind.UNREACHABLE
    if isinstance(exc, (ServerTimeoutError, asyncio.TimeoutError)):
        return _RequestFailureKind.TIMEOUT
    if isinstance(exc, (ServerDisconnectedError, ClientConnectionResetError, ClientOSError)):
        return _RequestFailureKind.PEER_DROP
    if isinstance(exc, ClientError):
        # An aiohttp error we do not recognise. Retried rather than failed fast, so a new
        # exception type in a future aiohttp is reported after retries instead of raised at once.
        return _RequestFailureKind.PEER_DROP
    return _RequestFailureKind.FATAL


_UNREACHABLE_HELP_TEXT = """Nothing accepted a connection at that address, so this is a configuration or startup problem rather than load.
1. Check the URL. For the policy model that is `policy_base_url` in `env.yaml`.
    - Verify with: curl <base_url>/models
    - A `localhost` URL means you serve the model yourself (e.g. vLLM) before collecting rollouts.
2. If the endpoint is still starting up, this resolves on its own once it is serving."""

_RESOURCE_HELP_TEXT = """Too many open connections and not enough sockets due to the file descriptor limit being hit.
    - Increase ulimit.
    - Bash example: https://github.com/NVIDIA-NeMo/RL/blob/de55be7777bbf034c04e41c40382c44725e8aa4b/ray.sub#L81
    - Python example: https://github.com/NVIDIA-NeMo/Gym/blob/c74ffddb3d8190cd717508b0830916b19a26e6cd/nemo_gym/server_utils.py#L495"""

_PEER_DROP_HELP_TEXT = """Depending on the serving framework and config, the server may be overloaded and is dropping connections.
    - Increase adapter server replicas.
    - If this is a file descriptor problem rather than load, it surfaces separately as `resource`."""

_TIMEOUT_HELP_TEXT = """The endpoint accepted the connection but did not answer inside the deadline.
    - Check whether the endpoint is saturated, or raise the timeout for this call."""

_FAILURE_KIND_HELP_TEXT = {
    _RequestFailureKind.UNREACHABLE: _UNREACHABLE_HELP_TEXT,
    _RequestFailureKind.RESOURCE: _RESOURCE_HELP_TEXT,
    _RequestFailureKind.PEER_DROP: _PEER_DROP_HELP_TEXT,
    _RequestFailureKind.TIMEOUT: _TIMEOUT_HELP_TEXT,
    _RequestFailureKind.FATAL: "",
}

# Reporting on the first failure is right for one request and wrong for sixteen thousand, so the
# full diagnostic goes out once per endpoint and kind and is then summarised on this interval.
# Keyed by (origin, kind) -> (when it was last reported, failures suppressed since). Grows with the
# number of distinct endpoints a process talks to, not with the number of requests.
_FAILURE_REPORT_INTERVAL_SECONDS: float = 30.0
_FAILURE_REPORT_STATE: Dict[Tuple[str, _RequestFailureKind], Tuple[float, int]] = {}


def _request_origin(url: str) -> str:
    """`scheme://host:port` for `url`. Failures are grouped and reported per origin."""
    split = urlsplit(url)
    return f"{split.scheme}://{split.netloc}" if split.netloc else url


def _next_failure_report(origin: str, kind: _RequestFailureKind, now: float) -> Optional[int]:
    """Whether to report this failure, and what to say.

    Returns 0 the first time an (origin, kind) pair is seen, meaning print the full diagnostic; a
    positive count once the interval has passed, meaning report that many suppressed failures; and
    `None` in between, meaning print nothing.
    """
    state = _FAILURE_REPORT_STATE.get((origin, kind))
    if state is None:
        _FAILURE_REPORT_STATE[(origin, kind)] = (now, 0)
        return 0

    last_reported, suppressed = state
    suppressed += 1
    if now - last_reported < _FAILURE_REPORT_INTERVAL_SECONDS:
        _FAILURE_REPORT_STATE[(origin, kind)] = (last_reported, suppressed)
        return None

    _FAILURE_REPORT_STATE[(origin, kind)] = (now, 0)
    return suppressed


def _report_request_failure(url: str, exc: BaseException, kind: _RequestFailureKind, now: float) -> None:
    """Print the diagnostic for a failed request, at most once per endpoint and kind per interval."""
    origin = _request_origin(url)
    suppressed = _next_failure_report(origin, kind, now)
    if suppressed is None:
        return

    if suppressed:
        print(
            f"[request_retry url={origin} kind={kind.value}] "
            f"{suppressed} more `{type(exc).__name__}` failures in the last "
            f"{_FAILURE_REPORT_INTERVAL_SECONDS:.0f}s. Still retrying.",
            flush=True,
        )
        return

    help_text = _FAILURE_KIND_HELP_TEXT[kind]
    print(
        f"[request_retry url={url} kind={kind.value} error={type(exc).__name__}] {exc}"
        + (f"\n{help_text}" if help_text else ""),
        flush=True,
    )


async def request(
    method: str, url: str, _internal: bool = False, **kwargs: Unpack[_RequestOptions]
) -> ClientResponse:  # pragma: no cover
    # Faster JSON dumps than the default aiohttp json
    if kwargs.get("json"):
        kwargs["data"] = orjson.dumps(kwargs.pop("json"))
        kwargs.setdefault("headers", dict())
        kwargs["headers"]["Content-Type"] = "application/json"

    client = get_global_aiohttp_client()
    num_tries = 1
    while True:
        try:
            return await client.request(method=method, url=url, **kwargs)
        except (ServerDisconnectedError, ClientOSError) as e:
            # Both were already retried without a limit, and still are; the kind only selects the
            # diagnostic. `ClientConnectorError` arrives here as a `ClientOSError` subclass, which
            # is why a refused connection used to be reported as an overloaded server.
            _report_request_failure(url, e, _classify_request_failure(e), time.monotonic())

            await asyncio.sleep(0.5)
        except Exception as e:
            if _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG:
                print_exc()

            # Don't increment internal since we know we are ok. If we are not, the head server will shut everything down anyways.
            if not _internal:
                kind = _classify_request_failure(e)
                help_text = _FAILURE_KIND_HELP_TEXT[kind]
                print(
                    f"""Hit an exception while making a request (try {num_tries}, kind {kind.value}): {type(e)}: {e}
Sleeping 0.5s and retrying...
"""
                    + (f"{help_text}\n" if help_text else "")
                )
                if num_tries >= MAX_NUM_TRIES:
                    raise e

                num_tries += 1

            await asyncio.sleep(0.5)


async def raise_for_status(response: ClientResponse) -> None:  # pragma: no cover
    if not response.ok:
        content = await response.content.read()
        if _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG:
            print(f"""Request info: {response.request_info}
Response content: {content}""")

        try:
            response.raise_for_status()
        except ClientResponseError as e:
            # Set the response content here so we have access to it down the line.
            e.response_content = content
            raise e


async def get_response_json(response: ClientResponse) -> Any:
    return orjson.loads(await response.read())


DEFAULT_HEAD_SERVER_PORT = 11000

ServerStatus = Union[Literal["success"], Literal["connection_error"], Literal["timeout"], Literal["unknown_error"]]


class ServerClient(BaseModel):
    head_server_config: BaseServerConfig

    global_config_dict: DictConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def load_head_server_config(cls) -> BaseServerConfig:
        global_config_dict = get_global_config_dict()
        server_config_dict = global_config_dict[HEAD_SERVER_KEY_NAME]
        config = BaseServerConfig.model_validate(server_config_dict)
        return config

    @classmethod
    def load_from_global_config(cls, head_server_config: Optional[BaseServerConfig] = None) -> "ServerClient":
        if head_server_config is None:
            head_server_config = cls.load_head_server_config()

        # It's critical we use requests here instead of the global httpx client since a FastAPI server may be run downstream of this function call.
        head_server_url = f"http://{head_server_config.host}:{head_server_config.port}"
        try:
            response = requests.get(
                f"{head_server_url}/global_config_dict_yaml",
            )
        except ConnectionError as e:
            raise ValueError(
                f"Could not connect to the head server at {head_server_url}. Perhaps you are not running a server or your head server is on a different port?"
            ) from e

        global_config_dict_yaml = response.content.decode()
        global_config_dict = OmegaConf.create(json.loads(global_config_dict_yaml))

        return cls(head_server_config=head_server_config, global_config_dict=global_config_dict)

    def _build_server_base_url(self, server_config_dict: OmegaConf) -> str:
        return f"http://{server_config_dict.host}:{server_config_dict.port}"

    async def request(
        self, server_name: str, url_path: str, method: str, **kwargs: Unpack[_RequestOptions]
    ) -> ClientResponse:
        server_config_dict = get_first_server_config_dict(self.global_config_dict, server_name)
        base_url = self._build_server_base_url(server_config_dict)

        json_obj = kwargs.get("json")
        if "json" in kwargs:
            if isinstance(json_obj, BaseModel):
                json_obj = json_obj.model_dump(exclude_unset=True)
                kwargs["json"] = json_obj

        observability_enabled = self.global_config_dict.get(OBSERVABILITY_ENABLED_KEY_NAME, False)
        server_entry = self.global_config_dict.get(server_name)
        rollout_id = current_rollout_id()
        if observability_enabled and server_entry is not None and "resources_servers" in server_entry:
            if url_path == "/verify":
                rollout_id = rollout_id or maybe_rollout_id_from_run_body(json_obj)
            if rollout_id is not None and not url_path.startswith(f"/{ROLLOUT_PATH_PREFIX}/"):
                url_path = f"{rollout_path_prefix(rollout_id)}{url_path}"

        if (
            rollout_id is not None
            and observability_enabled
            and server_entry is not None
            and "responses_api_models" in server_entry
            and url_path.partition("?")[0] in {"/v1/responses", "/v1/chat/completions", "/v1/messages"}
            and not url_path.startswith(f"/{ROLLOUT_PATH_PREFIX}/")
        ):
            url_path = f"{rollout_path_prefix(rollout_id)}{url_path}"

        kwargs.setdefault("timeout", internal_client_timeout())
        return await request(method=method, url=f"{base_url}{url_path}", _internal=True, **kwargs)

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

    def poll_for_status(self, server_name: str) -> ServerStatus:  # pragma: no cover
        if server_name == HEAD_SERVER_KEY_NAME:
            server_config_dict = self.global_config_dict[HEAD_SERVER_KEY_NAME]
        else:
            server_config_dict = get_first_server_config_dict(self.global_config_dict, server_name)

        try:
            requests.get(self._build_server_base_url(server_config_dict), timeout=5)
            # We don't check the status code since there may not be a route at /
            return "success"
        except requests.exceptions.ConnectionError:
            return "connection_error"
        except requests.exceptions.Timeout:
            return "timeout"
        except Exception:
            return "unknown_error"


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


_NEMO_GYM_STARTED_RAY_CLUSTER: bool = False


def initialize_ray() -> None:
    """
    Initialize ray cluster in a process.
    We store the Ray address in the global config dict so that child processes can connect to it.
    This avoids the need to start a new Ray cluster in each child process.
    Note: This function will modify the global config dict - update `ray_head_node_address`
    """

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
        global _NEMO_GYM_STARTED_RAY_CLUSTER
        _NEMO_GYM_STARTED_RAY_CLUSTER = True

    ray.init(**ray_init_kwargs)

    if not ray_head_node_address:
        with open_dict(global_config_dict):
            global_config_dict["ray_head_node_address"] = ray.get_runtime_context().gcs_address
        print(f"Started Ray cluster at {global_config_dict['ray_head_node_address']}")


def maybe_ray_cluster_exit():  # pragma: no cover
    global _NEMO_GYM_STARTED_RAY_CLUSTER

    if not _NEMO_GYM_STARTED_RAY_CLUSTER:
        return

    print("Shutting down Ray cluster spun up by NeMo Gym...")
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


class SimpleServer(BaseServer):
    server_client: ServerClient

    @abstractmethod
    def setup_webserver(self) -> FastAPI:
        pass

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
            request.session[SESSION_ID_KEY] = request.session.get(SESSION_ID_KEY, str(uuid4()))

            response: Response = await call_next(request)
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
                if _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG:
                    print(response_content)

                return JSONResponse(content=response_content, status_code=500)
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
        global_config_dict = get_global_config_dict()

        initialize_ray()

        is_main_fastapi_proc = not is_nemo_gym_fastapi_worker()

        server_config = cls.load_config_from_global_config()
        server_client = ServerClient(
            head_server_config=ServerClient.load_head_server_config(),
            global_config_dict=global_config_dict,
        )
        server = cls(config=server_config, server_client=server_client)

        if global_config_dict[DRY_RUN_KEY_NAME]:
            return

        app = server.setup_webserver()
        # After the app is fully built so subclass routes are present. Only resources servers expose tools over MCP,
        # so gating the lazy import on their config keeps the MCP SDK out of agent/model processes that never need it.
        if getattr(getattr(server, "config", None), "expose_tools_over_mcp", False):
            from nemo_gym.mcp_auto_exposure import maybe_auto_expose

            maybe_auto_expose(server, app)
        server.setup_liveness(app)
        server.set_ulimit()
        server.prefix_server_logs()
        server.setup_exception_middleware(app)

        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc):
            print(
                f"""Hit validation exception! Errors: {json.dumps(exc.errors(), indent=4)}
Full body: {json.dumps(exc.body, indent=4)}
"""
            )
            return await request_validation_exception_handler(request, exc)

        profiling_config = ProfilingMiddlewareConfig.model_validate(global_config_dict)
        if profiling_config.profiling_enabled:
            server.setup_profiling(app, profiling_config)

        uvicorn_logging_cfg = UvicornLoggingConfig.model_validate(global_config_dict)
        if not uvicorn_logging_cfg.uvicorn_logging_show_200_ok:

            class No200Filter(LoggingFilter):
                def filter(self, record: LogRecord) -> bool:
                    msg = record.getMessage()
                    return not msg.strip().endswith("200")

            uvicorn_logger = getLogger("uvicorn.access")
            uvicorn_logger.addFilter(No200Filter())

            if is_main_fastapi_proc:
                print(
                    "Adding a uvicorn logging filter so that the logs aren't spammed with 200 OK messages. This is to help errors pop up better and filter out noise."
                )

        uvicorn_kwargs = dict(
            host=server.config.host,
            port=server.config.port,
            # We add a very small graceful shutdown timeout so when we shutdown we cancel all inflight requests and there are no lingering requests (requests are cancelled)
            timeout_graceful_shutdown=0.5,
            # Some workers may take a while for imports and setup_webserver.
            timeout_worker_healthcheck=30,
            # Ensure server keepalive > client keepalive
            timeout_keep_alive=30,
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

            uvicorn_kwargs["app"] = f"{module_import_str}:app"
            uvicorn_kwargs["workers"] = server.config.num_workers
        else:
            uvicorn_kwargs["app"] = app

        if is_main_fastapi_proc:
            uvicorn.run(**uvicorn_kwargs)

        return app


class HeadServer(BaseServer):
    config: BaseServerConfig
    _server_instances: List[dict] = []

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_liveness(app)
        app.get("/global_config_dict_yaml")(self.global_config_dict_yaml)
        app.get("/server_instances")(self.get_server_instances)

        return app

    def get_server_instances(self) -> List[dict]:
        return self._server_instances

    def set_server_instances(self, instances: List) -> None:
        self._server_instances = instances

    @classmethod
    def run_webserver(cls) -> Tuple[uvicorn.Server, Thread, "HeadServer"]:  # pragma: no cover
        config = ServerClient.load_head_server_config()
        server = cls(config=config)

        app = server.setup_webserver()

        config = uvicorn.Config(
            app,
            host=server.config.host,
            port=server.config.port,
        )
        uvicorn_server = uvicorn.Server(config=config)

        thread = Thread(target=uvicorn_server.run, daemon=True)
        thread.start()

        return uvicorn_server, thread, server

    async def global_config_dict_yaml(self) -> str:
        return OmegaConf.to_yaml(get_global_config_dict())


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


def rollout_path_prefix(rollout_id: Optional[str]) -> str:
    """Return the leading model-server path prefix for a rollout, if available."""
    return f"/{ROLLOUT_PATH_PREFIX}/{rollout_id}" if rollout_id else ""


def apply_rollout_prefix(base_url: str, rollout_id: Optional[str]) -> str:
    """Append a rollout prefix to a model-server root URL."""
    if not rollout_id:
        return base_url
    return base_url.rstrip("/") + rollout_path_prefix(rollout_id)


def setup_server_client(head_server_config: Optional[BaseServerConfig] = None) -> ServerClient:  # pragma: no cover
    server_client = ServerClient.load_from_global_config(head_server_config)

    # We set this rollout global aiohttp client to use the same max connections as the underlying head server global config.
    if not is_global_aiohttp_client_setup():
        set_global_aiohttp_client(cfg=GlobalAIOHTTPAsyncClientConfig.model_validate(server_client.global_config_dict))

    return server_client
