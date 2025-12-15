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
import resource
from abc import abstractmethod
from contextlib import asynccontextmanager
from io import StringIO
from logging import Filter as LoggingFilter
from logging import LogRecord, getLogger
from os import getenv
from pathlib import Path
from threading import Thread
from traceback import print_exc
from typing import Literal, Optional, Tuple, Type, Union, Unpack
from uuid import uuid4

import ray
import requests
import uvicorn
import yappi
from aiohttp import (
    ClientResponse,
    ClientResponseError,
    ClientSession,
    ClientTimeout,
    DummyCookieJar,
    ServerDisconnectedError,
    TCPConnector,
)
from aiohttp.client import _RequestOptions
from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import BaseModel, ConfigDict
from requests.exceptions import ConnectionError
from starlette.middleware.sessions import SessionMiddleware

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import (
    BaseRunServerInstanceConfig,
    BaseServerConfig,
)
from nemo_gym.global_config import (
    HEAD_SERVER_KEY_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
    get_global_config_dict,
)


_GLOBAL_AIOHTTP_CLIENT: Union[None, ClientSession] = None
_GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG: bool = False


class GlobalAIOHTTPAsyncClientConfig(BaseModel):
    global_aiohttp_connector_limit: int = 100 * 1024
    global_aiohttp_connector_limit_per_host: int = 1024

    global_aiohttp_client_request_debug: bool = False


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


def set_global_aiohttp_client(cfg: GlobalAIOHTTPAsyncClientConfig) -> ClientSession:  # pragma: no cover
    assert not is_global_aiohttp_client_setup(), (
        "There is already a global aiohttp client setup. Please refactor your code or call `global_aiohttp_client_exit` if you want to explicitly re-make the client!"
    )

    client_session = ClientSession(
        connector=TCPConnector(
            limit=cfg.global_aiohttp_connector_limit,
            limit_per_host=cfg.global_aiohttp_connector_limit_per_host,
        ),
        timeout=ClientTimeout(),
        cookie_jar=DummyCookieJar(),
    )

    global _GLOBAL_AIOHTTP_CLIENT
    _GLOBAL_AIOHTTP_CLIENT = client_session

    global _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG
    _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG = cfg.global_aiohttp_client_request_debug

    return _GLOBAL_AIOHTTP_CLIENT


def is_global_aiohttp_client_setup() -> bool:  # pragma: no cover
    return _GLOBAL_AIOHTTP_CLIENT is not None


def global_aiohttp_client_exit():  # pragma: no cover
    if not is_global_aiohttp_client_setup():
        return

    global _GLOBAL_AIOHTTP_CLIENT
    asyncio.run(_GLOBAL_AIOHTTP_CLIENT.close())

    _GLOBAL_AIOHTTP_CLIENT = None


atexit.register(global_aiohttp_client_exit)


# This is not intended to be changed. If you want to increase this, we should probably figure out how to improve server-side robustness.
MAX_NUM_TRIES = 3


async def request(
    method: str, url: str, _internal: bool = False, **kwargs: Unpack[_RequestOptions]
) -> ClientResponse:  # pragma: no cover
    client = get_global_aiohttp_client()
    num_tries = 1
    while True:
        try:
            return await client.request(method=method, url=url, **kwargs)
        except ServerDisconnectedError:
            await asyncio.sleep(0.5)
        except Exception as e:
            if _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG:
                print_exc()

            # Don't increment internal since we know we are ok. If we are not, the head server will shut everything down anyways.
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


async def raise_for_status(response: ClientResponse) -> None:  # pragma: no cover
    if not response.ok:
        content = await response.content.read()
        print(f"""Request info: {response.request_info}
Response content: {content}""")

        try:
            response.raise_for_status()
        except ClientResponseError as e:
            # Set the response content here so we have access to it down the line.
            e.response_content = content
            raise e


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

        if "json" in kwargs:
            json_obj = kwargs["json"]
            if isinstance(json_obj, BaseModel):
                kwargs["json"] = json_obj.model_dump(exclude_unset=True)

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


class ProfilingMiddlewareInputConfig(BaseModel):
    # Relative to the Gym root dir.
    profiling_results_dirpath: Optional[str] = None


class ProfilingMiddlewareConfig(ProfilingMiddlewareInputConfig):
    profiling_enabled: bool = False


class UvicornLoggingConfig(BaseModel):
    # Default to False for regular use cases.
    uvicorn_logging_show_200_ok: bool = False


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
    ray_head_node_address = global_config_dict.get("ray_head_node_address")
    ray_init_kwargs = dict(ignore_reinit_error=True)

    if ray_head_node_address:
        print(f"Connecting to Ray cluster at specified address: {ray_head_node_address}")
        ray_init_kwargs["address"] = ray_head_node_address
    else:
        print("Starting Ray cluster...")

    ray.init(**ray_init_kwargs)

    if not ray_head_node_address:
        with open_dict(global_config_dict):
            global_config_dict["ray_head_node_address"] = ray.get_runtime_context().gcs_address
        print(f"Started Ray cluster at {global_config_dict['ray_head_node_address']}")


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
        # The multiple middleware execution order described in https://fastapi.tiangolo.com/tutorial/middleware/#multiple-middleware-execution-order
        # Says that if you register middlewares A and then B,
        # - at request time: They execute B first then A
        # - at response time: They return to A first and then B
        # So for adding session IDs, that middleware must run after SessionMiddleware, so it must be registered before it.

        @app.middleware("http")
        async def add_session_id(request: Request, call_next):  # pragma: no cover
            # If session_id not present, assign one
            if SESSION_ID_KEY not in request.session:
                request.session[SESSION_ID_KEY] = str(uuid4())

            response: Response = await call_next(request)
            return response

        session_middleware_key = self.get_session_middleware_key()
        app.add_middleware(SessionMiddleware, secret_key=session_middleware_key, session_cookie=session_middleware_key)

    def setup_exception_middleware(self, app: FastAPI) -> None:  # pragma: no cover
        @app.middleware("http")
        async def exception_handling_middleware(request: Request, call_next):
            try:
                return await call_next(request)
            except Exception as e:
                print_exc()
                print(
                    f"🚨 Caught an exception printed above in {self.config.name} ({self.__class__.__name__}). If you expect this to be fed back into this model, the exception repr i.e. `repr(e)` is returned to the model. However, please make sure this exception is caught in your server and returned to the model as appropriate. See https://fastapi.tiangolo.com/tutorial/handling-errors/#use-httpexception"
                )
                return JSONResponse(content=repr(e), status_code=500)
            except:
                print_exc()
                print(
                    f"🚨 Caught an unknown exception printed above in {self.config.name} ({self.__class__.__name__}). If you expect this to be fed back into this model, nothing meaningful is returned to the model. Please make sure this exception is caught in your server and returned to the model as appropriate. See https://fastapi.tiangolo.com/tutorial/handling-errors/#use-httpexception"
                )
                return JSONResponse(content="An unknown error occurred", status_code=500)

    def setup_profiling(self, app: FastAPI, profiling_config: ProfilingMiddlewareConfig) -> None:  # pragma: no cover
        base_profile_dir = Path(PARENT_DIR) / profiling_config.profiling_results_dirpath
        server_profile_path = (base_profile_dir / self.get_session_middleware_key()).with_suffix(".log")

        base_profile_dir.mkdir(parents=True, exist_ok=True)

        main_app_lifespan = app.router.lifespan_context

        def _dump_yappi_stats() -> str:
            buffer = StringIO()
            yappi.get_func_stats().print_all(
                out=buffer,
                columns={
                    0: ("name", 200),
                    1: ("ncall", 10),
                    2: ("tsub", 8),
                    3: ("ttot", 8),
                    4: ("tavg", 8),
                },
            )

            buffer.seek(0)
            res = ""
            past_header = False
            for line in buffer:
                if not past_header or self.config.entrypoint in line:
                    res += line

                if line.startswith("name"):
                    past_header = True

            return res

        @asynccontextmanager
        async def lifespan_wrapper(app):
            yappi.set_clock_type("CPU")
            yappi.start()
            print(f"🔍 Enabled profiling for {self.config.name}")

            async with main_app_lifespan(app) as maybe_state:
                yield maybe_state

            print(f"🛑 Stopping profiler for {self.config.name}. Check {server_profile_path} for the metrics!")
            yappi.stop()

            with open(server_profile_path, "w") as f:
                f.write(_dump_yappi_stats())

        app.router.lifespan_context = lifespan_wrapper

        @app.get("/stats")
        def stats():
            return Response(_dump_yappi_stats())

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

    @classmethod
    def run_webserver(cls) -> None:  # pragma: no cover
        global_config_dict = get_global_config_dict()

        initialize_ray()

        server_config = cls.load_config_from_global_config()
        server_client = ServerClient(
            head_server_config=ServerClient.load_head_server_config(),
            global_config_dict=global_config_dict,
        )
        server = cls(config=server_config, server_client=server_client)

        app = server.setup_webserver()
        server.set_ulimit()
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

            print(
                "Adding a uvicorn logging filter so that the logs aren't spammed with 200 OK messages. This is to help errors pop up better and filter out noise."
            )

        uvicorn.run(
            app,
            host=server.config.host,
            port=server.config.port,
            # We add a very small graceful shutdown timeout so when we shutdown we cancel all inflight requests and there are no lingering requests (requests are cancelled)
            timeout_graceful_shutdown=0.5,
        )


class HeadServer(BaseServer):
    config: BaseServerConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        app.get("/global_config_dict_yaml")(self.global_config_dict_yaml)

        return app

    @classmethod
    def run_webserver(cls) -> Tuple[uvicorn.Server, Thread]:  # pragma: no cover
        config = ServerClient.load_head_server_config()
        server = cls(config=config)

        app = server.setup_webserver()

        config = uvicorn.Config(
            app,
            host=server.config.host,
            port=server.config.port,
        )
        server = uvicorn.Server(config=config)

        thread = Thread(target=server.run, daemon=True)
        thread.start()

        return server, thread

    async def global_config_dict_yaml(self) -> str:
        return OmegaConf.to_yaml(get_global_config_dict())
