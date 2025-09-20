# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import json
from abc import abstractmethod
from asyncio import _get_running_loop, get_event_loop
from os import getenv
from threading import Thread
from typing import Any, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import requests
import uvicorn
import uvloop
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from fastapi import FastAPI, Request, Response
from httpx import AsyncClient, Cookies, Limits, Response
from httpx._types import (
    CookieTypes,
    HeaderTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestFiles,
)
from httpx_aiohttp import AiohttpTransport
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from requests.exceptions import ConnectionError
from starlette.middleware.sessions import SessionMiddleware

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


class NeMoGymStatelessCookies(Cookies):
    def extract_cookies(self, response):
        pass


class NeMoGymGlobalAsyncClient(AsyncClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._cookies = NeMoGymStatelessCookies(self._cookies)


# We create a single global httpx client as recommended by https://www.python-httpx.org/async/
# ```
# In order to get the most benefit from connection pooling, make sure you're not instantiating multiple client instances - for example by using async with inside a "hot loop". This can be achieved either by having a single scoped client that's passed throughout wherever it's needed, or by having a single global client instance.
# ```
#
# We use no timeout since various api or model calls may take an indefinite amount of time.
_GLOBAL_HTTPX_CLIENT: Optional[NeMoGymGlobalAsyncClient] = None


class GlobalHTTPXAsyncClientConfig(BaseModel):
    global_httpx_max_connections: int = 1500
    global_httpx_max_retries: int = 0


def get_global_httpx_client(
    global_config_dict_parser_config: Optional[GlobalConfigDictParserConfig] = None,
    global_config_dict_parser_cls: Type[GlobalConfigDictParser] = GlobalConfigDictParser,
) -> NeMoGymGlobalAsyncClient:
    if _GLOBAL_HTTPX_CLIENT is not None:
        return _GLOBAL_HTTPX_CLIENT

    # Initialize the event loop which is used in aiohttp.ClientSession below
    loop = _get_running_loop()
    if loop is None:
        uvloop.install()
        loop = get_event_loop()

    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=global_config_dict_parser_config,
        global_config_dict_parser_cls=global_config_dict_parser_cls,
    )
    cfg = GlobalHTTPXAsyncClientConfig.model_validate(global_config_dict)

    limits = Limits(
        max_keepalive_connections=cfg.global_httpx_max_connections,
        max_connections=cfg.global_httpx_max_connections,
    )
    client_session = ClientSession(
        connector=TCPConnector(
            limit=limits.max_connections,
            keepalive_timeout=limits.keepalive_expiry,
            ssl=None,
            local_addr=None,
        ),
        loop=loop,
        timeout=ClientTimeout(
            total=None,
            connect=None,
            sock_connect=None,
            sock_read=None,
        ),
    )
    transport = AiohttpTransport(
        retries=cfg.global_httpx_max_retries,
        limits=limits,
        client=client_session,
    )

    client = NeMoGymGlobalAsyncClient(
        limits=limits,
        transport=transport,
        timeout=None,
    )

    global _GLOBAL_HTTPX_CLIENT
    _GLOBAL_HTTPX_CLIENT = client

    return client


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

    async def get(
        self,
        server_name: str,
        url_path: str,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        **kwargs,
    ) -> Response:
        """
        This function definition is directly copied from httpx._client.AsyncClient. We omit some kwargs since they are most likely not used. We omit the url arg and replace it with the `server_name` and `url_path` args below.

        Args:
            server_name: str
                The name of the server you are trying to call.
            url_path: str
                The URL path in the server you are trying to call e.g. "/v1/responses".

        """
        server_config_dict = get_first_server_config_dict(self.global_config_dict, server_name)
        base_url = self._build_server_base_url(server_config_dict)
        return await get_global_httpx_client(base_url).get(
            f"{base_url}{url_path}",
            params=params,
            headers=headers,
            cookies=cookies,
            **kwargs,
        )

    async def post(
        self,
        server_name: str,
        url_path: str,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: Any | BaseModel | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        **kwargs,
    ) -> Response:
        """
        This function definition is directly copied from httpx._client.AsyncClient. We omit some kwargs since they are most likely not used. We omit the url arg and replace it with the `server_name` and `url_path` args below.

        Args:
            server_name: str
                The name of the server you are trying to call.
            url_path: str
                The URL path in the server you are trying to call e.g. "/v1/responses".

        """
        server_config_dict = get_first_server_config_dict(self.global_config_dict, server_name)
        base_url = self._build_server_base_url(server_config_dict)
        return await get_global_httpx_client(base_url).post(
            f"{base_url}{url_path}",
            content=content,
            data=data,
            files=files,
            json=json.model_dump(exclude_unset=True) if isinstance(json, BaseModel) else json,
            params=params,
            headers=headers,
            cookies=cookies,
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

    @classmethod
    def run_webserver(cls) -> None:  # pragma: no cover
        server_config = cls.load_config_from_global_config()
        server_client = ServerClient(
            head_server_config=ServerClient.load_head_server_config(),
            global_config_dict=get_global_config_dict(),
        )
        server = cls(config=server_config, server_client=server_client)

        app = server.setup_webserver()

        uvicorn.run(
            app,
            host=server.config.host,
            port=server.config.port,
            # We don't have any explicit lifespan logic, so instead of defaulting to "auto"
            # We just turn lifespan off
            lifespan="off",
            # We set loop none here since a server instance requires a server_client, which will init
            loop="none",
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
