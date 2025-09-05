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
from os import getenv
from threading import Thread
from typing import Any, Optional, Type

import requests
import uvicorn
from fastapi import FastAPI
from httpx import AsyncClient, AsyncHTTPTransport, Limits, Response
from httpx._types import (
    HeaderTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestFiles,
)
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from requests.exceptions import ConnectionError

from nemo_gym.config_types import (
    BaseRunServerConfig,
    BaseServerConfig,
)
from nemo_gym.global_config import (
    HEAD_SERVER_KEY_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    get_first_server_config_dict,
    get_global_config_dict,
)


# We create a single global httpx client as recommended by https://www.python-httpx.org/async/
# ```
# In order to get the most benefit from connection pooling, make sure you're not instantiating multiple client instances - for example by using async with inside a "hot loop". This can be achieved either by having a single scoped client that's passed throughout wherever it's needed, or by having a single global client instance.
# ```
#
# In principle, we use no timeout since various api or model calls may take an indefinite amount of time. Right now, we have no timeout, even for connection errors which may be problematic. We may want to revisit more granular httpx.Timeout later on.
#
# Eventually, we may also want to parameterize the max connections. For now, we set the max connections to just some very large number.
#
# It's critical that this client is NOT used before uvicorn.run is called. Under the hood, this async client will start and use an event loop, and store a handle to that specific event loop. When uvicorn.run is called, it will replace the event loop policy with its own. So the handle that the async client has is now outdated.
GLOBAL_HTTPX_CLIENT = AsyncClient(
    limits=Limits(max_keepalive_connections=1500, max_connections=1500),
    transport=AsyncHTTPTransport(retries=3),
    timeout=None,
)


DEFAULT_HEAD_SERVER_PORT = 11000


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

    async def get(
        self,
        server_name: str,
        url_path: str,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        **kwargs,
    ) -> Response:
        """
        This function definition is directly copied from httpx._client.AsyncClient. We omit some kwargs since they are most likely not used. We omit the url arg and replace it with the `server_name` and `url_path` args below.

        Parameters
        ----------
        server_name: str
            The name of the server you are trying to call.
        url_path: str
            The URL path in the server you are trying to call e.g. "/v1/responses".

        """
        server_config_dict = get_first_server_config_dict(self.global_config_dict, server_name)
        return await GLOBAL_HTTPX_CLIENT.get(
            f"http://{server_config_dict.host}:{server_config_dict.port}{url_path}",
            params=params,
            headers=headers,
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
        **kwargs,
    ) -> Response:
        """
        This function definition is directly copied from httpx._client.AsyncClient. We omit some kwargs since they are most likely not used. We omit the url arg and replace it with the `server_name` and `url_path` args below.

        Parameters
        ----------
        server_name: str
            The name of the server you are trying to call.
        url_path: str
            The URL path in the server you are trying to call e.g. "/v1/responses".

        """
        server_config_dict = get_first_server_config_dict(self.global_config_dict, server_name)
        return await GLOBAL_HTTPX_CLIENT.post(
            f"http://{server_config_dict.host}:{server_config_dict.port}{url_path}",
            content=content,
            data=data,
            files=files,
            json=json.model_dump(exclude_unset=True) if isinstance(json, BaseModel) else json,
            params=params,
            headers=headers,
            **kwargs,
        )


class BaseServer(BaseModel):
    """
    All instances of BaseServer are queryable using ServerClient.
    """

    config: BaseRunServerConfig

    @classmethod
    def load_config_from_global_config(cls) -> "BaseRunServerConfig":
        config_path_str = getenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME)
        global_config_dict = get_global_config_dict()
        server_config_dict = get_first_server_config_dict(global_config_dict, config_path_str)

        server_config_cls: Type[BaseRunServerConfig] = cls.model_fields["config"].annotation
        server_config = server_config_cls.model_validate(server_config_dict)

        return server_config


class SimpleServer(BaseServer):
    server_client: ServerClient

    @abstractmethod
    def setup_webserver(self) -> FastAPI:
        pass

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
            # TODO eventually we want to make this FastAPI server served across multiple processes or workers.
            # Right now this will always use one process.
            # workers=server.config.num_fastapi_workers,
            # We don't have any explicit lifespan logic, so instead of defaulting to "auto"
            # We just turn lifespan off
            lifespan="off",
        )


class HeadServer(BaseServer):
    config: BaseServerConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        app.get("/global_config_dict_yaml")(self.global_config_dict_yaml)

        return app

    @classmethod
    def run_webserver(cls) -> Thread:  # pragma: no cover
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

        return thread

    async def global_config_dict_yaml(self) -> str:
        return OmegaConf.to_yaml(get_global_config_dict())
