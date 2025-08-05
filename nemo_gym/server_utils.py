from typing import Any, List, Type, Literal, Optional, Union

from abc import abstractmethod

from os import getenv

from pathlib import Path

import requests

from threading import Thread

from socket import socket

import json

import hydra

from omegaconf import DictConfig, OmegaConf, open_dict

from pydantic import BaseModel, TypeAdapter, ConfigDict, ValidationError

from httpx import Limits, AsyncClient, AsyncHTTPTransport, Response
from httpx._types import (
    HeaderTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestFiles,
)

from fastapi import FastAPI

import uvicorn

from nemo_gym import PARENT_DIR


"""
We create a single global httpx client as recommended by https://www.python-httpx.org/async/
```
In order to get the most benefit from connection pooling, make sure you're not instantiating multiple client instances - for example by using async with inside a "hot loop". This can be achieved either by having a single scoped client that's passed throughout wherever it's needed, or by having a single global client instance.
```

In principle, we use no timeout since various api or model calls may take an indefinite amount of time. Right now, we have no timeout, even for connection errors which may be problematic. We may want to revisit more granular httpx.Timeout later on.

Eventually, we may also want to parameterize the max connections. For now, we set the max connections to just some very large number.

It's critical that this client is NOT used before uvicorn.run is called. Under the hood, this async client will start and use an event loop, and store a handle to that specific event loop. When uvicorn.run is called, it will replace the event loop policy with its own. So the handle that the async client has is now outdated.
"""
GLOBAL_HTTPX_CLIENT = AsyncClient(
    limits=Limits(max_keepalive_connections=1500, max_connections=1500),
    transport=AsyncHTTPTransport(retries=3),
    timeout=None,
)


_GLOBAL_CONFIG_DICT = None
NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME = "NEMO_GYM_CONFIG_DICT"
NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME = "NEMO_GYM_CONFIG_PATH"
CONFIG_PATHS_KEY_NAME = "config_paths"
ENTRYPOINT_KEY_NAME = "entrypoint"
DEFAULT_HOST_KEY_NAME = "default_host"
HEAD_SERVER_KEY_NAME = "head_server"
NEMO_GYM_RESERVED_TOP_LEVEL_KEYS = [
    CONFIG_PATHS_KEY_NAME,
    ENTRYPOINT_KEY_NAME,
    DEFAULT_HOST_KEY_NAME,
    HEAD_SERVER_KEY_NAME,
]

DEFAULT_HEAD_SERVER_PORT = 11000


def get_global_config_dict() -> DictConfig:
    """
    This function provides a handle to the global configuration dict `global_config_dict`. We try to have one source of truth for everything in NeMo gym.
    This config is resolved once and only once, immediately on a run command.

    On first initialization, the global config dict will be loaded from the following sources in order of priority (later items are higher priority):
    1. Configuration yamls specified in `config_paths` parameter.
    2. Configuration (usually sensitive values like API keys, etc) from a local `.env.yaml` file.
    3. Command line argument configuration.

    Validation is performed on the passed in configs:
    1. If a host or port is not provided for a server, defaults will be provided. Ports are resolved by the OS.
    2. If there are server reference configs, the respective server names and types will be validated against the remainder of the config.

    Then, the global config dict will be cached and reused.

    If this function is run by a child server of the main proc, that child will have been spun up with an environment variable with key NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME. The config dict will be read directly off this variable, cached, and returned with no additional validation.
    """
    global _GLOBAL_CONFIG_DICT
    if _GLOBAL_CONFIG_DICT is not None:
        return _GLOBAL_CONFIG_DICT

    nemo_gym_config_dict_str_from_env = getenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME)
    if nemo_gym_config_dict_str_from_env:
        global_config_dict = OmegaConf.create(nemo_gym_config_dict_str_from_env)

        _GLOBAL_CONFIG_DICT = global_config_dict

        return global_config_dict

    # This function is just to get the config object out of the hydra main call.
    # Need a closure. We simply use an outer ref of a list
    config_list = []

    @hydra.main(config_path=None, version_base=None)
    def inner_hydra_wrapper(cfg: DictConfig) -> DictConfig:
        config_list.append(cfg)

    inner_hydra_wrapper()

    global_config_dict: DictConfig = config_list[0]

    # Load the env.yaml config. We load it early so that people can use it to conveniently store config paths.
    dotenv_path = Path(PARENT_DIR) / "env.yaml"
    dotenv_extra_config = DictConfig({})
    if dotenv_path.exists():
        dotenv_extra_config = OmegaConf.load(dotenv_path)

    merged_config_for_config_paths = OmegaConf.merge(
        dotenv_extra_config, global_config_dict
    )
    ta = TypeAdapter(List[str])
    config_paths = merged_config_for_config_paths.get(CONFIG_PATHS_KEY_NAME) or []
    config_paths = ta.validate_python(config_paths)

    # ----- Load extra configs from config paths -----
    extra_configs: List[DictConfig] = []
    if config_paths:
        for config_path in config_paths:
            config_path = Path(config_path)
            # Assume relative to the parent dir
            if not config_path.is_absolute():
                config_path = PARENT_DIR / config_path

            extra_config = OmegaConf.load(config_path)
            extra_configs.append(extra_config)

    extra_configs.append(dotenv_extra_config)

    # Merge config dicts
    # global_config_dict is the last config arg here since we want command line args to override everything else.
    global_config_dict = OmegaConf.merge(*extra_configs, global_config_dict)

    # Get the non-reserved top level items
    non_reserved_items = [
        (key, v)
        for key, v in global_config_dict.items()
        if key not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
    ]

    # Do one pass to get the available servers and server types.
    server_refs: List[ServerRef] = []
    for server_name, server_type_config_dict in non_reserved_items:
        if not isinstance(server_type_config_dict, DictConfig):
            continue

        server_ref = is_server_ref(
            {"type": list(server_type_config_dict)[0], "name": server_name}
        )
        if server_ref:
            server_refs.append(server_ref)

    default_host = global_config_dict.get(DEFAULT_HOST_KEY_NAME) or "127.0.0.1"
    for _, server_type_config_dict in non_reserved_items:
        if not isinstance(server_type_config_dict, DictConfig):
            continue

        if not any(
            server_type in server_type_config_dict
            for server_type in (
                "responses_api_models",
                "resources_servers",
                "responses_api_agents",
            )
        ):
            continue

        server_type_config_dict: DictConfig
        for server_config_dict in server_type_config_dict.values():
            server_config_dict: DictConfig

            for server_instance_config_dict in server_config_dict.values():
                server_instance_config_dict: DictConfig

                for v in server_instance_config_dict.values():
                    maybe_server_ref = is_server_ref(v)
                    if not maybe_server_ref:
                        continue

                    assert maybe_server_ref in server_refs, (
                        f"Could not find {maybe_server_ref} in the list of available servers: {server_refs}"
                    )

                # Populate the host and port values if they are not present in the config.
                with open_dict(server_instance_config_dict):
                    if not server_instance_config_dict.get("host"):
                        server_instance_config_dict["host"] = default_host
                    if not server_instance_config_dict.get("port"):
                        server_instance_config_dict["port"] = find_open_port()

    if not global_config_dict.get(HEAD_SERVER_KEY_NAME):
        with open_dict(global_config_dict):
            global_config_dict[HEAD_SERVER_KEY_NAME] = {
                "host": default_host,
                "port": DEFAULT_HEAD_SERVER_PORT,
            }

    _GLOBAL_CONFIG_DICT = global_config_dict

    return global_config_dict


def get_first_server_config_dict(
    global_config_dict: DictConfig, top_level_path: str
) -> DictConfig:
    # Traverse three levels deep total
    server_config_dict = global_config_dict[top_level_path]
    server_config_dict = list(server_config_dict.values())[0]
    server_config_dict = list(server_config_dict.values())[0]

    return server_config_dict


def find_open_port() -> int:  # pragma: no cover
    with socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


class ModelServerRef(BaseModel):
    type: Literal["responses_api_models"]
    name: str


class ResourcesServerRef(BaseModel):
    type: Literal["resources_servers"]
    name: str


class AgentServerRef(BaseModel):
    type: Literal["responses_api_agents"]
    name: str


ServerRef = Union[ModelServerRef, ResourcesServerRef, AgentServerRef]
ServerRefTypeAdapter = TypeAdapter(ServerRef)


def is_server_ref(config_dict: DictConfig) -> Optional[ServerRef]:
    try:
        return ServerRefTypeAdapter.validate_python(config_dict)
    except ValidationError:
        return None


class BaseServerConfig(BaseModel):
    host: str
    port: int


class BaseRunServerConfig(BaseServerConfig):
    entrypoint: str


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
    def load_from_global_config(
        cls, head_server_config: Optional[BaseServerConfig] = None
    ) -> "ServerClient":
        if head_server_config is None:
            head_server_config = cls.load_head_server_config()

        # It's critical we use requests here instead of the global httpx client since a FastAPI server may be run downstream of this function call.
        response = requests.get(
            f"http://{head_server_config.host}:{head_server_config.port}/global_config_dict_yaml",
        )

        global_config_dict_yaml = response.content.decode()
        global_config_dict = OmegaConf.create(json.loads(global_config_dict_yaml))

        return cls(
            head_server_config=head_server_config, global_config_dict=global_config_dict
        )

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
        server_config_dict = get_first_server_config_dict(
            self.global_config_dict, server_name
        )
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
        json: Any | None = None,
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
        server_config_dict = get_first_server_config_dict(
            self.global_config_dict, server_name
        )
        return await GLOBAL_HTTPX_CLIENT.post(
            f"http://{server_config_dict.host}:{server_config_dict.port}{url_path}",
            content=content,
            data=data,
            files=files,
            json=json,
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
        server_config_dict = get_first_server_config_dict(
            global_config_dict, config_path_str
        )

        server_config_cls: Type[BaseRunServerConfig] = cls.model_fields[
            "config"
        ].annotation
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
