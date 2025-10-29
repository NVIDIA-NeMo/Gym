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
from os import getenv
from pathlib import Path
from socket import socket
from typing import ClassVar, List, Optional, Tuple, Type

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import BaseModel, ConfigDict, TypeAdapter

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import (
    ServerInstanceConfig,
    is_server_ref,
    maybe_get_server_instance_config,
)


_GLOBAL_CONFIG_DICT = None
NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME = "NEMO_GYM_CONFIG_DICT"
NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME = "NEMO_GYM_CONFIG_PATH"
CONFIG_PATHS_KEY_NAME = "config_paths"
ENTRYPOINT_KEY_NAME = "entrypoint"
DEFAULT_HOST_KEY_NAME = "default_host"
HEAD_SERVER_KEY_NAME = "head_server"
DISALLOWED_PORTS_KEY_NAME = "disallowed_ports"
HEAD_SERVER_DEPS_KEY_NAME = "head_server_deps"
NEMO_GYM_RESERVED_TOP_LEVEL_KEYS = [
    CONFIG_PATHS_KEY_NAME,
    ENTRYPOINT_KEY_NAME,
    DEFAULT_HOST_KEY_NAME,
    HEAD_SERVER_KEY_NAME,
    DISALLOWED_PORTS_KEY_NAME,
    HEAD_SERVER_DEPS_KEY_NAME,
]

POLICY_BASE_URL_KEY_NAME = "policy_base_url"
POLICY_API_KEY_KEY_NAME = "policy_api_key"  # pragma: allowlist secret
POLICY_MODEL_NAME_KEY_NAME = "policy_model_name"

DEFAULT_HEAD_SERVER_PORT = 11000


class GlobalConfigDictParserConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dotenv_path: Optional[Path] = None
    initial_global_config_dict: Optional[DictConfig] = None
    skip_load_from_cli: bool = False
    skip_load_from_dotenv: bool = False

    NO_MODEL_GLOBAL_CONFIG_DICT: ClassVar[DictConfig] = DictConfig(
        {
            POLICY_BASE_URL_KEY_NAME: "",
            POLICY_API_KEY_KEY_NAME: "",
            POLICY_MODEL_NAME_KEY_NAME: "",
        }
    )


class GlobalConfigDictParser(BaseModel):
    def parse_global_config_dict_from_cli(self) -> DictConfig:
        # This function is just to get the config object out of the hydra main call.
        # Need a closure. We simply use an outer ref of a list
        config_list = []

        @hydra.main(config_path=None, version_base=None)
        def inner_hydra_wrapper(cfg: DictConfig) -> DictConfig:
            config_list.append(cfg)

        inner_hydra_wrapper()

        global_config_dict: DictConfig = config_list[0]

        return global_config_dict

    def load_extra_config_paths(self, config_paths: List[str]) -> Tuple[List[str], List[DictConfig]]:
        """
        Returns the new total config_paths and the extra configs
        """
        config_paths = config_paths.copy()

        extra_configs: List[DictConfig] = []
        for config_path in config_paths:
            config_path = Path(config_path)
            # Assume relative to the parent dir
            if not config_path.is_absolute():
                config_path = PARENT_DIR / config_path

            extra_config = OmegaConf.load(config_path)
            for new_config_path in extra_config.get(CONFIG_PATHS_KEY_NAME) or []:
                if new_config_path not in config_paths:
                    config_paths.append(new_config_path)
            extra_configs.append(extra_config)

        return config_paths, extra_configs

    def filter_for_server_instance_configs(self, global_config_dict: DictConfig) -> List[ServerInstanceConfig]:
        # Get the non-reserved top level items
        non_reserved_items = [
            (key, v) for key, v in global_config_dict.items() if key not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
        ]

        # Do one pass to get the server instance configs
        server_instance_configs: List[ServerInstanceConfig] = []
        for server_name, server_type_config_dict in non_reserved_items:
            maybe_server_instance_config = maybe_get_server_instance_config(
                name=server_name, server_type_config_dict=server_type_config_dict
            )
            if maybe_server_instance_config is not None:
                server_instance_configs.append(maybe_server_instance_config)

        return server_instance_configs

    def validate_and_populate_defaults(
        self,
        server_instance_configs: List[ServerInstanceConfig],
        default_host: str,
        initial_disallowed_ports: Optional[List[int]] = None,
    ) -> List[int]:
        server_refs = [c.get_server_ref() for c in server_instance_configs]

        disallowed_ports = initial_disallowed_ports.copy() if initial_disallowed_ports is not None else []

        for server_instance_config in server_instance_configs:
            run_server_config_dict = server_instance_config.get_inner_run_server_config_dict()

            # Check server refs
            for v in run_server_config_dict.values():
                maybe_server_ref = is_server_ref(v)
                if not maybe_server_ref:
                    continue

                assert maybe_server_ref in server_refs, (
                    f"Could not find {maybe_server_ref} in the list of available servers: {server_refs}"
                )

            # Populate the host and port values if they are not present in the config.
            with open_dict(run_server_config_dict):
                if not run_server_config_dict.get("host"):
                    run_server_config_dict["host"] = default_host
                if not run_server_config_dict.get("port"):
                    port = find_open_port(
                        disallowed_ports=disallowed_ports,
                    )
                    run_server_config_dict["port"] = port
                    disallowed_ports.append(port)  # Disallow newly allocated port.
                else:
                    # Port already exists, add it to the disallowed list.
                    disallowed_ports.append(run_server_config_dict["port"])

        return disallowed_ports

    def parse(self, parse_config: Optional[GlobalConfigDictParserConfig] = None) -> DictConfig:
        if parse_config is None:
            parse_config = GlobalConfigDictParserConfig()

        global_config_dict = (
            DictConfig(dict()) if parse_config.skip_load_from_cli else self.parse_global_config_dict_from_cli()
        )

        # Command line overrides function input.
        initial_global_config_dict = OmegaConf.create(parse_config.initial_global_config_dict or dict())
        global_config_dict: DictConfig = OmegaConf.merge(initial_global_config_dict, global_config_dict)

        # Load the env.yaml config. We load it early so that people can use it to conveniently store config paths.
        dotenv_path = parse_config.dotenv_path or Path(PARENT_DIR) / "env.yaml"
        dotenv_extra_config = DictConfig({})
        if dotenv_path.exists() and not parse_config.skip_load_from_dotenv:
            dotenv_extra_config = OmegaConf.load(dotenv_path)

        merged_config_for_config_paths = OmegaConf.merge(dotenv_extra_config, global_config_dict)
        ta = TypeAdapter(List[str])
        config_paths = merged_config_for_config_paths.get(CONFIG_PATHS_KEY_NAME) or []
        config_paths = ta.validate_python(config_paths)

        config_paths, extra_configs = self.load_extra_config_paths(config_paths)

        # Dot env overrides previous configs
        extra_configs.append(dotenv_extra_config)

        # Merge config dicts
        # global_config_dict is the last config arg here since we want command line args to override everything else.
        global_config_dict = OmegaConf.merge(*extra_configs, global_config_dict)

        # Update the config paths after postprocessing
        if config_paths:
            with open_dict(global_config_dict):
                global_config_dict[CONFIG_PATHS_KEY_NAME] = config_paths

        server_instance_configs = self.filter_for_server_instance_configs(global_config_dict)

        # Do one pass through all the configs validate and populate various configs for our servers.
        default_host = global_config_dict.get(DEFAULT_HOST_KEY_NAME) or "127.0.0.1"

        head_server_config = global_config_dict.get(HEAD_SERVER_KEY_NAME, {})
        head_server_port = head_server_config.get("port", DEFAULT_HEAD_SERVER_PORT)

        initial_disallowed_ports = [head_server_port] if head_server_port is not None else []
        disallowed_ports = self.validate_and_populate_defaults(
            server_instance_configs, default_host, initial_disallowed_ports
        )

        # Populate head server defaults
        if not global_config_dict.get(HEAD_SERVER_KEY_NAME):
            with open_dict(global_config_dict):
                global_config_dict[HEAD_SERVER_KEY_NAME] = {
                    "host": default_host,
                    "port": DEFAULT_HEAD_SERVER_PORT,
                }

        # Store final list of disallowed ports.
        with open_dict(global_config_dict):
            global_config_dict[DISALLOWED_PORTS_KEY_NAME] = disallowed_ports

        return global_config_dict

    def parse_no_environment(
        self,
        initial_global_config_dict: Optional[DictConfig] = None,
    ) -> DictConfig:
        return self.parse(
            parse_config=GlobalConfigDictParserConfig(
                dotenv_path=None,
                initial_global_config_dict=initial_global_config_dict,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
            )
        )


def get_global_config_dict(
    global_config_dict_parser_config: Optional[GlobalConfigDictParserConfig] = None,
    global_config_dict_parser_cls: Type[GlobalConfigDictParser] = GlobalConfigDictParser,
) -> DictConfig:
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

    set_global_config_dict(
        global_config_dict_parser_config=global_config_dict_parser_config,
        global_config_dict_parser_cls=global_config_dict_parser_cls,
    )

    return _GLOBAL_CONFIG_DICT


def set_global_config_dict(
    global_config_dict_parser_config: Optional[GlobalConfigDictParserConfig] = None,
    global_config_dict_parser_cls: Type[GlobalConfigDictParser] = GlobalConfigDictParser,
) -> None:
    global _GLOBAL_CONFIG_DICT
    global_config_dict = global_config_dict_parser_cls().parse(global_config_dict_parser_config)

    _GLOBAL_CONFIG_DICT = global_config_dict


def get_first_server_config_dict(global_config_dict: DictConfig, top_level_path: str) -> DictConfig:
    # Traverse three levels deep total
    server_config_dict = global_config_dict[top_level_path]
    server_config_dict = list(server_config_dict.values())[0]
    server_config_dict = list(server_config_dict.values())[0]

    return server_config_dict


def find_open_port(
    disallowed_ports: Optional[List[int]] = None,
    max_retries: int = 50,
) -> int:  # pragma: no cover
    if disallowed_ports is None:
        disallowed_ports = []

    # Find an open port that doesn't conflict with disallowed ports.
    for _ in range(max_retries):
        with socket() as s:
            s.bind(("", 0))  # Bind to a free port provided by the host.
            port = s.getsockname()[1]

            if port not in disallowed_ports:
                return port

    raise RuntimeError(
        f"Unable to find an open port that doesn't conflict with disallowed ports "
        f"{disallowed_ports} after {max_retries} attempts"
    )
