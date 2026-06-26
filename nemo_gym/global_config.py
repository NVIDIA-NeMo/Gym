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
import logging
import sys
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from difflib import get_close_matches
from importlib import import_module
from os import environ, getenv
from pathlib import Path
from platform import python_version
from random import randint
from socket import gethostbyname, gethostname, socket
from typing import ClassVar, List, Optional, Tuple, Type

import hydra
import rich
import wandb
import wandb.util
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf, open_dict
from openai import __version__ as openai_version
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from ray import __version__ as ray_version
from wandb import Run

from nemo_gym import CACHE_DIR, PARENT_DIR, RESULTS_DIR, WORKING_DIR
from nemo_gym.config_types import (
    ConfigMissingValuesError,
    ConfigPathNotFoundError,
    MalformedConfigPathsError,
    NoServerInstancesError,
    ServerInstanceConfig,
    ServerRefNotFoundError,
    WANDBConfig,
    is_almost_server,
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
PYTHON_VERSION_KEY_NAME = "python_version"
PIP_INSTALL_VERBOSE_KEY_NAME = "pip_install_verbose"
USE_ABSOLUTE_IP = "use_absolute_ip"
UV_PIP_SET_PYTHON_KEY_NAME = "uv_pip_set_python"
SKIP_VENV_IF_PRESENT_KEY_NAME = "skip_venv_if_present"
HF_TOKEN_KEY_NAME = "hf_token"
RAY_HEAD_NODE_ADDRESS_KEY_NAME = "ray_head_node_address"
PORT_RANGE_LOW_KEY_NAME = "port_range_low"
PORT_RANGE_HIGH_KEY_NAME = "port_range_high"
DRY_RUN_KEY_NAME = "dry_run"
UV_CACHE_DIR_KEY_NAME = "uv_cache_dir"
UV_VENV_DIR_KEY_NAME = "uv_venv_dir"
INHERIT_FROM_KEY_NAME = "_inherit_from"
COPY_KEY_NAME = "_copy"
DELETE_KEY_KEY_NAME = "_delete_key"

# Sentinel returned by _recursive_index_dict_using_path when a referenced swap/copy/inherit path is
# unset (a '???' leaf or ancestor). Distinct object so callers can branch on it without mistaking a
# real config value (e.g. a literal "???" or a DictConfig) for "missing".
_MISSING_REF = object()
NEMO_GYM_LOG_DIR_KEY_NAME = "nemo_gym_log_dir"
VERBOSE_KEY_NAME = "verbose"
JSON_OUTPUT_KEY_NAME = "json"
QUERY_KEY_NAME = "query"
NEMO_GYM_RESERVED_TOP_LEVEL_KEYS = [
    CONFIG_PATHS_KEY_NAME,
    ENTRYPOINT_KEY_NAME,
    DEFAULT_HOST_KEY_NAME,
    HEAD_SERVER_KEY_NAME,
    DISALLOWED_PORTS_KEY_NAME,
    HEAD_SERVER_DEPS_KEY_NAME,
    PYTHON_VERSION_KEY_NAME,
    PIP_INSTALL_VERBOSE_KEY_NAME,
    USE_ABSOLUTE_IP,
    UV_PIP_SET_PYTHON_KEY_NAME,
    SKIP_VENV_IF_PRESENT_KEY_NAME,
    HF_TOKEN_KEY_NAME,
    RAY_HEAD_NODE_ADDRESS_KEY_NAME,
    PORT_RANGE_LOW_KEY_NAME,
    PORT_RANGE_HIGH_KEY_NAME,
    DRY_RUN_KEY_NAME,
    UV_CACHE_DIR_KEY_NAME,
    UV_VENV_DIR_KEY_NAME,
    INHERIT_FROM_KEY_NAME,
    COPY_KEY_NAME,
    NEMO_GYM_LOG_DIR_KEY_NAME,
    VERBOSE_KEY_NAME,
    JSON_OUTPUT_KEY_NAME,
    QUERY_KEY_NAME,
]

# Data keys
TASK_INDEX_KEY_NAME = "_ng_task_index"
ROLLOUT_INDEX_KEY_NAME = "_ng_rollout_index"
RESPONSES_CREATE_PARAMS_KEY_NAME = "responses_create_params"
RESPONSE_KEY_NAME = "response"
AGENT_REF_KEY_NAME = "agent_ref"
SKILLS_REF_KEY_NAME = "skills_ref"

POLICY_BASE_URL_KEY_NAME = "policy_base_url"
POLICY_API_KEY_KEY_NAME = "policy_api_key"  # pragma: allowlist secret
POLICY_MODEL_NAME_KEY_NAME = "policy_model_name"
POLICY_MODEL_KEY_NAME = "policy_model"

DEFAULT_HEAD_SERVER_PORT = 11000


# W&B
# Increase row limit since some of our rollouts are pretty hefty
wandb.util.VALUE_BYTES_LIMIT = 10_000_000
_WANDB_RUN: Optional[Run] = None


def get_wandb_run() -> Optional[Run]:
    return _WANDB_RUN


# HuggingFace
def get_hf_token() -> Optional[str]:  # pragma: no cover
    return get_global_config_dict().get(HF_TOKEN_KEY_NAME)


# OmegaConf new resolvers
OmegaConf.register_new_resolver("inherit_from", lambda a: f"${{inherit_from:{a}}}")
OmegaConf.register_new_resolver("copy", lambda a: f"${{copy:{a}}}")


class GlobalConfigDictParserConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dotenv_path: Optional[Path] = None
    initial_global_config_dict: Optional[DictConfig] = None
    skip_load_from_cli: bool = False
    skip_load_from_dotenv: bool = False

    hide_secrets: bool = False

    # This is a shorthand we use for config resolution use cases that shouldn't require a model
    # e.g. data loading, etc
    NO_MODEL_GLOBAL_CONFIG_DICT: ClassVar[DictConfig] = DictConfig(
        {
            POLICY_BASE_URL_KEY_NAME: "",
            POLICY_API_KEY_KEY_NAME: "",
            POLICY_MODEL_NAME_KEY_NAME: "",
            POLICY_MODEL_KEY_NAME: {"responses_api_models": {"dummy_model": {"entrypoint": "app.py"}}},
        }
    )


class GlobalConfigDictParser(BaseModel):
    def parse_global_config_dict_from_cli(self) -> DictConfig:
        # We need to monkeypatch hydra here so that it doesn't use Hydra help so that we can use our own help down the line
        hydra_main_module = import_module("hydra.main")
        original_get_args_parser = hydra_main_module.get_args_parser

        def new_get_args_parser():
            parser: ArgumentParser = original_get_args_parser()
            # Set the conflict handlers to resolve so we can disable the help.
            parser.conflict_handler = "resolve"
            for action_group in parser._action_groups:
                action_group.conflict_handler = "resolve"

            parser.add_argument("--help", "-h", action="store_false", default=False)

            # Reset to the original conflict_handler error scheme
            parser.conflict_handler = "error"
            for action_group in parser._action_groups:
                action_group.conflict_handler = "error"

            return parser

        hydra_main_module.get_args_parser = new_get_args_parser

        # This function is just to get the config object out of the hydra main call.
        # Need a closure. We simply use an outer ref of a list
        config_list = []

        @hydra.main(config_path=None, version_base=None)
        def inner_hydra_wrapper(cfg: DictConfig) -> DictConfig:
            config_list.append(cfg)

        inner_hydra_wrapper()

        # Hydra installs a console log handler on stdout; move it to stderr so command stdout stays machine-readable
        # (e.g. `gym ... --json`). Diagnostics belong on stderr; only the requested data goes to stdout.
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout:
                handler.setStream(sys.stderr)

        global_config_dict: DictConfig = config_list[0]

        return global_config_dict

    def load_extra_config_paths(self, config_paths: List[str]) -> Tuple[List[str], List[DictConfig]]:
        """
        Returns the new total config_paths and the extra configs
        """
        config_paths = config_paths.copy()

        extra_configs: List[DictConfig] = []
        duplicate_config_paths: List[str] = []
        # Just a careful note here that we explicitly mutate config_paths as it is being appended to
        for config_path in config_paths:
            original_entry = config_path
            config_path = Path(config_path)
            # Check cwd first for user's local configs, then install location
            searched_locations = [config_path]
            if not config_path.is_absolute():
                cwd_path = Path.cwd() / config_path
                install_path = PARENT_DIR / config_path
                # cwd and the install root coincide when run from the repo; list each location once.
                searched_locations = [cwd_path] if cwd_path == install_path else [cwd_path, install_path]
                config_path = cwd_path if cwd_path.exists() else install_path

            try:
                extra_config = OmegaConf.load(config_path)
            except FileNotFoundError as e:
                searched = "\n".join(f"  - {p}" for p in searched_locations)
                raise ConfigPathNotFoundError(
                    f"""config_paths entry '{original_entry}' was not found. Looked in:
{searched}
Check the path is spelled correctly and is relative to your working directory or the Gym install root."""
                ) from e
            for new_config_path in extra_config.get(CONFIG_PATHS_KEY_NAME) or []:
                if new_config_path not in config_paths:
                    config_paths.append(new_config_path)
                else:
                    duplicate_config_paths.append(new_config_path)
            extra_configs.append(extra_config)

        if duplicate_config_paths:
            duplicate_config_paths_str = "".join(f"- {p}\n" for p in duplicate_config_paths)
            print(f"""Found configs that reference the same source config path. You may want to double check whether the configs you have need to use different configs for the same server.
In cases like these, you may want to consider using the `inherit_from` OmegaConf directive e.g. '++my_specific_server=${{inherit_from:generic_server}}' and then overriding config parameters in `my_specific_server`.
Duplicate config paths:
{duplicate_config_paths_str}""")

        return config_paths, extra_configs

    def filter_for_server_instance_configs(self, global_config_dict: DictConfig) -> List[ServerInstanceConfig]:
        # Get the non-reserved top level items
        non_reserved_items = [
            (key, v) for key, v in global_config_dict.items() if key not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
        ]

        # Do one pass to get the server instance configs
        server_instance_configs: List[ServerInstanceConfig] = []
        for server_name, server_type_config_dict in non_reserved_items:
            maybe_server_instance_config, _ = maybe_get_server_instance_config(
                name=server_name, server_type_config_dict=server_type_config_dict
            )
            if maybe_server_instance_config is not None:
                server_instance_configs.append(maybe_server_instance_config)

        return server_instance_configs

    def raise_on_no_server_instances(self, global_config_dict: DictConfig) -> None:
        """Fail fast if a run has no server instances to start.

        Without this, `ng_run` with an empty/omitted `config_paths` starts the head server and Ray
        and then hangs with nothing to run. We catch it before Ray initialises with an actionable
        message instead.
        """
        if self.filter_for_server_instance_configs(global_config_dict):
            return

        raise NoServerInstancesError(
            """No server instances are configured, so there is nothing to run. Pass one or more configs, e.g.:
  gym env start --config resources_servers/<env>/configs/<env>.yaml --config responses_api_models/<model>/configs/<model>.yaml"""
        )

    def validate_and_populate_defaults(
        self,
        server_instance_configs: List[ServerInstanceConfig],
        default_host: str,
        port_range_low: int,
        port_range_high: int,
        initial_disallowed_ports: Optional[List[int]] = None,
    ) -> List[int]:
        server_refs = [c.get_server_ref() for c in server_instance_configs]

        disallowed_ports = initial_disallowed_ports.copy() if initial_disallowed_ports is not None else []

        for server_instance_config in server_instance_configs:
            run_server_config_dict = server_instance_config.get_inner_run_server_config_dict()

            # Check server refs
            for field_name, v in run_server_config_dict.items():
                maybe_server_ref = is_server_ref(v)
                if not maybe_server_ref:
                    continue

                if maybe_server_ref not in server_refs:
                    same_type_names = [ref.name for ref in server_refs if ref.type == maybe_server_ref.type]
                    suggestions = get_close_matches(maybe_server_ref.name, same_type_names, n=3, cutoff=0.6)
                    if suggestions:
                        hint = "Did you mean: " + ", ".join(repr(s) for s in suggestions) + "?"
                    else:
                        available = ", ".join(repr(n) for n in sorted(same_type_names)) or "(none)"
                        hint = f"Available {maybe_server_ref.type}: {available}"
                    raise ServerRefNotFoundError(
                        f"""In server instance '{server_instance_config.name}', field '{field_name}' references {maybe_server_ref.type}/'{maybe_server_ref.name}', which is not defined in the merged config.
{hint}"""
                    )

            # Populate the host and port values if they are not present in the config.
            with open_dict(run_server_config_dict):
                if not run_server_config_dict.get("host"):
                    run_server_config_dict["host"] = default_host
                if not run_server_config_dict.get("port"):
                    port = _find_open_port_using_range(
                        disallowed_ports=disallowed_ports,
                        port_range_low=port_range_low,
                        port_range_high=port_range_high,
                    )
                    run_server_config_dict["port"] = port
                    disallowed_ports.append(port)  # Disallow newly allocated port.
                else:
                    # Port already exists, add it to the disallowed list.
                    disallowed_ports.append(run_server_config_dict["port"])

        return disallowed_ports

    def collect_missing_value_paths(self, config: DictConfig) -> List[str]:
        """Return the dotted paths of every unset (OmegaConf '???') leaf, without raising.

        We convert to a plain container with `resolve=False, throw_on_missing=False` so that
        neither MISSING values nor unresolved interpolations (`${...}`) cause an exception — then
        walk the plain structure. Iterating or indexing the live DictConfig would raise.
        """
        container = OmegaConf.to_container(config, resolve=False, throw_on_missing=False)
        return self._walk_missing_value_paths(container)

    def _walk_missing_value_paths(self, node, prefix: str = "") -> List[str]:
        missing_paths: List[str] = []
        if isinstance(node, dict):
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if value == MISSING:
                    missing_paths.append(path)
                else:
                    missing_paths.extend(self._walk_missing_value_paths(value, path))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                path = f"{prefix}[{i}]"
                if value == MISSING:
                    missing_paths.append(path)
                else:
                    missing_paths.extend(self._walk_missing_value_paths(value, path))
        return missing_paths

    def raise_on_missing_values(self, global_config_dict: DictConfig) -> None:
        """Fail fast with one actionable error listing every unset '???' value.

        Without this, the first unset value surfaces deep in the run pipeline as an opaque
        omegaconf MissingMandatoryValue, one field at a time and with no override guidance.
        """
        missing_paths = self.collect_missing_value_paths(global_config_dict)
        if not missing_paths:
            return

        missing_list = "\n".join(f"  - {p}" for p in missing_paths)
        override_examples = "\n".join(f"  ++{p}=<value>" for p in missing_paths[:3])
        raise ConfigMissingValuesError(
            f"""{len(missing_paths)} required config value(s) are unset (still '???') after merging:
{missing_list}

Provide each value via a CLI override, in env.yaml, or in a config you pass via config_paths.
For example, on the command line:
{override_examples}"""
        )

    def _recursively_hide_secrets(self, dict_config: DictConfig) -> None:
        with open_dict(dict_config):
            self._recursively_hide_secrets_helper(dict_config)

    def _recursively_hide_secrets_helper(self, dict_config: DictConfig) -> None:
        for k, v in list(dict_config.items()):
            if isinstance(v, (DictConfig, dict)):
                self._recursively_hide_secrets_helper(v)
            elif isinstance(v, (ListConfig, list)):
                if "token" in k or "key" in k:
                    dict_config[k] = ["****"] * len(v)
                else:
                    for inner_v in v:
                        if isinstance(inner_v, (DictConfig, dict)):
                            self._recursively_hide_secrets_helper(inner_v)
            else:
                if "token" in k or "key" in k:
                    dict_config[k] = "****"

    def _recursively_swap_keys(self, dict_config: DictConfig) -> None:
        frozen_dict_config = deepcopy(dict_config)
        with open_dict(dict_config):
            self._recursively_swap_keys_helper(dict_config, dict_config, frozen_dict_config)

    def _recursively_swap_keys_helper(
        self, dict_config: DictConfig, original_dict_config: DictConfig, frozen_dict_config: DictConfig
    ) -> None:
        # items_ex(resolve=False) yields raw values: directive strings like "${inherit_from:...}"
        # come back unresolved (so the swap detection below still matches), and a missing ('???')
        # leaf is returned as-is instead of raising MissingMandatoryValue mid-swap. Any genuinely
        # unset values are reported together by raise_on_missing_values, which runs after this pass.
        for k, v in list(dict_config.items_ex(resolve=False)):
            is_delete_property = isinstance(v, DictConfig) and DELETE_KEY_KEY_NAME in v

            if is_delete_property:
                keys_to_delete = v.pop(DELETE_KEY_KEY_NAME).split(",")
                keys_to_delete = set(map(str.strip, keys_to_delete))

                # Delete first so we don't resolve the deleted keys
                # but only delete keys that are present in case the key-to-delete comes from a downstream inherit or swap
                existing_keys = set(k for k in keys_to_delete if k in v)
                for key in existing_keys:
                    v.pop(key)
                keys_to_delete -= existing_keys

            if isinstance(v, (DictConfig, dict)):
                self._recursively_swap_keys_helper(v, original_dict_config, frozen_dict_config)
            elif isinstance(v, (ListConfig, list)):
                # Iterate without resolving so a missing ('???') element doesn't raise mid-swap (it's
                # a scalar, so it's skipped here and reported later by raise_on_missing_values).
                for i in range(len(v)):
                    if isinstance(v, ListConfig) and OmegaConf.is_missing(v, i):
                        continue
                    inner_v = v[i]
                    if isinstance(inner_v, (DictConfig, dict)):
                        self._recursively_swap_keys_helper(inner_v, original_dict_config, frozen_dict_config)

            # e.g. ${inherit_from:grpo.num_prompts_per_step}
            is_swap_str = isinstance(v, str) and v.startswith("${inherit_from:")
            is_swap_property = isinstance(v, DictConfig) and INHERIT_FROM_KEY_NAME in v
            is_swap = is_swap_str or is_swap_property

            is_copy_str = isinstance(v, str) and v.startswith("${copy:")
            is_copy_property = isinstance(v, DictConfig) and COPY_KEY_NAME in v
            is_copy = is_copy_str or is_copy_property
            if not (is_swap or is_copy):
                continue

            if is_swap_str:
                path_to_swap = v.removeprefix("${inherit_from:").removesuffix("}")
            elif is_swap_property:
                path_to_swap = v.pop(INHERIT_FROM_KEY_NAME)

            if is_copy_str:
                path_to_swap = v.removeprefix("${copy:").removesuffix("}")
            elif is_copy_property:
                path_to_swap = v.pop(COPY_KEY_NAME)

            path_to_swap = path_to_swap.split(".")

            # Pop the swapped value. A '???' leaf or ancestor on the source path comes back as the
            # _MISSING_REF sentinel (not a dict/str), so guard the .pop()/.merge() that would crash on it.
            dict_containing_key_to_swap = self._recursive_index_dict_using_path(
                original_dict_config, path_to_swap[:-1]
            )
            if is_swap and dict_containing_key_to_swap is not _MISSING_REF:
                # Pop with a default since multiple configs may refer to the same path
                # We don't want to pop if it's just a copy
                dict_containing_key_to_swap.pop(path_to_swap[-1], None)

            swapped_value = self._recursive_index_dict_using_path(frozen_dict_config, path_to_swap)

            # If the source (leaf or an ancestor) is unset, the target inherits MISSING; skip the
            # property-merge and _delete_key handling and let raise_on_missing_values report it.
            if dict_containing_key_to_swap is _MISSING_REF or swapped_value is _MISSING_REF:
                dict_config[k] = "???"
                continue

            if is_swap_property or is_copy_property:
                swapped_value = OmegaConf.merge(swapped_value, v)

            dict_config[k] = swapped_value

            # TODO We may want to recurse again after swap since we are not guaranteed to traverse the swapped-from value before hitting this swap.

            if is_delete_property:
                # Enforce that every key-to-delete exists
                for key in keys_to_delete:
                    dict_config[k].pop(key)

    def _recursive_index_dict_using_path(self, dict_config: DictConfig, path: List[str]) -> "DictConfig | object":
        for k in path:
            # Use _get_node so a referenced value that is unset ('???') can be detected without
            # resolving it (indexing a MISSING leaf would raise an opaque error). A genuinely
            # absent key still errors clearly.
            node = dict_config._get_node(k) if isinstance(dict_config, DictConfig) else None
            if node is None:
                raise ValueError(f"Path specified does not exist in config: {path}")

            # The referenced value (or an ancestor of it) is unset. Return the _MISSING_REF sentinel
            # so the caller makes the swap/copy/inherit target '???' too (instead of calling .pop()/
            # OmegaConf.merge() on a bare string and crashing); raise_on_missing_values then reports
            # it with an actionable message instead of an opaque interpolation error.
            if OmegaConf.is_missing(dict_config, k):
                return _MISSING_REF

            dict_config = dict_config[k]

        return dict_config

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
        # Check cwd first for user's local env.yaml, then fall back to PARENT_DIR
        if parse_config.dotenv_path:
            dotenv_path = parse_config.dotenv_path
        else:
            cwd_env_yaml = Path.cwd() / "env.yaml"
            dotenv_path = cwd_env_yaml if cwd_env_yaml.exists() else PARENT_DIR / "env.yaml"

        dotenv_extra_config = DictConfig({})
        if dotenv_path.exists() and not parse_config.skip_load_from_dotenv:
            dotenv_extra_config = OmegaConf.load(dotenv_path)

        merged_config_for_config_paths = OmegaConf.merge(dotenv_extra_config, global_config_dict)
        ta = TypeAdapter(List[str])
        config_paths = merged_config_for_config_paths.get(CONFIG_PATHS_KEY_NAME) or []
        try:
            config_paths = ta.validate_python(config_paths)
        except ValidationError as e:
            raise MalformedConfigPathsError(
                f"""'{CONFIG_PATHS_KEY_NAME}' must be a list of paths. Got: {config_paths!r}.
Pass each config with --config (it builds the list for you), e.g.:
  gym env start --config resources_servers/<env>/configs/<env>.yaml"""
            ) from e

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

        self._recursively_swap_keys(global_config_dict)

        # Fail fast with one actionable error if any required value is still '???'. Runs *after*
        # _recursively_swap_keys so that _delete_key/_inherit_from/_copy have been applied first —
        # a '???' in a deleted or overwritten branch is not reported. Otherwise the first unset
        # value surfaces as an opaque MissingMandatoryValue deep in the pipeline.
        self.raise_on_missing_values(global_config_dict)

        # TODO @bxyu-nvidia: We need a better way of handling dummy model configs
        with open_dict(global_config_dict):
            for top_level_value in global_config_dict.values():
                if not (
                    isinstance(top_level_value, (DictConfig))
                    and "responses_api_models" in top_level_value
                    # We check `len(top_level_value) > 1` in case the policy model is inherited from.
                    and (len(top_level_value) > 1 or len(top_level_value["responses_api_models"]) > 1)
                    and "dummy_model" in top_level_value["responses_api_models"]
                ):
                    continue

                dummy_value = top_level_value["responses_api_models"].pop("dummy_model")
                actual_key = next(iter(top_level_value["responses_api_models"]))
                top_level_value["responses_api_models"][actual_key] = OmegaConf.merge(
                    dummy_value, top_level_value["responses_api_models"][actual_key]
                )

        # Almost-server detection and reporting
        almost_servers = self.detect_and_report_almost_servers(global_config_dict)

        if almost_servers:
            rich.print("[yellow]═══════════════════════════════════════════════════[/yellow]")
            rich.print("[yellow]Configuration Warnings: Almost-Servers Detected[/yellow]")
            rich.print("[yellow]═══════════════════════════════════════════════════[/yellow]")

            for server_name, error in almost_servers:
                rich.print(format_almost_server_warning(server_name, error))

            rich.print("[yellow]═══════════════════════════════════════════════════[/yellow]\n")

            error_on_almost_servers = global_config_dict.get("error_on_almost_servers", True)
            if error_on_almost_servers:
                config_dict_to_log = deepcopy(global_config_dict)
                self._recursively_hide_secrets(config_dict_to_log)
                config_to_log_yaml = OmegaConf.to_yaml(config_dict_to_log)

                error_msg = f"""Found {len(almost_servers)} almost-server(s) with validation errors. Fix the issues above or set error_on_almost_servers=false to bypass this error.
Found global config dict yaml:
{config_to_log_yaml}"""

                raise ValueError(error_msg)

        server_instance_configs = self.filter_for_server_instance_configs(global_config_dict)

        with open_dict(global_config_dict):
            use_absolute_ip = global_config_dict.setdefault(USE_ABSOLUTE_IP, False)
        if use_absolute_ip:
            default_host = gethostbyname(gethostname())
        else:
            # Do one pass through all the configs validate and populate various configs for our servers.
            default_host = global_config_dict.get(DEFAULT_HOST_KEY_NAME) or "127.0.0.1"

        head_server_config = global_config_dict.get(HEAD_SERVER_KEY_NAME, {})
        head_server_port = head_server_config.get("port", DEFAULT_HEAD_SERVER_PORT)

        initial_disallowed_ports = [head_server_port] if head_server_port is not None else []

        with open_dict(global_config_dict):
            port_range_low = global_config_dict.setdefault(PORT_RANGE_LOW_KEY_NAME, 10_001)
            port_range_high = global_config_dict.setdefault(PORT_RANGE_HIGH_KEY_NAME, 20_000)

        disallowed_ports = self.validate_and_populate_defaults(
            server_instance_configs=server_instance_configs,
            default_host=default_host,
            initial_disallowed_ports=initial_disallowed_ports,
            port_range_low=port_range_low,
            port_range_high=port_range_high,
        )

        with open_dict(global_config_dict):
            # Populate head server defaults
            if not global_config_dict.get(HEAD_SERVER_KEY_NAME):
                global_config_dict[HEAD_SERVER_KEY_NAME] = {
                    "host": default_host,
                    "port": DEFAULT_HEAD_SERVER_PORT,
                }

            # Store final list of disallowed ports.
            global_config_dict[DISALLOWED_PORTS_KEY_NAME] = disallowed_ports

            # Constrain sensitive package versions
            global_config_dict[HEAD_SERVER_DEPS_KEY_NAME] = [
                # The ray version is very sensitive. The children ray versions must exactly match those of the parent ray.
                # The ray extra [default] should also exactly match the extra in the top-level Gym pyproject.toml.
                f"ray[default]=={ray_version}",
                # OpenAI version is also sensitive since it changes so often and may introduce subtle incompatibilities.
                f"openai=={openai_version}",
            ]

            # Constrain python version since ray is sensitive to this.
            global_config_dict[PYTHON_VERSION_KEY_NAME] = python_version()

            # Skip venv setup is opt-in and defaults to False.
            global_config_dict.setdefault(SKIP_VENV_IF_PRESENT_KEY_NAME, False)

            global_config_dict.setdefault(DRY_RUN_KEY_NAME, False)

            # UV related configuration
            # UV caching directory overrides to local folders.
            global_config_dict.setdefault(UV_CACHE_DIR_KEY_NAME, str(CACHE_DIR / "uv"))
            # Set the appropriate environment variable here, and matche the config
            environ["UV_CACHE_DIR"] = global_config_dict[UV_CACHE_DIR_KEY_NAME]
            # By default, build the directories in their individual folders using the root repository
            # e.g. WORKING_DIR/responses_api_models/my_server
            global_config_dict.setdefault(UV_VENV_DIR_KEY_NAME, str(WORKING_DIR))

        if parse_config.hide_secrets:  # pragma: no cover
            self._recursively_hide_secrets(global_config_dict)

        # Set up W&B and log config. This must happen at the very last step.
        wandb_config = WANDBConfig.model_validate(global_config_dict)
        if wandb_config.is_available:  # pragma: no cover
            environ["WANDB_API_KEY"] = wandb_config.wandb_api_key

            global _WANDB_RUN
            _WANDB_RUN = wandb.init(
                project=wandb_config.wandb_project,
                name=wandb_config.wandb_name,
                dir=str(RESULTS_DIR / "wandb"),
            )

            # Log params
            config_dict_to_log = deepcopy(global_config_dict)
            self._recursively_hide_secrets(config_dict_to_log)
            _WANDB_RUN.config.update(OmegaConf.to_container(config_dict_to_log))

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

    def detect_and_report_almost_servers(
        self,
        global_config_dict: DictConfig,
    ) -> List[Tuple[str, ValidationError]]:
        non_reserved_items = [
            (key, v) for key, v in global_config_dict.items() if key not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
        ]

        almost_servers = []

        # Try to get config with error capture.
        for server_name, server_type_config_dict in non_reserved_items:
            config, error = maybe_get_server_instance_config(
                name=server_name, server_type_config_dict=server_type_config_dict
            )

            # Failed validation but looks like a server = almost-server
            if config is None and error is not None:
                if is_almost_server(server_type_config_dict):
                    almost_servers.append((server_name, error))

        return almost_servers


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

        _apply_verbosity(global_config_dict)
        return global_config_dict

    set_global_config_dict(
        global_config_dict_parser_config=global_config_dict_parser_config,
        global_config_dict_parser_cls=global_config_dict_parser_cls,
    )

    _apply_verbosity(_GLOBAL_CONFIG_DICT)
    return _GLOBAL_CONFIG_DICT


def _apply_verbosity(global_config_dict: DictConfig) -> None:
    """Set logging to DEBUG when `verbose` is in the config. Runs in the CLI process and, because the
    config dict is forwarded to every spun-up server, in each server process too."""
    if global_config_dict.get(VERBOSE_KEY_NAME):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)


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

    global_config_dict = get_global_config_dict()

    return _find_open_port_using_range(
        disallowed_ports=disallowed_ports,
        max_retries=max_retries,
        port_range_low=global_config_dict[PORT_RANGE_LOW_KEY_NAME],
        port_range_high=global_config_dict[PORT_RANGE_HIGH_KEY_NAME],
    )


def _find_open_port_using_range(
    disallowed_ports: List[int],
    port_range_low: int,
    port_range_high: int,
    max_retries: int = 50,
) -> int:  # pragma: no cover
    # Find an open port that doesn't conflict with disallowed ports.

    with socket() as s:
        for _ in range(max_retries):
            # Pick a random port in our range that is not disallowed
            port = None
            while port is None or port in disallowed_ports:
                port = randint(port_range_low, port_range_high)

            try:
                s.bind(("", port))
                return port
            except OSError:
                pass

    raise RuntimeError(
        f"Unable to find an open port that doesn't conflict with disallowed ports "
        f"{disallowed_ports} after {max_retries} attempts"
    )


def format_almost_server_warning(server_name: str, error: ValidationError) -> str:
    """Format user-friendly warning. Union literal errors are consolidated.
    Union discriminator noise is filtered out. Explanation:
    Pydantic validation is quirky- it will report all failures in the union if any union member fails. Example:
    If an agent server contains an invalid license, it will not only show the error for the invalid license in ResponsesAPIAgentServerInstanceConfig, but also missing values for ResponsesAPIModelServerInstanceConfig `responses_api_models` and ResourcesServerInstanceConfig `resources_servers`.
    """

    errors = error.errors()

    # Identify the actual server type from the error (excluding Union discriminator noise)
    server_type_keys = ["responses_api_models", "resources_servers", "responses_api_agents"]
    actual_server_type = None

    # Example error structure: ('ResponsesAPIAgentServerInstanceConfig', 'responses_api_agents', 'simple_agent', 'datasets', 0, 'license')
    for err in errors:
        loc = err["loc"]
        # loc[1] is the actual server type key.
        # Skip "missing" errors from the irrelevant Union variants.
        if len(loc) > 1 and loc[1] in server_type_keys and err["type"] != "missing":
            actual_server_type = loc[1]
            break

    # Fallback: if all errors are "missing", check the input dict for the actual server type.
    if not actual_server_type:  # pragma: no cover
        for err in errors:
            if "input" in err and isinstance(err["input"], dict):
                for key in server_type_keys:
                    if key in err["input"]:
                        actual_server_type = key
                        break
                if actual_server_type:
                    break

    # Filter out Union discriminator false positives.
    filtered_errors = []
    for err in errors:
        loc = err["loc"]

        # Filter out "Field required" errors from wrong Union variants.
        if (
            err["type"] == "missing"
            and len(loc) > 1
            and loc[1] in server_type_keys
            and actual_server_type
            and loc[1] != actual_server_type
        ):
            continue

        filtered_errors.append(err)

    # Group errors by location to consolidate Union literals.
    error_groups = defaultdict(list)

    for err in filtered_errors:
        loc = err["loc"]

        # Check if literal union error (starts with "literal[").
        if loc and isinstance(loc[-1], str) and loc[-1].startswith("literal["):
            # Group without the literal type prefix.
            base_loc = loc[:-1]
            error_groups[base_loc].append(err)
        else:
            error_groups[loc].append(err)

    error_details = []
    for loc, errs in error_groups.items():
        if len(errs) > 1 and all(isinstance(e["loc"][-1], str) and e["loc"][-1].startswith("literal[") for e in errs):
            # Consolidate errors for literals into "Must be one of: X, Y, Z" format.
            loc_str = " -> ".join(str(item) for item in loc)
            valid_options = []
            for e in errs:
                literal_str = e["loc"][-1]
                if literal_str.startswith("literal["):
                    value = literal_str[8:-2]  # Remove "literal['" and "']"
                    valid_options.append(value)

            if valid_options:
                options_str = "', ".join(valid_options)
                error_details.append(f"  - {loc_str}: Must be one of: {options_str}'")
            else:
                error_details.append(f"  - {loc_str}: {errs[0]['msg']}")

        else:
            err = errs[0]
            loc_str = " -> ".join(str(item) for item in err["loc"])
            error_details.append(f"  - {loc_str}: {err['msg']}")

    error_str = "\n".join(error_details)

    return f"""
    Almost-Server Detected: '{server_name}'
    This server configuration failed validation:

{error_str}

    This server will NOT be started.
    """
