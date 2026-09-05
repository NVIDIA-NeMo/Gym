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
import re
import sys
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from difflib import get_close_matches
from importlib import import_module
from os import environ, getenv
from pathlib import Path
from platform import python_version
from random import randint
from socket import gethostbyname, gethostname, socket
from typing import ClassVar, Dict, List, Optional, Set, Tuple, Type

import hydra
import rich
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf, open_dict
from omegaconf.errors import InterpolationResolutionError
from openai import __version__ as openai_version
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from ray import __version__ as ray_version

from nemo_gym import CACHE_DIR, RESULTS_DIR, WORKING_DIR, _resolve_under_cwd_or_install, component_search_roots
from nemo_gym._config_aliases import LEGACY_AGENT_ALIASES, legacy_config_path_alias
from nemo_gym.config_types import (
    AgentCompositionError,
    AlmostServerError,
    ConfigError,
    ConfigInterpolationError,
    ConfigMissingValuesError,
    ConfigPathNotFoundError,
    InheritPathNotFoundError,
    MalformedConfigPathsError,
    NoServerInstancesError,
    ServerInstanceConfig,
    ServerRefNotFoundError,
    UnsupportedAgentOverrideError,
    UnsupportedAgentPairingError,
    UnsupportedModelPairingError,
    is_almost_server,
    is_server_ref,
    maybe_get_server_instance_config,
)
from nemo_gym.exporters import setup_exporters
from nemo_gym.secret_utils import recursively_hide_secrets
from nemo_gym.telemetry.setup import (
    TELEMETRY_KEY_NAME,
    configure_telemetry_env,
    server_venv_requirements,
    telemetry_config_from_global_config,
)


logger = logging.getLogger(__name__)

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
UVICORN_TIMEOUT_WORKER_HEALTHCHECK = "uvicorn_timeout_worker_healthcheck"
MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME = "model_endpoint_readiness_timeout_seconds"
ALLOW_OPENAI_VERSION_SKEW_KEY_NAME = "allow_openai_version_skew"
UV_CACHE_DIR_KEY_NAME = "uv_cache_dir"
UV_VENV_DIR_KEY_NAME = "uv_venv_dir"
RESULTS_DIR_KEY_NAME = "results_dir"
CACHE_DIR_KEY_NAME = "cache_dir"
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
OBSERVABILITY_ENABLED_KEY_NAME = "observability_enabled"
MODEL_CALL_CAPTURE_DIR_KEY_NAME = "model_call_capture_dir"
# Run-wide training-token capture settings.
# See ``nemo_gym/token_id_capture/config.py``.
TOKEN_ID_CAPTURE_BLOCK = "token_id_capture"
COMPONENT_NAME_KEY_NAME = "component_name"
SKIP_VERIFICATION_KEY_NAME = "skip_verification"
SKIP_VERIFICATION_REWARD_KEY_NAME = "skip_verification_reward"
ALLOW_UNSUPPORTED_PAIRING_KEY_NAME = "allow_unsupported_pairing"
ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME = "NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING"
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
    MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME,
    ALLOW_OPENAI_VERSION_SKEW_KEY_NAME,
    UV_CACHE_DIR_KEY_NAME,
    UV_VENV_DIR_KEY_NAME,
    RESULTS_DIR_KEY_NAME,
    CACHE_DIR_KEY_NAME,
    INHERIT_FROM_KEY_NAME,
    COPY_KEY_NAME,
    NEMO_GYM_LOG_DIR_KEY_NAME,
    VERBOSE_KEY_NAME,
    JSON_OUTPUT_KEY_NAME,
    QUERY_KEY_NAME,
    OBSERVABILITY_ENABLED_KEY_NAME,
    MODEL_CALL_CAPTURE_DIR_KEY_NAME,
    TOKEN_ID_CAPTURE_BLOCK,
    COMPONENT_NAME_KEY_NAME,
    SKIP_VERIFICATION_KEY_NAME,
    SKIP_VERIFICATION_REWARD_KEY_NAME,
    TELEMETRY_KEY_NAME,
    ALLOW_UNSUPPORTED_PAIRING_KEY_NAME,
]

AGENT_SERVER_TYPE_KEY_NAME = "responses_api_agents"
# Carried over from the environment's agent instance onto the composed agent; every other key is dropped.
_COMPOSED_AGENT_CARRY_OVER_KEYS = ("resources_server", "model_server", "datasets")
# Declared on a resources server: the agent types it is known to score correctly. Absent means any harness.
ALLOWED_AGENTS_KEY_NAME = "allowed_agents"
# Declared on a resources server: the model adapter types that can carry its workload. Absent means any adapter.
ALLOWED_MODEL_TYPES_KEY_NAME = "allowed_model_types"
MODEL_SERVER_TYPE_KEY_NAME = "responses_api_models"
RESOURCES_SERVER_TYPE_KEY_NAME = "resources_servers"


@dataclass(frozen=True)
class _AgentInstance:
    """A top-level agent instance, with its single agent type already unwrapped."""

    name: str
    agent_type: str
    server_config: DictConfig


# Data keys
TASK_INDEX_KEY_NAME = "_ng_task_index"
ROLLOUT_INDEX_KEY_NAME = "_ng_rollout_index"
# Resume re-dispatch attempt counter (0 on the first attempt); distinguishes retries of the same
# (task, rollout) so their captured model calls stay separable.
ATTEMPT_INDEX_KEY_NAME = "_ng_attempt_index"
# An explicit capture id replaces the task and rollout derivation.
# Set it when dispatches reuse task and rollout indices.
# Otherwise two dispatches would share one capture key.
ROLLOUT_ID_KEY_NAME = "_ng_rollout_id"
RESPONSES_CREATE_PARAMS_KEY_NAME = "responses_create_params"
RESPONSE_KEY_NAME = "response"
AGENT_REF_KEY_NAME = "agent_ref"
# The config instance that declares the row's dataset (a resources server normally; the agent
# itself for self-contained environments). Stamped into derived artifacts at collate/load time;
# resolved to an agent at dispatch time. See the dataset-decoupling RFC.
TASK_SOURCE_KEY_NAME = "task_source"
SKILLS_REF_KEY_NAME = "skills_ref"
REWARD_KEY_NAME = "reward"

# Metric key names. `RewardProfiler` builds its metric names from these prefixes and suffixes, and
# consumers of `*_aggregate_metrics.json` (e.g. `gym eval compare`) parse them back out -- so they
# live here, where both sides can import them without pulling in pandas/scipy/wandb.
MEAN_STAT_NAME = "mean"
MAX_STAT_NAME = "max"
MIN_STAT_NAME = "min"
MEDIAN_STAT_NAME = "median"
STD_STAT_NAME = "std"
SEM_STAT_NAME = "sem"
P25_STAT_NAME = "p25"
P75_STAT_NAME = "p75"
CI_LOW_95_STAT_NAME = "ci_low_95"
CI_HIGH_95_STAT_NAME = "ci_high_95"
HISTOGRAM_STAT_NAME = "histogram"

# `<stat>/<field>`, e.g. `mean/reward`.
STAT_SEPARATOR = "/"
MEAN_PREFIX = f"{MEAN_STAT_NAME}{STAT_SEPARATOR}"
MAX_PREFIX = f"{MAX_STAT_NAME}{STAT_SEPARATOR}"
MIN_PREFIX = f"{MIN_STAT_NAME}{STAT_SEPARATOR}"
MEDIAN_PREFIX = f"{MEDIAN_STAT_NAME}{STAT_SEPARATOR}"
STD_PREFIX = f"{STD_STAT_NAME}{STAT_SEPARATOR}"
SEM_PREFIX = f"{SEM_STAT_NAME}{STAT_SEPARATOR}"
P25_PREFIX = f"{P25_STAT_NAME}{STAT_SEPARATOR}"
P75_PREFIX = f"{P75_STAT_NAME}{STAT_SEPARATOR}"
CI_LOW_95_PREFIX = f"{CI_LOW_95_STAT_NAME}{STAT_SEPARATOR}"
CI_HIGH_95_PREFIX = f"{CI_HIGH_95_STAT_NAME}{STAT_SEPARATOR}"

# `<stat>_across_repeats/mean/<field>`: one repeat's estimate aggregated over the run's repeats.
ACROSS_REPEATS_MARKER = f"_across_repeats{STAT_SEPARATOR}"
MEAN_ACROSS_REPEATS_PREFIX = f"{MEAN_STAT_NAME}{ACROSS_REPEATS_MARKER}"
MEDIAN_ACROSS_REPEATS_PREFIX = f"{MEDIAN_STAT_NAME}{ACROSS_REPEATS_MARKER}"
STD_ACROSS_REPEATS_PREFIX = f"{STD_STAT_NAME}{ACROSS_REPEATS_MARKER}"
MIN_ACROSS_REPEATS_PREFIX = f"{MIN_STAT_NAME}{ACROSS_REPEATS_MARKER}"
MAX_ACROSS_REPEATS_PREFIX = f"{MAX_STAT_NAME}{ACROSS_REPEATS_MARKER}"
SE_ACROSS_REPEATS_PREFIX = f"se{ACROSS_REPEATS_MARKER}"
CI_LOW_95_ACROSS_REPEATS_PREFIX = f"{CI_LOW_95_STAT_NAME}{ACROSS_REPEATS_MARKER}"
CI_HIGH_95_ACROSS_REPEATS_PREFIX = f"{CI_HIGH_95_STAT_NAME}{ACROSS_REPEATS_MARKER}"

# Suffixes `compute_pass_majority_metrics` appends to a pass@k metric name.
STD_DEV_ACROSS_RUNS_SUFFIX = f"{STAT_SEPARATOR}std_dev_across_runs"
STD_ERR_ACROSS_RUNS_SUFFIX = f"{STAT_SEPARATOR}std_err_across_runs"
AVG_SAMPLE_STD_DEV_SUFFIX = f"{STAT_SEPARATOR}avg_sample_std_dev"

# Per-task keys in `group_level_metrics`.
ROLLOUT_INFOS_KEY_NAME = "rollout_infos"
NUM_ROLLOUTS_KEY_NAME = "num_rollouts"
EXPECTED_NUM_ROLLOUTS_KEY_NAME = "expected_num_rollouts"
MISSING_NUM_ROLLOUTS_KEY_NAME = "missing_num_rollouts"

POLICY_BASE_URL_KEY_NAME = "policy_base_url"
POLICY_API_KEY_KEY_NAME = "policy_api_key"  # pragma: allowlist secret
POLICY_MODEL_NAME_KEY_NAME = "policy_model_name"
POLICY_MODEL_KEY_NAME = "policy_model"

DEFAULT_HEAD_SERVER_PORT = 11000


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
    # Static inspection avoids network and process side effects. Assigned ports are placeholders,
    # not evidence that a runtime is ready.
    offline: bool = False

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


def _load_config_yaml(config_path):
    """`OmegaConf.load`, converting a YAML syntax error into a clean `ConfigError` naming file + line/column.

    `FileNotFoundError` is left to propagate so callers can report a missing-path error themselves.
    """
    from yaml import YAMLError

    try:
        return OmegaConf.load(config_path)
    except YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
        problem = getattr(e, "problem", None) or str(e).splitlines()[0]
        raise ConfigError(f"Malformed YAML in '{config_path}'{location}: {problem}") from e


def _nemo_gym_openai_requirement() -> Optional[str]:
    """Return nemo-gym's own openai requirement string (name plus specifier, as declared in its metadata).

    Returns ``None`` when the requirement cannot be determined (nemo-gym not
    installed as a distribution, ``packaging`` unavailable, only marker'd
    requirements found).
    """
    try:
        # Lazy imports: `packaging` is not a declared nemo-gym dependency; any
        # import or lookup failure means "constraint unknown".
        from importlib.metadata import requires

        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name

        for req_str in requires("nemo-gym") or []:
            req = Requirement(req_str)
            if canonicalize_name(req.name) == "openai" and req.marker is None:
                return req_str
        return None
    except Exception:
        return None


def _openai_version_matches_nemo_gym_constraint(version: str) -> bool:
    """True when `version` satisfies nemo-gym's own openai requirement.

    head_server_deps normally pins the parent process's openai version into every
    sub-venv for consistency. When the parent environment ships an openai release
    outside nemo-gym's own constraint (e.g. the base image preinstalls a newer
    openai than nemo-gym's cap allows), that pin makes every
    sub-venv resolution unsatisfiable — and the dry-run prefetch then bakes
    venvs that contain nothing but pip. The parser uses this to fail fast (or,
    with the explicit opt-in, to fall back to nemo-gym's own resolution) instead
    of emitting an impossible pin. Returns True (preserving the original
    pin-the-parent behavior) when the constraint cannot be determined.
    """
    req_str = _nemo_gym_openai_requirement()
    if req_str is None:
        return True
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version

        return Requirement(req_str).specifier.contains(Version(version), prereleases=True)
    except Exception:
        return True


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
        Returns the new total config_paths and the extra configs, ordered for merging.

        Two rules decide precedence:

        - A config named in another config's ``config_paths`` is *inner*. The config
          that pulled it in overrides it, however deep the nesting goes.
        - Configs listed together are siblings, in the order they were listed. A later
          sibling overrides an earlier one, and so does everything it pulled in.

        The returned configs are ordered so that a left-to-right ``OmegaConf.merge``
        produces both rules: the include tree flattened so that a config follows
        everything it pulled in, with each subtree kept contiguous.
        """
        config_paths = config_paths.copy()
        # The entries the caller passed; everything appended below was pulled in by one
        # of them, directly or transitively.
        root_count = len(config_paths)
        # Entries pulled in by each entry, parallel to config_paths, in listed order.
        children: List[List[int]] = [[] for _ in config_paths]

        extra_configs: List[DictConfig] = []
        duplicate_config_paths: List[str] = []
        # Just a careful note here that we explicitly mutate config_paths as it is being appended to
        for index, config_path in enumerate(config_paths):
            original_entry = config_path
            config_path = Path(config_path)
            # Search NEMO_GYM_EXTRA_ROOTS, cwd, then the install root (see _resolve_under_cwd_or_install).
            if config_path.is_absolute():
                searched_locations = [config_path]
            else:
                searched_locations = [root / config_path for root in component_search_roots()]
            config_path = _resolve_under_cwd_or_install(original_entry)
            if not config_path.exists() and (canonical_entry := legacy_config_path_alias(original_entry)):
                canonical_path = _resolve_under_cwd_or_install(canonical_entry)
                if canonical_path.exists():
                    logger.warning(f"Config path `{original_entry}` is deprecated; use `{canonical_entry}`.")
                    config_paths[index] = canonical_entry
                    config_path = canonical_path

            try:
                extra_config = _load_config_yaml(config_path)
            except FileNotFoundError as e:
                searched = "\n".join(f"  - {p}" for p in searched_locations)
                raise ConfigPathNotFoundError(
                    f"""config_paths entry '{original_entry}' was not found. Looked in:
{searched}
Check the path is spelled correctly and is relative to your working directory, an extra root
(NEMO_GYM_EXTRA_ROOTS / --search-dir), or the Gym install root."""
                ) from e
            for new_config_path in extra_config.get(CONFIG_PATHS_KEY_NAME) or []:
                if new_config_path not in config_paths:
                    config_paths.append(new_config_path)
                    children.append([])
                    children[index].append(len(config_paths) - 1)
                else:
                    duplicate_config_paths.append(new_config_path)
            extra_configs.append(extra_config)

        if duplicate_config_paths:
            duplicate_config_paths_str = "".join(f"- {p}\n" for p in duplicate_config_paths)
            # Diagnostics go to stderr so that `--json` output on stdout stays machine-readable.
            print(
                f"""Found configs that reference the same source config path. You may want to double check whether the configs you have need to use different configs for the same server.
In cases like these, you may want to consider using the `inherit_from` OmegaConf directive e.g. '++my_specific_server=${{inherit_from:generic_server}}' and then overriding config parameters in `my_specific_server`.
Duplicate config paths:
{duplicate_config_paths_str}""",
                file=sys.stderr,
            )

        # Flatten the include tree so that every config merges after -- and therefore
        # overrides -- the ones it pulled in, and each subtree stays contiguous in
        # listed order, so a later sibling and its own includes override an earlier
        # sibling's. Iterative post-order, since include chains can nest arbitrarily.
        merge_order: List[int] = []
        pending: List[Tuple[int, bool]] = [(root, False) for root in reversed(range(root_count))]
        while pending:
            index, children_visited = pending.pop()
            if children_visited:
                merge_order.append(index)
                continue
            pending.append((index, True))
            pending.extend((child, False) for child in reversed(children[index]))
        extra_configs = [extra_configs[i] for i in merge_order]

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

        Without this, `gym env start` with an empty/omitted `config_paths` starts the head server and Ray
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
        skip_verification: Optional[bool] = None,
        skip_verification_reward: Optional[float] = None,
        probe_ports: bool = True,
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
                    if probe_ports:
                        port = _find_open_port_using_range(
                            disallowed_ports=disallowed_ports,
                            port_range_low=port_range_low,
                            port_range_high=port_range_high,
                        )
                    else:
                        # Offline resolution must not imply that a runnable port was allocated.
                        port = -1
                    run_server_config_dict["port"] = port
                    if probe_ports:
                        disallowed_ports.append(port)  # Disallow newly allocated port.
                else:
                    # Port already exists, add it to the disallowed list.
                    disallowed_ports.append(run_server_config_dict["port"])

                if server_instance_config.SERVER_TYPE == "responses_api_agents":
                    if skip_verification is not None and SKIP_VERIFICATION_KEY_NAME not in run_server_config_dict:
                        run_server_config_dict[SKIP_VERIFICATION_KEY_NAME] = skip_verification
                    if (
                        skip_verification_reward is not None
                        and SKIP_VERIFICATION_REWARD_KEY_NAME not in run_server_config_dict
                    ):
                        run_server_config_dict[SKIP_VERIFICATION_REWARD_KEY_NAME] = skip_verification_reward

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

    def _agent_instances(self, global_config_dict: DictConfig) -> List[_AgentInstance]:
        """Return every top-level agent instance in the config."""
        instances: List[_AgentInstance] = []
        for name, value in global_config_dict.items_ex(resolve=False):
            if name in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS or not isinstance(value, DictConfig):
                continue
            if AGENT_SERVER_TYPE_KEY_NAME not in value:
                continue
            agents = value[AGENT_SERVER_TYPE_KEY_NAME]
            # Not our error to report: the type config pins exactly one, and almost-server detection flags it.
            if not isinstance(agents, DictConfig) or len(agents) != 1:
                continue
            agent_type = str(next(iter(agents)))
            server_config = agents._get_node(agent_type)
            # An unset block is read as a node rather than resolved: resolving raises MissingMandatoryValue,
            # which is not a ConfigError, so the CLI would print a traceback instead of the usual report.
            if not isinstance(server_config, DictConfig):
                continue
            instances.append(_AgentInstance(name=str(name), agent_type=agent_type, server_config=server_config))
        return instances

    @staticmethod
    def _resources_server_reference(server_config: DictConfig) -> Optional[DictConfig]:
        """The agent's `resources_server` block, or None when it declares none.

        Selected rather than indexed because a self-contained agent omits the key, which a struct-mode
        config rejects outright.
        """
        reference = OmegaConf.select(server_config, "resources_server")
        return reference if isinstance(reference, DictConfig) else None

    @staticmethod
    def _model_server_reference(server_config: DictConfig) -> Optional[DictConfig]:
        """The agent's `model_server` block, or None when it declares none."""
        reference = OmegaConf.select(server_config, "model_server")
        return reference if isinstance(reference, DictConfig) else None

    def raise_on_unsupported_model_pairings(self, global_config_dict: DictConfig) -> None:
        """Reject resource/model bindings that explicitly disallow the selected model adapter.

        This runs while parsing the merged config, before Ray or any server subprocess starts. A resources
        server opts in with ``allowed_model_types``; existing configs that do not declare it stay unrestricted.
        """
        if pairing_override_enabled(global_config_dict):
            return

        restrictions: List[List[str]] = []
        rejected: List[Tuple[_AgentInstance, str, str, str, List[str]]] = []
        for agent in self._agent_instances(global_config_dict):
            resources_reference = self._resources_server_reference(agent.server_config)
            if not resources_reference or resources_reference.get("type") != RESOURCES_SERVER_TYPE_KEY_NAME:
                continue
            resources_server_name = resources_reference.get("name")
            allowed = allowed_model_types_for(global_config_dict, resources_server_name)
            if allowed is None:
                continue

            model_reference = self._model_server_reference(agent.server_config)
            if not model_reference or model_reference.get("type") != MODEL_SERVER_TYPE_KEY_NAME:
                continue
            model_server_name = model_reference.get("name")
            model_type = model_type_for(global_config_dict, model_server_name)
            # ``dummy_model`` is injected for parser clients that only need benchmark data or static config
            # inspection (for example, ``gym eval prepare``). It is not a runtime model selection.
            if model_type is None or model_type == "dummy_model":
                continue

            restrictions.append(allowed)
            if model_type not in allowed:
                rejected.append((agent, str(resources_server_name), str(model_server_name), model_type, allowed))

        if not rejected:
            return

        rejected_list = "\n".join(
            f"  - agent '{agent.name}' uses model server '{model_server_name}' (type '{model_type}'), "
            f"but resources server '{resources_server_name}' accepts only: {', '.join(allowed)}"
            for agent, resources_server_name, model_server_name, model_type, allowed in rejected
        )
        supported = [
            model_type
            for model_type in restrictions[0]
            if all(model_type in restriction for restriction in restrictions[1:])
        ]
        if supported:
            remedy = (
                f"Select a compatible model adapter (for example, --model-type {supported[0]}). "
                f"Supported model types: {', '.join(supported)}."
            )
        else:
            remedy = "No single model type satisfies all of these resources servers."
        raise UnsupportedModelPairingError(
            "The selected Gym model-server adapter is not compatible with this workload:\n"
            f"{rejected_list}\n\n"
            f"{remedy} Or pass --allow-unsupported-pairing (or set "
            f"{ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME}=1) to bypass the check."
        )

    def _runs_against_a_resources_server(self, server_config: DictConfig) -> bool:
        """True when the agent has a task to hand over, so another agent can take its place.

        Self-contained agents own their environment and declare no `resources_server`; swapping one out
        would leave the incoming agent with nothing to bind to.
        """
        return self._resources_server_reference(server_config) is not None

    def _is_unbound_agent(self, server_config: DictConfig) -> bool:
        """True when the agent declares a `resources_server` but leaves its name unset, marking it a swap source."""
        # Absent is not unset: self-contained agents omit the key entirely and must never match.
        reference = self._resources_server_reference(server_config)
        return reference is not None and OmegaConf.is_missing(reference, "name")

    def compose_unbound_agent(
        self, global_config_dict: DictConfig, held_agent_overrides: Optional[DictConfig] = None
    ) -> None:
        """Rehost every other agent instance on the config's unbound agent, then drop that agent.

        `held_agent_overrides` are command line overrides keyed by the name each instance is renamed to.
        """
        instances = self._agent_instances(global_config_dict)
        sources = [instance for instance in instances if self._is_unbound_agent(instance.server_config)]
        if not sources:
            return

        if len(sources) > 1:
            raise AgentCompositionError(
                f"{len(sources)} agent instances leave their 'resources_server' unset, so there is no single "
                f"agent to compose onto the others: {', '.join(sorted(repr(s.name) for s in sources))}. "
                f"Load exactly one standalone agent config, or bind the others in your own config."
            )

        source = sources[0]
        targets = [
            instance
            for instance in instances
            if instance.name != source.name and self._runs_against_a_resources_server(instance.server_config)
        ]
        if not targets:
            raise AgentCompositionError(
                f"Agent instance '{source.name}' leaves its 'resources_server' unset, but the merged config "
                f"defines no other agent instance to rehost it on. Select an environment, benchmark, or "
                f"resources server alongside the agent so there is a task for it to run."
            )

        self._raise_on_unsupported_pairing(global_config_dict, source, targets)

        renames = {target.name: self._composed_instance_name(target, source.agent_type) for target in targets}
        self._raise_on_name_collision(global_config_dict, renames, source.name)

        # Struct mode would reject the key removals below - we need open_dict to allow it.
        with open_dict(global_config_dict):
            # delete the source instance before any renames to avoid corner cases
            global_config_dict.pop(source.name)
            for target in targets:
                composed = deepcopy(source.server_config)
                self._carry_over_agent_bindings(target.server_config, composed)
                self._apply_held_agent_override(
                    held_agent_overrides, renames[target.name], source.agent_type, composed
                )

                # extract the target instance, remove the old agent config, add the new agent
                # and add the whole thing back to the config under the new name
                instance = global_config_dict.pop(target.name)
                agents = instance[AGENT_SERVER_TYPE_KEY_NAME]
                agents.pop(target.agent_type)
                agents[source.agent_type] = composed
                global_config_dict[renames[target.name]] = instance

            self._raise_on_outdated_routing(global_config_dict, renames)
            self._route_rows_stamped_before_the_swap(global_config_dict, renames)

        self._raise_on_unapplied_agent_overrides(held_agent_overrides, set(renames.values()))

    @staticmethod
    def _composed_instance_name(target: _AgentInstance, agent_type: str) -> str:
        """Rename the instance after swapping the agent.

        Substituting the trailing agent type keeps the environment prefix that makes the name readable
        (`gpqa_mcqa_simple_agent` -> `gpqa_mcqa_hermes_agent`); names not ending in their agent type just
        gain the suffix. Safe because routing resolves `task_source` through the resources server edge,
        not through this name.
        """
        stem = target.name.removesuffix(f"_{target.agent_type}").removesuffix(target.agent_type).rstrip("_")
        if stem == target.name:
            # The name does not end in its agent type, so fall back to the generic suffix most of them
            # share; without this the whole old name survives and the result carries two agent names.
            stem = target.name.removesuffix("_agent").rstrip("_")
        return f"{stem}_{agent_type}" if stem else agent_type

    @staticmethod
    def _raise_on_name_collision(global_config_dict: DictConfig, renames: dict, source_name: str) -> None:
        # The source instance is dropped by the composition, so its name is free to reuse.
        taken = set(global_config_dict) - set(renames) - {source_name}
        clashes = sorted(
            f"{old} -> {new}" for old, new in renames.items() if new in taken or list(renames.values()).count(new) > 1
        )
        if clashes:
            raise AgentCompositionError(
                f"Composing would give two instances the same name: {', '.join(clashes)}. "
                f"Rename the environment's agent instance or compose the config manually."
            )

    @staticmethod
    def _raise_on_outdated_routing(global_config_dict: DictConfig, renames: Dict[str, str]) -> None:
        """Reject routing that sends rows to an instance the swap renamed away.

        Destinations name a server that has to exist: `agent_name`, `agent_map` values and `fan_out`
        entries. Their keys are matching bases read off the data, so those may name the old instance.
        """
        outdated = []
        selected = global_config_dict.get("agent_name")
        if selected in renames:
            outdated.append(("agent_name", selected))

        declared = global_config_dict.get("agent_map")
        if isinstance(declared, DictConfig):
            outdated += [(f"agent_map[{key}]", value) for key, value in declared.items() if value in renames]

        listed = global_config_dict.get("fan_out")
        if isinstance(listed, DictConfig):
            outdated += [
                (f"fan_out[{key}]", agent) for key, agents in listed.items() for agent in agents if agent in renames
            ]
        if not outdated:
            return
        listing = "\n".join(f"  - {where}: '{name}' is now '{renames[name]}'" for where, name in outdated)
        raise AgentCompositionError(
            f"""Routing names agent instances that no longer exist once the agent is swapped:
{listing}

Use the name the composed config reports."""
        )

    @staticmethod
    def _route_rows_stamped_before_the_swap(global_config_dict: DictConfig, renames: Dict[str, str]) -> None:
        """Map the pre-swap instance name onto the composed one, for rows stamped before it happened.

        Only a matching base is added, so a route the user declared still wins.
        """
        declared = global_config_dict.get("agent_map")
        routes = dict(renames)
        if isinstance(declared, DictConfig):
            routes.update({str(key): value for key, value in declared.items()})
        global_config_dict["agent_map"] = routes

    @staticmethod
    def apply_legacy_agent_aliases(global_config_dict: DictConfig) -> None:
        """Route legacy reasoning-gym agent names to their canonical instances."""
        declared = global_config_dict.get("agent_map")
        routes = dict(declared) if isinstance(declared, DictConfig) else {}
        active_aliases = {}
        for legacy, canonical in LEGACY_AGENT_ALIASES.items():
            destination = routes.get(canonical, canonical)
            if legacy not in global_config_dict and destination in global_config_dict:
                active_aliases[legacy] = destination
        if not active_aliases:
            return

        deprecated_uses = set()
        selected = global_config_dict.get("agent_name")
        if selected in active_aliases:
            deprecated_uses.add(str(selected))
            global_config_dict["agent_name"] = active_aliases[selected]

        for key, destination in list(routes.items()):
            if destination in active_aliases:
                deprecated_uses.add(str(destination))
                routes[key] = active_aliases[destination]
        for legacy, destination in active_aliases.items():
            routes.setdefault(legacy, destination)
        global_config_dict["agent_map"] = routes

        fan_out = global_config_dict.get("fan_out")
        if isinstance(fan_out, DictConfig):
            for key, destinations in fan_out.items():
                if not isinstance(destinations, (list, ListConfig)):
                    continue
                replacements = [active_aliases.get(destination, destination) for destination in destinations]
                deprecated_uses.update(destination for destination in destinations if destination in active_aliases)
                fan_out[key] = replacements

        if deprecated_uses:
            replacements = ", ".join(f"`{legacy}` -> `{active_aliases[legacy]}`" for legacy in sorted(deprecated_uses))
            logger.warning(f"Legacy agent names are deprecated; use {replacements}.")

    def _raise_on_unsupported_pairing(
        self, global_config_dict: DictConfig, source: _AgentInstance, targets: List[_AgentInstance]
    ) -> None:
        """Reject swapping `source` onto any target whose resources server does not declare support for it.

        Compatibility is declared verifier-side because that is where it is known: an environment's author
        knows which harnesses score their task correctly, while a generic harness cannot know that for every
        environment. A server that declares nothing accepts any harness.
        """
        if pairing_override_enabled(global_config_dict):
            return

        restrictions: List[Set[str]] = []
        rejected: List[Tuple[_AgentInstance, List[str]]] = []
        for target in targets:
            reference = self._resources_server_reference(target.server_config)
            allowed = allowed_agents_for(global_config_dict, reference.get("name") if reference else None)
            if allowed is None:
                continue
            restrictions.append(set(allowed))
            if source.agent_type not in allowed:
                rejected.append((target, allowed))
        if not rejected:
            return

        # Report every rejected instance at once: a config can bring in several, and fixing them one
        # error at a time means one full re-resolve per instance.
        rejected_list = "\n".join(
            f"  - {target.name} uses {target.server_config['resources_server']['name']} "
            f"and accepts {', '.join(allowed)}"
            for target, allowed in rejected
        )
        supported = sorted(set.intersection(*restrictions))
        remedy = f"Select one of: {', '.join(supported)}." if supported else "No single agent satisfies all of them."
        raise UnsupportedAgentPairingError(
            f"""'{source.agent_type}' is not declared compatible with {len(rejected)} of the agent instance(s) """
            f"""it would replace, so it cannot be scored correctly:
{rejected_list}

{remedy} Or pass --allow-unsupported-pairing (or set {ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME}=1) to bypass \
the check."""
        )

    def _composed_instance_names(self, config_dict: DictConfig) -> set:
        """The instance names composition will produce, worked out before it runs."""
        instances = self._agent_instances(config_dict)
        sources = [instance for instance in instances if self._is_unbound_agent(instance.server_config)]
        if len(sources) != 1:
            return set()
        return {
            self._composed_instance_name(target, sources[0].agent_type)
            for target in instances
            if target.name != sources[0].name
        }

    def _hold_back_composed_agent_overrides(
        self, cli_global_config_dict: DictConfig, config_dict: DictConfig
    ) -> DictConfig:
        """Take command line overrides naming an instance composition is about to create out of the dict.

        That instance does not exist yet, so merging them now would build a partial server beside it.
        Returned to be applied to the composed agent instead.
        """
        composed_names = self._composed_instance_names(config_dict)
        held = OmegaConf.create({})
        for name in list(cli_global_config_dict.keys()):
            if name not in composed_names:
                continue
            with open_dict(cli_global_config_dict), open_dict(held):
                held[name] = cli_global_config_dict.pop(name)
        return held

    @staticmethod
    def _apply_held_agent_override(
        held_agent_overrides: Optional[DictConfig], name: str, agent_type: str, composed: DictConfig
    ) -> None:
        """Merge the override held for `name` onto the composed agent, in place, after the bindings."""
        if held_agent_overrides is None:
            return
        override = OmegaConf.select(held_agent_overrides, f"{name}.{AGENT_SERVER_TYPE_KEY_NAME}.{agent_type}")
        if not isinstance(override, DictConfig):
            return
        # Struct mode is what makes a field the agent does not declare an error rather than a silent add.
        OmegaConf.set_struct(composed, True)
        composed.merge_with(override)

    @staticmethod
    def _raise_on_unapplied_agent_overrides(held_agent_overrides: Optional[DictConfig], composed_names: set) -> None:
        """Report held overrides that named an instance composition did not produce."""
        unapplied = sorted(name for name in (held_agent_overrides or {}) if name not in composed_names)
        if not unapplied:
            return
        raise UnsupportedAgentOverrideError(
            "Command line overrides name agent instances that do not exist:\n"
            + "\n".join(f"  - {name}" for name in unapplied)
            + "\nUse the instance name the composed config reports, e.g. from `gym env resolve`."
        )

    @staticmethod
    def _carry_over_agent_bindings(original: DictConfig, composed: DictConfig) -> None:
        """Move the environment's bindings onto the composed agent config, in place.

        A binding left explicitly unset is carried over still unset, so it is reported rather than
        silently resolving to whatever the incoming agent happens to declare. It needs its own branch
        because OmegaConf reports a '???' value as absent.
        """
        for key in _COMPOSED_AGENT_CARRY_OVER_KEYS:
            if key in original:
                composed[key] = deepcopy(original[key])
            elif OmegaConf.is_missing(original, key):
                composed[key] = MISSING

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
                raise InheritPathNotFoundError(f"Path specified does not exist in config: {path}")

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

        cli_global_config_dict = (
            DictConfig(dict()) if parse_config.skip_load_from_cli else self.parse_global_config_dict_from_cli()
        )
        # @bxyu-nvidia: Hydra returns `cli_global_config_dict` as a struct, but this causes problems in downstream swapping/merging.
        OmegaConf.set_struct(cli_global_config_dict, False)

        # Command line overrides function input.
        initial_global_config_dict = OmegaConf.create(parse_config.initial_global_config_dict or dict())
        global_config_dict: DictConfig = OmegaConf.merge(initial_global_config_dict, cli_global_config_dict)

        # Load the env.yaml config. We load it early so that people can use it to conveniently store config paths.
        # Search NEMO_GYM_EXTRA_ROOTS, cwd, then the install root.
        if parse_config.dotenv_path:
            dotenv_path = parse_config.dotenv_path
        else:
            dotenv_path = _resolve_under_cwd_or_install("env.yaml")

        dotenv_extra_config = DictConfig({})
        if dotenv_path.exists() and not parse_config.skip_load_from_dotenv:
            dotenv_extra_config = _load_config_yaml(dotenv_path)

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

        # Returned in merge order: a config follows everything it pulled in, so it
        # overrides them; siblings in listed order, so a later one overrides earlier.
        config_paths, extra_configs = self.load_extra_config_paths(config_paths)

        # Dot env overrides previous configs
        extra_configs.append(dotenv_extra_config)

        # Merge config dicts
        # global_config_dict is the last config arg here since we want command line args to override everything else.
        # An agent override naming an instance no config defines is addressed to one composition is about to
        # create, so it is held back from this merge, which has nowhere to put it, and applied there instead.
        held_agent_overrides = self._hold_back_composed_agent_overrides(
            cli_global_config_dict, OmegaConf.merge(*extra_configs, initial_global_config_dict)
        )
        # Rebuilt rather than reused: the command line dict was trimmed after the early merge above.
        global_config_dict = OmegaConf.merge(
            *extra_configs, OmegaConf.merge(initial_global_config_dict, cli_global_config_dict)
        )

        # Update the config paths after postprocessing
        if config_paths:
            with open_dict(global_config_dict):
                global_config_dict[CONFIG_PATHS_KEY_NAME] = config_paths

        self._recursively_swap_keys(global_config_dict)

        # Must run after the swap above (inherited bindings must exist to carry over) and before the
        # missing-value check below (it removes the unbound agent instance that still carries '???').
        self.compose_unbound_agent(global_config_dict, held_agent_overrides)
        self.apply_legacy_agent_aliases(global_config_dict)

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
            # Diagnostics go to stderr so that `--json` output on stdout stays machine-readable.
            rich.print("[yellow]═══════════════════════════════════════════════════[/yellow]", file=sys.stderr)
            rich.print("[yellow]Configuration Warnings: Almost-Servers Detected[/yellow]", file=sys.stderr)
            rich.print("[yellow]═══════════════════════════════════════════════════[/yellow]", file=sys.stderr)

            for server_name, error in almost_servers:
                rich.print(format_almost_server_warning(server_name, error), file=sys.stderr)

            rich.print("[yellow]═══════════════════════════════════════════════════[/yellow]\n", file=sys.stderr)

            error_on_almost_servers = global_config_dict.get("error_on_almost_servers", True)
            if error_on_almost_servers:
                config_dict_to_log = deepcopy(global_config_dict)
                recursively_hide_secrets(config_dict_to_log)
                config_to_log_yaml = OmegaConf.to_yaml(config_dict_to_log)

                error_msg = f"""Found {len(almost_servers)} almost-server(s) with validation errors. Fix the issues above or set error_on_almost_servers=false to bypass this error.
Found global config dict yaml:
{config_to_log_yaml}"""

                raise AlmostServerError(error_msg)

        server_instance_configs = self.filter_for_server_instance_configs(global_config_dict)
        self.raise_on_unsupported_model_pairings(global_config_dict)

        with open_dict(global_config_dict):
            use_absolute_ip = global_config_dict.setdefault(USE_ABSOLUTE_IP, False)
        if use_absolute_ip and not parse_config.offline:
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
            skip_verification=global_config_dict.get(SKIP_VERIFICATION_KEY_NAME),
            skip_verification_reward=global_config_dict.get(SKIP_VERIFICATION_REWARD_KEY_NAME),
            probe_ports=not parse_config.offline,
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
            head_server_deps = [
                # The ray version is very sensitive. The children ray versions must exactly match those of the parent ray.
                # The ray extra [default] should also exactly match the extra in the top-level Gym pyproject.toml.
                f"ray[default]=={ray_version}",
            ]
            # OpenAI version is also sensitive since it changes so often and may introduce subtle
            # incompatibilities — but only pin the parent's version when nemo-gym's own constraint
            # accepts it; otherwise the sub-venv resolutions are unsatisfiable and the venvs come
            # out empty (see _openai_version_matches_nemo_gym_constraint).
            allow_openai_skew = global_config_dict.setdefault(ALLOW_OPENAI_VERSION_SKEW_KEY_NAME, False)
            if not isinstance(allow_openai_skew, bool):
                raise ConfigError(
                    f"{ALLOW_OPENAI_VERSION_SKEW_KEY_NAME} must be a boolean (true/false), got {allow_openai_skew!r}."
                )
            if _openai_version_matches_nemo_gym_constraint(openai_version):
                head_server_deps.append(f"openai=={openai_version}")
            elif allow_openai_skew:
                logging.warning(
                    f"Not pinning the parent process's openai=={openai_version} into server venvs: "
                    f"it does not satisfy nemo-gym's own constraint ({_nemo_gym_openai_requirement()}). "
                    "Server venvs resolve openai from nemo-gym's constraint instead, so the parent and "
                    "the servers exchange requests across the HTTP boundary with different openai "
                    f"versions ({ALLOW_OPENAI_VERSION_SKEW_KEY_NAME}=true)."
                )
            else:
                raise ConfigError(
                    f"The parent process runs openai=={openai_version}, which does not satisfy "
                    f"nemo-gym's own openai constraint ({_nemo_gym_openai_requirement()}). Pinning it "
                    "into the server venvs would make every server venv resolution unsatisfiable "
                    "(and the dry-run prefetch would bake venvs containing nothing but pip). "
                    "Install an openai version compatible with nemo-gym in the parent environment, "
                    f"or set {ALLOW_OPENAI_VERSION_SKEW_KEY_NAME}=true to let server venvs resolve "
                    "openai from nemo-gym's own constraint (the parent and the servers then run "
                    "different openai versions across the HTTP boundary)."
                )
            # Telemetry, when enabled. Server venvs install nemo-gym[dev], not
            # nemo-gym[telemetry], so without this the orchestrator would export spans and
            # every server process would silently have no telemetry at all — a trace with a
            # hole in the middle. Translating the config into the environment first is what
            # lets server_venv_requirements() see that telemetry is on.
            configure_telemetry_env(telemetry_config_from_global_config(global_config_dict))
            head_server_deps.extend(server_venv_requirements())

            global_config_dict[HEAD_SERVER_DEPS_KEY_NAME] = head_server_deps

            # Constrain python version since ray is sensitive to this.
            global_config_dict[PYTHON_VERSION_KEY_NAME] = python_version()

            # Skip venv setup is opt-in and defaults to False.
            global_config_dict.setdefault(SKIP_VENV_IF_PRESENT_KEY_NAME, False)

            global_config_dict.setdefault(DRY_RUN_KEY_NAME, False)

            # How long `gym env start` waits for the model endpoints named in the config to accept
            # a connection. Generous because vLLM can take minutes to load weights; 0 skips it.
            global_config_dict.setdefault(MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME, 600)

            # Artifact roots. `results_dir` is where servers write run artifacts;
            # `cache_dir` is where they keep reusable setup trees (clones, venvs,
            # toolchains). Overridable independently, so a run can point results at
            # a shared filesystem while caches stay on fast local (or baked
            # container) storage. Normalized to absolute here: children resolve
            # relative paths against their own cwd, which differs per process.
            for dir_key_name, dir_default in (
                (RESULTS_DIR_KEY_NAME, RESULTS_DIR),
                (CACHE_DIR_KEY_NAME, CACHE_DIR),
            ):
                dir_value = global_config_dict.setdefault(dir_key_name, str(dir_default))
                if not isinstance(dir_value, str) or not dir_value:
                    raise ConfigError(f"{dir_key_name} must be a non-empty path string, got {dir_value!r}.")
                global_config_dict[dir_key_name] = str(Path(dir_value).expanduser().resolve())

            # UV related configuration
            # UV caching directory overrides to local folders, under the cache root.
            global_config_dict.setdefault(
                UV_CACHE_DIR_KEY_NAME, str(Path(global_config_dict[CACHE_DIR_KEY_NAME]) / "uv")
            )
            # Runtime subprocesses inherit the configured cache directory.
            if not parse_config.offline:
                environ["UV_CACHE_DIR"] = global_config_dict[UV_CACHE_DIR_KEY_NAME]
            # By default, build the directories in their individual folders using the root repository
            # e.g. WORKING_DIR/responses_api_models/my_server
            # Deliberately anchored at WORKING_DIR rather than the cache root: venv
            # trees don't relocate safely once built, and the key is independently
            # overridable for deployments that want them elsewhere.
            global_config_dict.setdefault(UV_VENV_DIR_KEY_NAME, str(WORKING_DIR))

        if parse_config.hide_secrets:  # pragma: no cover
            recursively_hide_secrets(global_config_dict)

        # Set up exporters and log config. This must happen at the very last step.
        if not parse_config.offline:  # pragma: no cover
            setup_exporters(global_config_dict)

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


def maybe_get_global_config_dict() -> Optional[DictConfig]:
    """The global config dict when this process already has one; never triggers a CLI parse.

    Returns the cached dict, or the one the parent injected via
    NEMO_GYM_CONFIG_DICT (caching it), or None in a bare process. Library code
    that only wants to *consult* the config should use this instead of
    `get_global_config_dict`, which falls through to a full CLI/hydra parse.
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

    return None


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
    existing_global_config_dict = maybe_get_global_config_dict()
    if existing_global_config_dict is not None:
        return existing_global_config_dict

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
    try:
        global_config_dict = global_config_dict_parser_cls().parse(global_config_dict_parser_config)
    except InterpolationResolutionError as e:
        # Same class of user error as an unset '???' (see raise_on_missing_values), so report it the same
        # way instead of letting omegaconf's traceback reach the top level. Covers both a missing `${key}`
        # (InterpolationKeyError) and a failing resolver such as `${oc.env:VAR}`, which carries its own
        # message and so is passed through as-is.
        match = re.search(r"Interpolation key '([^']+)' not found", str(e))
        if not match:
            raise ConfigInterpolationError(str(e)) from e
        key = match.group(1)
        raise ConfigInterpolationError(
            f"""Config value '{e.full_key}' references '{key}', which is not set after merging.

Provide it via a CLI override, in env.yaml, or in a config you pass via config_paths.
For example, on the command line:
  ++{key}=<value>"""
        ) from e

    _GLOBAL_CONFIG_DICT = global_config_dict


def get_first_server_config_dict(global_config_dict: DictConfig, top_level_path: str) -> DictConfig:
    # Traverse three levels deep total
    server_config_dict = global_config_dict[top_level_path]
    server_config_dict = list(server_config_dict.values())[0]
    server_config_dict = list(server_config_dict.values())[0]

    return server_config_dict


def allowed_agents_for(global_config_dict: DictConfig, resources_server_name: Optional[str]) -> Optional[List[str]]:
    """The agent types `resources_server_name` declares support for, or None when it declares none.

    A bare string is accepted as a single entry: an `++...allowed_agents=name` override arrives before the
    server model is validated, and iterating it would read the name as its characters.
    """
    instance = global_config_dict.get(resources_server_name) if resources_server_name else None
    if not isinstance(instance, DictConfig) or RESOURCES_SERVER_TYPE_KEY_NAME not in instance:
        return None
    servers = instance[RESOURCES_SERVER_TYPE_KEY_NAME]
    if not isinstance(servers, DictConfig) or len(servers) != 1:
        return None
    implementation = next(iter(servers))
    declared = servers[implementation].get(ALLOWED_AGENTS_KEY_NAME)
    if not declared:
        return None
    if isinstance(declared, str):
        return [declared]
    if not isinstance(declared, ListConfig):
        raise ConfigError(
            f"'{resources_server_name}.{RESOURCES_SERVER_TYPE_KEY_NAME}.{implementation}."
            f"{ALLOWED_AGENTS_KEY_NAME}' must be a list of agent types, got {type(declared).__name__}."
        )
    return [str(name) for name in declared]


def allowed_model_types_for(
    global_config_dict: DictConfig, resources_server_name: Optional[str]
) -> Optional[List[str]]:
    """The model adapter types `resources_server_name` declares support for, or None when unrestricted."""
    instance = global_config_dict.get(resources_server_name) if resources_server_name else None
    if not isinstance(instance, DictConfig) or RESOURCES_SERVER_TYPE_KEY_NAME not in instance:
        return None
    servers = instance[RESOURCES_SERVER_TYPE_KEY_NAME]
    if not isinstance(servers, DictConfig) or len(servers) != 1:
        return None
    implementation = next(iter(servers))
    declared = servers[implementation].get(ALLOWED_MODEL_TYPES_KEY_NAME)
    if not declared:
        return None
    if isinstance(declared, str):
        return [declared]
    if not isinstance(declared, ListConfig):
        raise ConfigError(
            f"'{resources_server_name}.{RESOURCES_SERVER_TYPE_KEY_NAME}.{implementation}."
            f"{ALLOWED_MODEL_TYPES_KEY_NAME}' must be a list of model types, got {type(declared).__name__}."
        )
    return [str(name) for name in declared]


def model_type_for(global_config_dict: DictConfig, model_server_name: Optional[str]) -> Optional[str]:
    """The single model adapter type hosted by `model_server_name`, or None when it cannot be determined."""
    instance = global_config_dict.get(model_server_name) if model_server_name else None
    if not isinstance(instance, DictConfig) or MODEL_SERVER_TYPE_KEY_NAME not in instance:
        return None
    models = instance[MODEL_SERVER_TYPE_KEY_NAME]
    if not isinstance(models, DictConfig) or len(models) != 1:
        return None
    return str(next(iter(models)))


def pairing_override_enabled(global_config_dict: DictConfig) -> bool:
    """True when a declared agent/model compatibility guard has been explicitly waived."""
    return bool(global_config_dict.get(ALLOW_UNSUPPORTED_PAIRING_KEY_NAME)) or getenv(
        ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME, ""
    ).lower() not in ("", "0", "false")


def agents_by_resources_server(global_config_dict: DictConfig) -> Dict[str, List[str]]:
    """Invert the agent -> resources_server edges of a merged config.

    Returns {resources server instance name: [agent instance names referencing it]}. This is the
    lookup that routes task_source-stamped rows to an agent (and, transitionally, lets collate
    dual-stamp a legacy agent_ref). Template placeholders (``name: ???``) and malformed blocks are
    skipped: they are not routable candidates.
    """
    result: Dict[str, List[str]] = defaultdict(list)
    for instance_name, block in global_config_dict.items():
        if not isinstance(block, DictConfig) or "responses_api_agents" not in block:
            continue
        try:
            inner = get_first_server_config_dict(global_config_dict, instance_name)
            rs_name = (inner.get("resources_server") or {}).get("name")
        except Exception:
            continue
        if isinstance(rs_name, str):
            result[rs_name].append(str(instance_name))
    return result


def dataset_agent_pins(global_config_dict: DictConfig, instance_name: str) -> List[str]:
    """Distinct dataset-level ``agent:`` pins declared by one server instance, in declaration order."""
    try:
        inner = get_first_server_config_dict(global_config_dict, instance_name)
    except Exception:
        return []
    pins: List[str] = []
    for dataset in inner.get("datasets") or []:
        if not isinstance(dataset, (dict, DictConfig)):
            continue
        pin = dataset.get("agent")
        if isinstance(pin, str) and pin not in pins:
            pins.append(pin)
    return pins


def resolve_dataset_agent(
    global_config_dict: DictConfig, declaring_instance_name: str, pin: Optional[str] = None
) -> str:
    """Resolve the agent that runs a dataset declared by ``declaring_instance_name``.

    Single source of truth for dataset -> agent routing, shared by benchmark discovery,
    preparation, manifest validation, and rollout dispatch, so they can never disagree.
    First hit wins:

    1. ``pin`` (the dataset's ``agent:`` key) — validated: it must name the declaring agent
       itself, or an agent referencing the declaring resources server. Anything else is a hard
       error, never a silent re-route (rows carry only the declaring instance name).
    2. The declaring agent block itself.
    3. The unique agent referencing the declaring resources server; zero or 2+ is a hard error.
    """
    block = global_config_dict.get(declaring_instance_name)
    is_agent = isinstance(block, DictConfig) and "responses_api_agents" in block

    if is_agent:
        if pin is not None and pin != declaring_instance_name:
            raise ConfigError(
                f"dataset declared by agent {declaring_instance_name!r} pins agent {pin!r}, but rows are "
                f"dispatched to the declaring agent, so the pin would silently not apply. Drop the `agent:` "
                "key, or declare the dataset on the resources server and pin there."
            )
        return str(declaring_instance_name)

    candidates = agents_by_resources_server(global_config_dict).get(str(declaring_instance_name), [])
    if pin is not None:
        if pin not in candidates:
            raise ConfigError(
                f"the dataset pins agent {pin!r}, but no agent of that name references resources server "
                f"{declaring_instance_name!r} (rows are dispatched along the agent -> resources server edge). "
                f"Agents referencing it: {sorted(candidates) if candidates else 'none'}."
            )
        return pin
    if not candidates:
        raise ConfigError(
            f"no agent in the running config references resources server {declaring_instance_name!r}. "
            "Point an agent's `resources_server` at it, or declare the dataset on the agent."
        )
    if len(candidates) > 1:
        raise ConfigError(
            f"{len(candidates)} agents reference this resources server ({sorted(candidates)}). "
            "Pin the harness with an `agent:` key on the dataset declaration, or pass "
            f"+agent_map={{{declaring_instance_name}: <agent>}} to pick one at run time."
        )
    return candidates[0]


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
