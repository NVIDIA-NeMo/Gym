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
"""Ray support for Gym servers: detecting Ray, starting or joining a cluster, and shutting down a cluster Gym started."""

import ast
import sys
from importlib.util import find_spec
from pathlib import Path

from omegaconf import DictConfig, open_dict

from nemo_gym.config.global_config import (
    RAY_HEAD_NODE_ADDRESS_KEY_NAME,
    get_global_config_dict,
)
from nemo_gym.server_runtime.entrypoint_flags import _declared_class_flag, entrypoint_class_flag
from nemo_gym.utils.server_process import (
    _has_injected_global_config_env,
)


_NEMO_GYM_STARTED_RAY_CLUSTER: bool = False


def ray_is_installed() -> bool:
    """Whether Ray can be imported in this process, checked without importing it."""
    return find_spec("ray") is not None


def _get_ray():
    """Import Ray only for processes configured to use it."""
    try:
        import ray
    except ImportError as e:
        raise ImportError(
            "This process needs Ray, but Ray isn't installed in its environment. Install the `nemo-gym[ray]` extra."
        ) from e

    return ray


def initialize_ray() -> None:
    """
    Initialize ray cluster in a process.
    We store the Ray address in the global config dict so that child processes can connect to it.
    This avoids the need to start a new Ray cluster in each child process.
    Note: This function will modify the global config dict - update `ray_head_node_address`
    """

    ray = _get_ray()
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

    ray.init(**ray_init_kwargs)

    if not ray_head_node_address:
        global _NEMO_GYM_STARTED_RAY_CLUSTER
        _NEMO_GYM_STARTED_RAY_CLUSTER = True
        with open_dict(global_config_dict):
            global_config_dict["ray_head_node_address"] = ray.get_runtime_context().gcs_address
        print(f"Started Ray cluster at {global_config_dict['ray_head_node_address']}")


def maybe_ray_cluster_exit():  # pragma: no cover
    global _NEMO_GYM_STARTED_RAY_CLUSTER

    if not _NEMO_GYM_STARTED_RAY_CLUSTER:
        return

    print("Shutting down Ray cluster spun up by NeMo Gym...")
    ray = sys.modules.get("ray")
    if ray is not None:
        ray.shutdown()

    _NEMO_GYM_STARTED_RAY_CLUSTER = False


def _connect_server_to_ray(server_class: type, global_config_dict: DictConfig) -> None:
    """Join the run's Ray cluster if this server declares that it uses Ray."""
    if not server_class.ray_enabled:
        return
    if _has_injected_global_config_env() and not global_config_dict.get(RAY_HEAD_NODE_ADDRESS_KEY_NAME):
        # Gym only launches servers without a cluster address when it decided no server needs Ray. Fail instead
        # of letting initialize_ray start a private cluster in this process and in each of its workers.
        raise RuntimeError(
            f"{server_class.__module__}.{server_class.__name__} uses Ray, but this run has no Ray cluster. "
            f"Declare `ray_enabled = True` on {server_class.__name__} itself so `gym` starts Ray for it, and "
            "install the `nemo-gym[ray]` extra where `gym` runs."
        )
    initialize_ray()


def _declared_ray_enabled(class_node: ast.ClassDef) -> bool | None:
    """The literal `ray_enabled` assigned in a class body, or None when there is none."""
    return _declared_class_flag(class_node, "ray_enabled")


def entrypoint_ray_enabled(entrypoint_fpath: Path) -> bool | None:
    """The `ray_enabled` declaration of the server an entrypoint starts, read from source.

    The supervisor must decide whether to start Ray before any server exists; see `entrypoint_class_flag` for
    what the source alone can settle.
    """
    return entrypoint_class_flag(entrypoint_fpath, "ray_enabled")
