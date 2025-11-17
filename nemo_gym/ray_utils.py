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
import os
import sys
from collections import defaultdict
from time import sleep
from typing import Any, Dict, List, Optional

import ray.util.state
from ray.actor import ActorClass
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_gym.global_config import (
    RAY_NUM_GPUS_PER_NODE_KEY_NAME,
    get_global_config_dict,
)


def _lookup_node_id_with_free_gpus(num_gpus: int, node_list: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    cfg = get_global_config_dict()
    node_avail_gpu_dict = defaultdict(int)
    node_states = ray.util.state.list_nodes(
        cfg["ray_head_node_address"],
        detail=True,
    )
    for state in node_states:
        assert state.node_id is not None
        node_avail_gpu_dict[state.node_id] += state.resources_total.get("GPU", 0)
    while True:
        retry = False
        node_used_gpu_dict = defaultdict(int)
        actor_states = ray.util.state.list_actors(
            cfg["ray_head_node_address"],
            detail=True,
        )
        for state in actor_states:
            if state.state == "PENDING_CREATION" or state.node_id is None:
                retry = True
                break
            node_used_gpu_dict[state.node_id] += state.required_resources.get("GPU", 0)
        if retry:
            sleep(2)
            continue
        break
    for node_id, avail_num_gpus in node_avail_gpu_dict.items():
        used_num_gpus = node_used_gpu_dict[node_id]
        if used_num_gpus + num_gpus <= avail_num_gpus:
            return node_id
    return None


def spinup_single_ray_gpu_node_worker(
    worker_cls: ActorClass,
    num_gpus: int,
    *worker_args,
    **worker_kwargs,
):  # pragma: no cover
    cfg = get_global_config_dict()
    # nodes = cfg.get(RAY_GPU_NODES_KEY_NAME, None)
    num_gpus_per_node = cfg.get(RAY_NUM_GPUS_PER_NODE_KEY_NAME, 8)
    assert num_gpus >= 1, f"Must request at least 1 GPU node for spinning up {worker_cls}"
    assert num_gpus <= num_gpus_per_node, (
        f"Requested {num_gpus} > {num_gpus_per_node} GPU nodes for spinning up {worker_cls}"
    )
    node_id = _lookup_node_id_with_free_gpus(num_gpus)
    if node_id is None:
        raise RuntimeError(f"Cannot find {num_gpus} available Ray GPU nodes for spinning up {worker_cls}")
    worker_options = {}
    worker_options["num_gpus"] = num_gpus
    worker_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
        node_id=node_id,
        soft=False,
    )
    py_exec = sys.executable
    worker_runtime_env = {
        "py_executable": py_exec,
        "env_vars": {
            **os.environ,
        },
    }
    worker_options["runtime_env"] = worker_runtime_env
    worker = worker_cls.options(**worker_options).remote(*worker_args, **worker_kwargs)
    return worker
