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
from typing import Dict, List, Optional

import ray
import ray.util.state
from ray.actor import ActorClass, ActorProxy
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)

from nemo_gym.global_config import (
    RAY_GPU_NODES_KEY_NAME,
    RAY_NUM_GPUS_PER_NODE_KEY_NAME,
    get_global_config_dict,
)


def _prepare_ray_worker_env_vars() -> Dict[str, str]:  # pragma: no cover
    worker_env_vars = {
        **os.environ,
    }
    pop_env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_JOB_ID",
        "RAY_RAYLET_PID",
    ]
    for k in pop_env_vars:
        worker_env_vars.pop(k, None)
    return worker_env_vars


def _start_global_ray_gpu_scheduling_helper(node_id: Optional[str] = None) -> ActorProxy:  # pragma: no cover
    cfg = get_global_config_dict()
    helper_options = {
        "name": "_NeMoGymRayGPUSchedulingHelper",
        "num_cpus": 0,
    }
    if node_id is not None:
        helper_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=True,
        )
    helper = _NeMoGymRayGPUSchedulingHelper.options(**helper_options).remote(cfg)
    ray.get(helper._post_init.remote())
    return helper


def get_global_ray_gpu_scheduling_helper() -> ActorProxy:  # pragma: no cover
    cfg = get_global_config_dict()
    while True:
        try:
            get_actor_args = {
                "name": "_NeMoGymRayGPUSchedulingHelper",
            }
            ray_namespace = cfg.get("ray_namespace", None)
            if ray_namespace is None:
                ray_namespace = "nemo_gym"
            get_actor_args["namespace"] = ray_namespace
            worker = ray.get_actor(**get_actor_args)
            return worker
        except ValueError:
            sleep(3)


@ray.remote
class _NeMoGymRayGPUSchedulingHelper:  # pragma: no cover
    def __init__(self, cfg):
        self.cfg = cfg
        self.avail_gpus_dict = defaultdict(int)
        self.used_gpus_dict = defaultdict(int)
        # Maps node_id -> PlacementGroup for nodes with PG-based reservations.
        # When a PG is available for a node, spinup_single_ray_gpu_node_worker
        # uses PlacementGroupSchedulingStrategy instead of NodeAffinitySchedulingStrategy.
        self.node_pg_dict: Dict[str, object] = {}

    def _post_init(self) -> None:
        # If value of RAY_GPU_NODES_KEY_NAME is None, then Gym will use all Ray GPU nodes
        # for scheduling GPU actors.
        # Otherwise if value of RAY_GPU_NODES_KEY_NAME is a list, then Gym will only use
        # the listed Ray GPU nodes for scheduling GPU actors.
        allowed_gpu_nodes = self.cfg.get(RAY_GPU_NODES_KEY_NAME, None)
        if allowed_gpu_nodes is not None:
            allowed_gpu_nodes = set(allowed_gpu_nodes)

        print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper: post init: allow gpus = {allowed_gpu_nodes}", flush=True)

        head = self.cfg["ray_head_node_address"]
        node_states = ray.util.state.list_nodes(head, detail=True, limit=10000)
        for state in node_states:
            assert state.node_id is not None
            avail_num_gpus = state.resources_total.get("GPU", 0)
            if allowed_gpu_nodes is not None and state.node_id not in allowed_gpu_nodes:
                continue
            self.avail_gpus_dict[state.node_id] += avail_num_gpus

        print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper: post init: avail gpus = {self.avail_gpus_dict} (intermediate)", flush=True)

        default_num_gpus_per_node = self.cfg.get(RAY_NUM_GPUS_PER_NODE_KEY_NAME, 8)
        if allowed_gpu_nodes is not None:
            for node_id in allowed_gpu_nodes:
                if node_id in self.avail_gpus_dict:
                    continue
                print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper: post init: warning: ray state API did not return info for node={repr(node_id)}", flush=True)
                self.avail_gpus_dict[node_id] = default_num_gpus_per_node

        print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper: post init: avail gpus = {self.avail_gpus_dict}", flush=True)

    def set_gpu_pgs(self, node_ids: list, pgs: list) -> None:
        """Register PlacementGroup reservations for judge nodes.

        Called after _post_init with the PG objects directly (can't go through OmegaConf).
        PGs and node_ids are ordered the same: pgs[i] reserves GPUs on node_ids[i].
        """
        if len(node_ids) != len(pgs):
            raise ValueError(
                f"node_ids and pgs must have the same length, got {len(node_ids)} and {len(pgs)}"
            )

        for node_id, pg in zip(node_ids, pgs):
            if not hasattr(pg, "bundle_specs") or pg.bundle_specs is None:
                raise ValueError(f"PlacementGroup for node {node_id} is missing bundle_specs")
            self.node_pg_dict[node_id] = pg
            bundle_gpus = sum(bundle.get("GPU", 0) for bundle in pg.bundle_specs)
            self.avail_gpus_dict[node_id] = bundle_gpus

        print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper: set_gpu_pgs: {len(self.node_pg_dict)} nodes have PG reservations", flush=True)

    def alloc_gpu_node(self, num_gpus: int, desc: Optional[str]) -> Optional[str]:
        print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper: alloc gpu [{desc}]: avail gpus = {self.avail_gpus_dict}", flush=True)
        print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper: alloc gpu [{desc}]: used gpus  = {self.used_gpus_dict}", flush=True)
        for node_id, avail_num_gpus in self.avail_gpus_dict.items():
            used_num_gpus = self.used_gpus_dict[node_id]
            if used_num_gpus + num_gpus <= avail_num_gpus:
                self.used_gpus_dict[node_id] += num_gpus
                print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper: alloc gpu [{desc}]: node = {node_id} gpus = {num_gpus}", flush=True)
                return node_id
        print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper: alloc gpu [{desc}]: no available node", flush=True)
        return None

    def get_pg_for_node(self, node_id: str) -> Optional[object]:
        """Return the PlacementGroup for a node, or None if no PG reservation exists."""
        return self.node_pg_dict.get(node_id, None)


def lookup_ray_node_id_to_ip_dict() -> Dict[str, str]:  # pragma: no cover
    cfg = get_global_config_dict()
    head = cfg["ray_head_node_address"]
    id_to_ip = {}
    node_states = ray.util.state.list_nodes(head, limit=10000)
    for state in node_states:
        id_to_ip[state.node_id] = state.node_ip
    return id_to_ip


def lookup_current_ray_node_id() -> str:  # pragma: no cover
    return ray.get_runtime_context().get_node_id()


def lookup_current_ray_node_ip() -> str:  # pragma: no cover
    return lookup_ray_node_id_to_ip_dict()[lookup_current_ray_node_id()]


def spinup_single_ray_gpu_node_worker(
    worker_cls: ActorClass,
    num_gpus: int,
    *worker_args,
    **worker_kwargs,
) -> ActorProxy:  # pragma: no cover
    cfg = get_global_config_dict()

    num_gpus_per_node = cfg.get(RAY_NUM_GPUS_PER_NODE_KEY_NAME, 8)
    assert num_gpus >= 1, f"Must request at least 1 GPU node for spinning up {worker_cls}"
    assert num_gpus <= num_gpus_per_node, (
        f"Requested {num_gpus} > {num_gpus_per_node} GPU nodes for spinning up {worker_cls}"
    )

    helper = get_global_ray_gpu_scheduling_helper()
    node_id = ray.get(helper.alloc_gpu_node.remote(num_gpus, f"{worker_cls}"))
    if node_id is None:
        raise RuntimeError(f"Cannot find an available Ray node with {num_gpus} GPUs to spin up {worker_cls}")

    pg = ray.get(helper.get_pg_for_node.remote(node_id))

    worker_options = {}
    worker_options["num_gpus"] = num_gpus
    if pg is not None:
        worker_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
        )
        worker_options["num_cpus"] = 0
    else:
        worker_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False,
        )
    worker_options["runtime_env"] = {
        "py_executable": sys.executable,
        "env_vars": _prepare_ray_worker_env_vars(),
    }
    worker = worker_cls.options(**worker_options).remote(*worker_args, **worker_kwargs)
    return worker
