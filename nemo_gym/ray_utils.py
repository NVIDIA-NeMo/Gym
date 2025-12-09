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
from typing import Dict, Optional, Set

import ray
import ray.util.state
from ray.actor import ActorClass, ActorProxy
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

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
            if ray_namespace is not None:
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

    def _post_init(self) -> None:
        # If value of RAY_GPU_NODES_KEY_NAME is None, then Gym will use all Ray GPU nodes
        # for scheduling GPU actors.
        # Otherwise if value of RAY_GPU_NODES_KEY_NAME is a list, then Gym will only use
        # the listed Ray GPU nodes for scheduling GPU actors.
        allowed_gpu_nodes = self.cfg.get(RAY_GPU_NODES_KEY_NAME, None)
        if allowed_gpu_nodes is not None:
            allowed_gpu_nodes = set(
                [node["node_id"] if "node_id" in node else node for node in allowed_gpu_nodes]
            )
        print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper.alloc_gpu_node: allowed nodes = {allowed_gpu_nodes}", flush=True)

        head = self.cfg["ray_head_node_address"]
        node_states = ray.util.state.list_nodes(head, detail=True)
        for state in node_states:
            assert state.node_id is not None
            print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper.alloc_gpu_node: node state = {state}", flush=True)
            avail_num_gpus = state.resources_total.get("GPU", 0)
            print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper.alloc_gpu_node: avail num gpus = {avail_num_gpus}", flush=True)
            if allowed_gpu_nodes is not None and state.node_id not in allowed_gpu_nodes:
                print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper.alloc_gpu_node: not an allowed node", flush=True)
                continue
            self.avail_gpus_dict[state.node_id] += avail_num_gpus
            if False:
                num_gpus_per_node = cfg.get(RAY_NUM_GPUS_PER_NODE_KEY_NAME, 8)
                self.avail_gpus_dict[state.node_id] = num_gpus_per_node
        print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper.alloc_gpu_node: avail node gpus = {self.avail_gpus_dict}", flush=True)

    def alloc_gpu_node(self, num_gpus: int) -> Optional[str]:
        for node_id, avail_num_gpus in self.avail_gpus_dict.items():
            used_num_gpus = self.used_gpus_dict[node_id]
            print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper.alloc_gpu_node: node = {repr(node_id)} avail = {avail_num_gpus} used = {used_num_gpus}", flush=True)
            if used_num_gpus + num_gpus <= avail_num_gpus:
                self.used_gpus_dict[node_id] += num_gpus
                print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper.alloc_gpu_node: free node = {repr(node_id)}", flush=True)
                return node_id
        print(f"DEBUG: _NeMoGymRayGPUSchedulingHelper.alloc_gpu_node: no free node", flush=True)
        return None


def lookup_ray_node_id_to_ip_dict() -> Dict[str, str]:  # pragma: no cover
    cfg = get_global_config_dict()
    head = cfg["ray_head_node_address"]
    id_to_ip = {}
    node_states = ray.util.state.list_nodes(head)
    for state in node_states:
        id_to_ip[state.node_id] = state.node_ip
    return id_to_ip


def lookup_current_ray_node_id() -> str:  # pragma: no cover
    return ray.get_runtime_context().get_node_id()


def lookup_current_ray_node_ip() -> str:  # pragma: no cover
    return lookup_ray_node_id_to_ip_dict()[lookup_current_ray_node_id()]


def debug_dump_ray_node_state(pattern=None):  # pragma: no cover
    cfg = get_global_config_dict()
    head = cfg["ray_head_node_address"]
    # head = "auto"
    node_states = ray.util.state.list_nodes(
        head,
        detail=True,
    )
    n = len(node_states)
    for i, state in enumerate(node_states):
        print(f"DEBUG: debug_dump_ray_node_state: [{i}/{n}]: {state}", flush=True)


def debug_dump_ray_actor_state(pattern=None):  # pragma: no cover
    cfg = get_global_config_dict()
    head = cfg["ray_head_node_address"]
    # head = "auto"
    actor_states = ray.util.state.list_actors(
        head,
        detail=True,
    )
    n = len(actor_states)
    for i, state in enumerate(actor_states):
        if pattern is not None:
            if str(state.class_name).find(pattern) < 0:
                continue
        print(f"DEBUG: debug_dump_ray_actor_state: [{i}/{n}]: {state}", flush=True)


def _lookup_ray_node_with_free_gpus(
    num_gpus: int, allowed_gpu_nodes: Optional[Set[str]] = None
) -> Optional[str]:  # pragma: no cover
    cfg = get_global_config_dict()
    head = cfg["ray_head_node_address"]

    node_avail_gpu_dict = defaultdict(int)
    node_states = ray.util.state.list_nodes(head, detail=True)
    for state in node_states:
        assert state.node_id is not None
        if allowed_gpu_nodes is not None and state.node_id not in allowed_gpu_nodes:
            continue
        node_avail_gpu_dict[state.node_id] += state.resources_total.get("GPU", 0)
    print(f"DEBUG: _lookup_ray_node_with_free_gpus: avail = {node_avail_gpu_dict}", flush=True)

    while True:
        retry = False
        node_used_gpu_dict = defaultdict(int)
        actor_states = ray.util.state.list_actors(head, detail=True)
        for state in actor_states:
            if state.state == "DEAD":
                continue
            if state.state == "PENDING_CREATION" or state.node_id is None:
                print(f"DEBUG: _lookup_ray_node_with_free_gpus: debug: actor state = {state}", flush=True)
                # retry = True
                # break
                pass
            if state.node_id is not None:
                node_used_gpu_dict[state.node_id] += state.required_resources.get("GPU", 0)
        print(f"DEBUG: _lookup_ray_node_with_free_gpus: used  = {node_used_gpu_dict}", flush=True)
        if retry:
            sleep(2)
            continue
        break

    for node_id, avail_num_gpus in node_avail_gpu_dict.items():
        used_num_gpus = node_used_gpu_dict[node_id]
        print(
            f"DEBUG: _lookup_ray_node_with_free_gpus: node id = {node_id} req = {num_gpus} used = {used_num_gpus} avail = {avail_num_gpus}",
            flush=True,
        )
        if num_gpus + used_num_gpus <= avail_num_gpus:
            print(f"DEBUG: _lookup_ray_node_with_free_gpus: node id = {node_id} free", flush=True)
            return node_id
    return None


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
    node_id = ray.get(helper.alloc_gpu_node.remote(num_gpus))
    if node_id is None:
        raise RuntimeError(f"Cannot find an available Ray node with {num_gpus} GPUs to spin up {worker_cls}")

    print(f"DEBUG: spinup_single_ray_gpu_node_worker: node id = {node_id}", flush=True)
    print(f"DEBUG: spinup_single_ray_gpu_node_worker: py exec = {sys.executable}", flush=True)
    worker_options = {}
    worker_options["num_gpus"] = num_gpus
    worker_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
        node_id=node_id,
        soft=False,
    )
    worker_options["runtime_env"] = {
        "py_executable": sys.executable,
        "env_vars": _prepare_ray_worker_env_vars(),
    }
    worker = worker_cls.options(**worker_options).remote(*worker_args, **worker_kwargs)
    print(f"DEBUG: spinup_single_ray_gpu_node_worker: worker  = {worker}", flush=True)
    return worker
