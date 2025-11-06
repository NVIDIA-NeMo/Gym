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
import os
import sys
from typing import Optional

from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
)

from nemo_gym.server_utils import (
    get_global_config_dict,
)


def spinup_single_ray_gpu_node_worker(worker_cls, runtime_env: Optional[dict] = None):
    cfg = get_global_config_dict()
    cfg.setdefault("_ray_state", {})
    cfg["_ray_state"].setdefault("spunup_node_ids", {})
    nodes = cfg.get("ray_nodes", [])
    num_gpus_per_node = cfg.get("ray_num_gpus_per_node", 0)
    for node in nodes:
        if node["node_id"] in cfg["_ray_state"]["spunup_node_ids"]:
            continue
        worker_options = {}
        worker_options["num_gpus"] = num_gpus_per_node
        worker_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
            node_id=node["node_id"],
            soft=False,
        )
        py_exec = sys.executable
        venv_path = os.environ.get("VIRTUAL_ENV", None)
        uv_project_path = os.environ.get("UV_PROJECT_ENVIRONMENT", None)
        print(f"DEBUG: spinup_single_ray_gpu_node_worker: py exec         = {py_exec}", flush=True)
        print(f"DEBUG: spinup_single_ray_gpu_node_worker: venv path       = {venv_path}", flush=True)
        print(f"DEBUG: spinup_single_ray_gpu_node_worker: uv project path = {uv_project_path}", flush=True)
        worker_runtime_env = {
            "py_executable": py_exec,
        }
        if venv_path is not None or uv_project_path is not None:
            print(f"DEBUG: spinup_single_ray_gpu_node_worker: override env vars", flush=True)
            worker_runtime_env["env_vars"] = {
                **os.environ,
            }
        if runtime_env is not None:
            worker_runtime_env |= runtime_env
        # print(f"DEBUG: spinup_single_ray_gpu_node_worker: worker runtime env = {worker_runtime_env}", flush=True)
        worker_options["runtime_env"] = worker_runtime_env
        worker = worker_cls.options(**worker_options).remote()
        cfg["_ray_state"]["spunup_node_ids"][node["node_id"]] = {
            "worker_cls_name": f"{worker_cls}",
        }
        return worker
    raise RuntimeError(
        f"No available Ray GPU nodes for spinning up {worker_cls}"
    )
