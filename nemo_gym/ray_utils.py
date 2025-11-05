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
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
)

from nemo_gym.server_utils import (
    get_global_config_dict,
)


def spinup_single_ray_gpu_node_worker(worker_cls):
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
        worker = worker_cls.options(**worker_options).remote()
        cfg["_ray_state"]["spunup_node_ids"][node["node_id"]] = {
            "worker_cls_name": worker_cls.__name__,
        }
        return worker
    raise RuntimeError(
        "No available Ray GPU nodes for spinning up {worker_cls.__name__}"
    )
