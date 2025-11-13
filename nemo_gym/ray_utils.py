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
from typing import Optional

from nemo_gym.server_utils import (
    get_global_config_dict,
)


def spinup_single_ray_gpu_node_worker(worker_cls, num_gpus: Optional[int] = None):  # pragma: no cover
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    cfg = get_global_config_dict()
    nodes = cfg.get("ray_gpu_nodes", [])
    num_gpus_per_node = cfg.get("ray_num_gpus_per_node", 1)
    if num_gpus is None:
        num_gpus = num_gpus_per_node
    for node in nodes:
        worker_options = {}
        worker_options["num_gpus"] = num_gpus
        worker_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
            node_id=node["node_id"],
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
        worker = worker_cls.options(**worker_options).remote()
        return worker
    raise RuntimeError(f"No available Ray GPU nodes for spinning up {worker_cls}")
