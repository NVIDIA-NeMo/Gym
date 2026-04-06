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
import subprocess

import ray


ray.init(address="auto")


@ray.remote(num_cpus=0, num_gpus=0)
def get_gpu_info():
    output = subprocess.check_output(["nvidia-smi"], text=True)
    return output


node_resource_keys = []
for node in ray.nodes():
    node_resource_key = [k for k in node["Resources"].keys() if k.startswith("node:")][0]
    node_resource_keys.append(node_resource_key)
tasks = [
    get_gpu_info.options(resources={node_resource_key: 0.01}).remote() for node_resource_key in node_resource_keys
]

results = ray.get(tasks)

for i, res in enumerate(results):
    print(f"\n===== Node {i} =====\n")
    print(res)
