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
import asyncio
import json

from tqdm.auto import tqdm

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import ServerClient


async def main(examples: list) -> None:
    tasks = []
    for example in examples:
        task = server_client.post(
            server_name="swebench_resources_server",
            url_path="/verify",
            json=example,
        )
        tasks.append(task)

    num_resolved = 0
    num_total = 0
    pbar = tqdm(total=len(examples))
    write_file = open("temp2.jsonl", "w")
    for future in asyncio.as_completed(tasks):
        result = await future
        data = await result.json()

        num_resolved += data["resolved"]
        num_total += 1

        resolved_pct = 100 * num_resolved / num_total
        pbar.set_description_str(desc=f"Resolved: {num_resolved} / {num_total} ({resolved_pct:.2f}%)")
        pbar.update(1)
        write_file.write(json.dumps(data) + "\n")
    write_file.close()


if __name__ == "__main__":
    global_config_dict = get_global_config_dict()

    with open(global_config_dict["benchmark_jsonl"]) as f:
        examples = list(map(json.loads, f))

    limit = global_config_dict.get("limit")
    if limit:
        examples = examples[:limit]

    for example in examples:
        example |= {
            "response": {
                "output": [],
                "id": "",
                "created_at": 0,
                "model": "",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        }

    server_client = ServerClient.load_from_global_config()

    asyncio.run(main(examples))
