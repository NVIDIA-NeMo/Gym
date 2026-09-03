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

import asyncio
import json
from pathlib import Path

from tqdm.auto import tqdm

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import ServerClient
from resources_servers.swebench_pro.client import EMPTY_RESPONSE


async def main(examples: list[dict], concurrency: int, output_fpath: Path) -> None:
    client = ServerClient.load_from_global_config()
    semaphore = asyncio.Semaphore(concurrency)

    async def verify(example: dict) -> dict:
        async with semaphore:
            response = await client.post(
                server_name="swebench_pro_resources_server",
                url_path="/verify",
                json=example,
            )
            return await response.json()

    tasks = [asyncio.create_task(verify(example)) for example in examples]
    resolved = 0
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w", encoding="utf-8") as output:
        progress = tqdm(asyncio.as_completed(tasks), total=len(tasks))
        for completed in progress:
            result = await completed
            resolved += int(result.get("resolved", False))
            output.write(json.dumps(result) + "\n")
            finished = progress.n + 1
            progress.set_description(f"Resolved: {resolved}/{finished} ({100 * resolved / finished:.2f}%)")


if __name__ == "__main__":
    config = get_global_config_dict()
    with open(config["benchmark_jsonl"], encoding="utf-8") as benchmark:
        rows = [json.loads(line) for line in benchmark]
    if limit := config.get("limit"):
        rows = rows[: int(limit)]
    for row in rows:
        row |= {"response": EMPTY_RESPONSE}

    asyncio.run(
        main(
            rows,
            concurrency=int(config.get("concurrency", 4)),
            output_fpath=Path(config.get("output_fpath", "results/swebench_pro_golden.jsonl")),
        )
    )
