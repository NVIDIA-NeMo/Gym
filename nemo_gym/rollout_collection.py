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
import asyncio
import json
from asyncio import Semaphore
from collections import Counter
from contextlib import nullcontext
from itertools import chain, repeat
from typing import Optional

from pydantic import BaseModel
from tqdm.asyncio import tqdm

from nemo_gym.server_utils import ServerClient, get_global_config_dict


class RolloutCollectionConfig(BaseModel):
    agent_name: str
    input_jsonl_fpath: str
    output_jsonl_fpath: str
    limit: Optional[int] = None
    num_repeats: Optional[int] = None
    num_samples_in_parallel: Optional[int] = None


async def _collect_rollouts(config: RolloutCollectionConfig):  # pragma: no cover
    with open(config.input_jsonl_fpath) as input_dataset:
        rows = list(map(json.loads, input_dataset))
    print(f"Found {len(rows)} rows!")

    if config.limit:
        previous_length = len(rows)
        rows = rows[: config.limit]
        print(f"Limiting rows from {previous_length} to {len(rows)}!")

    if config.num_repeats:
        previous_length = len(rows)
        rows = list(chain.from_iterable(repeat(row, config.num_repeats) for row in rows))
        print(f"Repeating rows (in a pattern of abc to aabbcc) from {previous_length} to {len(rows)}!")

    server_client = ServerClient.load_from_global_config()

    semaphore = nullcontext()
    if config.num_samples_in_parallel:
        semaphore = Semaphore(config.num_samples_in_parallel)

    metrics = Counter()
    with open(config.output_jsonl_fpath, "a") as f:

        async def _post_coroutine(i: int, row: dict) -> None:
            async with semaphore:
                response = await server_client.post(
                    server_name=config.agent_name, url_path="/run", json=row, headers={"i": str(i)}
                )
                result = response.json()
                f.write(json.dumps(result) + "\n")
                metrics.update({k: v for k, v in result.items() if isinstance(v, (int, float))})

        tasks = list(map(_post_coroutine, range(len(rows)), rows))
        await tqdm.gather(*tasks, desc="Collecting rollouts")

    avg_metrics = {k: v / len(tasks) for k, v in metrics.items()}
    print(json.dumps(avg_metrics, indent=4))


def collect_rollouts():  # pragma: no cover
    config = RolloutCollectionConfig.model_validate(get_global_config_dict())
    asyncio.run(_collect_rollouts(config))
