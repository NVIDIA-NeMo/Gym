from typing import Optional

import json

from collections import Counter

from itertools import chain, repeat

from contextlib import nullcontext

import asyncio
from asyncio import Semaphore

from tqdm.asyncio import tqdm

from pydantic import BaseModel

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
        rows = list(
            chain.from_iterable(repeat(row, config.num_repeats) for row in rows)
        )
        print(
            f"Repeating rows (in a pattern of abc to aabbcc) from {previous_length} to {len(rows)}!"
        )

    server_client = ServerClient.load_from_global_config()

    semaphore = nullcontext()
    if config.num_samples_in_parallel:
        semaphore = Semaphore(config.num_samples_in_parallel)

    async def _post_coroutine(row: dict):
        async with semaphore:
            return await server_client.post(
                server_name=config.agent_name, url_path="/run", json=row
            )

    tasks = list(map(_post_coroutine, rows))

    metrics = Counter()
    pbar = tqdm.as_completed(tasks, desc="Collecting rollouts")
    with open(config.output_jsonl_fpath, "a") as f:
        for future in pbar:
            result = await future
            result = result.json()
            f.write(json.dumps(result) + "\n")
            metrics += Counter(
                {k: v for k, v in result.items() if isinstance(v, (int, float))}
            )

    avg_metrics = {k: v / len(tasks) for k, v in metrics.items()}
    avg_metrics.setdefault("reward", 0.0)
    print(json.dumps(avg_metrics, indent=4))


def collect_rollouts():  # pragma: no cover
    config = RolloutCollectionConfig.model_validate(get_global_config_dict())
    asyncio.run(_collect_rollouts(config))
