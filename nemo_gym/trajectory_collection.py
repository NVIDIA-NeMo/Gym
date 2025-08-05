from typing import Optional

import json

from collections import Counter

import asyncio

from tqdm.asyncio import tqdm

from pydantic import BaseModel

from nemo_gym.server_utils import ServerClient, get_global_config_dict


class TrajectoryCollectionConfig(BaseModel):
    agent_name: str
    input_jsonl_fpath: str
    output_jsonl_fpath: str
    limit: Optional[int] = None


async def _collect_trajectories(config: TrajectoryCollectionConfig):  # pragma: no cover
    with open(config.input_jsonl_fpath) as input_dataset:
        rows = list(map(json.loads, input_dataset))

    if config.limit:
        rows = rows[: config.limit]

    server_client = ServerClient.load_from_global_config()
    tasks = [
        server_client.post(server_name=config.agent_name, url_path="/run", json=d)
        for d in rows
    ]

    metrics = Counter()
    pbar = tqdm.as_completed(tasks, desc="Collecting trajectories")
    with open(config.output_jsonl_fpath, "a") as f:
        for future in pbar:
            result = await future
            result = result.json()
            f.write(json.dumps(result) + "\n")
            metrics += Counter(
                {k: v for k, v in result.items() if isinstance(v, (int, float))}
            )

    avg_metrics = {k: v / len(tasks) for k, v in metrics.items()}
    print(json.dumps(avg_metrics, indent=4))


def collect_trajectories():  # pragma: no cover
    config = TrajectoryCollectionConfig.model_validate(get_global_config_dict())
    asyncio.run(_collect_trajectories(config))
