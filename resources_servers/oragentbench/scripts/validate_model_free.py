# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive the server's model-free validation modes over a benchmark JSONL.

Start the server with ``validation_mode`` set (``reference`` for the gold sweep, or one of
the negative controls), then post every row to ``/verify`` with an empty response. Writes
one verify response per line and prints the tally by status and stratum. ``+validation_mode``
selects another mode per row without restarting a server already in a validation mode;
``+resume_jsonl`` skips tasks already recorded in an earlier output.

    python resources_servers/oragentbench/scripts/validate_model_free.py \
        +benchmark_jsonl=resources_servers/oragentbench/data/benchmark.jsonl \
        +output_jsonl=results/oragentbench_reference.jsonl +concurrency=6 \
        +validation_mode=no_action
"""

import asyncio
import json
from collections import Counter

from tqdm.auto import tqdm

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import ServerClient


EMPTY_RESPONSE = {
    "output": [],
    "id": "",
    "created_at": 0,
    "model": "",
    "object": "response",
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}


async def main(examples: list, output_jsonl: str, concurrency: int, server_name: str) -> None:
    server_client = ServerClient.load_from_global_config()
    semaphore = asyncio.Semaphore(concurrency)

    async def one(example: dict) -> dict:
        async with semaphore:
            result = await server_client.post(server_name=server_name, url_path="/verify", json=example)
            if result.status != 200:
                raise RuntimeError(
                    f"/verify returned HTTP {result.status} for {example['task_name']}: {await result.text()}"
                )
            return await result.json()

    tally: Counter = Counter()
    passed = 0
    with open(output_jsonl, "w") as f, tqdm(total=len(examples)) as pbar:
        for future in asyncio.as_completed([one(e) for e in examples]):
            data = await future
            passed += int(data["reward"])
            tally[(data["difficulty"], data["status"], int(data["reward"]))] += 1
            pbar.set_description_str(f"passed {passed}/{pbar.n + 1}")
            pbar.update(1)
            f.write(json.dumps(data) + "\n")
    print(f"passed {passed} / {len(examples)}")
    for key, count in sorted(tally.items()):
        print(f"  difficulty={key[0]} status={key[1]} reward={key[2]}: {count}")


if __name__ == "__main__":
    config = get_global_config_dict()
    with open(config["benchmark_jsonl"]) as f:
        rows = [json.loads(line) for line in f]
    limit = config.get("limit")
    if limit:
        rows = rows[: int(limit)]
    resume_from = config.get("resume_jsonl")
    if resume_from:
        with open(resume_from) as f:
            done = {json.loads(line)["task_name"] for line in f if line.strip()}
        rows = [row for row in rows if row["task_name"] not in done]
        print(f"resuming: {len(done)} tasks already recorded, {len(rows)} to run")
    mode = config.get("validation_mode")
    for row in rows:
        row["response"] = EMPTY_RESPONSE
        if mode:
            row["validation_mode"] = mode
    asyncio.run(
        main(
            rows, config["output_jsonl"], int(config.get("concurrency", 6)), config.get("server_name", "oragentbench")
        )
    )
