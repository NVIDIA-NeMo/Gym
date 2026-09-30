# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare task identities and instructions from digest-verified packages."""

import asyncio
import json
from pathlib import Path


from nemo_gym import server_utils
from resources_servers.terminal_bench_4.task import PackageLoader


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark.jsonl"


async def load_instructions(tasks: list[dict], task_download_dir: str | None = None) -> dict[str, str]:
    loader = PackageLoader(task_download_dir)
    owned_client = not server_utils.is_global_aiohttp_client_setup()
    if owned_client:
        server_utils.set_global_aiohttp_client(server_utils.GlobalAIOHTTPAsyncClientConfig())
    try:
        instructions = {}
        for task in tasks:
            package = await loader.load("terminal-bench/" + task["name"], task["ref"])
            instructions[task["name"]] = package.instruction
        return instructions
    finally:
        if owned_client:
            await server_utils.get_global_aiohttp_client().close()
            server_utils._GLOBAL_AIOHTTP_CLIENT = None


def prepare(
    task_names: list[str] | None = None,
    category: str | None = None,
    task_download_dir: str | None = None,
) -> Path:
    manifest = json.loads((BENCHMARK_DIR / "manifest.json").read_text())
    tasks = manifest["tasks"]
    if len(tasks) != 66 or len({task["name"] for task in tasks}) != 66:
        raise ValueError("Terminal-Bench 4.0 must contain 66 unique tasks")
    if task_names is not None:
        unknown = set(task_names) - {task["name"] for task in tasks}
        if unknown:
            raise ValueError(f"Unknown TB4 task names: {sorted(unknown)}")
        tasks = [task for task in tasks if task["name"] in task_names]
    if category is not None:
        if category not in {"cpu", "compose", "gpu"}:
            raise ValueError(f"Unknown TB4 category: {category}")
        tasks = [task for task in tasks if task["category"] == category]
    if not tasks:
        raise ValueError("The TB4 task selection is empty")
    instructions = asyncio.run(load_instructions(tasks, task_download_dir))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as output:
        for task in tasks:
            output.write(
                json.dumps(
                    {
                        "task_id": f"terminal-bench/{task['name']}",
                        "task_name": f"terminal-bench/{task['name']}",
                        "task_ref": task["ref"],
                        "dataset_ref": manifest["ref"],
                        "responses_create_params": {
                            "input": [{"role": "user", "content": instructions[task["name"]]}]
                        },
                    }
                )
                + "\n"
            )
    return OUTPUT_PATH


if __name__ == "__main__":
    prepare()
