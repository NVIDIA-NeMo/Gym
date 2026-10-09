# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write Gym rows for a Harbor dataset.

Each row names one task by ``harbor_dataset`` alias and ``task_name``, and carries the task instruction as the
user message, so any session agent can run it. Task-level agent settings from ``task.toml`` travel in reserved
``responses_create_params.metadata`` keys for the Harbor harness agent. Run it with the server's venv, which has
Harbor installed::

    python resources_servers/harbor_tasks/prepare.py --alias hello \\
        --path resources_servers/harbor_tasks/data/tasks --output resources_servers/harbor_tasks/data/example.jsonl
    python resources_servers/harbor_tasks/prepare.py --alias terminal_bench \\
        --name terminal-bench --version 2.0 --output data/terminal_bench.jsonl

The alias must match a key under the server's ``harbor_datasets`` config with the same source.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from harbor.models.job.config import DatasetConfig
from harbor.models.task.task import Task

from nemo_gym.sandbox.adapters.harbor import HARBOR_AGENT_TIMEOUT_METADATA_KEY, HARBOR_AGENT_USER_METADATA_KEY
from resources_servers.harbor_tasks.app import download_dataset_tasks, unsupported_features


def task_row(alias: str, task: Task) -> dict[str, Any]:
    """Build the Gym row for one supported single-step Harbor task."""
    metadata = {}
    if task.config.agent.timeout_sec is not None:
        metadata[HARBOR_AGENT_TIMEOUT_METADATA_KEY] = str(task.config.agent.timeout_sec)
    if task.config.agent.user is not None:
        metadata[HARBOR_AGENT_USER_METADATA_KEY] = str(task.config.agent.user)
    responses_create_params: dict[str, Any] = {"input": [{"role": "user", "content": task.instruction}]}
    if metadata:
        responses_create_params["metadata"] = metadata
    return {
        "task_id": task.name,
        "harbor_dataset": alias,
        "task_name": task.name,
        "responses_create_params": responses_create_params,
    }


async def build_rows(
    alias: str,
    dataset: DatasetConfig,
    *,
    skip_unsupported: bool = False,
    allow_unenforced_network_policy: bool = False,
) -> list[dict[str, Any]]:
    """Rows for every task in the dataset; an unsupported task raises unless ``skip_unsupported``."""
    rows = []
    for path in await download_dataset_tasks(dataset):
        task = Task(path)
        unsupported = unsupported_features(task, allow_unenforced_network_policy=allow_unenforced_network_policy)
        if unsupported:
            message = f"Harbor task {task.name!r} uses unsupported features: {', '.join(unsupported)}"
            if not skip_unsupported:
                raise ValueError(message)
            print(f"Skipping: {message}")
            continue
        rows.append(task_row(alias, task))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alias", required=True, help="harbor_datasets key the rows refer to")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", type=Path, help="local directory of Harbor task directories")
    source.add_argument("--name", help="Harbor registry dataset name")
    parser.add_argument("--version", help="Harbor registry dataset version")
    parser.add_argument("--task-names", nargs="*", help="only these tasks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-unsupported", action="store_true")
    parser.add_argument("--allow-unenforced-network-policy", action="store_true")
    args = parser.parse_args()

    dataset = DatasetConfig(path=args.path, name=args.name, version=args.version, task_names=args.task_names)
    rows = asyncio.run(
        build_rows(
            args.alias,
            dataset,
            skip_unsupported=args.skip_unsupported,
            allow_unenforced_network_policy=args.allow_unenforced_network_policy,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
