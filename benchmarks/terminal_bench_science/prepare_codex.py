# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write Terminal-Bench-Science rows that carry the official instruction as their prompt.

Harnesses behind an Environment Server (``codex.yaml``) receive the row's
``responses_create_params``, not the resources seed, so each row embeds the task's
``instruction.md`` as one user message, read from the pinned package exactly as the
resources server reads it. Loading packages verifies their content hashes, which needs
``pathspec``: run this with the resources server's environment, for example::

    resources_servers/terminal_bench_4/.venv/bin/python -m benchmarks.terminal_bench_science.prepare_codex \\
        --task-names mri-harmonization eeg-erp-recovery
"""

import argparse
import asyncio
import json
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.terminal_bench_science.prepare import BENCHMARK_DIR, load_manifest, select_tasks, task_row
from nemo_gym import global_config


OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark_codex.jsonl"
# Every v0.1 task gives its agent 8 hours; codex.yaml sets the harness timeout to match.
AGENT_TIMEOUT_SEC = 28800


async def _load_tasks(names_and_refs: list[tuple[str, str]], download_dir: Path | None) -> list:
    try:
        from resources_servers.terminal_bench_4.task import PackageLoader
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Loading pinned task packages needs the terminal_bench_4 server's requirements "
            "(pathspec); run this module with resources_servers/terminal_bench_4/.venv/bin/python"
        ) from error
    loader = PackageLoader(download_dir)
    return await asyncio.gather(*(loader.load(name, ref) for name, ref in names_and_refs))


def prepare(
    *,
    task_names: list[str] | None = None,
    category: str | None = None,
    domain: str | None = None,
    task_download_dir: str | None = None,
) -> Path:
    """Write one prompt-bearing row per selected task; filters combine as in ``prepare.py``."""
    manifest = load_manifest()
    tasks = select_tasks(manifest, task_names=task_names, category=category, domain=domain)
    rows = [task_row(manifest, task) for task in tasks]
    packages = asyncio.run(
        _load_tasks(
            [(row["task_name"], row["task_ref"]) for row in rows],
            Path(task_download_dir) if task_download_dir else None,
        )
    )
    for row, package in zip(rows, packages):
        if package.config.agent.timeout_sec != AGENT_TIMEOUT_SEC:
            raise ValueError(
                f"{row['task_name']} has a {package.config.agent.timeout_sec}s agent budget; "
                f"codex.yaml assumes {AGENT_TIMEOUT_SEC}s"
            )
        row["responses_create_params"] = {"input": [{"role": "user", "content": package.instruction}]}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
    return OUTPUT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-names", nargs="+")
    parser.add_argument("--category")
    parser.add_argument("--domain")
    parser.add_argument("--task-download-dir")
    args = parser.parse_args()
    # Package downloads use Gym's shared HTTP client, which would otherwise parse this
    # script's arguments as a Gym run config.
    if global_config._GLOBAL_CONFIG_DICT is None:
        global_config._GLOBAL_CONFIG_DICT = OmegaConf.create({})
    print(
        prepare(
            task_names=args.task_names,
            category=args.category,
            domain=args.domain,
            task_download_dir=args.task_download_dir,
        )
    )


if __name__ == "__main__":
    main()
