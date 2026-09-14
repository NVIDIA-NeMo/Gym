# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare source rows for the terminal_bench_2_1 benchmark."""

import json
import tomllib
from glob import glob
from pathlib import Path
from subprocess import run


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark.jsonl"


def prepare() -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    repo_path = BENCHMARK_DIR / "terminal-bench-2-1"
    if not repo_path.exists():
        run(
            "git clone https://github.com/harbor-framework/terminal-bench-2-1".split(),
            cwd=str(BENCHMARK_DIR),
        )

    num_samples = 0
    f_out = open(OUTPUT_PATH, "w")
    for task_dir in glob(f"{repo_path}/tasks/*"):
        task_dir = repo_path / "tasks" / task_dir
        if not task_dir.is_dir():
            continue

        with open(task_dir / "task.toml", "rb") as file:
            task_toml = tomllib.load(file)

        sample = {
            "responses_create_params": {
                "input": [{"role": "user", "content": (task_dir / "instruction.md").read_text()}]
            },
            "task_name": task_toml["task"]["name"],
            "docker_image": task_toml["environment"]["docker_image"],
            "task_folder": str(task_dir.relative_to(BENCHMARK_DIR.parent.parent)),
        }

        f_out.write(json.dumps(sample) + "\n")
        num_samples += 1
    f_out.close()

    assert num_samples == 89, num_samples

    return OUTPUT_PATH


if __name__ == "__main__":
    prepare()
