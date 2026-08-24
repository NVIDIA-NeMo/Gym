# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare source rows for the terminal_bench_2_1 benchmark."""

import json
import tomllib
from glob import glob
from pathlib import Path
from subprocess import run


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_PATH = BENCHMARK_DIR / "data" / "example.jsonl"


def prepare() -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    repo_path = BENCHMARK_DIR / "terminal-bench-2-1"
    if not repo_path.exists():
        run(
            "git clone https://github.com/harbor-framework/terminal-bench-2-1".split(),
            cwd=str(BENCHMARK_DIR),
        )

    f_out = open(OUTPUT_PATH, "w")
    for task_dir in glob(f"{repo_path}/tasks/*"):
        sample = dict()

        task_dir = repo_path / "tasks" / task_dir

        solution_files = list((task_dir / "solution").iterdir())
        assert len(solution_files) == 1, solution_files
        sample["solve.sh"] = (task_dir / "solution" / "solve.sh").read_text()

        test_files = list((task_dir / "tests").iterdir())
        assert len(test_files) == 2, test_files
        sample["test.sh"] = (task_dir / "tests" / "test.sh").read_text()
        sample["test_outputs.py"] = (task_dir / "tests" / "test_outputs.py").read_text()

        sample["responses_create_params"] = {
            "input": [{"role": "user", "content": (task_dir / "instruction.md").read_text()}]
        }

        with open(task_dir / "task.toml", "rb") as file:
            task_toml = tomllib.load(file)

        sample["task.name"] = task_toml["task"]["name"]
        sample["docker_image"] = task_toml["environment"]["docker_image"]

        f_out.write(json.dumps(sample) + "\n")

    return OUTPUT_PATH


if __name__ == "__main__":
    prepare()
