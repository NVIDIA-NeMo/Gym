# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Prepare the Terminal-Bench 2.1 input JSONL for the harbor agent.

Emits one Gym row per task (``instance_id: "terminal_bench::<task>"``) from a
checkout of the terminal-bench-2-1 tasks repository, pinned for
reproducibility. The same checkout's ``tasks/`` directory is what
``harbor_datasets.terminal_bench.local_dataset_path`` must point at (via
``TERMINAL_BENCH_2_1_TASKS_DIR``) in
``configs/harbor_agent_opensandbox.yaml``.

The per-row sampling parameters mirror the Qwen3.6 model-card Terminal-Bench
setup (thinking mode: temperature 1.0, top_p 0.95); adjust for other models.
"""

import json
import os
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/harbor-framework/terminal-bench-2-1.git"
PINNED_COMMIT = "36d417f56c293b8271b306a0e4c566f58e98c153"

AGENT_DIR = Path(__file__).parent
OUTPUT_FPATH = AGENT_DIR / "example" / "terminal_bench_2_1_input.jsonl"
DEFAULT_CHECKOUT_DIR = AGENT_DIR / "data" / "terminal-bench-2-1"

DATASET_ALIAS = "terminal_bench"
RESPONSES_CREATE_PARAMS = {"input": [], "temperature": 1.0, "top_p": 0.95}


def _ensure_tasks_dir() -> Path:
    """Return the TB 2.1 tasks directory, cloning the pinned repo if needed."""
    env_dir = os.environ.get("TERMINAL_BENCH_2_1_TASKS_DIR")
    if env_dir:
        tasks_dir = Path(env_dir)
        if not tasks_dir.exists():
            raise FileNotFoundError(f"TERMINAL_BENCH_2_1_TASKS_DIR does not exist: {tasks_dir}")
        return tasks_dir

    tasks_dir = DEFAULT_CHECKOUT_DIR / "tasks"
    if not tasks_dir.exists():
        DEFAULT_CHECKOUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO_URL, str(DEFAULT_CHECKOUT_DIR)], check=True)
        subprocess.run(["git", "-C", str(DEFAULT_CHECKOUT_DIR), "checkout", PINNED_COMMIT], check=True)
    return tasks_dir


def prepare() -> Path:
    """Write one Gym input row per Terminal-Bench 2.1 task."""
    tasks_dir = _ensure_tasks_dir()
    task_names = sorted(p.name for p in tasks_dir.iterdir() if (p / "task.toml").exists())
    if not task_names:
        raise RuntimeError(f"No tasks with task.toml found under {tasks_dir}")

    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FPATH, "w") as f:
        for task_name in task_names:
            row = {
                "instance_id": f"{DATASET_ALIAS}::{task_name}",
                "responses_create_params": RESPONSES_CREATE_PARAMS,
                "agent_ref": {"name": "harbor_agent"},
            }
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(task_names)} tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
