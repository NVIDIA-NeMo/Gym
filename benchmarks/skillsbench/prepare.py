# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare the SkillsBench benchmark for the `benchflow_agent`.

Clones the SkillsBench task-definition repo at a pinned commit into ``data/`` (this
checkout is the runtime ``tasks_dir`` for the agent) and writes one Gym JSONL row
per task. Each row carries an ``instance_id`` of the form ``skillsbench::<task>``;
the agent runs that single task through BenchFlow. The task instruction lives inside
the task's container, so ``responses_create_params.input`` is empty.

Run via ``ng_prepare_benchmark "+config_paths=[benchmarks/skillsbench/config.yaml]"``.
"""

import json
import shutil
import subprocess
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
REPO_DIR = DATA_DIR / "skillsbench_repo"
OUTPUT_FPATH = DATA_DIR / "skillsbench_benchmark.jsonl"

REPO_URL = "https://github.com/benchflow-ai/skillsbench.git"
# Pinned commit for reproducibility.
SKILLSBENCH_COMMIT = "312d07e15e5398f6eda32ee1bb86e492ab18edd1"  # pragma: allowlist secret

# Top-level config key of the agent that serves these rows (see config.yaml). Used as
# the `agent_ref.name` so `ng_collect_rollouts` routes each row to the right server.
AGENT_INSTANCE_NAME = "skillsbench_benchflow_agent"

# Tasks to skip (mirrors the current eval setup).
EXCLUDED_TASKS = {"multilingual-video-dubbing"}


def _ensure_repo(repo_dir: Path) -> None:
    """Clone SkillsBench at the pinned commit into ``repo_dir`` (idempotent).

    Reuses an existing checkout already at the pinned commit; otherwise (re)clones.
    The checkout persists because it is the agent's runtime ``tasks_dir``.
    """
    if repo_dir.exists():
        head = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if head.returncode == 0 and head.stdout.strip() == SKILLSBENCH_COMMIT:
            print(f"SkillsBench already at {SKILLSBENCH_COMMIT}; reusing {repo_dir}")
            return
        print(f"Removing stale SkillsBench checkout at {repo_dir}")
        shutil.rmtree(repo_dir)

    print(f"Cloning {REPO_URL} at {SKILLSBENCH_COMMIT}...")
    subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", SKILLSBENCH_COMMIT], check=True)


def _discover_task_names(repo_dir: Path) -> list[str]:
    """Return sorted SkillsBench task directory names (those with a task.toml)."""
    tasks_root = repo_dir / "tasks"
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"No tasks/ directory found in SkillsBench checkout at {tasks_root}")
    return sorted(
        task_dir.name
        for task_dir in tasks_root.iterdir()
        if task_dir.is_dir() and (task_dir / "task.toml").exists() and task_dir.name not in EXCLUDED_TASKS
    )


def prepare() -> Path:
    """Clone SkillsBench and write one JSONL row per task. Returns the JSONL path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_repo(REPO_DIR)

    task_names = _discover_task_names(REPO_DIR)
    if not task_names:
        raise RuntimeError(f"No SkillsBench tasks found under {REPO_DIR / 'tasks'}")

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        for task_name in task_names:
            row = {
                "instance_id": f"skillsbench::{task_name}",
                "responses_create_params": {"input": []},
                "agent_ref": {"name": AGENT_INSTANCE_NAME},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(task_names)} SkillsBench tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
