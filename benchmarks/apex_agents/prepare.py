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
"""Prepare the Apex Agents benchmark JSONL.

Downloads the ``mercor/apex-agents`` HuggingFace dataset and converts it into
the NeMo-Gym benchmark JSONL format: each row has an empty
``responses_create_params.input`` (the agent builds the actual prompt from the
top-level ``prompt`` field at runtime) plus task metadata and the rubric at the
top level so the Apex resources server can pick them up via ``verifier_metadata``.

The heavy world/task/gold assets are *not* materialized here — they are fetched
per-task by the agent server at rollout time (by ``world_id`` / ``task_id``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "apex_agents.jsonl"

HF_DATASET = "mercor/apex-agents"


def convert_task(task: dict, worlds: dict) -> dict:
    """Convert one ``tasks_and_rubrics.json`` entry into a Gym JSONL row.

    ``worlds`` maps ``world_id`` -> world descriptor (from
    ``world_descriptions.json``) so the row can carry the human-readable
    ``world_name`` alongside the id.
    """
    world = worlds.get(task["world_id"], {})
    return {
        # Empty input: the Apex agent constructs the user prompt from the
        # top-level ``prompt`` field at runtime.
        "responses_create_params": {"input": []},
        "task_id": task["task_id"],
        "world_id": task["world_id"],
        "world_name": world.get("world_name", ""),
        "domain": task.get("domain", ""),
        "task_name": task.get("task_name", ""),
        "prompt": task["prompt"],
        "rubric": task.get("rubric", []),
        "has_task_input_files": bool(task.get("task_input_files")),
    }


def prepare() -> Path:
    from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Pass HF_TOKEN explicitly — ``mercor/apex-agents`` is a gated dataset and
    # ``hf_hub_download`` does not always pick the token up from the env.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tasks_path = hf_hub_download(HF_DATASET, "tasks_and_rubrics.json", repo_type="dataset", token=token)
    worlds_path = hf_hub_download(HF_DATASET, "world_descriptions.json", repo_type="dataset", token=token)

    tasks = json.loads(Path(tasks_path).read_text())
    worlds = {w["world_id"]: w for w in json.loads(Path(worlds_path).read_text())}

    with OUTPUT_FPATH.open("w") as f:
        for task in tasks:
            f.write(json.dumps(convert_task(task, worlds)) + "\n")

    print(f"Wrote {len(tasks)} tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
