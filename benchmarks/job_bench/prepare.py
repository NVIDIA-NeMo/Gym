# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FPATH = DATA_DIR / "job_bench.jsonl"


def prepare() -> Path:
    from huggingface_hub import snapshot_download

    split = os.environ.get("JOB_BENCH_SPLIT", "main")
    source_dir = "dataset" if split == "main" else "dataset_easy"
    root = (
        Path(
            snapshot_download(
                "JobBench/job-bench",
                repo_type="dataset",
                allow_patterns=f"{source_dir}/**",
            )
        )
        / source_dir
    )
    tasks = sorted(root.glob("*/task[0-9]*"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w", encoding="utf-8") as output:
        for task in tasks:
            task_id = f"{task.parent.name}/{task.name}"
            prompt = """=== TASK FOLDER ===
/workspace/task

=== INSTRUCTIONS ===
1. Read TASK_INSTRUCTIONS.txt in the task folder
2. Read the files named in its Reference Files section
3. Complete the task as specified
4. Save only final deliverables in the output directory

=== OUTPUT DIRECTORY ===
/workspace/output

All reference files are in /workspace/task. Only access /workspace or search online for needed references.
If information conflicts, explain and justify the chosen approach. Use appropriate tools to read office files."""
            output.write(
                json.dumps(
                    {
                        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
                        "task_id": task_id,
                        "task_dir": str(task),
                        "rubrics_file": str(task / "RUBRICS.json"),
                    }
                )
                + "\n"
            )
    print(f"Wrote {len(tasks)} {split} tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
