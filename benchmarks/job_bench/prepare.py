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
            prompt = (
                "Read /workspace/task/TASK_INSTRUCTIONS.txt and the referenced files in /workspace/task. "
                "Complete the task and save only final deliverables in /workspace/output. "
                "Do not access paths outside /workspace. If required information is absent, search the web."
            )
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
