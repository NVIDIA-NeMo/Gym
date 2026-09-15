# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = DATA_DIR / "workspace_bench_lite.jsonl"


def prepare() -> Path:
    from huggingface_hub import snapshot_download

    root = (
        Path(
            snapshot_download(
                "Workspace-Bench/Workspace-Bench-Lite",
                repo_type="dataset",
                allow_patterns="task_lite_clean_en/**",
            )
        )
        / "task_lite_clean_en"
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for metadata_path in sorted(root.glob("*/metadata.json"), key=lambda path: int(path.parent.name)):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            task_id = str(metadata.get("absolute_id", metadata_path.parent.name))
            output.write(
                json.dumps(
                    {
                        "responses_create_params": {
                            "input": [
                                {
                                    "role": "user",
                                    "content": (
                                        f"{metadata['task']}\n\nRead source files from /workspace/input. "
                                        "Save every requested final deliverable in /workspace/output."
                                    ),
                                }
                            ]
                        },
                        "task_id": task_id,
                        "task_dir": str(metadata_path.parent),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote Workspace-Bench-Lite tasks to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    prepare()
