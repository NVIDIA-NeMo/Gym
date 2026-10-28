# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare benchmark JSONL for Gym; reference labels never enter policy input."""

import json
from pathlib import Path

from benchmarks.opencode_multimodal_benchmarks.canonical import validate_mcq_tasks
from benchmarks.opencode_multimodal_benchmarks.data import prepare as prepare_tasks
from benchmarks.opencode_multimodal_benchmarks.data import sha256_file


BENCHMARKS = {"vstar": "VStarBench", "hrbench8k": "HRBench8K", "vtcbench": "VTCBench", "babyvision": "BabyVision"}


def task_row(task):
    return {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": task["prompt"]},
                        *[{"type": "input_image", "image_url": path, "detail": "auto"} for path in task["images"]],
                    ],
                }
            ],
            "temperature": 0.6,
            "top_p": 0.95,
        },
        "verifier_metadata": {key: value for key, value in task.items() if key not in ("prompt", "images")},
    }


def prepare(benchmark, tsv, image_root=None, prepared_dir=None, limit=None, sha256=None):
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark}")
    if sha256 is not None and sha256_file(Path(tsv)) != sha256:
        raise ValueError(f"{benchmark} source dataset checksum mismatch")
    # The native launcher's Hydra overrides stringify Python None as "None".
    if isinstance(limit, str):
        limit = None if limit.lower() in ("none", "null") else int(limit)
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer or null")
    output = Path(f"benchmarks/opencode_multimodal_benchmarks/data/{benchmark}.jsonl")
    materialized = Path(prepared_dir) if prepared_dir else output.with_suffix("")
    tasks, _ = prepare_tasks(
        Path(tsv),
        BENCHMARKS[benchmark],
        materialized,
        image_root=Path(image_root) if image_root else None,
        answer_format=True,
    )
    if benchmark in ("vstar", "hrbench8k"):
        validate_mcq_tasks(tasks, BENCHMARKS[benchmark])
    tasks = [task | {"expected_samples": len(tasks[:limit])} for task in tasks[:limit]]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(task_row(task), ensure_ascii=False) + "\n" for task in tasks))
    temporary.replace(output)
    return output
