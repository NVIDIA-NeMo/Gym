# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrap existing SWE-bench Pro preparation in explicit native task rows."""

import json
from pathlib import Path

from benchmarks.swebench.pro.prepare import OUTPUT_FPATH, prepare


def prepare_native(*, source: Path = OUTPUT_FPATH, output: Path | None = None) -> Path:
    """Preserve task and grading data while separating the Responses request."""
    if not source.exists():
        prepare(output_fpath=source)
    output = output or source.with_name("swebench_pro_nooa.jsonl")
    with source.open() as reader, output.open("w") as writer:
        for line in reader:
            task_data = json.loads(line)
            params = task_data.pop("responses_create_params")
            writer.write(
                json.dumps(
                    {
                        "task_id": {"taskset": "swebench-pro-nooa", "task_id": task_data["instance_id"]},
                        "task_input": {"responses_create_params": params, "task_data": task_data},
                    }
                )
                + "\n"
            )
    return output


if __name__ == "__main__":
    prepare_native()
