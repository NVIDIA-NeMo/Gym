# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the pinned benchmark's verbatim prompts; no initial media injection."""

import json
import os
from pathlib import Path

from responses_api_agents.agentic_vbench_agent.core import dataset_rows, inventory


def prepare() -> Path:
    tasks = inventory(Path(os.environ["AGENTIC_VBENCH_ROOT"]))
    rows = dataset_rows(tasks, os.environ.get("AGENTIC_VBENCH_TASKS", "all"))
    output = Path(
        os.environ.get("AGENTIC_VBENCH_DATASET", Path(__file__).parent / "data/agentic_vbench_benchmark.jsonl")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Prepared {len(rows)} Agentic-VBench tasks: {output}")
    return output


if __name__ == "__main__":
    prepare()
