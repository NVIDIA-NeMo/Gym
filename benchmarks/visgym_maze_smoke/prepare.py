# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a small qualification set; this is not the full published benchmark."""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parents[1]
SEED_BASE = 900000


def prepare() -> Path:
    """Use explicit fixture task contracts and reserved deterministic evaluation seeds."""
    source = ROOT / "resources_servers/visgym/data/example.jsonl"
    templates = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    templates = templates[:1]
    rows = []
    for family_index, template in enumerate(templates):
        for offset in range(5):
            row = json.loads(json.dumps(template))
            row.pop("task_idx", None)
            row.pop("agent_ref", None)
            row["seed"] = SEED_BASE + 100 * family_index + offset
            row["task_id"] = f"visgym_maze_smoke:{row['env_id']}:{row['seed']}"
            row["task_metadata"] = {"suite": "visgym", "split": "benchmark", "qualification_only": True}
            rows.append(row)
    destination = BENCHMARK_DIR / "data/visgym_maze_smoke_benchmark.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return destination


if __name__ == "__main__":
    print(prepare())
