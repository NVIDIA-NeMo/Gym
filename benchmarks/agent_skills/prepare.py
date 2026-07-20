# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the checked-in agent-skills benchmark dataset."""

from __future__ import annotations

import json
from pathlib import Path


DATASET_PATH = Path(__file__).parent / "data/create_environment_validation.jsonl"


def prepare() -> Path:
    rows = []
    for line_number, line in enumerate(DATASET_PATH.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not (row.get("responses_create_params") or {}).get("input"):
            raise ValueError(f"Line {line_number} is missing responses_create_params.input")
        metadata = row.get("verifier_metadata") or {}
        if not metadata.get("task_id") or not metadata.get("check_suite_id"):
            raise ValueError(f"Line {line_number} is missing verifier task metadata")
        rows.append(row)
    if not rows:
        raise ValueError(f"No tasks found in {DATASET_PATH}")
    print(f"Validated {len(rows)} agent-skill benchmark task(s) in {DATASET_PATH}")
    return DATASET_PATH


if __name__ == "__main__":
    prepare()
