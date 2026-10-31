# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate synthetic task rows through Gym's actual task-data schema interface."""

import json
from copy import deepcopy
from pathlib import Path

from nemo_gym.task_data import TaskDataValidator, load_task_data_schema, validate_jsonl_rows


def test_task_data_preserves_legacy_bindings_and_rejects_bad_rows():
    directory = Path(__file__).parents[1]
    adapter = load_task_data_schema(directory)
    path = directory / "data/example.jsonl"
    lines = path.read_text().splitlines()
    report = validate_jsonl_rows("agentic_if_realworld", adapter, str(path), lines)
    assert report.clean and report.rows == 5
    original = json.loads(lines[0])
    for damage in ["missing", "malformed_hash", "misplaced"]:
        row = deepcopy(original)
        if damage == "missing":
            row["verifier_metadata"].pop("item_id")
        elif damage == "malformed_hash":
            row["verifier_metadata"]["input_sha256"] = "invalid"
        else:
            row.update(row.pop("verifier_metadata"))
        validator = TaskDataValidator(server_name="agentic_if_realworld", adapter=adapter, dataset_fpath=str(path))
        validator.validate_row(0, row)
        assert not validator.report.clean
        if damage == "misplaced":
            assert set(validator.report.misplaced_keys) == {"item_id", "dataset_sha256", "input_sha256"}
