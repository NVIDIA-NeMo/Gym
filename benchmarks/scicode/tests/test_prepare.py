# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``benchmarks/scicode/prepare.py``."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from benchmarks.scicode import prepare as scicode_prepare


def _entry(problem_id: str) -> dict:
    return {
        "problem_id": problem_id,
        "problem_name": f"problem-{problem_id}",
        "required_dependencies": "import numpy as np",
        "sub_steps": [],
    }


def test_prepare_uses_test_split_problem_ids(monkeypatch, tmp_path) -> None:
    dataset = {
        "validation": [_entry("validation-only")],
        "test": [_entry("test-1"), _entry("test-2")],
    }
    load_dataset = Mock(return_value=dataset)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr(scicode_prepare, "DATA_DIR", tmp_path)
    monkeypatch.setattr(scicode_prepare, "OUTPUT_FPATH", tmp_path / "scicode_benchmark.jsonl")
    monkeypatch.setattr(scicode_prepare, "get_global_config_dict", lambda: {})

    output_path = scicode_prepare.prepare()
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert [row["problem_id"] for row in rows] == ["test-1", "test-2"]
    assert [row["uuid"] for row in rows] == ["test-1", "test-2"]
    load_dataset.assert_called_once_with("SciCode1/SciCode", token=None)
