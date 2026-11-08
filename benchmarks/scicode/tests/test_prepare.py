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
        "sub_steps": [
            {
                "step_number": f"{problem_id}.1",
                "step_description_prompt": "Write f.",
                "step_background": "Background.",
                "ground_truth_code": "def f():\n    return 1",
                "function_header": "def f():",
                "test_cases": ["assert f() == 1"],
                "return_line": "    return result",
            }
        ],
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


def test_prepare_gives_agent_sessions_the_prompt_material_only(monkeypatch, tmp_path) -> None:
    load_dataset = Mock(return_value={"test": [_entry("test-1")]})
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr(scicode_prepare, "DATA_DIR", tmp_path)
    monkeypatch.setattr(scicode_prepare, "OUTPUT_FPATH", tmp_path / "scicode_benchmark.jsonl")
    monkeypatch.setattr(scicode_prepare, "get_global_config_dict", lambda: {})

    (row,) = [json.loads(line) for line in scicode_prepare.prepare().read_text().splitlines()]

    assert row["agent_context"] == {
        "values": {
            "problem_id": "test-1",
            "sub_steps": [
                {
                    "step_number": "test-1.1",
                    "step_description_prompt": "Write f.",
                    "step_background": "Background.",
                    "function_header": "def f():",
                    "return_line": "    return result",
                }
            ],
            "required_dependencies": "import numpy as np",
        }
    }
    # The verifier's fields stay on the row, out of the agent's context.
    assert row["sub_steps"][0]["test_cases"] == ["assert f() == 1"]
