# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the OSWorld task-definition parity preflight."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


SCRIPT = Path(__file__).parents[1] / "tools" / "check_task_input_parity.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_task_input_parity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compare_inputs_accepts_nested_gym_and_flat_tasks(tmp_path: Path) -> None:
    module = _module()
    task = {
        "id": "task-1",
        "instruction": "Do the task.",
        "config": [{"type": "launch", "parameters": {"command": ["app"]}}],
        "evaluator": {"func": "exact"},
    }
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_jsonl(left, [task])
    _write_jsonl(right, [{"verifier_metadata": {"task_id": "task-1", "osworld_task": task}}])

    report = module.compare_inputs(left, right)

    assert report["parity"] is True
    assert report["matched"] == 1


def test_compare_inputs_detects_config_order_mismatch(tmp_path: Path) -> None:
    module = _module()
    base = {
        "id": "task-vlc",
        "instruction": "Change the setting.",
        "evaluator": {"func": "check_setting"},
    }
    left_task = {**base, "config": [{"type": "execute"}, {"type": "launch"}]}
    right_task = {**base, "config": [{"type": "launch"}, {"type": "execute"}]}
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_jsonl(left, [left_task])
    _write_jsonl(right, [right_task])

    report = module.compare_inputs(left, right)

    assert report["parity"] is False
    assert report["mismatched"] == 1
    assert report["mismatches"][0]["task_id"] == "task-vlc"
    assert report["mismatches"][0]["changed_fields"] == ["config"]
