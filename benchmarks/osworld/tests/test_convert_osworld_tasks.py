# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for converting upstream OSWorld manifests into Gym training rows."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


SCRIPT = Path(__file__).parents[1] / "tools" / "convert_osworld_tasks.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("convert_osworld_tasks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_writes_agent_ref_for_direct_run_examples_contract(
    tmp_path: Path,
) -> None:
    module = _module()
    osworld_root = tmp_path / "OSWorld"
    examples = osworld_root / "evaluation_examples" / "examples" / "chrome"
    examples.mkdir(parents=True)
    (osworld_root / "evaluation_examples" / "test_nogdrive.json").write_text(
        json.dumps({"chrome": ["task-1"]}), encoding="utf-8"
    )
    (examples / "task-1.json").write_text(
        json.dumps({"id": "task-1", "instruction": "Change the setting."}),
        encoding="utf-8",
    )
    output = tmp_path / "tasks.jsonl"

    total, per_domain = module.convert(
        osworld_root,
        "test_nogdrive",
        output,
        agent_name="osworld_simple_agent",
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert total == 1
    assert per_domain == {"chrome": 1}
    assert row["agent_ref"] == {"name": "osworld_simple_agent"}
    assert row["responses_create_params"]["input"][0]["content"] == ("Change the setting.")
    provenance = json.loads(output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8"))
    assert provenance["rows"] == 1
    assert provenance["osworld_commit"] is None
    assert provenance["output"] == str(output.resolve())
    assert len(provenance["manifest_sha256"]) == 64
    assert len(provenance["output_sha256"]) == 64
    assert len(provenance["task_ids_sha256"]) == 64
