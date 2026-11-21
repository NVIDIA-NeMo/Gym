# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from resources_servers.legal_agent_bench.prepare import TOOL_RUNNER_SOURCE


def test_large_document_write_uses_stdin(tmp_path):
    content = "Legal document content\n" * 20_000
    output = tmp_path / "memo.txt"
    result = subprocess.run(
        [sys.executable, str(TOOL_RUNNER_SOURCE), "write"],
        input=json.dumps({"file_path": str(output), "content": content}),
        env={**os.environ, "WORKSPACE_DIR": str(tmp_path), "OUTPUT_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["metrics"]["files_written"] == 1
    assert output.read_text() == content


@pytest.mark.parametrize("case", ["valid", "outside", "invalid_entries", "string", "number", "object"])
def test_finish_checks_actual_output_files(tmp_path: Path, case: str) -> None:
    workspace = tmp_path / "workspace"
    output = workspace / "output"
    output.mkdir(parents=True)
    (output / "memo.docx").write_text("File existence only; grading checks its contents.")
    outside = workspace / "outside.txt"
    outside.write_text("Not a deliverable")
    (output / "escape.txt").symlink_to(outside)
    (output / "directory").mkdir()
    cases = {
        "valid": (["memo.docx", "output/memo.docx", str(output / "memo.docx")], []),
        "outside": (
            ["missing.docx", "outside.txt", "../outside.txt", str(outside), "escape.txt", "directory"],
            ["missing.docx", "outside.txt", "../outside.txt", str(outside), "escape.txt", "directory"],
        ),
        "invalid_entries": ([None, "", 42], ["None", "''", "42"]),
        "string": ("memo.docx", ["'memo.docx'"]),
        "number": (42, ["42"]),
        "object": ({"file": "memo.docx"}, ["{'file': 'memo.docx'}"]),
    }
    deliverables, expected_missing = cases[case]
    result = subprocess.run(
        [sys.executable, str(TOOL_RUNNER_SOURCE), "finish"],
        input=json.dumps({"summary": "Done", "deliverables": deliverables}),
        env={**os.environ, "WORKSPACE_DIR": str(workspace), "OUTPUT_DIR": str(output)},
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["metrics"]["missing_deliverables"] == expected_missing
