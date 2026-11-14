# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys

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
