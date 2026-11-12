# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component imports must work without vendor path injection or a repository cwd."""

import os
import subprocess
import sys
from pathlib import Path


def test_runtime_imports_outside_repository_use_component_packages(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    program = """
import sys
from pathlib import Path
from resources_servers.legal_agent_bench import prepare, verifier, environment_prepare
from resources_servers.legal_agent_bench.lab_harbor import judge, scoring, tools

root = Path(prepare.__file__).parent
assert str(root / 'vendor' / 'harvey_labs') not in sys.path
assert environment_prepare.PACKAGE_DIR == root
assert prepare.TOOL_RUNNER_SOURCE == root / 'lab_harbor' / 'container_tool_runner.py'
assert verifier.score_rubric is scoring.score_rubric
assert verifier.OpenAICompatibleJudge is judge.OpenAICompatibleJudge
assert callable(tools.get_all_tool_definitions)
for module in [judge, scoring, tools]:
    assert Path(module.__file__).is_relative_to(root)
    assert 'vendor' not in Path(module.__file__).relative_to(root).parts
vendor = root / 'vendor' / 'harvey_labs'
files = {
    path.relative_to(vendor).as_posix()
    for path in vendor.rglob('*')
    if path.is_file() and '__pycache__' not in path.relative_to(vendor).parts
}
assert files == {'LICENSE', 'README.md'}
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
