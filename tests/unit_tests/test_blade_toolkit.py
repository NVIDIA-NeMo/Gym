#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLAUDE_SKILLS = (
    "add-benchmark",
    "gh-stack",
    "nemo-gym-blade-analysis",
    "nemo-gym-debugging",
    "nemo-gym-docs",
    "nemo-gym-pivot-datasets",
    "nemo-gym-reward-profiling",
)
CODEX_COMPATIBILITY_LINKS = (
    "nemo-gym-blade-analysis",
    "nemo-gym-debugging",
    "nemo-gym-pivot-datasets",
    "nemo-gym-reward-profiling",
)
TOOLKIT = REPO_ROOT / ".agents/skills/nemo-gym-blade-analysis/scripts/blade_toolkit.py"


def test_agent_specific_skill_paths_resolve_to_canonical_skills():
    for skill in CLAUDE_SKILLS:
        canonical_skill = REPO_ROOT / ".agents/skills" / skill
        assert (REPO_ROOT / ".claude/skills" / skill).resolve() == canonical_skill.resolve()

    for skill in CODEX_COMPATIBILITY_LINKS:
        canonical_skill = REPO_ROOT / ".agents/skills" / skill
        assert (REPO_ROOT / ".codex/skills" / skill).resolve() == canonical_skill.resolve()


def test_make_shallow_keeps_only_high_level_metric_tables(tmp_path):
    source = tmp_path / "golden_report.md"
    output = tmp_path / "shallow.md"
    source.write_text(
        """# Example BLADE Report

## 1. Aggregate Metrics

| Model | Pass@1 |
|---|---:|
| strong | 80.0% |

### Task-Level Aggregate Detail

| Task | Evidence |
|---|---|
| task_nested_leak | should not appear |

```text
code block with task_nested_leak and diagnostic evidence
```

- Diagnostic bullet with task_nested_leak should be dropped.

## Workflow Funnel and Phase Distribution

| Phase | Count |
|---|---:|
| scored | 10 |

## Dominant Failure Modes

| Failure | Count |
|---|---:|
| wrong_tool | 7 |

### task_123

| Task | Tool |
|---|---|
| task_123 | createSecretEvidence |
""",
    )

    subprocess.run(
        [sys.executable, "-S", str(TOOLKIT), "make-shallow", "--input", str(source), "--output", str(output)],
        check=True,
    )

    shallow = output.read_text()
    assert "| Model | Pass@1 |" in shallow
    assert "| Phase | Count |" in shallow
    assert "## Dominant Failure Modes" in shallow
    assert "wrong_tool" not in shallow
    assert "task_nested_leak" not in shallow
    assert "Diagnostic bullet" not in shallow
    assert "code block with task_nested_leak" not in shallow
    assert "task_123" not in shallow
    assert "createSecretEvidence" not in shallow
