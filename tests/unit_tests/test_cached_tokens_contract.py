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
"""Agents report cache reads the harness reported, or None; a literal zero needs a known-zero comment."""

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / "responses_api_agents"
ALLOW_MARKER = "known-zero"
# `cached_tokens=0`, `"cached_tokens": 0`, `cached_tokens = 0` and `.get("cached_tokens", 0)`, also for
# `cached_input_tokens`.
LITERAL_ZERO = re.compile(r"""\bcached_(?:input_)?tokens(?:['"]?\s*[:=]|['"]\s*,)\s*0\b""")
# Fixed by NVIDIA-NeMo/Gym#5227; remove once it merges.
PENDING = {Path("claude_code_agent/app.py")}


def _is_test_file(relative: Path) -> bool:
    return "tests" in relative.parts or relative.name.startswith("test_") or relative.name.endswith("_test.py")


@pytest.mark.parametrize(
    ("line", "flagged"),
    [
        ("input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),", True),
        ('"input_tokens_details": {"cached_tokens": 0},', True),
        ("cached_tokens = 0", True),
        ('cached_tokens = usage.get("cached_tokens", 0)', True),
        ('cached_tokens=usage.get("cached_input_tokens", 0)', True),
        ("input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),  # known-zero", False),
        ('cached_tokens = usage.get("cached_tokens")', False),
        ("cached_tokens=cache_read if cache_read > 0 else None", False),
        ("cached_tokens=10", False),
    ],
)
def test_literal_zero_pattern(line: str, flagged: bool) -> None:
    assert bool(LITERAL_ZERO.search(line) and ALLOW_MARKER not in line) is flagged


def test_agents_do_not_report_a_literal_zero_for_cache_reads() -> None:
    violations = []
    for path in sorted(AGENTS_DIR.rglob("*.py")):
        relative = path.relative_to(AGENTS_DIR)
        if _is_test_file(relative) or relative in PENDING:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if LITERAL_ZERO.search(line) and ALLOW_MARKER not in line:
                violations.append(f"responses_api_agents/{relative}:{number}: {line.strip()}")

    assert not violations, (
        "Report the cache reads the harness reported, or None when it reported none; 0 reads as a measured "
        f"'no cache hits'. For a provable zero, explain why and end the line with `# {ALLOW_MARKER}`:\n"
        + "\n".join(violations)
    )
