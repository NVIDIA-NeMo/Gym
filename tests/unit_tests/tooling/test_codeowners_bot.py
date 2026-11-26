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
"""The automation bot may own enforced CODEOWNERS lines, but never protected paths."""

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
BOT = "@svcnvidia-nemo-ci"
PROTECTED_PATTERNS = {".github/", "docker/", "/pyproject.toml", "/uv.lock"}
REQUIRED_CHECK_NAME = "Autobot codeowners"


def _enforced_rules() -> dict[str, list[str]]:
    rules: dict[str, list[str]] = {}
    for line in (REPO_ROOT / ".github" / "CODEOWNERS").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pattern, *owners = line.split()
            rules[pattern] = owners
    return rules


RULES = _enforced_rules()


@pytest.mark.parametrize("pattern", sorted(PROTECTED_PATTERNS))
def test_protected_path_is_still_owned_by_a_human_team(pattern: str) -> None:
    assert pattern in RULES, f"protected pattern {pattern} is no longer an enforced CODEOWNERS rule"
    assert BOT not in RULES[pattern], f"{BOT} must not own protected pattern {pattern}"
    assert RULES[pattern], f"protected pattern {pattern} has no owner"


@pytest.mark.parametrize("pattern", sorted(set(RULES) - PROTECTED_PATTERNS))
def test_every_other_enforced_rule_includes_the_bot(pattern: str) -> None:
    assert BOT in RULES[pattern], f"{pattern} must list {BOT} so its approval can satisfy the owning teams"


def test_merge_queue_workflow_reports_the_required_check_name() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "codeowners-merge-queue.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare `on` key as True
    assert set(triggers) == {"merge_group"}
    assert [job["name"] for job in workflow["jobs"].values()] == [REQUIRED_CHECK_NAME]
    assert workflow["permissions"] == {}
