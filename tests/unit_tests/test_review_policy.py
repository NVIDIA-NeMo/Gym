# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_legacy_review_workflows_are_removed() -> None:
    workflows = ROOT / ".github/workflows"
    assert not list(workflows.glob("*claude*review*"))
    for path in (*workflows.glob("*.yml"), *workflows.glob("*.yaml")):
        text = path.read_text()
        for retired in ("_claude_review", "claude-code-action", "auto-review:", "/claude review", "/claude strict-review"):
            assert retired not in text, f"{path.name} contains {retired}"


def test_formal_review_rubric_is_inert_and_uses_formal_submission() -> None:
    rubric = (ROOT / ".agents/skills/pr-review/SKILL.md").read_text()
    metadata = yaml.safe_load(rubric.split("---", 2)[1])
    assert metadata["name"] == "pr-review"
    assert metadata["disable-model-invocation"] is True
    assert metadata["user_invocable"] is False
    assert "mode=light" in rubric
    assert "mode=strict" in rubric
    assert "Do not run GitHub commands or" in rubric
    assert "Never approve an incomplete review" in rubric
    assert "automatic job" not in rubric
    assert "gh pr diff" not in rubric


def test_review_guidance_uses_only_formal_commands() -> None:
    guidance = (ROOT / "CONTRIBUTING.md").read_text().split("## Pull-request reviews", 1)[1]
    assert "`/review`" in guidance
    assert "`/review mode=strict`" in guidance
    assert "protected `main`" in guidance
    assert "/claude" not in guidance
    assert "redirect" not in guidance
