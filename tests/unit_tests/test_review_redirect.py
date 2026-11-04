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

import os
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_review_redirect_is_least_privilege_and_does_not_execute_review() -> None:
    """Require the complete trusted-command gate and notice-only permissions."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/claude-review.yml").read_text())
    job = workflow["jobs"]["redirect-to-review"]
    condition = job["if"]
    assert " ".join(condition.split()) == (
        "github.event_name == 'issue_comment' && "
        "github.event.issue.pull_request && "
        "github.event.comment.user.type != 'Bot' && "
        'contains(fromJSON(\'["OWNER", "MEMBER", "COLLABORATOR"]\'), '
        "github.event.comment.author_association) && "
        "(github.event.comment.body == '/claude review' || "
        "github.event.comment.body == '/claude strict-review')"
    )
    assert job["permissions"] == {"pull-requests": "write"}
    assert "uses" not in job
    assert "secrets" not in job
    assert len(job["steps"]) == 1
    step = job["steps"][0]
    assert "uses" not in step
    assert step["env"]["COMMENT_BODY"] == "${{ github.event.comment.body }}"
    assert step["run"].splitlines() == [
        "# Actions string equality ignores case; enforce literal commands here.",
        'case "$COMMENT_BODY" in',
        "  '/claude review'|'/claude strict-review')",
        '    gh pr comment "$PR_NUMBER" --repo "$REPO" --body "$NOTICE"',
        "    ;;",
        "esac",
    ]
    assert "${{" not in step["run"]
    assert "model=claude" in step["env"]["NOTICE"]
    assert "model=codex" in step["env"]["NOTICE"]
    assert "/review help" in step["env"]["NOTICE"]
    assert job["env"]["RETIRED_COMMAND"] == (
        "${{ github.event.comment.body == '/claude strict-review' && '/claude strict-review' || '/claude review' }}"
    )
    assert job["env"]["REVIEW_COMMAND"] == (
        "${{ github.event.comment.body == '/claude strict-review' && '/review mode=strict' || '/review' }}"
    )


def test_review_redirect_workflow_has_no_other_execution_or_permissions() -> None:
    """Reject automatic review triggers and additional privileged jobs."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/claude-review.yml").read_text())
    assert workflow["permissions"] == {}
    assert set(workflow["jobs"]) == {"redirect-to-review"}
    events = workflow.get("on", workflow.get(True))
    assert events == {"issue_comment": {"types": ["created"]}}


@pytest.mark.parametrize(
    "comment_body, accepted",
    [
        ("/claude review", True),
        ("/claude strict-review", True),
        ("/CLAUDE REVIEW", False),
        ("/CLAUDE STRICT-REVIEW", False),
        ("/Claude Review", False),
        ("/Claude Strict-Review", False),
        ("/claude review ", False),
        ("/claude strict-review ", False),
        (" /claude review", False),
        (" /claude strict-review", False),
        ("/claude review\n", False),
        ("/claude strict-review\n", False),
        ("`/claude review`", False),
        ("`/claude strict-review`", False),
        ("> /claude review", False),
        ("> /claude strict-review", False),
    ],
)
@pytest.mark.parametrize(
    "notice",
    [
        "Use /review",
        "Use /review mode=strict",
        "$(touch injected) `touch injected` ' \" ;\n/review",
    ],
)
def test_notice_requires_exact_command_and_is_passed_as_one_literal_argument(
    tmp_path: Path, notice: str, comment_body: str, accepted: bool
) -> None:
    """Only literal commands may send a notice, without shell evaluation."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/claude-review.yml").read_text())
    script = workflow["jobs"]["redirect-to-review"]["steps"][0]["run"]
    gh = tmp_path / "gh"
    gh.write_text('#!/bin/sh\nprintf \'%s\\0\' "$@" >> "$CAPTURE"\n')
    gh.chmod(0o755)
    capture = tmp_path / "args"
    env = dict(
        os.environ,
        PS1="",
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        PR_NUMBER="123",
        REPO="NVIDIA-NeMo/example",
        NOTICE=notice,
        COMMENT_BODY=comment_body,
        CAPTURE=str(capture),
    )
    env.pop("BASH_ENV", None)
    env.pop("ENV", None)
    subprocess.run(["/bin/bash", "-eu", "-c", script], cwd=tmp_path, env=env, check=True)
    if accepted:
        assert capture.read_bytes().split(b"\0") == [
            b"pr",
            b"comment",
            b"123",
            b"--repo",
            b"NVIDIA-NeMo/example",
            b"--body",
            notice.encode(),
            b"",
        ]
    else:
        assert not capture.exists()
    assert not (tmp_path / "injected").exists()
