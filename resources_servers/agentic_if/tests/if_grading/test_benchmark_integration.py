# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen benchmark semantics through the canonical scorer contracts."""

import pytest

from resources_servers.agentic_if.if_grading.checkers.run import evidence_steps
from resources_servers.agentic_if.if_grading.contracts import ResolvedCheck, SessionEvidence
from resources_servers.agentic_if.if_grading.engine import grade
from resources_servers.agentic_if.if_grading.judge import Judge


def message_with_call(tool, args, **marks):
    """One assistant message holding one call, as canonical evidence records it."""
    call = {"call_id": "c", "invocation_id": "r", "tool": tool, "model_tool": tool, "args": args, "model_args": args}
    return {"id": "m", "session_id": "root", "role": "assistant", "calls": [call | marks]}


def test_change_marks_a_call_carries_survive_without_workspace_diffs():
    marks = {"changed_repo": True, "changed_paths": ["src/a.py"], "created_paths": [], "test_output_paths": []}
    evidence = SessionEvidence(status="complete", steps=[message_with_call("bash", {"command": "opaque"}, **marks)])
    call = evidence_steps(evidence)[0]["calls"][0]
    assert call["changed_repo"] is True
    assert call["changed_paths"] == ["src/a.py"]


NEW_FILE_DIFF = "diff --git a/x.py b/x.py\nnew file mode 100644\n--- /dev/null\n+++ b/x.py\n@@ -0,0 +1 @@\n+x\n"


def test_correlated_result_diff_attributes_each_grouped_call():
    calls = [
        {
            "call_id": key,
            "invocation_id": "r",
            "tool": "bash",
            "model_tool": "Run",
            "args": {"command": "opaque"},
            "model_args": {"cmd": "opaque"},
        }
        for key in ["a", "b"]
    ]
    evidence = SessionEvidence(
        status="complete",
        steps=[
            {"id": "m", "session_id": "root", "role": "assistant", "calls": calls},
            {
                "id": "ra",
                "session_id": "root",
                "role": "tool",
                "tool": "bash",
                "call_id": "b",
                "invocation_id": "r",
                "workspace_diff": NEW_FILE_DIFF,
            },
            {
                "id": "rb",
                "session_id": "root",
                "role": "tool",
                "tool": "bash",
                "call_id": "a",
                "invocation_id": "r",
                "workspace_diff": NEW_FILE_DIFF,
            },
        ],
    )
    marked = evidence_steps(evidence)[0]["calls"]
    assert marked[1]["changed_paths"] == ["x.py"]
    assert marked[0]["changed_paths"] == []


def test_planted_file_facts_resolve_against_the_runtime_workdir():
    check = ResolvedCheck(
        id="planted",
        kind="calls",
        parameters={
            "value": {
                "mode": "must",
                "tool": "read",
                "arg": {"field": "path", "regex": "AGENTS\\.md$", "planted": True},
            },
            "facts": {"planted_files": ["AGENTS.md"]},
        },
        scope={"places": ["run"]},
    )
    evidence = SessionEvidence(
        status="complete",
        steps=[message_with_call("read", {"path": "/task/docs/AGENTS.md"})],
        facts={"workdir": "/task"},
    )
    judge = Judge(lambda body: pytest.fail("a calls check asks no judge"))
    assert grade([check], evidence, judge=judge).checks[0].status == "fail"
    evidence.steps[0].calls[0].args["path"] = "/task/AGENTS.md"
    assert grade([check], evidence, judge=judge).checks[0].status == "pass"
