# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The scorer boundary rejects ambiguous scope and preserves actual actor evidence."""

import pytest
from pydantic import ValidationError

from resources_servers.agentic_if.if_grading.contracts import ResolvedCheck, SessionEvidence


CALL = {"call_id": "c", "invocation_id": "i", "tool": "read", "model_tool": "read", "args": {}, "model_args": {}}


@pytest.mark.parametrize(
    "fields,match",
    [
        ({"unknown_rule": {}}, "Extra inputs"),
        ({"instruction_text": "Append me to the prompt"}, "Extra inputs"),
        ({"rubric": None}, "rubric"),
        ({"rubric": ""}, "rubric"),
        ({"rubric": "   "}, "rubric"),
        ({"scope": {"activation": "event"}}, "event_id"),
        ({"scope": {"activation": "session_start", "event_id": "not-active"}}, "event_id"),
        ({"scope": {"places": []}}, "places"),
        ({"kind": "arbitrary_python", "parameters": {"code": "print('bad')"}}, "kind"),
    ],
)
def test_invalid_check_is_rejected_before_execution(fields, match):
    with pytest.raises(ValidationError, match=match):
        ResolvedCheck(**{"id": "a", "kind": "judge", "rubric": "Be concise."} | fields)


@pytest.mark.parametrize(
    "fields,match",
    [
        ({"events": [{"id": "e", "step_index": 0}, {"id": "e", "step_index": 0}]}, "event.*unique"),
        ({"steps": [], "events": [{"id": "e", "step_index": 0}]}, "step_index"),
        ({"status": "incomplete", "steps": []}, "issue"),
        ({"steps": [{"id": "m", "session_id": "root", "role": "assistant", "calls": [CALL, CALL]}]}, "call.*unique"),
    ],
)
def test_invalid_evidence_is_rejected(fields, match):
    evidence = {
        "status": "complete",
        "steps": [{"id": "m", "session_id": "root", "role": "user", "text": "New rule."}],
        "sessions": [{"id": "root", "parent_id": None}],
    }
    with pytest.raises(ValidationError, match=match):
        SessionEvidence(**evidence | fields)


@pytest.mark.parametrize(
    "fields",
    [
        # A child session may reuse its parent's local message and call IDs.
        {
            "status": "complete",
            "steps": [
                {"id": "msg1", "session_id": session, "role": "assistant", "text": "Inspect.", "calls": [CALL]}
                for session in ["root", "child"]
            ],
            "sessions": [{"id": "root", "parent_id": None}, {"id": "child", "parent_id": "root"}],
        },
        # Incomplete evidence is valid once it says what is missing.
        {"status": "incomplete", "steps": [], "issues": ["child export missing"]},
    ],
)
def test_reused_child_session_ids_and_explained_incomplete_evidence_are_valid(fields):
    SessionEvidence(**fields)
