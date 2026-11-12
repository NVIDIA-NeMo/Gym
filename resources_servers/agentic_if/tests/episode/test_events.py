# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Planted-event firings bind to the exact captured tool result; the profile plugin's load receipt is required."""

from resources_servers.agentic_if.episode.opencode_hooks import attach_events, check_hook_receipts
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence, SessionInfo, SessionStep
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


def evidence():
    return SessionEvidence(
        status="complete",
        sessions=[SessionInfo(id="root"), SessionInfo(id="child", parent_id="root")],
        steps=[
            SessionStep(
                id="a", session_id="root", role="tool", tool="read", call_id="same", invocation_id="root", output="ok"
            ),
            SessionStep(
                id="b",
                session_id="child",
                role="tool",
                tool="read",
                call_id="same",
                invocation_id="child",
                output="NOTICE",
                error=True,
            ),
        ],
    )


SPEC = RuntimeSpec(
    harness="opencode",
    task_backend="swe",
    events=[{"id": "notice", "match": {"tool": "read"}, "action": "fail", "text": "NOTICE"}],
)
RECORDS = [
    {"kind": "loaded", "plants": 1},
    {
        "kind": "fired",
        "event_id": "notice",
        "index": 0,
        "tool": "read",
        "action": "fail",
        "session_id": "child",
        "call_id": "same",
        "call_number": 2,
    },
]


def test_event_receipt_is_correlated_to_the_exact_session_and_call():
    source = evidence()
    result = attach_events(source, SPEC.events, RECORDS)
    assert result.status == "complete"
    assert result.events[0].id == "notice"
    assert result.events[0].step_index == 1
    assert result.steps[1].planted is True
    assert source.events == []
    assert attach_events(source, SPEC.events, []).status == "incomplete"
    bad = [RECORDS[0], RECORDS[1] | {"call_id": "missing"}]
    assert attach_events(source, SPEC.events, bad).status == "incomplete"


def test_the_profile_load_receipt_must_name_the_staged_profile():
    loaded = [{"kind": "loaded", "profile_sha256": "staged"}]
    complete = check_hook_receipts(
        evidence(), records=loaded, profile_sha256="staged", events=SPEC.events, event_records=RECORDS
    )
    assert complete.status == "complete" and complete.events[0].id == "notice"
    for records, profile in (([], "staged"), (loaded, "other"), ([{"kind": "loaded", "profile_sha256": None}], None)):
        missing = check_hook_receipts(evidence(), records=records, profile_sha256=profile, events=[], event_records=[])
        assert missing.status == "incomplete"
        assert missing.issues == ["hook_receipt_error:missing live hook load receipt"]
