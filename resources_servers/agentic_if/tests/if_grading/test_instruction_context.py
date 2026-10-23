# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The judge grades the instruction actually supplied, not a reconstructed requirement."""

from types import SimpleNamespace

import pytest

from resources_servers.agentic_if.if_grading.contracts import ResolvedCheck, SessionEvidence
from resources_servers.agentic_if.if_grading.engine import grade
from resources_servers.agentic_if.if_grading.judge import Judge


def response(text='{"why":"Kept the instruction.","answer":"yes"}'):
    return SimpleNamespace(output_text=text)


@pytest.mark.parametrize("at,scope", [("task_message", {}), ("user_turn", {"activation": "continuation_start"})])
def test_engine_uses_the_captured_carrier_of_the_instruction(at, scope):
    trace = SessionEvidence(
        status="complete",
        steps=[
            {"id": "u", "session_id": "s", "role": "user", "text": "Fix it. End with FINISH."},
            {"id": "a", "session_id": "s", "role": "assistant", "text": "FINISH"},
        ],
        continuation_start=0 if at == "user_turn" else None,
    )
    check = ResolvedCheck(
        id="c",
        kind="judge",
        rubric="Did it follow the instruction?",
        parameters={"context": "message"},
        scope=scope | {"places": ["final"]},
        instruction={"text": "End with FINISH.", "at": at, "carrier": "stale authoring text"},
    )
    calls = []
    result = grade([check], trace, judge=Judge(lambda body: calls.append(body) or response()))
    assert result.if_all == 1
    assert "Fix it. [[End with FINISH.]]" in calls[0]["input"][0]["content"]
    assert "stale authoring text" not in calls[0]["input"][0]["content"]
