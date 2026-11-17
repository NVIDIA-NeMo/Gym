# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delegated conversations retain their grading relationships."""

import pytest

from resources_servers.agentic_if.if_grading import engine
from resources_servers.agentic_if.if_grading.configuration import load_config
from resources_servers.agentic_if.if_grading.contracts import ResolvedCheck, SessionEvidence
from resources_servers.agentic_if.if_grading.judge import Judge
from resources_servers.agentic_if.if_grading.rendering import render


def delegated():
    return [
        {"session_id": "root", "role": "user", "text": "PARENT TASK"},
        {"session_id": "root", "role": "assistant", "text": "Delegate", "calls": [{"tool": "task", "args": {}}]},
        {"session_id": "child", "role": "user", "text": "CHILD SUBTASK"},
        {"session_id": "child", "role": "assistant", "text": "Child response", "calls": []},
        {"session_id": "root", "role": "assistant", "text": "Root final", "calls": []},
    ]


def test_message_contexts_show_only_their_own_session():
    for kind in ["message_and_request", "messages_after_request", "message_and_conversation"]:
        rendered = render(delegated(), kind, 4, load_config())
        assert "PARENT TASK" in rendered and "Root final" in rendered
        assert "CHILD SUBTASK" not in rendered and "Child response" not in rendered
    assert "Delegate" in render(delegated(), "messages_after_request", 4, load_config())


def test_whole_run_judge_context_distinguishes_sessions():
    rendered = render(delegated(), "run", None, load_config())
    assert "session=root" in rendered and "session=child" in rendered


def test_user_language_is_inferred_from_the_same_session(monkeypatch):
    seen = []

    def detect(text):
        seen.append(text)
        return "English", None

    monkeypatch.setattr(engine.language, "detect", detect)
    monkeypatch.setattr(engine.language, "assess", lambda *_: {"ok": True, "clear": True, "detail": "matches"})
    steps = delegated()
    # The parent answer follows the child conversation in merged evidence.
    steps[1]["calls"] = []
    evidence = SessionEvidence(status="complete", steps=[{"id": str(i), **step} for i, step in enumerate(steps)])
    check = ResolvedCheck(id="language", kind="language", parameters={"value": "user"}, scope={"places": ["final"]})
    judge = Judge(lambda body: pytest.fail("a clear language check asks no judge"))
    result = engine.grade([check], evidence, judge=judge)
    assert result.checks[0].status == "pass"
    assert seen == ["PARENT TASK"]
