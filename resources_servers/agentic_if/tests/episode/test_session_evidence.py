# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One episode's evidence, built from the exports and receipts the plugin bundle collected."""

from copy import deepcopy

from resources_servers.agentic_if.episode.session_evidence import build_session_evidence, row_prompt_texts
from resources_servers.agentic_if.if_grading.contracts import SessionInfo
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec
from resources_servers.agentic_if.tests.support import (
    REPLAY_RECEIPT,
    answered_export,
    call,
    export,
    message,
    observation,
    text,
    tool_observation,
)


PROFILE = "p" * 64
LOADED = [{"kind": "loaded", "profile_sha256": PROFILE}]


def build(
    spec=None,
    *,
    exports,
    sessions=None,
    hooks=LOADED,
    events=(),
    calls=(),
    replay=None,
    artifacts=None,
    task_text=None,
):
    exports = {document["info"]["id"]: document for document in exports}
    if sessions is None:
        sessions = [SessionInfo(id=key, parent_id=value["info"].get("parentID")) for key, value in exports.items()]
    return build_session_evidence(
        spec or RuntimeSpec(harness="opencode", task_backend="swe"),
        exports=exports,
        sessions=sessions,
        hook_receipts=list(hooks),
        event_records=list(events),
        call_records=list(calls),
        profile_sha256=PROFILE,
        replay=replay,
        artifacts=artifacts or {},
        task_text=task_text,
    )


def test_a_plain_session_binds_its_model_calls_and_receipts():
    source = answered_export()
    before = deepcopy(source)
    built = build(exports=[source], calls=[observation(), tool_observation()])
    evidence = built.evidence
    assert evidence.status == "complete", evidence.issues
    assert [step.role for step in evidence.steps] == ["user", "system", "assistant", "tool", "assistant"]
    assert evidence.steps[1].text == "Explain each command."
    actual = evidence.steps[2].calls[0]
    assert (actual.tool, actual.model_tool, actual.model_args) == ("bash", "RunCommand", {"command": "pwd"})
    assert built.root_export["info"]["id"] == "root"
    assert built.enriched_exports == {"root": source} and source == before


def test_a_receipt_that_binds_to_no_call_keeps_the_raw_exports_and_is_a_gap():
    receipt = {
        "kind": "tool_receipt",
        "session_id": "root",
        "call_id": "missing",
        "opencode_tool": "bash",
        "metadata": {},
    }
    built = build(exports=[answered_export()], hooks=LOADED + [receipt], calls=[observation(), tool_observation()])
    assert built.enriched_exports is None
    assert built.evidence.status == "incomplete"
    assert any(issue.startswith("hook_execution_receipts:") for issue in built.evidence.issues)


def test_missing_plugin_or_model_records_make_the_evidence_incomplete():
    assert (
        "hook_receipt_error:missing live hook load receipt"
        in build(exports=[answered_export()], hooks=[]).evidence.issues
    )
    unobserved = build(exports=[answered_export()], calls=[]).evidence
    assert unobserved.status == "incomplete"
    assert "missing_model_observation:root:a" in unobserved.issues


def test_a_delegated_child_session_is_graded_beside_its_root():
    child = answered_export("child", "root")
    calls = [observation(), tool_observation(), observation("child", tool="ChildShell"), tool_observation("child")]
    evidence = build(exports=[answered_export(), child], calls=calls).evidence
    assert evidence.status == "complete", evidence.issues
    assert [(session.id, session.parent_id) for session in evidence.sessions] == [("root", None), ("child", "root")]
    missing = build(
        exports=[answered_export()],
        sessions=[SessionInfo(id="root"), SessionInfo(id="child", parent_id="root")],
        calls=calls[:2],
    ).evidence
    assert "missing_session:child" in missing.issues


def test_a_planted_event_binds_to_the_call_it_failed():
    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        events=[{"id": "notice", "match": {"tool": "bash"}, "action": "fail", "text": "NOTICE"}],
    )
    fired = [
        {"kind": "loaded", "plants": 1},
        {
            "kind": "fired",
            "event_id": "notice",
            "tool": "bash",
            "action": "fail",
            "session_id": "root",
            "call_id": "c1",
        },
    ]
    evidence = build(
        spec, exports=[answered_export()], events=fired, calls=[observation(), tool_observation()]
    ).evidence
    assert evidence.status == "complete", evidence.issues
    assert evidence.events[0].id == "notice" and evidence.steps[evidence.events[0].step_index].planted


def test_tool_steps_show_what_the_model_saw_and_capture_gaps_are_explicit():
    presented = tool_observation(value="RunCommand failed: denied")
    evidence = build(exports=[answered_export()], calls=[observation(), presented]).evidence
    assert evidence.status == "complete", evidence.issues
    assert next(step for step in evidence.steps if step.role == "tool").output == "RunCommand failed: denied"
    changed = answered_export()
    changed["messages"][1]["parts"][0]["text"] = "Not what the model said"
    gap = build(exports=[changed], calls=[observation(), tool_observation()]).evidence
    assert gap.status == "incomplete"
    assert "training_capture:root:captured_and_exported_assistant_output_disagree" in gap.issues


def test_the_row_s_prompt_texts_are_its_system_message_and_its_user_messages():
    messages = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": "Be terse."}, {"type": "input_text", "text": "!"}],
        },
        {"role": "user", "content": "Fix it"},
    ]
    assert row_prompt_texts(messages) == ("Be terse.\n!", "Fix it")
    assert row_prompt_texts(messages[1:] * 2) == (None, "Fix it\n\nFix it")


def first_user_text(value):
    source = answered_export()
    source["messages"][0]["parts"] = [text(value)]
    return source


def test_the_root_session_must_start_from_the_row_s_literal_task():
    calls = [observation(), tool_observation()]
    exact = build(exports=[answered_export()], calls=calls, task_text="Fix it").evidence
    assert exact.status == "complete", exact.issues
    # The harness may frame the task with its own text.
    framed = build(exports=[first_user_text("<task>\nFix it\n</task>")], calls=calls, task_text="Fix it").evidence
    assert framed.status == "complete", framed.issues
    for export_text in ("Fix that", "Fix", ""):
        changed = build(exports=[first_user_text(export_text)], calls=calls, task_text="Fix it").evidence
        assert changed.status == "incomplete" and "literal_task_not_presented:root" in changed.issues
    unchecked = build(exports=[first_user_text("Anything")], calls=calls).evidence
    assert unchecked.status == "complete", unchecked.issues


def test_a_root_session_without_a_user_message_did_not_present_the_task():
    source = answered_export()
    source["messages"] = source["messages"][1:]
    evidence = build(exports=[source], calls=[observation(), tool_observation()], task_text="Fix it").evidence
    assert "literal_task_not_presented:root" in evidence.issues


def prefix_export():
    return export(
        message("p-user", "user", text("Fix it"), session="ses_prefix"),
        message("p-reply", "assistant", text("Earlier decision"), session="ses_prefix", created=2),
        session="ses_prefix",
    )


def continuation_spec():
    return RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        continuation={
            "mode": "restore",
            "prefix_export": prefix_export(),
            "workspace_diff": "",
            "source_task_sha256": "a" * 64,
            "source_tools_sha256": "b" * 64,
        },
    )


def test_a_continuation_grades_from_its_fresh_user_message_without_receipts_for_the_prefix():
    root = prefix_export()
    root["messages"] += [
        message("fresh-user", "user", text("Continue"), session="ses_prefix", created=10),
        message("fresh-reply", "assistant", text("DONE"), session="ses_prefix", created=11),
    ]
    call_record = {
        "session_id": "ses_prefix",
        "assistant_message_id": "fresh-reply",
        "execution_tools": {},
        "request": {"messages": []},
        "response": {"choices": [{"message": {"role": "assistant", "content": "DONE"}}]},
    }
    built = build(
        continuation_spec(),
        exports=[root],
        calls=[call_record],
        replay=REPLAY_RECEIPT,
        artifacts={"prefix_export": "/artifacts/prefix-export.json"},
    )
    evidence = built.evidence
    assert evidence.status == "complete", evidence.issues
    assert evidence.continuation_start == 2
    assert evidence.facts["replay"]["tool_outputs_match"] is True and evidence.facts["replay"]["prefix_messages"] == 2
    assert evidence.artifacts["prefix_export"] == "/artifacts/prefix-export.json"


def test_a_continuation_must_present_the_row_s_task_after_its_restored_prefix():
    root = prefix_export()
    root["messages"] += [
        message("fresh-user", "user", text("Continue"), session="ses_prefix", created=10),
        message("fresh-reply", "assistant", text("DONE"), session="ses_prefix", created=11),
    ]
    call_record = {
        "session_id": "ses_prefix",
        "assistant_message_id": "fresh-reply",
        "execution_tools": {},
        "request": {"messages": []},
        "response": {"choices": [{"message": {"role": "assistant", "content": "DONE"}}]},
    }

    def evidence(task_text, document=root):
        return build(
            continuation_spec(),
            exports=[document],
            calls=[call_record],
            replay=REPLAY_RECEIPT,
            artifacts={"prefix_export": "/artifacts/prefix-export.json"},
            task_text=task_text,
        ).evidence

    assert evidence("Continue").status == "complete"
    # The prefix's own task does not stand in for the fresh turn's.
    assert "literal_task_not_presented:ses_prefix" in evidence("Fix it").issues
    unanswered = deepcopy(root)
    unanswered["messages"] = unanswered["messages"][:2]
    assert "literal_task_not_presented:ses_prefix" in evidence("Continue", unanswered).issues


def test_a_continuation_without_its_root_or_replay_receipt_is_incomplete():
    assert "missing_replay_receipt" in build(continuation_spec(), exports=[prefix_export()]).evidence.issues
    no_root = build(continuation_spec(), exports=[], sessions=[], replay=REPLAY_RECEIPT).evidence
    assert "missing_continued_root_session" in no_root.issues and "missing_export" in no_root.issues


def test_a_rejected_call_keeps_the_model_s_own_attempt():
    source = export(
        message(
            "a",
            "assistant",
            call("call-1", "invalid", input={"tool": "made_up_tool", "error": "unavailable"}, output="Tool error"),
        )
    )
    observed = {
        "session_id": "root",
        "assistant_message_id": "a",
        "execution_tools": {},
        "request": {"messages": []},
        "response": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{"id": "call-1", "function": {"name": "made_up_tool", "arguments": "{}"}}],
                    }
                }
            ]
        },
    }
    evidence = build(exports=[source], calls=[observed]).evidence
    attempted = evidence.steps[0].calls[0]
    assert (attempted.model_tool, attempted.model_args, attempted.rejected) == ("made_up_tool", {}, True)
