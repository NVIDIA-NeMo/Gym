# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode exports must retain the observations needed to grade a real session."""

from copy import deepcopy

import pytest

from resources_servers.agentic_if.episode.opencode_evidence import opencode_evidence
from resources_servers.agentic_if.if_grading.contracts import SessionInfo
from resources_servers.agentic_if.tests.support import call, export, message, text


def untimed_child():
    child = export(message("c", "assistant", text("Child"), session="child"), session="child", parent="root")
    del child["messages"][0]["info"]["time"]
    return child


def test_multiple_calls_stay_in_one_message_and_keep_matching_results():
    source = export(
        message("u", "user", text("Fix it")),
        message(
            "a",
            "assistant",
            text("Inspect both"),
            call("one"),
            call("two", output="second"),
            created=2,
        ),
    )
    before = deepcopy(source)
    evidence = opencode_evidence(source)
    assert evidence.status == "complete"
    assert [s.role for s in evidence.steps] == ["user", "assistant", "tool", "tool"]
    assert [c.call_id for c in evidence.steps[1].calls] == ["one", "two"]
    assert [(s.call_id, s.output) for s in evidence.steps[2:]] == [("one", "contents"), ("two", "second")]
    assert evidence.steps[1].calls[0].args["path"] == "/app/a.py"
    assert source == before


def test_rejected_calls_keep_actor_interface_and_opencode_execution_facts():
    source = export(
        message(
            "a",
            "assistant",
            call(
                "bad",
                status="error",
                output=None,
                input={
                    "filePath": "/app/a.py",
                    "__nemo_gym_profile": {
                        "configured_args": {"path": "/app/a.py", "extra": True},
                        "names": {"filePath": "path"},
                    },
                },
                error="The read tool was called with invalid arguments: filePath",
            ),
        )
    )
    evidence = opencode_evidence(source)
    actual = evidence.steps[0].calls[0]
    assert actual.model_tool == "read"
    assert actual.model_args == {"path": "/app/a.py", "extra": True}
    assert actual.args["path"] == "/app/a.py"
    assert actual.rejected
    assert evidence.steps[1].output.endswith("path")
    assert evidence.steps[1].error


def test_installed_interface_normalization_keeps_nested_args_fixed_values_and_false():
    args = {"location": {"file": "/app/x"}, "oldText": "old", "newText": "new", "all": False}
    source = export(message("a", "assistant", call("edit", "edit", input=args)))
    evidence = opencode_evidence(
        source,
        interfaces={
            "edit": {
                "args_map": {
                    "filePath": "location.file",
                    "oldString": "oldText",
                    "newString": "newText",
                    "replaceAll": "all",
                },
                "fixed": {"unrelated": 0},
            }
        },
    )
    actual = evidence.steps[0].calls[0]
    assert actual.model_args == args
    assert actual.args["path"] == "/app/x"
    assert actual.args["old_string"] == "old"
    assert actual.args["content"] == "new"
    assert actual.args["replace_all"] is False
    assert actual.args["unrelated"] == 0


def test_private_reasoning_is_not_scored_and_input_text_is_not_rewritten():
    source = export(
        message("u", "user", text("Literal <think>example</think>")),
        message(
            "a",
            "assistant",
            {"type": "reasoning", "text": "private scratch"},
            text("<think>private embedded</think>Visible"),
            {"type": "text", "text": "synthetic", "synthetic": True},
            {"type": "text", "text": "ignored", "ignored": True},
        ),
    )
    evidence = opencode_evidence(source)
    assert evidence.steps[0].text == "Literal <think>example</think>"
    assert evidence.steps[1].text == "Visible"
    assert "private" not in evidence.model_dump_json()


def test_children_keep_parentage_and_colliding_ids_without_corrupting_order():
    root = export(
        message("a", "assistant", text("Root"), call("same", end=5), created=1),
        message("z", "assistant", text("Finished"), created=7),
    )
    child = export(
        message(
            "a", "assistant", text("Child"), call("same", output="child result", end=4), session="child", created=2
        ),
        session="child",
        parent="root",
    )
    evidence = opencode_evidence(root, child_exports=[child])
    assert evidence.status == "complete"
    assert [(s.id, s.parent_id) for s in evidence.sessions] == [("root", None), ("child", "root")]
    assert [(s.session_id, s.role) for s in evidence.steps] == [
        ("root", "assistant"),
        ("child", "assistant"),
        ("child", "tool"),
        ("root", "tool"),
        ("root", "assistant"),
    ]
    assert [(s.session_id, s.call_id, s.output) for s in evidence.steps if s.role == "tool"] == [
        ("child", "same", "child result"),
        ("root", "same", "contents"),
    ]


@pytest.mark.parametrize(
    "source, options, issue",
    [
        (None, {}, "missing_export"),
        (export(), {}, "missing_messages:root"),
        (export(message("a", "assistant", call("x", status="running"))), {}, "unfinished_tool:root:a:x"),
        (export(message("a", "assistant", call("x", output=None))), {}, "missing_tool_result:root:a:x"),
        (
            export(message("a", "assistant", text("done"))),
            {"expected_sessions": [SessionInfo(id="root"), SessionInfo(id="missing", parent_id="root")]},
            "missing_session:missing",
        ),
        (
            export(message("a", "assistant", text("Root"))),
            {"child_exports": [untimed_child()]},
            "missing_cross_session_time:child:c",
        ),
    ],
    ids=["export", "messages", "unfinished_tool", "tool_result", "child_export", "cross_session_time"],
)
def test_missing_observations_are_explicit_evidence_gaps(source, options, issue):
    evidence = opencode_evidence(source, **options)
    assert evidence.status == "incomplete"
    assert issue in evidence.issues


def test_missing_call_identity_is_not_replaced_with_a_random_id():
    part = call("x")
    del part["callID"]
    evidence = opencode_evidence(export(message("a", "assistant", part)))
    assert evidence.status == "incomplete"
    assert "missing_call_id:root:a" in evidence.issues
    assert evidence.steps[0].calls == []


def test_duplicate_message_identity_is_reported_without_crashing_other_evidence():
    evidence = opencode_evidence(
        export(message("a", "assistant", text("First")), message("a", "assistant", text("Second")))
    )
    assert evidence.status == "incomplete"
    assert "duplicate_message:root:a" in evidence.issues
    assert [s.text for s in evidence.steps] == ["First"]


def test_opencode_evidence_keeps_the_root_export_workdir():
    root = export(message("u", "user", text("hello")))
    root["info"]["directory"] = "/task/root"
    assert opencode_evidence(root).facts["workdir"] == "/task/root"


def test_workspace_receipt_stays_attached_to_its_tool_result():
    source = export(
        message(
            "a",
            "assistant",
            call(
                "x",
                metadata={
                    "nemo_gym_profile": {
                        "configured_args": {"filePath": "/app/a.py"},
                        "workspace_diff": "diff --git a/x b/x",
                    },
                },
            ),
        )
    )
    evidence = opencode_evidence(source)
    assert evidence.steps[1].workspace_diff == "diff --git a/x b/x"
