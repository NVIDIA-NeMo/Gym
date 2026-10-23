# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join real model observations to OpenCode exports by session/message/call identity."""

from copy import deepcopy

import pytest

from resources_servers.agentic_if.episode.opencode_evidence import opencode_evidence
from resources_servers.agentic_if.episode.opencode_export import export_output
from resources_servers.agentic_if.episode.opencode_observations import bind_observations, project_output
from resources_servers.agentic_if.tests.support import (
    call,
    export,
    message,
    observation,
    session_export,
    text,
    tool_observation,
)


def test_names_and_arguments_come_from_actual_model_call_without_mutating_the_raw_export():
    source = finished()
    before = deepcopy(source)
    evidence = bind_observations(opencode_evidence(source), [observation(), tool_observation()])
    assert evidence.status == "complete"
    actual = next(step for step in evidence.steps if step.role == "assistant").calls[0]
    assert actual.tool == "bash"
    assert actual.model_tool == "RunCommand"
    assert actual.model_args == {"command": "pwd"}
    assert next(step for step in evidence.steps if step.role == "tool").tool == "bash"
    output, issues = project_output(export_output(source), [observation(), tool_observation()], session_id="root")
    assert not issues
    actual_output = next(item for item in output if item.type == "function_call")
    assert actual_output.name == "RunCommand"
    assert actual_output.arguments == '{ "command": "pwd" }'
    assert source == before


def finished(session="root", parent=None):
    """The OpenCode session through the final reply that follows the tool result."""
    source = session_export(session, parent)
    source["messages"].append(message("z", "assistant", text("Done"), session=session, created=5))
    return source


HTTP_REJECTION = {"phase": "http_rejected", "status": 429}
# A model server that answered 200, then sent an error event before any model chunk (a rate-limited stream).
STREAM_REJECTION = {"phase": "stream_error", "status": 429, "message": "rate limit exceeded", "chunks": 0}


def rejected_observation(error=HTTP_REJECTION, **kwargs):
    result = observation(**kwargs)
    result.pop("response")
    result["transport_error"] = deepcopy(error)
    return result


@pytest.mark.parametrize("error", [HTTP_REJECTION, STREAM_REJECTION], ids=["http", "stream"])
def test_recovered_rejections_do_not_duplicate_the_exported_assistant_turn(error):
    source = finished()
    observed = [rejected_observation(error), rejected_observation(error), observation(), tool_observation()]
    before = deepcopy(observed)
    evidence = bind_observations(opencode_evidence(source), observed)
    assert evidence.status == "complete"
    assert not evidence.issues
    output, issues = project_output(export_output(source), observed, session_id="root")
    assert not issues
    assert next(item for item in output if item.type == "function_call").name == "RunCommand"
    assert observed == before  # raw failed attempts remain available for audit


@pytest.mark.parametrize(
    "kind",
    ["unmarked", "partial", "stream_after_chunks", "late", "other_message", "other_session", "duplicate"],
)
def test_only_a_proven_rejection_recovered_by_the_same_later_turn_is_ignored(kind):
    rejected = rejected_observation()
    observed = [rejected, observation(), tool_observation()]
    if kind == "unmarked":
        rejected.pop("transport_error")
    elif kind == "partial":
        rejected["transport_error"]["phase"] = "stream_interrupted"
    elif kind == "stream_after_chunks":
        # OpenCode already received part of this reply, so its retry is not a clean redo.
        rejected["transport_error"] = STREAM_REJECTION | {"chunks": 3}
    elif kind == "late":
        observed = [observation(), rejected, tool_observation()]
    elif kind == "other_message":
        rejected["assistant_message_id"] = "different"
    elif kind == "other_session":
        rejected["session_id"] = "child"
    else:
        rejected["response"] = deepcopy(observation()["response"])
    evidence = bind_observations(opencode_evidence(finished()), observed)
    assert evidence.status == "incomplete"
    assert evidence.issues
    if kind != "other_session":
        _, issues = project_output(export_output(finished()), observed, session_id="root")
        assert issues


def test_an_unrecovered_http_rejection_is_not_a_complete_model_turn():
    observed = [rejected_observation(), tool_observation()]
    evidence = bind_observations(opencode_evidence(finished()), observed)
    assert evidence.status == "incomplete"
    assert any("inconsistent_model_observation" in issue for issue in evidence.issues)
    _, issues = project_output(export_output(finished()), observed, session_id="root")
    assert "invalid_model_output_capture" in issues


def test_training_output_uses_observed_tool_error_not_the_error_opencode_stored():
    source = finished()
    source["messages"][1]["parts"][1]["state"].update(
        status="error", error="OpenCode bash permission details", output=None
    )
    output, issues = project_output(
        export_output(source), [observation(), tool_observation(value="RunCommand failed: denied")], session_id="root"
    )
    assert not issues
    assert next(item for item in output if item.type == "function_call_output").output == "RunCommand failed: denied"
    assert source["messages"][1]["parts"][1]["state"]["error"] == "OpenCode bash permission details"


@pytest.mark.parametrize(
    "change", ["missing_result", "changed_result", "changed_text", "wrong_session", "duplicate_call"]
)
def test_training_capture_gaps_are_explicit(change):
    observed = [observation(), tool_observation()]
    source = finished()
    if change == "missing_result":
        observed.pop()
    elif change == "changed_result":
        observed.append(tool_observation(value="compacted or changed"))
    elif change == "changed_text":
        source["messages"][1]["parts"][0]["text"] = "Not what the model said"
    elif change == "wrong_session":
        observed[1]["session_id"] = "child"
    else:
        observed.append(observation(identity="b"))
    _, issues = project_output(export_output(source), observed, session_id="root")
    assert issues


def test_training_projection_preserves_reasoning_and_scopes_child_calls():
    source = finished()
    source["messages"][1]["parts"].insert(0, {"type": "reasoning", "text": "private reasoning"})
    parent = observation()
    parent["response"]["choices"][0]["message"]["reasoning_content"] = "private reasoning"
    observed = [
        parent,
        tool_observation(),
        observation("child", tool="ChildShell"),
        tool_observation("child", "child result"),
    ]
    output, issues = project_output(export_output(source), observed, session_id="root")
    assert not issues
    assert next(item for item in output if item.type == "reasoning").summary[0].text == "private reasoning"
    assert next(item for item in output if item.type == "function_call_output").output == "/app"
    assert next(item for item in output if item.type == "function_call").name == "RunCommand"


def test_system_evidence_is_inserted_at_observation_time_not_backdated_into_replayed_prefix():
    source = export(
        message("prefix", "assistant", text("Earlier answer")),
        message("fresh", "user", text("Now explain commands")),
        message("a", "assistant", text("Inspect"), call("c1", "bash", input={"command": "pwd"})),
    )
    evidence = bind_observations(
        opencode_evidence(source), [observation()], prefix_messages=frozenset({("root", "prefix")})
    )
    assert evidence.status == "complete"
    assert [(step.role, step.text) for step in evidence.steps[:4]] == [
        ("assistant", "Earlier answer"),
        ("user", "Now explain commands"),
        ("system", "Explain each command."),
        ("assistant", "Inspect"),
    ]


def test_child_call_ids_are_correlated_with_their_session_not_a_global_call_id_map():
    evidence = opencode_evidence(session_export(), child_exports=[session_export("child", "root")])
    observations = [observation(), observation("child", tool="ChildShell")]
    result = bind_observations(evidence, observations)
    assert result.status == "complete"
    assert {step.session_id: step.calls[0].model_tool for step in result.steps if step.calls} == {
        "root": "RunCommand",
        "child": "ChildShell",
    }


@pytest.mark.parametrize("change", ["missing", "wrong_call", "wrong_message", "missing_response"])
def test_missing_or_uncorrelated_model_observations_cannot_be_graded_as_complete(change):
    observed = observation()
    if change == "wrong_call":
        observed["response"]["choices"][0]["message"]["tool_calls"][0]["id"] = "different"
    elif change == "wrong_message":
        observed["assistant_message_id"] = "different"
    elif change == "missing_response":
        observed.pop("response")
    result = bind_observations(opencode_evidence(session_export()), [] if change == "missing" else [observed])
    assert result.status == "incomplete"
    assert result.issues


def test_malformed_model_arguments_remain_visible_on_a_rejected_call():
    source = export(message("a", "assistant", call("c1", "bash", status="error", input={}, output="Invalid JSON")))
    observed = observation(arguments="{invalid")
    result = bind_observations(opencode_evidence(source), [observed])
    assert result.status == "complete"
    assert next(step for step in result.steps if step.calls).calls[0].model_args == "{invalid"
