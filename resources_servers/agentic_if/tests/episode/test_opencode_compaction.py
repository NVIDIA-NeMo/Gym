# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode's context management is not an actor reply or missing session history."""

from copy import deepcopy

import pytest

from resources_servers.agentic_if.episode.opencode_context import actor_export, compaction_context
from resources_servers.agentic_if.episode.opencode_evidence import opencode_evidence
from resources_servers.agentic_if.episode.opencode_export import export_output
from resources_servers.agentic_if.episode.opencode_observations import bind_observations, project_output
from resources_servers.agentic_if.if_grading.checkers.calls import check_after_event
from resources_servers.agentic_if.if_grading.checkers.run import evidence_steps
from resources_servers.agentic_if.tests.support import (
    call,
    message,
    observation,
    session_export,
    text,
    tool_observation,
)


def auto_continue(source):
    followup = message(
        "auto",
        "user",
        {
            "type": "text",
            "text": "Internal continuation",
            "synthetic": True,
            "metadata": {"compaction_continue": True},
        },
        created=7,
    )
    source["messages"].insert(-1, followup)
    return source


@pytest.mark.parametrize("genuine", [False, True])
def test_only_a_genuine_user_continuation_lifts_an_event_scoped_stop(genuine):
    source = compacted()
    if genuine:
        source["messages"].insert(-1, message("human", "user", text("Actually, continue."), created=7))
    else:
        auto_continue(source)
    source["messages"][-1]["parts"].append(call("after", "bash", input={"command": "pwd"}, end=9))
    before = deepcopy(source)
    evidence = opencode_evidence(source)
    assert evidence.status == "complete"
    stop = {"then": {"calls_avoid": {"tool": "any"}}}
    assert check_after_event(evidence_steps(evidence), stop, {}, {}, start=2)[0][0] is genuine
    assert [s.id for s in evidence.steps if s.role == "user"] == (["u", "human"] if genuine else ["u"])
    assert "auto" not in [m["info"]["id"] for m in actor_export(source)["messages"]]
    assert source == before


@pytest.mark.parametrize("defect", ["manual", "not_synthetic", "mixed_text", "not_adjacent"])
def test_real_or_unproven_user_messages_are_not_removed_by_text_similarity(defect):
    source = auto_continue(compacted())
    followup = source["messages"][-2]
    if defect == "manual":
        followup["parts"][0].pop("metadata")
        followup["parts"][0].pop("synthetic")
    elif defect == "not_synthetic":
        followup["parts"][0].pop("synthetic")
    elif defect == "mixed_text":
        followup["parts"].append(text("Actually, keep working."))
    else:
        source["messages"].insert(-2, message("human", "user", text("OK, proceed."), created=7))
    internal, _, _ = compaction_context(source)
    assert "auto" not in internal


def overflow_replay():
    source = compacted()
    source["messages"].insert(
        2,
        message(
            "instruction",
            "user",
            {
                "type": "text",
                "text": "Stop using tools.",
                "time": {"start": 4, "end": 4},
                "metadata": {"origin": "user"},
                "id": "original-part",
                "messageID": "instruction",
                "sessionID": "root",
            },
            created=4,
        ),
    )
    source["messages"][-3]["parts"][0]["overflow"] = True
    followup = deepcopy(source["messages"][2])
    followup["info"].update(id="auto", time={"created": 7})
    followup["parts"][0].update(id="copy-part", messageID="auto")
    source["messages"].insert(-1, followup)
    return source


def test_controlled_overflow_copy_is_not_a_new_user_instruction():
    source = overflow_replay()
    before = deepcopy(source)
    internal, records, issues = compaction_context(source, controlled_ids=frozenset({"summary", "auto"}))
    assert not issues
    assert "auto" in internal and "instruction" not in internal
    assert records[0]["followup_id"] == "auto"
    assert records[0]["replayed_user_id"] == "instruction"
    assert source == before


@pytest.mark.parametrize(
    "constraint",
    [{"tool": "any"}, {"tool": "bash", "arg": {"field": "command", "regex": "^pwd$"}}],
    ids=["any_call", "that_call"],
)
def test_overflow_replay_preserves_active_tool_constraint(constraint):
    source = overflow_replay()
    source["messages"][-1]["parts"].append(call("after", "bash", input={"command": "pwd"}, end=9))
    evidence = opencode_evidence(source, controlled_messages=frozenset({("root", "summary"), ("root", "auto")}))
    assert evidence.status == "complete"
    avoid = {"then": {"calls_avoid": constraint}}
    assert check_after_event(evidence_steps(evidence), avoid, {}, {}, start=3)[0][0] is False


def test_malformed_followup_remains_an_evidence_gap_not_a_parser_crash():
    source = overflow_replay()
    source["messages"][-2] = "bad record"
    result = opencode_evidence(source)
    assert result.status == "incomplete"


@pytest.mark.parametrize("defect", ["uncontrolled", "time", "metadata", "session", "duplicate", "missing_history"])
def test_ambiguous_overflow_copy_stays_visible_and_is_an_evidence_gap(defect):
    source = overflow_replay()
    if defect == "time":
        source["messages"][-2]["parts"][0]["time"] = {"start": 7, "end": 7}
    elif defect == "metadata":
        source["messages"][-2]["parts"][0]["metadata"] = {"origin": "new-user"}
    elif defect == "session":
        source["messages"][-2]["info"]["sessionID"] = "other"
    elif defect == "duplicate":
        source["messages"].insert(-1, deepcopy(source["messages"][-2]))
        source["messages"][-2]["info"]["id"] = "second-copy"
    elif defect == "missing_history":
        source["messages"] = source["messages"][2:]
    internal, _, issues = compaction_context(
        source, controlled_ids=frozenset() if defect == "uncontrolled" else frozenset({"summary", "auto"})
    )
    assert "auto" not in internal
    assert issues


def compacted():
    source = session_export()
    marker = message("marker", "user", {"type": "compaction", "auto": True}, created=5)
    summary = message("summary", "assistant", text("Internal summary, not the final answer."), created=6)
    summary["info"].update(summary=True, agent="compaction", parentID="marker", finish="stop")
    summary["info"]["time"]["completed"] = 7
    source["messages"].extend([marker, summary])
    source["messages"].append(message("final", "assistant", text("Done."), created=8))
    return source


def summary_observation():
    result = observation(identity="summary")
    result.update(source="compaction", execution_tools={})
    result["request"] = {"messages": [{"role": "user", "content": "Summarize"}], "tools": []}
    result["response"]["choices"][0]["message"] = {
        "role": "assistant",
        "content": "Internal summary, not the final answer.",
    }
    return result


def final_observation():
    result = observation(identity="final")
    result["request"]["messages"] += tool_observation()["request"]["messages"]
    result["response"]["choices"][0]["message"] = {"role": "assistant", "content": "Done."}
    return result


def test_opencode_compaction_summary_is_not_scored_as_an_actor_reply_or_lost_history():
    source = compacted()
    before = deepcopy(source)
    evidence = opencode_evidence(source)
    assert evidence.status == "complete"
    assert [s.text for s in evidence.steps if s.role == "assistant"] == ["Inspect", "Done."]
    assert [s.id for s in evidence.steps if s.role == "user"] == ["u"]
    assert evidence.facts["opencode_compactions"][0]["message_id"] == "summary"
    assert source == before


def test_summary_observation_is_joined_separately_from_actor_observations():
    result = bind_observations(
        opencode_evidence(compacted()), [observation(), summary_observation(), final_observation()]
    )
    assert result.status == "complete"
    assert not result.issues
    assert "Internal summary" not in " ".join(s.text for s in result.steps)


@pytest.mark.parametrize("defect", ["missing", "wrong_text", "wrong_session", "actor_kind", "tool_call"])
def test_unproven_internal_summary_remains_an_evidence_gap(defect):
    observed = summary_observation()
    if defect == "wrong_text":
        observed["response"]["choices"][0]["message"]["content"] = "Different summary"
    elif defect == "wrong_session":
        observed["session_id"] = "other"
    elif defect == "actor_kind":
        observed["source"] = "model"
    elif defect == "tool_call":
        observed["response"]["choices"][0]["message"]["tool_calls"] = observation()["response"]["choices"][0][
            "message"
        ]["tool_calls"]
    observations = [observation(), final_observation()] + ([] if defect == "missing" else [observed])
    result = bind_observations(opencode_evidence(compacted()), observations)
    assert result.status == "incomplete"
    assert result.issues


@pytest.mark.parametrize("defect", ["not_compaction", "missing_parent", "missing_history", "unfinished", "has_tools"])
def test_summary_flag_alone_never_makes_an_export_complete(defect):
    source = compacted()
    summary = source["messages"][-2]
    if defect == "not_compaction":
        summary["info"]["agent"] = "build"
    elif defect == "missing_parent":
        summary["info"]["parentID"] = "absent"
    elif defect == "missing_history":
        source["messages"] = source["messages"][2:]
    elif defect == "unfinished":
        summary["info"].pop("finish")
    elif defect == "has_tools":
        summary["parts"].append(deepcopy(session_export()["messages"][1]["parts"][1]))
    result = opencode_evidence(source)
    assert result.status == "incomplete"
    assert result.issues


def test_frozen_prefix_summary_needs_no_new_inference_receipt():
    source = compacted()
    prefix = frozenset(("root", m["info"]["id"]) for m in source["messages"][:-1])
    result = bind_observations(opencode_evidence(source), [final_observation()], prefix_messages=prefix)
    assert result.status == "complete"


def test_internal_summary_does_not_appear_in_actor_output_projection():
    source = compacted()
    before = deepcopy(source)
    output, issues = project_output(
        export_output(actor_export(source)),
        [observation(), summary_observation(), final_observation()],
        session_id="root",
    )
    assert not issues
    assert all("Internal summary" not in item.model_dump_json() for item in output)
    assert source == before


def test_compactors_truncated_history_is_not_an_actor_tool_observation():
    source = compacted()
    internal = summary_observation()
    internal["request"]["messages"] += tool_observation(value="Truncated for internal summarization")["request"][
        "messages"
    ]
    output, issues = project_output(
        export_output(actor_export(source)),
        [observation(), internal, final_observation()],
        session_id="root",
        compaction_messages=frozenset({("root", "summary")}),
    )
    assert not issues
    assert next(item.output for item in output if item.type == "function_call_output") == "/app"


def test_verified_compactor_can_receipt_the_exact_last_tool_result():
    source = compacted()
    internal = summary_observation()
    internal["request"]["messages"] += tool_observation()["request"]["messages"]
    final = final_observation()
    final["request"]["messages"] = []
    observations = [observation(), internal, final]
    evidence = bind_observations(opencode_evidence(source), observations)
    assert evidence.facts["observed_opencode_compactions"] == [{"session_id": "root", "message_id": "summary"}]
    output, issues = project_output(
        export_output(actor_export(source)),
        observations,
        session_id="root",
        compaction_messages=frozenset({("root", "summary")}),
    )
    assert not issues
    assert next(item.output for item in output if item.type == "function_call_output") == "/app"


def test_compactor_accepts_exact_receipted_error_presentation_not_raw_opencode_wording():
    from resources_servers.agentic_if.episode.opencode_hooks import merge_hook_receipts

    source = compacted()
    state = source["messages"][1]["parts"][1]["state"]
    state.pop("output")
    state.update(status="error", error="The bash tool was called with invalid arguments: required")
    presented = "RunCommand failed: Invalid arguments: required"
    enriched = merge_hook_receipts(
        {"root": source},
        [
            {
                "kind": "tool_presentation",
                "session_id": "root",
                "call_id": "c1",
                "opencode_tool": "bash",
                "opencode_error": state["error"],
                "model_error": presented,
            }
        ],
    )["root"]
    internal = summary_observation()
    internal["request"]["messages"] += tool_observation(value=presented)["request"]["messages"]
    final = final_observation()
    final["request"]["messages"] = []
    output, issues = project_output(
        export_output(actor_export(enriched)),
        [observation(), internal, final],
        session_id="root",
        compaction_messages=frozenset({("root", "summary")}),
    )
    assert not issues
    assert next(item.output for item in output if item.type == "function_call_output") == presented
    assert source["messages"][1]["parts"][1]["state"]["error"] == state["error"]


@pytest.mark.parametrize(
    ("full", "prefix", "omitted"),
    [("a" * 2001, "a" * 2000, 1), ("a" * 1999 + "😀z", "a" * 1999 + "\ud83d", 2)],
    ids=["ascii", "split-surrogate"],
)
@pytest.mark.parametrize("actor_receipt", [False, True])
def test_exact_stock_compaction_truncation_preserves_only_the_captured_view(full, prefix, omitted, actor_receipt):
    source = compacted()
    source["messages"][1]["parts"][1]["state"]["output"] = full
    before = deepcopy(source)
    captured = prefix + f"\n[Tool output truncated for compaction: omitted {omitted} chars]"
    internal = summary_observation()
    internal["request"]["messages"] += tool_observation(value=captured)["request"]["messages"]
    final = final_observation()
    # The actor's own receipt of the full result arrives in its next request.
    final["request"]["messages"] = tool_observation(value=full)["request"]["messages"] if actor_receipt else []
    observations = [observation(), internal, final]
    output, issues = project_output(
        export_output(actor_export(source)),
        observations,
        session_id="root",
        compaction_messages=frozenset({("root", "summary")}),
    )
    unsupported = "\ud83d" in captured and not actor_receipt
    assert bool(issues) is unsupported
    if unsupported:
        assert "missing_tool_observation:c1" in issues
    result = next(item for item in output if item.type == "function_call_output")
    result.model_dump_json()  # The collected receipt must cross the actual response boundary.
    assert result.output == (full if actor_receipt or unsupported else captured)
    assert source == before


@pytest.mark.parametrize(
    "defect", ["prefix", "count", "suffix", "changed_text", "unvalidated", "session", "no_response"]
)
def test_near_match_changed_or_unproven_compactor_text_does_not_fill_observation_gap(defect):
    source = compacted()
    source["messages"][1]["parts"][1]["state"]["output"] = "a" * 2001
    captured = "a" * 2000 + "\n[Tool output truncated for compaction: omitted 1 chars]"
    if defect == "prefix":
        captured = "b" + captured[1:]
    elif defect == "count":
        captured = captured.replace("omitted 1", "omitted 2")
    elif defect == "suffix":
        captured += " "
    elif defect == "changed_text":
        captured = "Truncated internally"
    internal = summary_observation()
    internal["request"]["messages"] += tool_observation(value=captured)["request"]["messages"]
    if defect == "session":
        internal["session_id"] = "other"
    elif defect == "no_response":
        internal.pop("response")
    final = final_observation()
    final["request"]["messages"] = []
    _, issues = project_output(
        export_output(actor_export(source)),
        [observation(), internal, final],
        session_id="root",
        compaction_messages=frozenset() if defect == "unvalidated" else frozenset({("root", "summary")}),
    )
    assert "missing_tool_observation:c1" in issues
