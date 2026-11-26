# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A prefix's rejected calls recorded without a receipt keep OpenCode's own form, only when the row lists them."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from resources_servers.agentic_if.episode.opencode_hooks import enrich_exports
from resources_servers.agentic_if.opencode_bundle import UnsupportedFeatureError, bundle_files
from resources_servers.agentic_if.opencode_bundle.boundary import compile_boundary
from resources_servers.agentic_if.opencode_bundle.profile import compile_profile
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec

from .harness import call, needs_node, run_boundary, upstream


TOOLS = ["bash", "edit", "read", "write"]
# OpenCode's recovery of a call to a tool the model lacks: its `invalid` tool, with this input and output.
RECOVERY = {"tool": "Missing", "error": "Model tried to call unavailable tool 'Missing'."}
RECOVERY_OUTPUT = "The arguments provided to the tool are invalid: Model tried to call unavailable tool 'Missing'."
OK = upstream(
    json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()
)


def prefix(*, receipted: bool = False) -> dict:
    """A recorded prefix: a shell call, then a call to a tool the model lacked, answered by OpenCode's `invalid` tool."""
    invalid = {
        "status": "completed",
        "input": RECOVERY,
        "output": RECOVERY_OUTPUT,
        "metadata": {"nemo_gym_profile": {"profile_sha256": "p" * 64}},
    }
    if receipted:
        invalid["metadata"]["nemo_gym_model_call"] = {
            "function": {"name": "Missing", "arguments": "{}"},
            "available_names": TOOLS,
        }
    return {
        "info": {"id": "ses_source", "directory": "/app"},
        "messages": [
            {"info": {"id": "msg_u", "role": "user"}, "parts": [{"type": "text", "text": "Fix it"}]},
            {
                "info": {"id": "msg_a", "role": "assistant"},
                "parts": [
                    {
                        "type": "tool",
                        "tool": "bash",
                        "callID": "call_ls",
                        "state": {"status": "completed", "input": {"command": "ls"}, "output": "a.py\n"},
                    },
                    {"type": "tool", "tool": "invalid", "callID": "call_bad", "state": invalid},
                ],
            },
        ],
    }


def spec(*, tools: dict | None = None, **continuation: object) -> RuntimeSpec:
    """A continuation row that selects TOOLS, unless ``tools`` says otherwise."""
    return RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        tools={"tool_names": {tool: tool for tool in TOOLS}} if tools is None else tools,
        continuation={
            "mode": "restore",
            "prefix_export": prefix(),
            "workspace_diff": "",
            "source_task_sha256": "a" * 64,
            "source_tools_sha256": "b" * 64,
            **continuation,
        },
    )


@pytest.mark.parametrize(
    "listed, receipted",
    [
        ([], False),
        (["call_bad", "call_bad"], False),
        (["call_ls"], False),
        (["call_gone"], False),
        (["call_bad"], True),
    ],
    ids=["empty", "duplicate", "not-rejected", "unknown", "receipted"],
)
def test_only_unreceipted_rejected_calls_of_the_prefix_can_be_listed(listed: list[str], receipted: bool) -> None:
    with pytest.raises(ValidationError, match="unreceipted_rejected_calls must list distinct call IDs"):
        spec(prefix_export=prefix(receipted=receipted), unreceipted_rejected_calls=listed)


def test_compiling_keeps_listed_calls_out_of_the_receipts_and_names_them_last() -> None:
    boundary = compile_boundary(spec(unreceipted_rejected_calls=["call_bad"]))
    assert boundary["historical_rejected_calls"] == {}
    assert list(boundary) == [
        *compile_boundary(spec(prefix_export=prefix(receipted=True))),
        "unreceipted_rejected_calls",
    ]
    assert boundary["unreceipted_rejected_calls"] == ["call_bad"]
    # A row without a tool selection leaves OpenCode's history as it is anyway, so it gets no key.
    assert "unreceipted_rejected_calls" not in compile_boundary(
        spec(tools={}, unreceipted_rejected_calls=["call_bad"])
    )


def test_an_unlisted_unreceipted_call_is_still_refused() -> None:
    export = prefix()
    second = deepcopy(export["messages"][1]["parts"][1])
    second["callID"] = "call_other"
    export["messages"][1]["parts"].append(second)
    with pytest.raises(ValueError, match="rejected prefix requires captured call evidence"):
        compile_boundary(spec(prefix_export=export, unreceipted_rejected_calls=["call_bad"]))


def test_a_row_that_renames_tools_cannot_list_them() -> None:
    # OpenCode's error for such a call lists the built-in tools.
    renamed = {"tool_names": {"bash": "shell", "read": "read"}, "tool_descriptions": {"bash": "Run.", "read": "Read."}}
    for compile_ in (compile_profile, compile_boundary, bundle_files):
        with pytest.raises(UnsupportedFeatureError, match="unreceipted_rejected_calls"):
            compile_(spec(tools=renamed, unreceipted_rejected_calls=["call_bad"]))


def opencode_request() -> dict:
    """The continuation's model request as OpenCode builds it from the imported prefix: the call in its own form."""
    return {
        "messages": [
            {"role": "user", "content": "Fix it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "invalid", "arguments": json.dumps(RECOVERY)},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_bad", "content": RECOVERY_OUTPUT},
            {"role": "user", "content": "Continue"},
        ],
        "tools": [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}} for name in TOOLS
        ],
    }


@needs_node
def test_the_boundary_sends_a_listed_call_as_opencode_recorded_it_in_the_root_session_only(tmp_path: Path) -> None:
    boundary = compile_boundary(spec(unreceipted_rejected_calls=["call_bad"]))
    calls = [
        call(opencode_request(), OK, session="new-root", assistant="a1"),
        call(opencode_request(), OK, session="child", assistant="c1"),
    ]
    run = run_boundary(tmp_path, boundary=boundary, calls=calls)
    assert json.loads(run.calls[0]["sent"]) == opencode_request()
    assert run.calls[1]["sent"] is None
    assert "prefix requires captured call evidence" in run.calls[1]["error"]["message"]


@needs_node
@pytest.mark.parametrize("listed", [["call_bad", "call_bad"], [""], "call_bad"], ids=["duplicate", "empty", "string"])
def test_the_boundary_refuses_to_load_a_malformed_list(tmp_path: Path, listed: object) -> None:
    boundary = compile_boundary(spec(unreceipted_rejected_calls=["call_bad"])) | {"unreceipted_rejected_calls": listed}
    run = run_boundary(tmp_path, boundary=boundary, calls=[call(opencode_request(), OK, session="new-root")])
    assert run.rows[0]["kind"] == "load_error"
    assert "invalid boundary.json: unreceipted_rejected_calls" in run.rows[0]["message"]
    assert run.calls[0]["sent"] is None


def restored(session: str, parent: str | None = None) -> dict:
    """The session OpenCode exports after importing the prefix: the recorded parts, under the continued session."""
    return {"info": {"id": session, "parentID": parent}, "messages": deepcopy(prefix()["messages"])}


def test_evidence_keeps_a_listed_call_as_recorded_and_invents_no_receipt() -> None:
    raw = {"new-root": restored("new-root")}
    enriched = enrich_exports(raw, [], spec=spec(unreceipted_rejected_calls=["call_bad"]), observations=[])
    assert enriched == raw
    # Only the root continues the prefix; a child with the same call ID has no recorded call to keep.
    with pytest.raises(ValueError, match="unobserved rejected model call"):
        enrich_exports(
            {"new-root": restored("new-root"), "child": restored("child", parent="new-root")},
            [],
            spec=spec(unreceipted_rejected_calls=["call_bad"]),
            observations=[],
        )
