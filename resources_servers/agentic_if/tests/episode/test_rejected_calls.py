# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rejected-call evidence retains what the model actually sent, not OpenCode's invalid-tool repair input."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from resources_servers.agentic_if.episode.opencode_hooks import enrich_exports
from resources_servers.agentic_if.episode.opencode_observations import preserve_rejected_calls
from resources_servers.agentic_if.opencode_bundle.boundary import prefix_model_calls
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


def rejected_call_export(session="source", *, parent=None):
    """A session whose model called a tool it does not have; OpenCode ran its `invalid` tool instead."""
    return {
        "info": {"id": session, "parentID": parent},
        "messages": [
            {"info": {"id": "u", "role": "user"}, "parts": [{"type": "text", "text": "Fix it"}]},
            {
                "info": {"id": "a", "role": "assistant"},
                "parts": [
                    {
                        "type": "tool",
                        "tool": "invalid",
                        "callID": "rejected",
                        "state": {
                            "status": "completed",
                            "input": {"tool": "Missing", "error": "Available tools: read"},
                            "output": "Available tools: read",
                            "metadata": {"nemo_gym_profile": {"workspace_diff": ""}},
                        },
                    }
                ],
            },
        ],
    }


def captured(arguments='{ "path": "source" }'):
    """The boundary's record of the call: the model's own name and raw arguments, and the tools it was offered."""
    return {
        "source": "model",
        "session_id": "source",
        "assistant_message_id": "a",
        "execution_tools": {},
        "request": {"messages": [], "tools": [{"type": "function", "function": {"name": "read"}}]},
        "response": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{"id": "rejected", "function": {"name": "Missing", "arguments": arguments}}],
                    }
                }
            ]
        },
    }


def selected(prefix=None):
    continuation = None
    if prefix is not None:
        continuation = {
            "mode": "restore",
            "prefix_export": prefix,
            "workspace_diff": "",
            "source_task_sha256": "a" * 64,
            "source_tools_sha256": "b" * 64,
        }
    return RuntimeSpec(
        harness="opencode", task_backend="swe", tools={"tool_names": {"read": "read"}}, continuation=continuation
    )


@pytest.mark.parametrize("arguments", ['{ "path": "source" }', "{"])
def test_captured_rejected_call_survives_export_and_continuation(arguments):
    raw = {"source": rejected_call_export()}
    saved = deepcopy(raw)
    enriched = enrich_exports(raw, [], spec=selected(), observations=[captured(arguments=arguments)])
    receipt = enriched["source"]["messages"][1]["parts"][0]["state"]["metadata"]["nemo_gym_model_call"]
    assert receipt == {"function": {"name": "Missing", "arguments": arguments}, "available_names": ["read"]}
    assert raw == saved  # the raw export stays an unchanged execution receipt

    # A continuation of that prefix restores the historical call in its root, from the frozen prefix alone.
    restored = rejected_call_export("new-root")
    again = enrich_exports({"new-root": restored}, [], spec=selected(enriched["source"]), observations=[])
    assert again["new-root"]["messages"][1]["parts"][0]["state"]["metadata"]["nemo_gym_model_call"] == receipt
    # A child cannot borrow the root's historical call, even with an identical call ID.
    child = rejected_call_export("child", parent="new-root")
    with pytest.raises(ValueError, match="unobserved rejected"):
        enrich_exports({"new-root": restored, "child": child}, [], spec=selected(enriched["source"]), observations=[])


def test_without_a_tool_selection_opencode_recovery_is_left_unchanged():
    raw = {"source": rejected_call_export()}
    assert enrich_exports(raw, [], spec=RuntimeSpec(harness="opencode", task_backend="swe"), observations=[]) == raw


@pytest.mark.parametrize("case", ["duplicate_call", "duplicate_names", "wrong_arguments"])
def test_invalid_prefix_receipts_fail_closed(case):
    export = rejected_call_export()
    part = export["messages"][1]["parts"][0]
    receipt = {"function": {"name": "Missing", "arguments": "{"}, "available_names": ["ReadSource"]}
    part["state"]["metadata"]["nemo_gym_model_call"] = receipt
    if case == "duplicate_call":
        export["messages"][1]["parts"].append(deepcopy(part))
    elif case == "duplicate_names":
        receipt["available_names"].append("ReadSource")
    else:
        receipt["function"]["arguments"] = {}
    with pytest.raises(ValueError, match="invalid or duplicate"):
        prefix_model_calls(SimpleNamespace(prefix_export=export))


def test_a_rejected_prefix_call_without_its_capture_fails_closed():
    with pytest.raises(ValueError, match="requires captured call evidence"):
        prefix_model_calls(SimpleNamespace(prefix_export=rejected_call_export()))


def test_export_does_not_infer_lost_model_arguments():
    with pytest.raises(ValueError, match="unobserved rejected"):
        preserve_rejected_calls({"source": rejected_call_export()}, [], None)


def test_ordinary_tool_error_does_not_require_rejection_receipts():
    export = rejected_call_export()
    part = export["messages"][1]["parts"][0]
    part["tool"] = "read"
    part["state"] = {"status": "error", "input": {"filePath": "source"}, "error": "File not found"}
    saved = deepcopy(export)
    preserve_rejected_calls({"source": export}, [], None)
    assert export == saved
