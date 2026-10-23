# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile-plugin receipts join OpenCode exports by exact session and call identity."""

from copy import deepcopy

import pytest


def export(session, tool="bash"):
    return {
        "info": {"id": session},
        "messages": [
            {
                "info": {"id": "message", "sessionID": session, "role": "assistant"},
                "parts": [
                    {
                        "type": "tool",
                        "tool": tool,
                        "callID": "same",
                        "state": {
                            "status": "error",
                            "input": {
                                "command": "exit 3",
                                "__nemo_gym_profile": {"configured_args": {"cmd": "exit 3"}},
                            },
                            "error": "failure",
                        },
                    }
                ],
            }
        ],
    }


def receipt(session, code):
    return {
        "kind": "tool_receipt",
        "session_id": session,
        "call_id": "same",
        "opencode_tool": "bash",
        "metadata": {"configured_args": {"cmd": "exit 3"}, "workspace_diff": f"diff {code}"},
    }


def test_failure_receipts_join_exact_sessions_without_overwriting_opencode_status_or_output():
    from resources_servers.agentic_if.episode.opencode_hooks import merge_hook_receipts

    originals = {"root": export("root"), "child": export("child")}
    saved = deepcopy(originals)
    result = merge_hook_receipts(originals, [receipt("child", 4), receipt("root", 3)])
    for session, code in [("root", 3), ("child", 4)]:
        state = result[session]["messages"][0]["parts"][0]["state"]
        assert state["metadata"]["nemo_gym_profile"]["workspace_diff"] == f"diff {code}"
        assert state["input"] == {"cmd": "exit 3"}
        assert state["status"] == "error" and state["error"] == "failure"
    assert originals == saved


def presentation(session="root"):
    return {
        "kind": "tool_presentation",
        "session_id": session,
        "call_id": "same",
        "opencode_tool": "bash",
        "opencode_error": "failure",
        "model_error": "RunCommand failed: failure",
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_presentation_receipt_preserves_execution_facts_and_raw_error(reverse):
    from resources_servers.agentic_if.episode.opencode_hooks import merge_hook_receipts
    from resources_servers.agentic_if.episode.opencode_profile import model_visible_tool_output

    original = {"root": export("root")}
    before = deepcopy(original)
    records = [receipt("root", 3), presentation()]
    result = merge_hook_receipts(original, records[::-1] if reverse else records)
    state = result["root"]["messages"][0]["parts"][0]["state"]
    assert state["status"] == "error" and state["error"] == "failure"
    assert state["metadata"]["nemo_gym_profile"]["workspace_diff"] == "diff 3"
    assert model_visible_tool_output(state) == "RunCommand failed: failure"
    assert original == before


def test_identical_replayed_presentation_is_idempotent():
    from resources_servers.agentic_if.episode.opencode_hooks import merge_hook_receipts

    records = [receipt("root", 3), presentation()]
    assert merge_hook_receipts({"root": export("root")}, records) == merge_hook_receipts(
        {"root": export("root")}, records + [presentation()]
    )


@pytest.mark.parametrize("defect", ["opencode_error", "session", "tool", "duplicate", "output", "status"])
def test_unbound_presentation_receipt_cannot_replace_opencode_error(defect):
    from resources_servers.agentic_if.episode.opencode_hooks import merge_hook_receipts

    original = {"root": export("root")}
    records = [receipt("root", 3), presentation()]
    if defect == "opencode_error":
        records[-1]["opencode_error"] = "different failure"
    elif defect == "session":
        records[-1]["session_id"] = "child"
    elif defect == "tool":
        records[-1]["opencode_tool"] = "read"
    elif defect == "duplicate":
        records.append(presentation() | {"model_error": "Conflicting presentation"})
    elif defect == "output":
        records[-1]["model_error"] = None
    else:
        original["root"]["messages"][0]["parts"][0]["state"]["status"] = "completed"
    with pytest.raises(ValueError, match="presentation"):
        merge_hook_receipts(original, records)


@pytest.mark.parametrize("case", ["duplicate", "wrong_tool", "missing_session"])
def test_incomplete_or_mismatched_receipts_do_not_make_a_replayable_export(case):
    from resources_servers.agentic_if.episode.opencode_hooks import merge_hook_receipts

    records = [receipt("root", 3)]
    if case == "duplicate":
        records.append(receipt("root", 4))
    if case == "wrong_tool":
        records[0]["opencode_tool"] = "write"
    if case == "missing_session":
        records[0]["session_id"] = "child"
    with pytest.raises(ValueError, match="unbound or duplicate hook execution receipt"):
        merge_hook_receipts({"root": export("root")}, records)
