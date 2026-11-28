# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence of a row that renames tools: the tool that ran (``tool``) beside the name the model saw and wrote
(``model_tool``): the name on the wire for a live call, the row's alias for a call in a restored prefix.
"""

from resources_servers.agentic_if.episode.session_evidence import build_session_evidence
from resources_servers.agentic_if.if_grading.contracts import SessionInfo
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec
from resources_servers.agentic_if.tests.support import REPLAY_RECEIPT, call, export, message, text


PROFILE = "p" * 64
TOOLS = {"tool_names": {"bash": "shell", "read": "read"}}
# The profile plugin's refusal of bash, the hidden ID of the tool the model sees as shell.
REFUSAL = "bash is unavailable or its arguments are invalid. Available tools: read, shell."


def build(spec, *, exports, hooks=(), calls=(), replay=None, artifacts=None):
    documents = {document["info"]["id"]: document for document in exports}
    return build_session_evidence(
        spec,
        exports=documents,
        sessions=[SessionInfo(id=key, parent_id=value["info"].get("parentID")) for key, value in documents.items()],
        hook_receipts=[{"kind": "loaded", "profile_sha256": PROFILE}, *hooks],
        event_records=[],
        call_records=list(calls),
        profile_sha256=PROFILE,
        replay=replay,
        artifacts=artifacts or {},
    ).evidence


def model_call(identity, request_messages, *, content, tool_calls=()):
    """The boundary's record of one actor call of the root session."""
    message_ = {"role": "assistant", "content": content}
    if tool_calls:
        message_["tool_calls"] = [
            {"id": ident, "type": "function", "function": {"name": name, "arguments": '{"command": "pwd"}'}}
            for ident, name in tool_calls
        ]
    return {
        "source": "model",
        "session_id": "root",
        "assistant_message_id": identity,
        "execution_tools": {"shell": "bash", "read": "read"},
        "request": {
            "messages": [{"role": "system", "content": "Use shell."}, *request_messages],
            "tools": [{"type": "function", "function": {"name": name}} for name in ("shell", "read")],
        },
        "response": {"choices": [{"message": message_}]},
    }


def live_session():
    """The model ran pwd under the alias shell, then wrote the hidden ID bash, which the profile plugin refused."""
    refused = {
        "type": "tool",
        "id": "part-c2",
        "callID": "c2",
        "tool": "bash",
        "state": {
            "status": "error",
            "input": {"command": "pwd"},
            "error": "Error: " + REFUSAL,
            "time": {"start": 4, "end": 5},
        },
    }
    root = export(
        message("u", "user", text("Fix it")),
        message(
            "a",
            "assistant",
            text("Inspect"),
            call("c1", "bash", input={"command": "pwd"}, output="/app", end=4),
            refused,
            created=2,
        ),
        message("z", "assistant", text("Done"), created=6),
    )
    receipt = {
        "kind": "tool_receipt",
        "session_id": "root",
        "call_id": "c2",
        "opencode_tool": "bash",
        "metadata": {"model_error": REFUSAL, "dispatch_rejected": True},
    }
    calls = [
        model_call(
            "a",
            [{"role": "user", "content": "Fix it"}],
            content="Inspect",
            tool_calls=[("c1", "shell"), ("c2", "bash")],
        ),
        model_call(
            "z",
            [
                {"role": "tool", "tool_call_id": "c1", "content": "/app"},
                {"role": "tool", "tool_call_id": "c2", "content": REFUSAL},
            ],
            content="Done",
        ),
    ]
    return root, receipt, calls


def test_a_live_call_keeps_the_tool_that_ran_and_the_name_the_model_wrote():
    root, receipt, calls = live_session()
    evidence = build(
        RuntimeSpec(harness="opencode", task_backend="swe", tools=TOOLS), exports=[root], hooks=[receipt], calls=calls
    )
    assert evidence.status == "complete", evidence.issues
    ran, refused = next(step for step in evidence.steps if step.calls).calls
    assert (ran.tool, ran.model_tool, ran.rejected) == ("bash", "shell", False)
    # A hidden ID ran nothing: OpenCode's tool part says bash, but it is a rejected call, like any unknown name.
    assert (refused.tool, refused.model_tool, refused.rejected) == ("invalid", "bash", True)
    results = {step.call_id: step for step in evidence.steps if step.role == "tool"}
    assert (results["c1"].tool, results["c1"].output) == ("bash", "/app")
    assert (results["c2"].tool, results["c2"].output, results["c2"].error) == ("invalid", REFUSAL, True)


def test_a_restored_prefix_call_is_shown_under_the_row_s_alias():
    prefix = export(
        message("p-user", "user", text("Fix it"), session="ses_prefix"),
        message(
            "p-reply",
            "assistant",
            text("Listing"),
            call("call_ls", "bash", input={"command": "ls"}, output="a.py\n", end=3),
            session="ses_prefix",
            created=2,
        ),
        session="ses_prefix",
    )
    spec = RuntimeSpec(
        harness="opencode",
        task_backend="swe",
        tools=TOOLS,
        continuation={
            "mode": "restore",
            "prefix_export": prefix,
            "workspace_diff": "",
            "source_task_sha256": "a" * 64,
            "source_tools_sha256": "b" * 64,
        },
    )
    root = export(*prefix["messages"], session="ses_prefix")
    root["messages"] += [
        message("fresh-user", "user", text("Continue"), session="ses_prefix", created=10),
        message("fresh-reply", "assistant", text("DONE"), session="ses_prefix", created=11),
    ]
    fresh = {
        "session_id": "ses_prefix",
        "assistant_message_id": "fresh-reply",
        "execution_tools": {"shell": "bash", "read": "read"},
        "request": {"messages": []},
        "response": {"choices": [{"message": {"role": "assistant", "content": "DONE"}}]},
    }
    evidence = build(
        spec,
        exports=[root],
        calls=[fresh],
        replay=REPLAY_RECEIPT,
        artifacts={"prefix_export": "/artifacts/prefix-export.json"},
    )
    assert evidence.status == "complete", evidence.issues
    # No model call of this run is bound to the prefix's call: the model saw it under the alias the row gives.
    restored = next(step for step in evidence.steps if step.calls).calls[0]
    assert (restored.tool, restored.model_tool) == ("bash", "shell")
