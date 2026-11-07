# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goose 1.54.0 session store and stream-json output conversion, with notices recorded from real runs."""

import json

import pytest

from responses_api_agents.goose_agent.conversion import (
    goose_messages_to_output,
    goose_stop_reason,
    max_concurrent,
    model_server_failure,
)


# Notices verbatim from Goose 1.54.0 runs against a stub model server (500, 429, connection refused, context overflow).
NOTICES = {
    "server_error": "Ran into this error: Server error: Server error (500 Internal Server Error) at "
    "http://127.0.0.1:18991/v1/chat/completions: boom.\n\nPlease retry if you think this is a transient or recoverable error.",
    "rate_limit": "Ran into this error: Rate limit exceeded: rate limited.\n\nPlease retry if you think this is a transient "
    "or recoverable error.",
    "network": "Network error: Could not connect to 127.0.0.1:9 — check your network connection and try again.\n\n"
    "Please resend your message to try again.",
    "not_found": "Ran into this error: Request failed: Resource not found (404) at "
    "http://127.0.0.1:18983/v1/chat/completions: model not found.\n\nPlease retry if you think this is a transient or "
    "recoverable error.",
    "context_limit": "Ran into this error trying to compact: Failed to compact: the base prompt (system prompt, tool schemas, "
    "and conversation) exceeds the model's effective context window.",
    "auth": "Ran into this error: Authentication error: Authentication failed for http://127.0.0.1:18991/v1/chat/completions. "
    "Status: 401 Unauthorized. Response: bad key.",
    "model_error": "Ran into this error: Execution error: boom.",
    "max_turns": "I've reached the maximum number of actions I can do without user input. Would you like me to continue?",
}


def call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "type": "toolRequest",
        "id": call_id,
        "toolCall": {"status": "success", "value": {"name": name, "arguments": arguments}},
    }


def result(call_id: str, text: str) -> dict:
    return {
        "type": "toolResponse",
        "id": call_id,
        "toolResult": {"status": "success", "value": {"content": [{"type": "text", "text": text}], "isError": False}},
    }


def stream(last_id: str, text: str) -> str:
    # A banner precedes the stream-json events. Model messages are persisted under their streamed id; Goose's notice is not.
    events = [
        {"type": "message", "message": {"id": i, "role": "assistant", "content": [{"type": "text", "text": t}]}}
        for i, t in (("m1", "Working."), (last_id, text))
    ]
    return "    goose is ready\n" + "\n".join(json.dumps(e) for e in events + [{"type": "complete"}])


SESSIONS = [
    {"messages": [{"message_id": "m1", "role": "assistant", "content": [{"type": "text", "text": "Working."}]}]}
]


def test_thinking_dedup_is_per_turn():
    think = {"type": "thinking", "thinking": "plan"}

    def assistant(text):
        return {"role": "assistant", "content": [think, {"type": "text", "text": text}]}

    def user(text):
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    items = goose_messages_to_output([user("task"), assistant("a"), assistant("b"), user("next"), assistant("c")])
    assert [item.type for item in items] == ["reasoning", "message", "message", "reasoning", "message"]


def test_goose_only_messages_are_not_output():
    def text(role, value, visible=True):
        return {"role": role, "content": [{"type": "text", "text": value}], "user_visible": visible}

    # Compaction hides earlier messages from the agent but keeps them visible, and adds Goose-only ones.
    messages = [
        {"role": "assistant", "content": [call("c1", "shell", {"command": "ls"})], "user_visible": True},
        {"role": "user", "content": [result("c1", "a.txt")], "user_visible": True},
        text("user", "SUMMARY", visible=False),
        text("assistant", "Your context was compacted.", visible=False),
        text("assistant", "Done."),
    ]
    items = goose_messages_to_output(messages)
    assert [item.type for item in items] == ["function_call", "function_call_output", "message"]
    assert items[0].call_id == items[1].call_id == "c1"
    assert json.loads(items[0].arguments) == {"command": "ls"}
    assert items[-1].content[0].text == "Done."


@pytest.mark.parametrize(
    ("notice", "reason", "failure"),
    [
        ("server_error", "model_server", ("model_request_error", 500)),
        ("rate_limit", "model_server", ("model_request_error", 429)),
        ("network", "model_server", ("model_transport_error", None)),
        ("not_found", "model_server", ("model_request_error", 404)),
        ("auth", "model_server", ("model_request_error", 401)),
        ("context_limit", "context_limit", None),
        ("model_error", "model_error", None),
        ("max_turns", "max_turns", None),
    ],
)
def test_unpersisted_last_message_is_goose_notice(notice, reason, failure):
    assert goose_stop_reason(stream("notice", NOTICES[notice]), SESSIONS) == (reason, NOTICES[notice])
    assert model_server_failure(NOTICES[notice]) == failure


def test_context_limit_after_compaction_is_a_system_notification():
    text = "Unable to continue: Context limit still exceeded after compaction. Try using a shorter message."
    block = {"type": "systemNotification", "notificationType": "inlineMessage", "msg": text}
    event = {"type": "message", "message": {"id": "notice", "role": "assistant", "content": [block]}}
    assert goose_stop_reason(json.dumps(event), SESSIONS) == ("context_limit", text)


def test_persisted_policy_text_is_never_a_notice():
    # The policy's own persisted answer is not Goose's notice, whatever its wording.
    sessions = [{"messages": [{"message_id": "m1"}, {"message_id": "m2"}]}]
    assert goose_stop_reason(stream("m2", NOTICES["max_turns"]), sessions) == (None, "")
    assert goose_stop_reason(stream("m2", "All done."), sessions) == (None, "")


def test_persisted_notices_are_error_blocks():
    # Goose persists a notice only as an error block (rejected credentials).
    sessions = [{"messages": [{"message_id": "m1"}]}]

    def last(content):
        return json.dumps({"type": "message", "message": {"id": "m1", "role": "assistant", "content": content}})

    assert goose_stop_reason(last([{"type": "error", "message": NOTICES["auth"]}]), sessions)[0] == "model_server"
    assert goose_stop_reason(last([{"type": "text", "text": NOTICES["auth"]}]), sessions) == (None, "")


def test_output_limit_is_a_length_stop():
    event = {
        "type": "message",
        "message": {"id": "m1", "role": "assistant", "content": [], "metadata": {"outputTokenLimitReached": True}},
    }
    assert goose_stop_reason(json.dumps(event), [{"messages": [{"message_id": "m1"}]}])[0] == "length"


def test_max_concurrent():
    assert max_concurrent([(0, 5), (1, 2), (6, 7)]) == 2
    assert max_concurrent([(0, 1), (1, 2)]) == 2  # Second-resolution timestamps: touching counts as overlap.
    assert max_concurrent([]) == 0
