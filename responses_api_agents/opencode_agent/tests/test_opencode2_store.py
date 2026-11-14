# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode 2 session stores read through the shared OpenCode 1 readers.

The rows below have the shapes OpenCode 2.2.7 saved in a real run (``session_v2``, ``session_message`` with the
message JSON in ``data``), trimmed to the columns the translator reads.
"""

import json
import sqlite3
from pathlib import Path

from nemo_gym.core import MODEL_CONTEXT_LIMIT, MODEL_REQUEST_ERROR, MODEL_TRANSPORT_ERROR
from nemo_gym.trajectory import ContextCompactionObservation, ToolCallObservation, TrajectoryRecord
from responses_api_agents.opencode_agent.app import OpenCodeAgent
from responses_api_agents.opencode_agent.artifacts import _parse_opencode_session
from responses_api_agents.opencode_agent.opencode2.store import (
    message_labels,
    reached_step_limit,
    tool_output_text,
    translate_store,
)
from responses_api_agents.opencode_agent.snapshot import snapshot


ROOT = "ses_root"
MODEL = {"id": "dummy_model", "providerID": "nemo_gym"}
TOKENS = {"input": 100, "output": 7, "reasoning": 2, "cache": {"read": 5, "write": 0}}


def user(text, created):
    return ("user", {"time": {"created": created}, "text": text, "files": []})


def assistant(created, content, *, finish="stop", error=None):
    data = {
        "time": {"created": created, "streamed": created + 1, "completed": created + 2},
        "agent": "build",
        "model": MODEL,
        "content": content,
        "finish": finish,
        "cost": 0,
        "tokens": TOKENS,
    }
    if error is not None:
        data["error"] = error
    return ("assistant", data)


def shell(call_id, output, *, exit_code=0):
    return {
        "type": "tool",
        "id": call_id,
        "name": "shell",
        "executed": False,
        "state": {
            "status": "completed",
            "input": {"command": "ls", "description": "list"},
            "content": [{"type": "text", "text": output}],
            "metadata": {"status": "completed", "truncated": False, "exit": exit_code},
        },
        "time": {"created": 10, "ran": 11, "completed": 12},
    }


def store(path, messages, *, sessions=((ROOT, None),), parts=()):
    """An OpenCode 2 store; ``messages`` holds ``(session, id, type, data)`` in order."""
    with sqlite3.connect(path) as con:
        con.execute("create table session_v2(id text, parent_id text, time_created integer)")
        con.execute(
            "create table session_message(id text, session_id text, type text, seq integer, time_created integer, "
            "data text)"
        )
        con.execute("create table session_part(message_id text, ordinal integer, data text)")
        con.executemany(
            "insert into session_v2 values (?, ?, ?)", [(sid, parent, n) for n, (sid, parent) in enumerate(sessions)]
        )
        con.executemany(
            "insert into session_message values (?, ?, ?, ?, ?, ?)",
            [
                (message_id, session, kind, seq, data.get("time", {}).get("created", 0), json.dumps(data))
                for seq, (session, message_id, kind, data) in enumerate(messages)
            ],
        )
        con.executemany("insert into session_part values (?, ?, ?)", parts)
    return path


def conversation(tmp_path, messages, **kwargs):
    """Translate, export and parse a store as a sandbox session's capture does."""
    translate_store(store(tmp_path / "opencode.db", messages, **kwargs), tmp_path / "sessions.db")
    snapshot(tmp_path / "sessions.db", tmp_path, harness_session_id=None, previous_message_ids=[])
    trajectory = TrajectoryRecord(task_id="t", rollout_id="r")
    bundle = _parse_opencode_session(
        tmp_path / "observations.db",
        "fallback",
        trajectory,
        require_terminal_finish=True,
        observation_schema_version="2.0",
        clock_domain="test",
    )
    return json.loads((tmp_path / "export.json").read_text()), bundle, trajectory


def test_a_run_reads_as_opencode_1_turns_with_the_tool_text_the_model_read(tmp_path):
    export, bundle, trajectory = conversation(
        tmp_path,
        [
            (ROOT, "msg_switch", "agent-switched", {"time": {"created": 1}, "agent": "build", "previous": "build"}),
            (ROOT, "msg_user", *user("Create out.txt", 2)),
            (
                ROOT,
                "msg_a1",
                *assistant(
                    3,
                    [{"type": "reasoning", "text": "think"}, shell("call_1", "HOME=/root\nout.txt\n")],
                    finish="tool-calls",
                ),
            ),
            (ROOT, "msg_a2", *assistant(9, [{"type": "text", "text": "DONE"}])),
            (ROOT, "msg_idle", "idle", {"time": {"created": 12}, "outcome": "succeeded"}),
        ],
    )
    assert [(m["info"]["role"], [p["type"] for p in m["parts"]]) for m in export["messages"]] == [
        ("user", ["text"]),
        ("assistant", ["step-start", "reasoning", "tool", "step-finish"]),
        ("assistant", ["step-start", "text", "step-finish"]),
    ]
    tool = export["messages"][1]["parts"][2]
    assert tool["tool"] == "shell" and tool["callID"] == "call_1"
    assert tool["state"]["output"] == "HOME=/root\nout.txt\n"
    assert export["messages"][1]["info"]["tokens"] == TOKENS
    (invocation,) = [r for r in bundle.records if r.kind == "agent_invocation"]
    assert invocation.status == "completed"
    assert [item.type for item in invocation.conversation] == [
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert invocation.conversation[3].output == "HOME=/root\nout.txt\n"
    (call,) = [r for r in bundle.records if isinstance(r, ToolCallObservation)]
    assert call.tool_name == "shell" and call.status == "completed"
    assert [turn.source_message_id for turn in trajectory.turns] == ["msg_a1", "msg_a2"]
    assert not trajectory.gaps


def test_a_failing_shell_command_is_a_failed_tool_call(tmp_path):
    _, bundle, _ = conversation(
        tmp_path,
        [
            (ROOT, "msg_user", *user("go", 1)),
            (ROOT, "msg_a1", *assistant(2, [shell("call_1", "boom", exit_code=2)], finish="tool-calls")),
            (ROOT, "msg_a2", *assistant(5, [{"type": "text", "text": "done"}])),
        ],
    )
    (call,) = [r for r in bundle.records if isinstance(r, ToolCallObservation)]
    assert (call.status, call.error_type) == ("failed", "tool_error")


def test_labeled_replies_take_the_label_the_model_call_carried(tmp_path):
    messages = [
        (ROOT, "msg_user", *user("go", 1)),
        (ROOT, "msg_a1", *assistant(2, [{"type": "text", "text": "done"}])),
    ]
    labels = message_labels(json.dumps({"label": "msg_gym_1", "session": ROOT, "message": "msg_a1"}) + "\nnot json\n")
    assert labels == {(ROOT, "msg_a1"): "msg_gym_1"}
    translate_store(store(tmp_path / "opencode.db", messages), tmp_path / "sessions.db", labels=labels)
    with sqlite3.connect(tmp_path / "sessions.db") as con:
        ids = [row[0] for row in con.execute("select id from message order by time_created, id")]
    assert ids == ["msg_user", "msg_gym_1"]


def test_provider_errors_keep_their_status_and_classification(tmp_path):
    def failed(error):
        return assistant(2, [], finish="error", error=error)

    cases = {
        "msg_429": failed({"type": "provider.rate-limit", "message": "slow down", "status": 429}),
        "msg_net": failed({"type": "provider.transport", "message": "socket hang up"}),
        "msg_ctx": failed(
            {
                "type": "provider.invalid-request",
                "message": "Bad request",
                "status": 400,
                "response": {"body": "This model's maximum context length is 32768 tokens."},
            }
        ),
        "msg_400": failed({"type": "provider.invalid-request", "message": "bad tool schema", "status": 400}),
    }
    messages = [(ROOT, "msg_user", *user("go", 1))] + [(ROOT, key, *value) for key, value in cases.items()]
    translate_store(store(tmp_path / "opencode.db", messages), tmp_path / "sessions.db")
    with sqlite3.connect(tmp_path / "sessions.db") as con:
        infos = {row[0]: json.loads(row[1]) for row in con.execute("select id, data from message")}
    assert infos["msg_429"]["error"] == {
        "name": "APIError",
        "data": {"message": "slow down", "type": "provider.rate-limit", "statusCode": 429, "isRetryable": True},
    }
    # OpenCode 1 saved a provider failure without a finish reason and a context overflow with "error".
    assert "finish" not in infos["msg_429"] and infos["msg_ctx"]["finish"] == "error"
    assert infos["msg_ctx"]["error"]["name"] == "ContextOverflowError"
    outcomes = {key: OpenCodeAgent.classify_outcome(None, infos[key]) for key in cases}
    assert (outcomes["msg_429"].failure_kind, outcomes["msg_429"].status_code) == (MODEL_REQUEST_ERROR, 429)
    assert outcomes["msg_net"].failure_kind == MODEL_TRANSPORT_ERROR
    assert outcomes["msg_ctx"].failure_kind == MODEL_CONTEXT_LIMIT
    assert (outcomes["msg_400"].failure_kind, outcomes["msg_400"].status_code) == (MODEL_REQUEST_ERROR, 400)


def test_a_failed_or_aborted_tool_records_the_json_the_model_read(tmp_path):
    rejected = {"type": "permission.rejected", "message": "This non-interactive run cannot ask the user"}
    tools = [
        {
            "type": "tool",
            "id": "call_1",
            "name": "shell",
            "state": {"status": "error", "input": {"command": "rm -rf /"}, "error": rejected},
            "time": {"created": 3, "completed": 4},
        },
        {
            "type": "tool",
            "id": "call_2",
            "name": "shell",
            "state": {
                "status": "error",
                "input": {"command": "sleep 99"},
                "error": {"type": "aborted", "message": "x"},
            },
            "time": {"created": 3, "completed": 4},
        },
    ]
    export, bundle, _ = conversation(
        tmp_path,
        [
            (ROOT, "msg_user", *user("go", 1)),
            (ROOT, "msg_a1", *assistant(2, tools, finish="tool-calls")),
            (ROOT, "msg_a2", *assistant(6, [{"type": "text", "text": "done"}])),
        ],
    )
    state = export["messages"][1]["parts"][1]["state"]
    assert state["error"] == json.dumps({"error": rejected, "content": []}, separators=(",", ":"))
    calls = {r.tool_call_id: r for r in bundle.records if isinstance(r, ToolCallObservation)}
    assert len(calls) == 2
    assert sorted((call.status, call.error_type) for call in calls.values()) == [
        ("cancelled", "tool_aborted"),
        ("failed", "tool_error"),
    ]


def test_tool_images_stay_out_of_the_recorded_text():
    content = [
        {"type": "text", "text": "Image read successfully"},
        {"type": "file", "uri": "data:image/png;base64,AAAA", "mime": "image/png", "name": "a.png"},
    ]
    assert tool_output_text(content) == "Image read successfully"
    assert tool_output_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"


def test_a_compaction_is_a_summary_reply_to_a_compaction_request(tmp_path):
    export, bundle, trajectory = conversation(
        tmp_path,
        [
            (ROOT, "msg_user", *user("go", 1)),
            (ROOT, "msg_a1", *assistant(2, [shell("call_1", "x")], finish="tool-calls")),
            (
                ROOT,
                "msg_compact",
                "compaction",
                {
                    "time": {"created": 5, "completed": 6},
                    "status": "completed",
                    "reason": "auto",
                    "summary": "Listed files.",
                    "model": MODEL,
                    "tokens": TOKENS,
                    "cost": 0,
                },
            ),
            (ROOT, "msg_a2", *assistant(7, [{"type": "text", "text": "done"}])),
        ],
    )
    roles = [(m["info"]["role"], m["info"].get("summary")) for m in export["messages"]]
    assert roles == [("user", None), ("assistant", None), ("user", None), ("assistant", True), ("assistant", None)]
    (compaction,) = [r for r in bundle.records if isinstance(r, ContextCompactionObservation)]
    assert (compaction.summary, compaction.trigger, compaction.source_message_ids) == (
        "Listed files.",
        "automatic",
        ["msg_compact"],
    )
    # The summary call is not a turn of the conversation.
    assert [turn.source_message_id for turn in trajectory.turns] == ["msg_a1", "msg_a2"]
    # Its tokens count toward usage, as OpenCode 1's summary message did.
    assert len([info for info in export["usage_messages"] if info["role"] == "assistant"]) == 3


def test_content_kept_in_part_rows_is_read_too(tmp_path):
    reply = assistant(2, [])
    parts = [("msg_a1", 0, json.dumps({"type": "text", "text": "from part rows"}))]
    export, _, _ = conversation(tmp_path, [(ROOT, "msg_user", *user("go", 1)), (ROOT, "msg_a1", *reply)], parts=parts)
    assert export["messages"][1]["parts"][1] == {"type": "text", "text": "from part rows"}


def test_a_subagent_session_is_a_child_invocation_spawned_by_its_call(tmp_path):
    spawn = {
        "type": "tool",
        "id": "call_sub",
        "name": "subagent",
        "state": {
            "status": "completed",
            "input": {"agent": "general", "prompt": "look"},
            "content": [{"type": "text", "text": '<subagent sessionID="ses_child" state="completed">'}],
            "metadata": {"sessionID": "ses_child", "status": "completed"},
        },
        "time": {"created": 3, "ran": 3, "completed": 8},
    }
    _, bundle, _ = conversation(
        tmp_path,
        [
            (ROOT, "msg_user", *user("go", 1)),
            (ROOT, "msg_a1", *assistant(2, [spawn], finish="tool-calls")),
            ("ses_child", "msg_cu", *user("look", 4)),
            ("ses_child", "msg_ca", *assistant(5, [{"type": "text", "text": "seen"}])),
            (ROOT, "msg_a2", *assistant(9, [{"type": "text", "text": "done"}])),
        ],
        sessions=((ROOT, None), ("ses_child", ROOT)),
    )
    invocations = {r.invocation_id: r for r in bundle.records if r.kind == "agent_invocation"}
    assert invocations["ses_child"].parent_invocation_id == ROOT
    assert invocations["ses_child"].spawned_by_tool_call_id == "call:msg_a1-0001"


def test_messages_with_one_creation_time_keep_their_sequence(tmp_path):
    messages = [
        (ROOT, "msg_z_user", *user("go", 5)),
        (ROOT, "msg_a_reply", *assistant(5, [{"type": "text", "text": "done"}])),
    ]
    translate_store(store(tmp_path / "opencode.db", messages), tmp_path / "sessions.db")
    with sqlite3.connect(tmp_path / "sessions.db") as con:
        ids = [row[0] for row in con.execute("select id from message order by time_created, id")]
    assert ids == ["msg_z_user", "msg_a_reply"]


def export_of(*roles):
    """An export whose messages have ``roles``: user, synthetic, compaction, assistant or summary."""
    messages = []
    for role in roles:
        if role in ("user", "synthetic", "compaction"):
            part = (
                {"type": "compaction"} if role == "compaction" else {"type": "text", "synthetic": role == "synthetic"}
            )
            messages.append({"info": {"role": "user"}, "parts": [part]})
        else:
            messages.append({"info": {"role": "assistant", "summary": role == "summary"}, "parts": []})
    return {"messages": messages}


def test_the_step_limit_counts_replies_since_the_last_prompt():
    assert not reached_step_limit(export_of("user", "assistant", "assistant"), None)
    assert reached_step_limit(export_of("user", "assistant", "assistant"), 2)
    assert not reached_step_limit(export_of("user", "assistant", "user", "assistant"), 2)
    # A compaction runs inside a step, and a synthetic continuation does not start a prompt.
    assert not reached_step_limit(export_of("user", "assistant", "compaction", "summary"), 2)
    assert reached_step_limit(export_of("user", "assistant", "synthetic", "assistant"), 2)


def test_an_empty_store_translates_to_empty_tables(tmp_path: Path):
    translate_store(store(tmp_path / "opencode.db", []), tmp_path / "sessions.db")
    with sqlite3.connect(tmp_path / "sessions.db") as con:
        assert con.execute("select count(*) from message").fetchone() == (0,)
        assert con.execute("select id, parent_id from session").fetchall() == [(ROOT, None)]
