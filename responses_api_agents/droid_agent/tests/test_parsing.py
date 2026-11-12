# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from responses_api_agents.droid_agent.parsing import parse_events


SESSION = "9bba6f55-69ea-40a8-9213-b5f408a4c6b9"
# Shortened from a real Droid 1.1.0 run against deepseek-ai/deepseek-v4.1-flash, with paths moved to /app.
DEEPSEEK_RUN = [
    {
        "type": "system",
        "subtype": "init",
        "cwd": "/app",
        "session_id": SESSION,
        "tools": ["Read"],
        "model": "custom:Gym-0",
    },
    {"type": "message", "role": "user", "id": "u1", "text": "Fix calc.py.", "timestamp": 1793730335714},
    {
        "type": "message",
        "role": "assistant",
        "id": "a1",
        "text": "I'll start by reading both files.\n\n",
        "timestamp": 1793730342218,
    },
    {
        "type": "tool_call",
        "id": "chatcmpl-tool-ae3",
        "toolId": "Read",
        "toolName": "Read",
        "parameters": {"file_path": "/app/calc.py"},
        "timestamp": 1793730342218,
    },
    {
        "type": "tool_call",
        "id": "chatcmpl-tool-941",
        "toolId": "Read",
        "toolName": "Read",
        "parameters": {"file_path": "/app/test_calc.py"},
        "timestamp": 1793730342218,
    },
    {
        "type": "tool_result",
        "id": "chatcmpl-tool-ae3",
        "toolId": "Read",
        "isError": False,
        "value": "def add(a, b):\n    return a - b\n",
        "timestamp": 1793730342232,
    },
    {
        "type": "tool_result",
        "id": "chatcmpl-tool-941",
        "toolId": "Read",
        "isError": False,
        "value": "from calc import add\n",
        "timestamp": 1793730342232,
    },
    {"type": "reasoning", "id": "r1", "text": "Simple fix.", "timestamp": 1793730349996},
    {"type": "reasoning", "id": "r1", "text": "Simple fix.", "timestamp": 1793730349996},
    {
        "type": "tool_call",
        "id": "chatcmpl-tool-858",
        "toolId": "Edit",
        "toolName": "Edit",
        "parameters": {"file_path": "/app/calc.py", "old_str": "a - b", "new_str": "a + b"},
        "timestamp": 1793730349996,
    },
    {
        "type": "tool_result",
        "id": "chatcmpl-tool-858",
        "toolId": "Edit",
        "isError": False,
        "value": {"success": True, "file_path": "/app/calc.py"},
        "timestamp": 1793730350010,
    },
    {
        "type": "tool_call",
        "id": "chatcmpl-tool-b35",
        "toolId": "Execute",
        "toolName": "Execute",
        "parameters": {"summary": "Run the calc test suite", "command": "python3 test_calc.py"},
        "timestamp": 1793730357434,
    },
    {
        "type": "tool_result",
        "id": "chatcmpl-tool-b35",
        "toolId": "Execute",
        "isError": False,
        "value": "TESTS PASS\n\n\n[Process exited with code 0]",
        "timestamp": 1793730357503,
    },
    {"type": "message", "role": "assistant", "id": "a2", "text": "Fixed `add`.", "timestamp": 1793730369946},
    {
        "type": "completion",
        "finalText": "Fixed `add`.",
        "numTurns": 4,
        "durationMs": 34358,
        "session_id": SESSION,
        "usage": {
            "input_tokens": 24118,
            "output_tokens": 466,
            "cache_read_input_tokens": 8192,
            "cache_creation_input_tokens": 0,
            "factory_credits": 0,
            "thinking_tokens": 0,
        },
    },
]


def stdout(events):
    return "\n".join(map(json.dumps, events)) + "\n"


def test_real_run_keeps_every_item_in_model_order():
    run = parse_events(stdout(DEEPSEEK_RUN), invocation_id="rollout")
    assert [item.type for item in run.items] == [
        "message",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
        "reasoning",
        "function_call",
        "function_call_output",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert run.items[1].arguments == json.dumps({"file_path": "/app/calc.py"})
    assert all(item.status == "completed" for item in run.items if item.type == "function_call")
    # A structured tool result reaches the response as the JSON the model saw.
    assert json.loads(run.items[7].output) == {"success": True, "file_path": "/app/calc.py"}
    assert run.items[-1].content[0].text == "Fixed `add`."
    assert run.completed and run.error is None and run.session_id == SESSION
    assert run.usage["input_tokens"] == 24118 and run.duration_ms == 34358
    assert run.conversation == run.items
    assert run.gaps == []


def test_tool_observations_carry_harness_timing_and_outcome():
    run = parse_events(stdout(DEEPSEEK_RUN), invocation_id="rollout")
    execute = next(tool for tool in run.tools if tool.tool_name == "Execute")
    assert execute.invocation_id == "rollout"
    assert execute.status == "completed"
    assert execute.duration_ms == 69
    assert execute.timing_source == "harness"
    assert len(run.tools) == 4


def test_nonzero_exit_and_tool_error_fail_the_tool_without_changing_its_output():
    events = [
        {"type": "tool_call", "id": "c1", "toolId": "Execute", "toolName": "Execute", "parameters": {}},
        {
            "type": "tool_result",
            "id": "c1",
            "toolId": "Execute",
            "isError": False,
            "value": "boom\n[Process exited with code 2]",
        },
        {"type": "tool_call", "id": "c2", "toolId": "Read", "toolName": "Read", "parameters": {}},
        {"type": "tool_result", "id": "c2", "toolId": "Read", "isError": True, "value": "no such file"},
        # Droid 1.1.0 reports a failed command with an error object and no value.
        {"type": "tool_call", "id": "c3", "toolId": "Execute", "toolName": "Execute", "parameters": {}},
        {
            "type": "tool_result",
            "id": "c3",
            "toolId": "Execute",
            "isError": True,
            "error": {"type": "tool_error", "message": "Error: Command terminated by signal: SIGKILL"},
        },
    ]
    run = parse_events(stdout(events), invocation_id="rollout")
    assert [tool.status for tool in run.tools] == ["failed", "failed", "failed"]
    assert run.items[1].output == "boom\n[Process exited with code 2]"
    assert run.items[5].output == "Error: Command terminated by signal: SIGKILL"
    assert [gap.code for gap in run.gaps] == ["tool_timing_unavailable"] * 3


def test_unfinished_tool_call_stays_incomplete():
    events = [
        {"type": "tool_call", "id": "c1", "toolId": "Execute", "toolName": "Execute", "parameters": {}, "timestamp": 5}
    ]
    run = parse_events(stdout(events), invocation_id="rollout")
    assert run.items[0].status == "incomplete"
    assert run.tools[0].status == "incomplete" and run.tools[0].started_at == 0.005
    assert not run.completed and run.usage is None


def test_error_event_is_kept_once():
    message = '404 {"status":404,"title":"Not Found"}'
    events = [
        {"type": "error", "source": "agent_loop", "message": message},
        {"type": "error", "source": "cli", "message": message},
    ]
    run = parse_events(stdout(events), invocation_id="rollout")
    assert run.error == message and not run.completed


def test_subagent_report_stays_in_the_conversation_with_gaps():
    events = [
        {"type": "message", "role": "user", "id": "u1", "text": "probe"},
        {
            "type": "tool_call",
            "id": "t1",
            "toolId": "Task",
            "toolName": "Task",
            "parameters": {"subagent_type": "explorer"},
        },
        {
            "type": "tool_result",
            "id": "t1",
            "toolId": "Task",
            "isError": False,
            "value": "Task launched in background.",
        },
        {"type": "message", "role": "assistant", "id": "a1", "text": "Waiting."},
        {"type": "message", "role": "user", "id": "task-completion:1", "text": "Background task completed."},
        {"type": "message", "role": "assistant", "id": "a2", "text": "Done."},
    ]
    run = parse_events(stdout(events), invocation_id="rollout")
    assert [item.type for item in run.items] == ["function_call", "function_call_output", "message", "message"]
    assert run.conversation[3].role == "user" and run.conversation[3].content == "Background task completed."
    codes = {gap.code for gap in run.gaps}
    assert {"subagent_hierarchy_unavailable", "injected_user_messages_omitted"} <= codes


def test_unparseable_lines_are_counted_not_dropped_silently():
    run = parse_events("not json\n[1, 2]\n" + stdout(DEEPSEEK_RUN[2:3]), invocation_id="rollout")
    assert run.unparseable_lines == 2
    assert run.gaps[-1].code == "agent_stdout_unparseable"
    assert run.items[0].content[0].text.startswith("I'll start")


def test_unicode_line_separators_inside_events_do_not_split_records():
    text = "a b c\u0085d"
    events = [
        {"type": "reasoning", "id": "r1", "text": text},
        {"type": "tool_call", "id": "c1", "toolName": "Read", "parameters": {"file_path": "/app/x"}},
        {"type": "tool_result", "id": "c1", "value": text},
        {"type": "message", "role": "assistant", "id": "a1", "text": text},
        {"type": "completion", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    # Droid writes raw UTF-8; CRLF delimiters and blank lines still separate records.
    lines = [json.dumps(event, ensure_ascii=False) for event in events]
    run = parse_events("\r\n".join(lines) + "\r\n\r\n", invocation_id="rollout")
    assert run.unparseable_lines == 0 and run.completed
    reasoning, _, output, message = run.items
    assert reasoning.summary[0].text == output.output == message.content[0].text == text
