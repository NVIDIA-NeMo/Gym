# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled native protocol inputs passed through real producers and collection.

These fixtures exercise the capture-enabled delivery path. They do not establish
that a deployed harness routes every request through capture or passes live QA.
"""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nemo_gym.base_responses_api_model import build_model_call_record
from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_collection import _attach_trajectory_record, _strip_capture_payloads
from nemo_gym.rollout_observability import TrajectoryRecord, join_model_call_observations
from responses_api_agents.codex_agent.app import parse_exec_jsonl
from responses_api_agents.hermes_agent.observability import HermesAgentObserver
from responses_api_agents.opencode_agent.app import _parse_opencode_session
from responses_api_agents.pi_agent.app import _build_pi_observations, parse_pi_events


HARNESSES = ("opencode", "pi", "codex", "hermes")
MODEL = ModelServerRef(type="responses_api_models", name="policy")
ARGS = {"command": "printf ok"}


def _exchanges(dialect: str, *, session: str | None) -> list[dict]:
    request_messages = [{"role": "user", "content": "Print ok, then report success."}]
    tool = {"id": "tool-1", "type": "function", "function": {"name": "bash", "arguments": json.dumps(ARGS)}}
    tool_output = {"role": "assistant", "content": None, "tool_calls": [tool]}
    result = {"role": "tool", "tool_call_id": "tool-1", "content": "ok"}
    records = []
    for index in range(2):
        usage = {
            "prompt_tokens": 10 + index,
            "completion_tokens": 5,
            "total_tokens": 15 + index,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
        response = {
            "id": f"response-{index}",
            "model": "fixture-model",
            "usage": usage,
            "choices": [
                {
                    "index": 0,
                    "message": tool_output if index == 0 else {"role": "assistant", "content": "Done"},
                    "finish_reason": "tool_calls" if index == 0 else "stop",
                }
            ],
        }
        request = {
            "model": "fixture-model",
            "messages": request_messages if index == 0 else [*request_messages, tool_output, result],
        }
        if dialect == "responses":
            response = {
                "id": f"response-{index}",
                "model": "fixture-model",
                "status": "completed",
                "output": [
                    {"type": "function_call", "call_id": "tool-1", "name": "bash", "arguments": json.dumps(ARGS)}
                ]
                if index == 0
                else [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done"}]}],
                "usage": {
                    "input_tokens": 10 + index,
                    "output_tokens": 5,
                    "total_tokens": 15 + index,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 2},
                },
            }
            request = {
                "model": "fixture-model",
                "input": request_messages
                if index == 0
                else [
                    *request_messages,
                    {"type": "function_call", "call_id": "tool-1", "name": "bash", "arguments": json.dumps(ARGS)},
                    {"type": "function_call_output", "call_id": "tool-1", "output": "ok"},
                ],
            }
        records.append(
            {
                "model_call_id": f"call-{index}",
                "client_session_id": session,
                "model_ref": MODEL.model_dump(),
                "dialect": dialect,
                "started_at": float(index + 1),
                "completed_at": float(index + 1.5),
                "latency_ms": 500.0,
                "status_code": 200,
                "request": request,
                "response": response,
            }
        )
    return records


def _opencode(directory: Path):
    path = directory / "opencode.db"
    with sqlite3.connect(path) as db:
        db.execute("create table session (id text, parent_id text, time_created integer)")
        db.execute("create table message (id text, session_id text, data text, time_created integer)")
        db.execute("create table part (id text, message_id text, session_id text, data text, time_created integer)")
        db.execute("insert into session values ('root', null, 0)")
        for index in range(2):
            message_id = f"message-{index}"
            db.execute(
                "insert into message values (?, ?, ?, ?)",
                (
                    message_id,
                    "root",
                    json.dumps(
                        {
                            "role": "assistant",
                            "time": {"created": 1000 + index * 1000, "completed": 1500 + index * 1000},
                        }
                    ),
                    index,
                ),
            )
            part = (
                {
                    "type": "tool",
                    "tool": "bash",
                    "callID": "tool-1",
                    "state": {
                        "status": "completed",
                        "input": ARGS,
                        "output": "ok",
                        "time": {"start": 1600, "end": 1800},
                    },
                }
                if index == 0
                else {"type": "text", "text": "Done"}
            )
            for number, value in enumerate([{"type": "step-start"}, part, {"type": "step-finish"}]):
                db.execute(
                    "insert into part values (?, ?, ?, ?, ?)",
                    (f"{message_id}-{number}", message_id, "root", json.dumps(value), number),
                )
    trajectory = TrajectoryRecord(task_id="0", rollout_id="0-0")
    return _parse_opencode_session(path, "root", trajectory), trajectory


def _pi():
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "responseId": "response-0",
                "content": [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": ARGS}],
            },
        },
        {"type": "tool_execution_start", "toolCallId": "tool-1", "toolName": "bash"},
        {"type": "tool_execution_end", "toolCallId": "tool-1", "toolName": "bash", "isError": False},
        {
            "type": "message_end",
            "message": {"role": "toolResult", "toolCallId": "tool-1", "content": [{"type": "text", "text": "ok"}]},
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "responseId": "response-1",
                "content": [{"type": "text", "text": "Done"}],
            },
        },
        {"type": "agent_end", "messages": [{"role": "assistant", "stopReason": "stop"}]},
    ]
    output, _ = parse_pi_events("\n".join(json.dumps(event) for event in events))
    return _build_pi_observations([(float(i), event) for i, event in enumerate(events)], "root", MODEL, output)


class _HermesProtocol:
    """Only the upstream hook interface; Gym's instrumentation runs unmodified."""

    def __init__(self):
        self.tool_start_callback = self.tool_complete_callback = None
        self._active_children = []
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)

    def _interruptible_api_call(self, params):
        return SimpleNamespace(id=params["response_id"])

    def _invoke_tool(self, name, arguments, task):
        return "ok"

    def _compress_context(self, messages, system_message, **kwargs):
        return messages, system_message


def _hermes():
    agent = _HermesProtocol()
    with (
        patch("responses_api_agents.hermes_agent.observability.time", return_value=1.0),
        patch("responses_api_agents.hermes_agent.observability.monotonic", return_value=1.0),
    ):
        observer = HermesAgentObserver(root_invocation_id="root", model_ref=MODEL).instrument(agent)
        agent._interruptible_api_call({"response_id": "response-0"})
        agent.tool_start_callback("tool-1", "bash", ARGS)
        agent._invoke_tool("bash", ARGS, "task")
        agent.tool_complete_callback("tool-1", "bash", ARGS, "ok")
        agent._interruptible_api_call({"response_id": "response-1"})
        return observer.finish(
            {
                "completed": True,
                "messages": [
                    {"role": "user", "content": "Print ok, then report success."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "tool-1", "function": {"name": "bash", "arguments": ARGS}}],
                    },
                    {"role": "tool", "tool_call_id": "tool-1", "content": "ok"},
                    {"role": "assistant", "content": "Done"},
                ],
            }
        )


def build_harness_record(harness: str, directory: Path) -> dict:
    """Build a collected record using real parser/observer and collector functions."""
    directory.mkdir(parents=True, exist_ok=True)
    calls = [
        build_model_call_record(exchange, call_index=i)
        for i, exchange in enumerate(
            _exchanges(
                "responses" if harness == "codex" else "chat", session="root" if harness == "opencode" else None
            )
        )
    ]
    row = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "_ng_rollout_id": "0-0",
        "reward": 1.0,
        "ng_model_call_capture": {"rollout_id": "0-0", "calls": [c.model_dump(mode="json") for c in calls]},
    }
    if harness == "codex":
        events = [
            {
                "type": "item.completed",
                "item": {
                    "id": "tool-1",
                    "type": "command_execution",
                    "command": "printf ok",
                    "aggregated_output": "ok",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {"type": "item.completed", "item": {"id": "answer", "type": "agent_message", "text": "Done"}},
            {"type": "turn.completed", "usage": {"input_tokens": 21, "output_tokens": 10}},
        ]
        output, usage = parse_exec_jsonl("\n".join(json.dumps(e) for e in events))
        row["response"] = {"output": [item.model_dump(mode="json") for item in output], "usage": usage}
        # Codex's producer returns response items, not canonical observations or turns.
    else:
        if harness == "opencode":
            observations, trajectory = _opencode(directory)
            row["ng_trajectory"] = trajectory.model_dump(mode="json")
        elif harness == "pi":
            observations = _pi()
        elif harness == "hermes":
            observations = _hermes()
        else:
            raise ValueError(harness)
        row["ng_agent_observations"] = join_model_call_observations(observations, calls).model_dump(mode="json")
    _attach_trajectory_record(row, row)
    _strip_capture_payloads(row)
    return row
