# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract fixture through Gym's real OpenCode parser and collector projection.

Synthetic protocol/SQLite inputs; not a live OpenCode qualification.
"""

import json
import sqlite3

from nemo_gym.base_responses_api_model import build_model_call_record
from nemo_gym.harness_capabilities.checker import P0, inspect_record
from nemo_gym.harness_capabilities.reader import hydrate_record
from nemo_gym.rollout_collection import _attach_trajectory_record
from nemo_gym.rollout_observability import join_model_call_observations
from responses_api_agents.opencode_sandboxed_agent.app import (
    parse_opencode_observations,
)


def test_current_opencode_projection(tmp_path):
    path = tmp_path / "opencode.db"
    with sqlite3.connect(path) as db:
        db.execute("create table session (id text, parent_id text, time_created integer)")
        db.execute("create table message (id text, session_id text, data text, time_created integer)")
        db.execute("create table part (id text, message_id text, session_id text, data text, time_created integer)")
        db.execute("insert into session values ('root', null, 0)")
        db.execute(
            "insert into message values (?, ?, ?, ?)",
            (
                "m1",
                "root",
                json.dumps(
                    {
                        "role": "assistant",
                        "time": {"created": 1000, "completed": 3000},
                    }
                ),
                1000,
            ),
        )
        db.execute(
            "insert into part values (?, ?, ?, ?, ?)",
            (
                "p1",
                "m1",
                "root",
                json.dumps(
                    {
                        "type": "tool",
                        "tool": "bash",
                        "callID": "tool-1",
                        "state": {
                            "status": "completed",
                            "input": {"command": "printf ok"},
                            "output": "ok",
                            "time": {"start": 2000, "end": 3000},
                        },
                    }
                ),
                2000,
            ),
        )
    observations = parse_opencode_observations(path, "root")
    exchange = {
        "model_call_id": "call-1",
        "client_session_id": "root",
        "model_ref": {"type": "responses_api_models", "name": "policy"},
        "dialect": "responses",
        "started_at": 1.0,
        "completed_at": 2.0,
        "latency_ms": 1000.0,
        "status_code": 200,
        "request": {
            "model": "model",
            "input": [{"role": "user", "content": "Print ok"}],
            "tools": [
                {
                    "type": "function",
                    "name": "bash",
                    "parameters": {"type": "object"},
                }
            ],
        },
        "response": {
            "id": "response-1",
            "model": "model",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "tool-1",
                    "name": "bash",
                    "arguments": '{"command":"printf ok"}',
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 2},
            },
        },
    }
    call = build_model_call_record(exchange, call_index=0)
    observations = join_model_call_observations(observations, [call])
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "ng_model_call_capture": {
            "rollout_id": "0-0",
            "calls": [call.model_dump(mode="json")],
        },
        "ng_agent_observations": observations.model_dump(mode="json"),
    }
    _attach_trajectory_record(record, record)
    result = inspect_record(hydrate_record(record))
    assert all(result["capabilities"][key]["verdict"] == "fulfilled" for key in (*P0, "C5")), result["findings"]
    assert result["capabilities"]["C3"]["verdict"] == "not_fulfilled"
