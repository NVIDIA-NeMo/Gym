# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family coverage, output separation, accounting, and consistent artifact capture."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, ToolCallObservation
from responses_api_agents.kilocode_agent.observability import read_kilo_observations
from responses_api_agents.kilocode_agent.sandbox import KiloArtifacts
from responses_api_agents.kilocode_agent.tests.test_app import _make_model_server_agent
from responses_api_agents.kilocode_agent.tests.test_sandbox import state


def family() -> list[dict]:
    sessions = []
    for sid, parent, child in [
        ("root", None, "child"),
        ("child", "root", "grandchild"),
        ("grandchild", "child", None),
    ]:
        parts = [
            {"id": sid + "-start", "type": "step-start"},
            {"id": sid + "-text", "type": "text", "text": sid + " answer"},
            {
                "id": sid + "-finish",
                "type": "step-finish",
                "tokens": {"input": 10, "output": 5, "cache": {"read": 3, "write": 1}, "reasoning": 2},
            },
        ]
        if child:
            parts.append(
                {
                    "id": sid + "-tool",
                    "type": "tool",
                    "tool": "task",
                    "callID": sid + "-task",
                    "state": {
                        "status": "completed",
                        "input": {"prompt": "delegate"},
                        "output": child + " answer",
                        "metadata": {"sessionId": child},
                        "time": {"start": 1000, "end": 2000},
                    },
                }
            )
        sessions.append(
            {
                "info": {"id": sid, "parentID": parent},
                "messages": [
                    {
                        "info": {"id": sid + "-user", "role": "user"},
                        "parts": [{"id": sid + "-prompt", "type": "text", "text": "delegate"}],
                    },
                    {
                        "info": {
                            "id": sid + "-assistant",
                            "role": "assistant",
                            "time": {"created": 1000, "completed": 2000},
                        },
                        "parts": parts,
                    },
                ],
            }
        )
    return sessions


def database(path: Path, sessions: list[dict]) -> None:
    """Minimal persisted session fixture; assertions concern Gym's observable results."""
    with sqlite3.connect(path) as connection:
        connection.execute("pragma journal_mode=wal")
        connection.execute("create table session (id text, parent_id text, time_created integer)")
        connection.execute("create table message (id text, session_id text, data text, time_created integer)")
        connection.execute(
            "create table part (id text, message_id text, session_id text, data text, time_created integer)"
        )
        for session in sessions:
            sid = session["info"]["id"]
            connection.execute("insert into session values (?, ?, 1)", (sid, session["info"].get("parentID")))
            for message in session["messages"]:
                mid = message["info"]["id"]
                connection.execute("insert into message values (?, ?, ?, 1)", (mid, sid, json.dumps(message["info"])))
                for part in message["parts"]:
                    connection.execute(
                        "insert into part values (?, ?, ?, ?, 1)", (part["id"], mid, sid, json.dumps(part))
                    )


def test_family_conversations_tool_links_and_inclusive_usage(tmp_path: Path) -> None:
    path = tmp_path / "kilo.db"
    database(path, family())
    observations, usage = read_kilo_observations(path, fallback_invocation_id="gym")
    invocations = {
        record.invocation_id: record for record in observations.records if isinstance(record, AgentInvocation)
    }
    assert set(invocations) == {"root", "child", "grandchild"}
    assert invocations["child"].parent_invocation_id == "root"
    assert invocations["grandchild"].spawned_by_tool_call_id == "child-task"
    assert all(record.conversation[1].content[0].text == sid + " answer" for sid, record in invocations.items())
    tools = [record for record in observations.records if isinstance(record, ToolCallObservation)]
    assert {tool.tool_call_id for tool in tools} == {"root-task", "child-task"}
    assert all(tool.duration_ms == 1000 and tool.status == "completed" for tool in tools)
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (42, 21, 63)
    assert usage.input_tokens_details.cached_tokens == 9
    assert usage.output_tokens_details.reasoning_tokens == 6
    assert not observations.gaps


async def test_response_keeps_root_output_and_family_usage(tmp_path: Path) -> None:
    path = tmp_path / "kilo.db"
    database(path, family())
    observations, usage = read_kilo_observations(path, fallback_invocation_id="gym")
    session = state()
    artifacts = KiloArtifacts(
        stdout=json.dumps({"type": "text", "sessionID": "root", "part": {"text": "root answer"}}),
        stderr="",
        exit_code=0,
        observations=observations,
        usage=usage,
        wall_time_s=1.25,
    )
    session.session.execute = AsyncMock(return_value=artifacts)
    session.session.artifacts = artifacts
    agent = _make_model_server_agent()
    response = await agent._sandbox_response(
        session, NeMoGymResponseCreateParamsNonStreaming(input="delegate"), rollout_id="rollout-1-a2"
    )
    assert response.usage == usage
    assert [item.content[0].text for item in response.output if item.type == "message"] == ["root answer"]
    retained = agent._sandbox_observations(session)
    invocations = [record for record in retained.records if isinstance(record, AgentInvocation)]
    assert len(invocations) == 3
    assert next(record for record in invocations if record.parent_invocation_id is None).duration_ms == 1250
    assert retained.records[-1].wall_time_s == 1.25


@pytest.mark.parametrize("damage", ["defaulted_details", "unfinished_step", "malformed_step"])
def test_partial_accounting_keeps_conversations_and_reports_gaps(tmp_path: Path, damage: str) -> None:
    sessions = family()
    parts = sessions[1]["messages"][1]["parts"]
    if damage == "defaulted_details":
        parts[2]["tokens"]["cache"]["read"] = parts[2]["tokens"]["reasoning"] = 0
    elif damage == "unfinished_step":
        parts.pop(2)
    else:
        parts[2].pop("tokens")
    path = tmp_path / "kilo.db"
    database(path, sessions)
    observations, usage = read_kilo_observations(path, fallback_invocation_id="gym")
    assert len([record for record in observations.records if isinstance(record, AgentInvocation)]) == 3
    assert usage.input_tokens_details.cached_tokens is None
    assert usage.output_tokens_details.reasoning_tokens is None
    gaps = {gap.code for gap in observations.gaps}
    assert "usage_details_unavailable" in gaps
    if damage != "defaulted_details":
        assert {"usage_totals_incomplete", "turn_model_call_scope_incomplete"} <= gaps


def test_snapshot_keeps_committed_wal_data(tmp_path: Path) -> None:
    path = tmp_path / "kilo.db"
    database(path, family())
    with sqlite3.connect(path) as connection:
        connection.execute("pragma wal_checkpoint(truncate)")
        connection.execute("insert into session values ('new-child', 'root', 2)")
        connection.commit()
        runner = Path(__file__).parents[1] / "sandbox_runner.py"
        subprocess.run(
            [sys.executable, "-I", str(runner), "--snapshot", str(tmp_path)], check=True, capture_output=True
        )
    observations, _ = read_kilo_observations(tmp_path / "observations.db", fallback_invocation_id="gym")
    assert "new-child" in {
        record.invocation_id for record in observations.records if isinstance(record, AgentInvocation)
    }


def test_native_child_errors_survive_observation_capture(tmp_path: Path) -> None:
    sessions = family()
    sessions[1]["messages"][1]["info"]["error"] = {"name": "APIError"}
    path = tmp_path / "kilo.db"
    database(path, sessions)
    observations, _ = read_kilo_observations(path, fallback_invocation_id="gym")
    child = next(
        record
        for record in observations.records
        if isinstance(record, AgentInvocation) and record.invocation_id == "child"
    )
    assert child.status == "failed" and child.error_type == "APIError"
