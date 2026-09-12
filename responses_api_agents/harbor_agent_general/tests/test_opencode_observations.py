# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3

from responses_api_agents.harbor_agent_general.app import HarborAgent


def test_reads_harbor_session_databases_for_multiple_steps(tmp_path):
    paths = []
    for name in ("first", "second"):
        path = tmp_path / name / "trajectory.json"
        paths.append(path)
        database = path.parent / "opencode/xdg-data/opencode/opencode.db"
        database.parent.mkdir(parents=True)
        with sqlite3.connect(database) as connection:
            connection.execute("create table session (id text, parent_id text, time_created integer)")
            connection.execute("create table message (id text, session_id text, data text, time_created integer)")
            connection.execute(
                "create table part (id text, message_id text, session_id text, data text, time_created integer)"
            )
            connection.execute("insert into session values (?, null, 1000)", (name,))
            connection.execute(
                "insert into message values ('message', ?, ?, 1000)",
                (name, json.dumps({"role": "assistant", "time": {"created": 1000, "completed": 2000}})),
            )
            connection.execute(
                "insert into part values ('part', 'message', ?, ?, 1000)",
                (name, json.dumps({"type": "text", "text": "Saved answer"})),
            )
    observations = HarborAgent.opencode_observations(paths)
    assert observations.source == "opencode"
    assert [record.invocation_id for record in observations.records] == ["first", "second"]
    assert all(record.conversation[0].content[0].text == "Saved answer" for record in observations.records)
    assert not observations.gaps


def test_missing_database_reports_observation_gap(tmp_path):
    observations = HarborAgent.opencode_observations([tmp_path / "trajectory.json"])
    assert "agent_artifact_unavailable" in [gap.code for gap in observations.gaps]


def test_corrupt_database_preserves_rollout_with_observation_gap(tmp_path):
    database = tmp_path / "opencode/xdg-data/opencode/opencode.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"invalid SQLite database")
    observations = HarborAgent.opencode_observations([tmp_path / "trajectory.json"])
    assert not observations.records
    assert [(gap.code, gap.detail) for gap in observations.gaps] == [("agent_artifact_unreadable", "DatabaseError")]
