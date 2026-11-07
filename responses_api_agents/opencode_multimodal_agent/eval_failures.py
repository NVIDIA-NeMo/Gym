# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation policy for completed, failed OpenCode attempts."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path


def terminal_model_result(database: Path) -> dict:
    """Read the last root assistant's result; damaged optional evidence proves no cause."""
    try:
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            roots = connection.execute("SELECT id FROM session WHERE parent_id IS NULL").fetchall()
            if len(roots) != 1:
                return {}
            rows = connection.execute(
                "SELECT data FROM message WHERE session_id=? ORDER BY time_created DESC, id DESC", (roots[0][0],)
            )
            for row in rows:
                message = json.loads(row[0])
                if isinstance(message, dict) and message.get("role") == "assistant":
                    error = message.get("error")
                    if not isinstance(error, dict) or not isinstance(error.get("data", {}), dict):
                        return {}
                    if not isinstance(message.get("time"), dict):
                        return {}
                    return message
    except (sqlite3.Error, json.JSONDecodeError, TypeError):
        return {}
    return {}


def episode_failure_kind(message: str) -> str | None:
    """Identify agent failures without treating missing inputs or judge failures as answers."""
    if "OpenCode failed (timeout=True," in message:
        return "agent_timeout"
    if any(
        marker in message
        for marker in (
            "OpenCode failed (timeout=False,",
            "OpenCode did not produce its session database",
            "OpenCode did not produce its native session database",
            "OpenCode recorded a model/runtime error",
            "Missing assistant turns or exceeded max_turns",
            "Unfinished OpenCode assistant turn",
            "Expected exactly one root OpenCode session",
            "OpenCode dropped initial images",
        )
    ):
        return "agent_run_error"
    return None
