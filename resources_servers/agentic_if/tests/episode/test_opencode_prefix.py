# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay helpers: which recorded commands are re-run, and the workspace diff a prefix recorded."""

import pytest

from resources_servers.agentic_if.episode.opencode_prefix import (
    ReplayError,
    recorded_commands,
    recorded_workspace_diff,
)


def tool(name: str, state: dict) -> dict:
    return {"type": "tool", "tool": name, "callID": f"c-{name}", "state": state}


def export(*parts: dict) -> dict:
    return {
        "info": {"id": "ses_1", "directory": "/workspace/repo"},
        "messages": [
            {"info": {"id": "m1", "role": "user"}, "parts": [{"type": "text", "text": "Fix it."}]},
            {"info": {"id": "m2", "role": "assistant"}, "parts": list(parts)},
        ],
    }


BASH = tool("bash", {"status": "completed", "input": {"command": "pytest -q"}, "output": "1 passed"})
EDIT = tool("edit", {"status": "completed", "input": {"filePath": "/workspace/repo/a.py"}, "output": "ok"})


def test_recorded_commands_keep_order_directory_and_skip_failed_calls() -> None:
    failed = tool("bash", {"status": "error", "input": {"command": "false"}})
    elsewhere = tool("bash", {"status": "completed", "input": {"command": "ls", "workdir": "/tmp"}, "output": "x"})
    commands = recorded_commands(export(BASH, failed, EDIT, elsewhere), default_workdir="/workspace/repo")
    assert [(c.command, c.workdir) for c in commands] == [("pytest -q", "/workspace/repo"), ("ls", "/tmp")]


@pytest.mark.parametrize("arguments, seconds", [({}, 120), ({"timeout": 2500}, 2.5), ({"timeout": 1}, 0.001)])
def test_bash_replay_keeps_its_command_deadline(arguments: dict, seconds: float) -> None:
    # OpenCode 1.17.11's default is 120000ms, not the episode's multi-hour limit.
    part = tool("bash", {"status": "completed", "input": {"command": "sleep 999", **arguments}, "output": ""})
    (command,) = recorded_commands(export(part), default_workdir="/workspace/repo")
    assert command.timeout_s == seconds


@pytest.mark.parametrize("timeout", [None, 0, -1, True, "120000", float("nan"), float("inf")])
def test_bash_replay_rejects_invalid_deadlines(timeout: object) -> None:
    part = tool("bash", {"status": "completed", "input": {"command": "sleep 999", "timeout": timeout}})
    with pytest.raises(ReplayError, match="timeout"):
        recorded_commands(export(part), default_workdir="/workspace/repo")


def stamped(name: str, status: str, diff: str | None) -> dict:
    stamp = {"profile_sha256": "h1"}
    if diff is not None:
        stamp["workspace_diff"] = diff
    meta = {"nemo_gym_profile": stamp} if status == "completed" else {}
    return tool(name, {"status": status, "input": {"command": "x"}, "output": "", "metadata": meta})


def session(*steps: list[dict]) -> dict:
    messages = [{"info": {"id": "u1", "role": "user"}, "parts": [{"type": "text", "text": "Fix it."}]}]
    for i, parts in enumerate(steps):
        messages.append({"info": {"id": f"a{i}", "role": "assistant"}, "parts": parts})
    return {"info": {"id": "ses_1", "directory": "/w"}, "messages": messages}


def test_recorded_diff_is_the_last_record_unless_a_later_file_change_has_none() -> None:
    export = session(
        [stamped("edit", "completed", "diff-1")],
        [stamped("read", "completed", None)],
        [stamped("bash", "completed", "diff-2")],
    )
    assert recorded_workspace_diff(export) == "diff-2"
    assert recorded_workspace_diff(session([stamped("read", "completed", None)])) is None
    failed = session([stamped("edit", "completed", "diff-1")], [stamped("bash", "error", None)])
    with pytest.raises(ReplayError, match="no record of its own"):
        recorded_workspace_diff(failed)
