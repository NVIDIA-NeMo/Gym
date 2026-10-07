# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture OpenCode output; the shared process supervisor owns deadlines and cleanup."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import TypedDict


class RunnerInput(TypedDict):
    """Harness input, independent of supervisor cleanup and runtime metadata."""

    directory: str
    prompt: str
    command: list[str]
    cwd: str
    env: dict[str, str]


def run(params: RunnerInput) -> int:
    """Run the harness with isolated files and let the supervisor reap descendants."""
    directory = Path(params["directory"])
    (directory / "runtime.json").write_text(
        json.dumps({"hostname": os.uname().nodename, "pid": os.getpid(), "python": sys.executable})
    )
    for key in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
        if key in params["env"]:
            Path(params["env"][key]).mkdir(parents=True, exist_ok=True)
    (directory / "prompt.txt").write_text(params["prompt"])
    stopping = False

    def interrupt(*_: object) -> None:
        nonlocal stopping
        # Defer TERM until Popen returns so a signal cannot lose the child handle.
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    with ExitStack() as stack:
        stdout = stack.enter_context((directory / "stdout.jsonl").open("wb"))
        stderr = stack.enter_context((directory / "stderr.log").open("wb"))
        stdin = stack.enter_context((directory / "prompt.txt").open("rb"))
        process = subprocess.Popen(
            params["command"],
            cwd=params["cwd"],
            env={**os.environ, **params["env"]},
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
        while True:
            if stopping:
                process.terminate()
                stopping = False
            try:
                return process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass


def snapshot(directory: Path, *, session_directory: Path | None = None) -> None:
    """Export root messages and retain the full session tree for observations."""
    session_directory = session_directory or directory
    database = session_directory / "data/opencode/opencode.db"
    input_path = directory / "input.json"
    params = json.loads(input_path.read_text()) if input_path.exists() else {}
    previous_ids = set(params.get("previous_message_ids", []))
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as source:
        with sqlite3.connect(directory / "observations.db") as destination:
            source.backup(destination)
        source.row_factory = sqlite3.Row
        sessions = source.execute(
            "select id from session where parent_id is null order by time_created, id"
        ).fetchall()
        if len(sessions) != 1:
            raise RuntimeError(f"Expected one OpenCode root session, found {len(sessions)}")
        rows = source.execute(
            "select id, data from message where session_id=? order by time_created, id", (sessions[0]["id"],)
        ).fetchall()
        session_id = sessions[0]["id"]
        expected_id = params.get("native_session_id")
        if expected_id is not None and expected_id != session_id:
            raise RuntimeError(f"OpenCode resumed unexpected session {session_id!r}")
        messages = []
        for row in rows:
            if row["id"] in previous_ids:
                continue
            info = json.loads(row["data"])
            parts = [
                json.loads(part[0])
                for part in source.execute(
                    "select data from part where message_id=? order by time_created, id", (row["id"],)
                )
            ]
            messages.append({"id": row["id"], "info": info, "parts": parts})
        all_messages = source.execute("select id, data from message order by time_created, id").fetchall()
        usage_messages = [json.loads(row["data"]) for row in all_messages if row["id"] not in previous_ids]
        (directory / "export.json").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "message_ids": [row["id"] for row in all_messages],
                    "activation_message_ids": [row["id"] for row in all_messages if row["id"] not in previous_ids],
                    "messages": messages,
                    "usage_messages": usage_messages,
                }
            )
        )


def main() -> None:
    if sys.argv[1] == "--snapshot":
        snapshot(Path(sys.argv[2]), session_directory=Path(sys.argv[3]) if len(sys.argv) > 3 else None)
        return
    params = json.loads(Path(sys.argv[1]).read_text())
    raise SystemExit(run(params))


if __name__ == "__main__":
    main()
