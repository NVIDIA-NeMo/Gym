# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


RUNNER = Path(__file__).parents[1] / "sandbox_runner.py"


def test_runner_records_complete_events_and_drops_streaming_duplicates(tmp_path):
    fake = tmp_path / "copilot.py"
    fake.write_text(
        "import json, os, sys\n"
        "assert sys.stdin.read() == ''\n"
        "print('not json')\n"
        "for kind in ('assistant.message_delta', 'tool.shell_output', 'assistant.message', 'result'):\n"
        "    print(json.dumps({'type': kind, 'data': {'env': os.environ.get('COPILOT_MODEL')}}))\n"
        "print('stderr diagnostics', file=sys.stderr)\n"
        "raise SystemExit(4)\n"
    )
    params = {
        "directory": str(tmp_path),
        "command": [sys.executable, str(fake)],
        "cwd": str(tmp_path),
        "env": {"COPILOT_MODEL": "m"},
    }
    (tmp_path / "input.json").write_text(json.dumps(params))
    finished = subprocess.run([sys.executable, "-I", str(RUNNER), str(tmp_path / "input.json")], timeout=30)
    assert finished.returncode == 4
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["type"] for _, event in events] == ["assistant.message", "result"]
    assert all(isinstance(observed_at, float) and event["data"]["env"] == "m" for observed_at, event in events)
    assert (tmp_path / "stderr.log").read_text() == "stderr diagnostics\n"
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert set(runtime) == {"hostname", "pid"}


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        (None, None),
        ("", None),
        ('{"type":"session.start"}\n', False),
        ('{"type":"session.start"}\n{"type":"session.compaction_start"}\n', True),
        ('{"type":"session.compaction_complete","data":{"success":false}}\n', True),
        ('{"type":"session.start"}\n{"truncated":', None),
    ],
)
def test_read_persisted_compaction_state(tmp_path, contents, expected):
    if contents is not None:
        events = tmp_path / "copilot-home/session-state/session-id/events.jsonl"
        events.parent.mkdir(parents=True)
        events.write_text(contents)
    finished = subprocess.run(
        [sys.executable, "-I", str(RUNNER), "--compaction-state", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert finished.returncode == 0, finished.stderr
    assert json.loads(finished.stdout) is expected


def test_detached_tool_holding_stdout_does_not_block_the_runner(tmp_path):
    fake = tmp_path / "copilot.py"
    child_pid = tmp_path / "tool.pid"
    fake.write_text(
        "import json, subprocess, sys\n"
        "tool = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(child_pid)!r}, 'w').write(str(tool.pid))\n"
        # The final event has no trailing newline.
        "sys.stdout.write(json.dumps({'type': 'result', 'sessionId': 's'}))\n"
        "sys.stdout.flush()\n"
    )
    params = {"directory": str(tmp_path), "command": [sys.executable, str(fake)], "cwd": str(tmp_path), "env": {}}
    (tmp_path / "input.json").write_text(json.dumps(params))
    started = time.monotonic()
    try:
        finished = subprocess.run([sys.executable, "-I", str(RUNNER), str(tmp_path / "input.json")], timeout=30)
        elapsed = time.monotonic() - started
    finally:
        if child_pid.exists():
            try:
                os.kill(int(child_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
    assert finished.returncode == 0
    assert elapsed < 20, f"runner waited {elapsed:.1f}s for a detached tool's stdout"
    captured = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(event["type"], event["sessionId"]) for _, event in captured] == [("result", "s")]
