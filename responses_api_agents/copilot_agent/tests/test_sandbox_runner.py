# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path


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
