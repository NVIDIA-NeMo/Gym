# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from responses_api_agents.claude_code_agent import sandbox_runner
from responses_api_agents.claude_code_agent.parsing import _invocation_outcome, parse_stream_json


def runner_input(tmp_path, code):
    child = tmp_path / "child.py"
    child.write_text(code)
    payload = {"directory": str(tmp_path), "command": [sys.executable, str(child)], "cwd": str(tmp_path), "env": {}}
    params = tmp_path / "input.json"
    params.write_text(json.dumps(payload))
    return [sys.executable, str(Path(sandbox_runner.__file__)), str(params)]


def test_large_json_stdout_cannot_be_corrupted_by_interleaved_stderr(tmp_path):
    command = runner_input(
        tmp_path,
        """import json,os
text="x"*300000
line=json.dumps({"type":"assistant","message":{"content":[{"type":"text","text":text}]}}).encode()+b"\\n"
os.write(1,line[:150000])
os.write(2,b'{"type":"result","subtype":"error_fake"}\\n'+b'diagnostic\\xff\\n')
os.write(1,line[150000:])
os.write(1,b'{"type":"result","subtype":"success","is_error":false}\\n')
""",
    )
    result = subprocess.run(command, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    raw = (tmp_path / "events.jsonl").read_text()
    assert "error_fake" not in raw
    output, metadata = parse_stream_json(raw)
    assert output[0].content[0].text == "x" * 300000
    assert _invocation_outcome(metadata, result.returncode) == ("completed", None)
    assert b"diagnostic\xff" in (tmp_path / "stderr.log").read_bytes()
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["hostname"] and runtime["pid"] > 0


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX termination")
def test_termination_preserves_partial_events(tmp_path):
    command = runner_input(
        tmp_path,
        """import os,time
os.write(1,b'{"type":"assistant","message":{"content":[{"type":"text","text":"partial"}]}}\\n')
time.sleep(60)
""",
    )
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "events.jsonl").exists() or (tmp_path / "events.jsonl").stat().st_size == 0:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.terminate()
        process.communicate(timeout=5)
        output, metadata = parse_stream_json((tmp_path / "events.jsonl").read_text())
        assert output[0].content[0].text == "partial"
        assert "subtype" not in metadata
        assert process.returncode != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
