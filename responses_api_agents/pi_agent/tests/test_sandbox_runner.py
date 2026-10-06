# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Linux supervisor, not a mocked cleanup acknowledgement."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym.agent_utils import process_supervisor
from responses_api_agents.pi_agent import sandbox_runner


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and /proc are required")


def launch(tmp_path, code, timeout=3, python=sys.executable, env=None):
    request = {
        "directory": str(tmp_path),
        "command": [python, "-c", code],
        "cwd": str(tmp_path),
        "env": env or {},
        "timeout": timeout,
        "cleanup_timeout": 2,
        "prompt": "task input",
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(request))
    process = subprocess.Popen(
        [
            python,
            "-I",
            process_supervisor.__file__,
            "--timeout",
            str(timeout),
            "--cleanup-timeout",
            "0.5",
            "--receipt",
            str(tmp_path / "cleanup.json"),
            "--",
            python,
            "-I",
            str(Path(sandbox_runner.__file__)),
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


def result(tmp_path, process):
    try:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
        return json.loads((tmp_path / "cleanup.json").read_text())
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_capture_and_stdin(tmp_path):
    process = launch(
        tmp_path,
        "import json,os,sys; os.write(1,b'\\xff\\n'); print('[]'); "
        "sys.stdout.write(json.dumps({'type':'test', 'prompt':sys.stdin.read()}))",
    )
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] == 0
    recorded_at, event = json.loads((tmp_path / "events.jsonl").read_text())
    assert recorded_at > 0
    assert event["prompt"] == "task input"


@pytest.mark.skipif(shutil.which("python3.8") is None, reason="Python 3.8 is not installed")
def test_python38_preserves_environment_and_reaps_detached_child(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_TEST_INHERITED", "inherited")
    monkeypatch.setenv("PI_TEST_OVERRIDE", "old")
    code = (
        "import json,os,pathlib,subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        "pathlib.Path('child.pid').write_text(str(p.pid)); "
        "print(json.dumps({'inherited':os.environ['PI_TEST_INHERITED'],"
        "'override':os.environ['PI_TEST_OVERRIDE'],'prompt':sys.stdin.read()}))"
    )
    process = launch(tmp_path, code, python=shutil.which("python3.8"), env={"PI_TEST_OVERRIDE": "new"})
    summary = result(tmp_path, process)
    assert summary["return_code"] == 0, summary
    assert summary["cleanup_confirmed"] is True
    _, event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event == {"inherited": "inherited", "override": "new", "prompt": "task input"}
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "child.pid").read_text()), 0)


@pytest.mark.parametrize("ending", ["natural", "timeout", "cancel"])
def test_detached_descendants_are_gone_before_receipt(tmp_path, ending):
    # The grandchild starts a new session, escaping process-group-only cleanup.
    code = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        "pathlib.Path('child.pid').write_text(str(p.pid)); " + ("time.sleep(60)" if ending != "natural" else "pass")
    )
    # Allow the nested interpreters to start before testing descendant cleanup,
    # including when subprocess coverage adds interpreter startup overhead.
    process = launch(tmp_path, code, timeout=3)
    if ending == "cancel":
        for _ in range(200):
            if (tmp_path / "child.pid").exists():
                break
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["timed_out"] is (ending == "timeout")
    pid = int((tmp_path / "child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_spawn_error_is_not_success(tmp_path):
    process = launch(tmp_path, "raise RuntimeError('failed')")
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] != 0
