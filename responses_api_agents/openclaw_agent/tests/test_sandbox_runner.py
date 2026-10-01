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

from responses_api_agents.openclaw_agent import sandbox_runner


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and /proc are required")


def launch(tmp_path, code, timeout=3):
    request = {
        "directory": str(tmp_path),
        "command": [sys.executable, "-c", code],
        "cwd": str(tmp_path),
        "env": {},
        "timeout": timeout,
        "cleanup_timeout": 2,
        "prompt": "task input",
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(request))
    process = subprocess.Popen(
        [sys.executable, "-I", str(Path(sandbox_runner.__file__)), str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


def result(tmp_path, process):
    try:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
        return json.loads((tmp_path / "result.json").read_text())
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_capture_and_prompt_file(tmp_path):
    process = launch(
        tmp_path, "import json,pathlib; print(json.dumps({'prompt':pathlib.Path('prompt.txt').read_text()}))"
    )
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] == 0
    event = json.loads((tmp_path / "stdout.log").read_text())
    assert event["prompt"] == "task input"


@pytest.mark.parametrize("ending", ["natural", "timeout", "cancel"])
def test_detached_descendants_are_gone_before_receipt(tmp_path, ending):
    # The grandchild starts a new session, escaping process-group-only cleanup.
    code = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        "pathlib.Path('child.pid').write_text(str(p.pid)); " + ("time.sleep(60)" if ending != "natural" else "pass")
    )
    process = launch(tmp_path, code, timeout=0.3 if ending == "timeout" else 3)
    if ending == "cancel":
        for _ in range(200):
            if (tmp_path / "child.pid").exists():
                break
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["timed_out"] is (ending != "natural")
    pid = int((tmp_path / "child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_spawn_error_is_not_success(tmp_path):
    process = launch(tmp_path, "raise RuntimeError('failed')")
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] != 0


@pytest.mark.skipif(shutil.which("python3.8") is None, reason="Python 3.8 is not installed")
def test_python38_preserves_environment_and_reaps_detached_child(tmp_path: Path) -> None:
    python = shutil.which("python3.8")
    code = (
        "import json,os,pathlib,subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        "pathlib.Path('child.pid').write_text(str(p.pid)); "
        "print(json.dumps([os.environ['OPENCLAW_TEST_INHERITED'],os.environ['OPENCLAW_TEST_OVERRIDE']]))"
    )
    request = {
        "directory": str(tmp_path),
        "command": [python, "-c", code],
        "cwd": str(tmp_path),
        "env": {"OPENCLAW_TEST_OVERRIDE": "new"},
        "timeout": 3,
        "cleanup_timeout": 2,
        "prompt": "task input",
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(request))
    process = subprocess.Popen(
        [python, "-I", str(Path(sandbox_runner.__file__)), str(path)],
        env={**os.environ, "OPENCLAW_TEST_INHERITED": "kept", "OPENCLAW_TEST_OVERRIDE": "old"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] == 0
    assert json.loads((tmp_path / "stdout.log").read_text()) == ["kept", "new"]
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "child.pid").read_text()), 0)


def test_stop_marker_prevents_process_launch(tmp_path):
    (tmp_path / "runner.stop").touch()
    process = launch(tmp_path, "open('should-not-exist', 'w').close()")
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] != 0
    assert not (tmp_path / "should-not-exist").exists()


def test_sigterm_during_spawn_does_not_lose_child_handle(tmp_path):
    # Deliver SIGTERM after the real child exists but before Popen returns to run().
    # Isolate signal handlers/subreaper state from pytest, and always reap the test child.
    driver = """
import json, os, runpy, signal, subprocess, sys
runner = runpy.run_path(sys.argv[1])
spawn = subprocess.Popen
children = []
def interrupted_spawn(*args, **kwargs):
    child = spawn(*args, **kwargs)
    children.append(child)
    os.kill(os.getpid(), signal.SIGTERM)
    return child
subprocess.Popen = interrupted_spawn
try:
    summary = runner['run']({
        'directory': sys.argv[2], 'cwd': sys.argv[2], 'env': {}, 'prompt': 'task',
        'command': [sys.executable, '-c', 'import time; time.sleep(60)'],
        'timeout': 5, 'cleanup_timeout': 2,
    })
    summary['child_alive'] = any(child.poll() is None for child in children)
    print(json.dumps(summary))
finally:
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait()
"""
    completed = subprocess.run(
        [sys.executable, "-c", driver, sandbox_runner.__file__, str(tmp_path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
        check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["timed_out"] is True
    assert summary["cleanup_confirmed"] is True
    assert summary["child_alive"] is False
