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

import pytest

from nemo_gym.sandbox import process_supervisor
from responses_api_agents.codex_agent import sandbox_runner


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and /proc are required")


def launch(tmp_path, code, timeout=3, python=sys.executable):
    request = {
        "directory": str(tmp_path),
        "command": [python, "-c", code],
        "cwd": str(tmp_path),
        "env": {},
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
            "--stop-file",
            str(tmp_path / "runner.stop"),
            "--receipt",
            str(tmp_path / "cleanup.json"),
            "--",
            python,
            "-I",
            sandbox_runner.__file__,
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
    process = launch(tmp_path, "import json,sys; print(json.dumps({'type':'test', 'prompt':sys.stdin.read()}))")
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] == 0
    recorded_at, event = json.loads((tmp_path / "events.jsonl").read_text())
    assert recorded_at > 0
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
    summary = runner['_supervise']([sys.executable, '-c', 'import time; time.sleep(60)'], timeout=5, cleanup_timeout=2)
    summary['child_alive'] = any(child.poll() is None for child in children)
    print(json.dumps(summary))
finally:
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait()
"""
    completed = subprocess.run(
        [sys.executable, "-c", driver, process_supervisor.__file__, str(tmp_path)],
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


def test_runtime_environment_does_not_inherit_provider_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_PROVIDER_API_KEY", "must-not-inherit")
    process = launch(
        tmp_path, "import os,json; print(json.dumps({'credential':os.environ.get('SANDBOX_PROVIDER_API_KEY')}))"
    )
    assert result(tmp_path, process)["cleanup_confirmed"]
    _, event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["credential"] is None


def test_spawned_child_is_reaped_when_popen_loses_handle(tmp_path):
    # A constructor failure after process creation must still drain the subreaper's children.
    driver = """
import json, os, runpy, subprocess, sys
runner = runpy.run_path(sys.argv[1])
spawn = subprocess.Popen
children = []
def lost_handle(*args, **kwargs):
    child = spawn(*args, **kwargs)
    children.append(child)
    raise OSError('launch handle lost after process creation')
subprocess.Popen = lost_handle
try:
    summary = runner['_supervise']([sys.executable, '-c', 'import time; time.sleep(60)'], timeout=5, cleanup_timeout=2)
    try:
        os.kill(children[0].pid, 0)
        summary['child_alive'] = True
    except ProcessLookupError:
        summary['child_alive'] = False
    print(json.dumps(summary))
finally:
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait()
"""
    completed = subprocess.run(
        [sys.executable, "-c", driver, process_supervisor.__file__, str(tmp_path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
        check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["return_code"] != 0
    assert summary["error"] == "launch handle lost after process creation"
    assert summary["cleanup_confirmed"] is True
    assert summary["child_alive"] is False


@pytest.mark.skipif(shutil.which("python3.8") is None, reason="Python 3.8 is not installed")
def test_python38_worker_captures_output_and_reaps_detached_children(tmp_path):
    code = (
        "import json,pathlib,subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        "pathlib.Path('child.pid').write_text(str(child.pid)); "
        "print(json.dumps({'prompt':sys.stdin.read()}))"
    )
    process = launch(tmp_path, code, python=shutil.which("python3.8"))
    receipt = result(tmp_path, process)
    assert receipt["cleanup_confirmed"] is True
    assert receipt["return_code"] == 0
    _, event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["prompt"] == "task input"
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "child.pid").read_text()), 0)
