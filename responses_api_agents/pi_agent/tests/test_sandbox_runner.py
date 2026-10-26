# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Linux supervisor, not a mocked cleanup acknowledgement."""

import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym.agent_utils import process_supervisor
from responses_api_agents.pi_agent import sandbox_runner


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision is required")


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
def test_python38_preserves_environment_and_task_child(tmp_path, monkeypatch):
    try:
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
        pid = int((tmp_path / "child.pid").read_text())
        assert Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    finally:
        if (tmp_path / "child.pid").exists():
            try:
                os.kill(int((tmp_path / "child.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("ending", ["natural", "timeout", "cancel"])
def test_task_descendants_survive_until_owner_teardown(tmp_path, ending):
    try:
        # A completed tool leaves its detached service adopted before close.
        tool = (
            "import pathlib,subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
            "start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
            "pathlib.Path('child.pid').write_text(str(p.pid))"
        )
        code = (
            "import subprocess,sys,time,pathlib; "
            f"subprocess.run([sys.executable,'-c',{tool!r}],check=True); "
            "pathlib.Path('ready').touch(); " + ("time.sleep(60)" if ending != "natural" else "pass")
        )
        # Allow the nested interpreters to start before testing harness termination,
        # including when subprocess coverage adds interpreter startup overhead.
        process = launch(tmp_path, code, timeout=3)
        if ending == "cancel":
            for _ in range(200):
                if (tmp_path / "ready").exists():
                    break
                time.sleep(0.01)
            process.send_signal(signal.SIGTERM)
        summary = result(tmp_path, process)
        assert summary["cleanup_confirmed"] is True
        assert summary["timed_out"] is (ending == "timeout")
        pid = int((tmp_path / "child.pid").read_text())
        assert Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    finally:
        if (tmp_path / "child.pid").exists():
            try:
                os.kill(int((tmp_path / "child.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_spawn_error_is_not_success(tmp_path):
    process = launch(tmp_path, "raise RuntimeError('failed')")
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] != 0


def test_last_lines_are_kept_without_waiting_for_a_detached_tool(tmp_path, monkeypatch):
    """Pi's last lines, such as its final ng_pi_outcome, can arrive just after a select that found nothing, and a
    tool Pi started can hold its stdout open after Pi exits."""
    started, waited = [], []
    popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        started.append(popen(*args, **kwargs))
        return started[-1]

    class LateSelector(selectors.DefaultSelector):
        def select(self, timeout=None):
            if timeout and not waited:
                # This wait ends empty, and only then does Pi write its last line and exit.
                (tmp_path / "go").touch()
                waited.append(started[0].wait(timeout=30))
                return []
            return super().select(timeout)

    pi = (
        "import json, os, pathlib, subprocess, sys, time\n"
        "deadline = time.monotonic() + 10\n"
        "while not os.path.exists('go') and time.monotonic() < deadline: time.sleep(0.005)\n"
        "tool = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'], start_new_session=True)\n"
        "pathlib.Path('tool.pid').write_text(str(tool.pid))\n"
        "print(json.dumps({'type': 'ng_pi_outcome'}))\n"
    )
    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    monkeypatch.setattr(selectors, "DefaultSelector", LateSelector)
    handler = signal.getsignal(signal.SIGTERM)
    began = time.monotonic()
    try:
        exit_code = sandbox_runner.run(
            {
                "directory": str(tmp_path),
                "command": [sys.executable, "-c", pi],
                "cwd": str(tmp_path),
                "env": {},
                "prompt": "",
            }
        )
        elapsed = time.monotonic() - began
    finally:
        signal.signal(signal.SIGTERM, handler)
        if (tmp_path / "tool.pid").exists():
            try:
                os.kill(int((tmp_path / "tool.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
    assert exit_code == 0
    events = [json.loads(line)[1] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events == [{"type": "ng_pi_outcome"}]
    # The tool still holds the pipe, so waiting for its end would take the tool's 10 s.
    assert elapsed < 5
