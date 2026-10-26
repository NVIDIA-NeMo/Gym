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


def test_streaming_events_are_dropped_and_logs_go_to_the_given_paths(tmp_path):
    # Pi's JSON mode repeats the partial message on every streaming delta; only completed events are kept.
    lines = [
        {"type": "message_start", "message": {"role": "assistant"}},
        {"type": "message_update", "message": {"role": "assistant", "content": "partial"}},
        {"type": "tool_execution_update", "toolCallId": "t", "partialResult": "out"},
        # outcome.mjs writes its record to stdout from Pi's message_end handler, ahead of Pi's own line.
        {"type": "ng_pi_outcome", "context_overflow": False},
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}},
    ]
    code = (
        "import json,sys; "
        f"[print(json.dumps(line), flush=True) for line in {lines!r}]; "
        "print('not json'); sys.stderr.write('pi stderr\\n')"
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    return_code = sandbox_runner.run(
        {
            "directory": str(tmp_path),
            "command": [sys.executable, "-c", code],
            "cwd": str(tmp_path),
            "env": {},
            "prompt": "task input",
            "transcript": str(logs / "pi.txt"),
            "stderr": str(logs / "pi_stderr.txt"),
        }
    )
    assert return_code == 0
    recorded = [json.loads(line)[1]["type"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert recorded == ["message_start", "ng_pi_outcome", "message_end"]
    transcript = (logs / "pi.txt").read_text().splitlines()
    assert [json.loads(line)["type"] for line in transcript[:3]] == ["message_start", "ng_pi_outcome", "message_end"]
    assert transcript[3:] == ["not json"]
    assert (logs / "pi_stderr.txt").read_text() == "pi stderr\n"
    assert not (tmp_path / "stderr.log").exists()


def test_oom_score_is_raised_for_pi_and_its_tools(tmp_path):
    score = tmp_path / "oom_score_adj"
    score.write_text("0\n")
    assert sandbox_runner.prefer_for_oom_kill(str(score)) == "0 -> 1000"
    assert score.read_text() == "1000"
    score.write_text("-997\n")
    assert sandbox_runner.prefer_for_oom_kill(str(score)) == "-997 -> 0"
    assert sandbox_runner.prefer_for_oom_kill(str(tmp_path / "missing")).startswith("unchanged:")
    code = "import sys; print(open('/proc/self/oom_score_adj').read().strip(), file=sys.stderr)"
    params = {
        "directory": str(tmp_path),
        "command": [sys.executable, "-c", code],
        "cwd": str(tmp_path),
        "env": {},
        "prompt": "",
        "raise_oom_score_adj": True,
    }
    before = Path("/proc/self/oom_score_adj").read_text().strip()
    try:
        assert sandbox_runner.run(params) == 0
    finally:
        # The test process is the runner here: restore what the runner raised (lowering may need privilege).
        try:
            Path("/proc/self/oom_score_adj").write_text(before)
        except OSError:
            pass
    record, child = (tmp_path / "stderr.log").read_text().splitlines()
    # The runner records what it changed, and Pi (here the child) inherits the score.
    assert record.startswith("oom_score_adj: ")
    if record.endswith("-> 1000"):
        assert child == "1000"
