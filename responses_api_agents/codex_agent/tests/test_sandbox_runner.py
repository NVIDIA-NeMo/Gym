# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Linux supervisor, not a mocked cleanup acknowledgement."""

import json
import os
import runpy
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemo_gym.agents import SUPERVISOR_SCRIPT
from responses_api_agents.codex_agent import sandbox_runner


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision is required")


def launch(tmp_path, code, timeout=3, python=sys.executable, **request_fields):
    request = {
        "directory": str(tmp_path),
        "command": [python, "-c", code],
        "cwd": str(tmp_path),
        "env": {},
        "timeout": timeout,
        "cleanup_timeout": 2,
        "prompt": "task input",
        **request_fields,
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(request))
    process = subprocess.Popen(
        [
            "bash",
            str(SUPERVISOR_SCRIPT),
            "--timeout",
            str(timeout),
            "--cleanup-timeout",
            "0.5",
            "--stop-file",
            str(tmp_path / "stop.request"),
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
        process = launch(tmp_path, code, timeout=1 if ending == "timeout" else 3)
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


@pytest.mark.parametrize(("start", "expected"), [("0", "1000"), ("500", "1000"), ("-997", "0"), ("-1000", "0")])
def test_prefer_for_oom_kill_raises_the_score_and_never_lowers_it(tmp_path, start, expected):
    path = tmp_path / "oom_score_adj"
    path.write_text(start + "\n")
    assert sandbox_runner.prefer_for_oom_kill(str(path)) == f"{start} -> {expected}"
    assert path.read_text() == expected
    assert sandbox_runner.prefer_for_oom_kill(str(tmp_path / "missing")).startswith("unchanged")


@pytest.mark.parametrize("raise_score", [False, True])
def test_codex_and_its_tools_inherit_the_raised_oom_score(tmp_path, raise_score):
    start = int(open("/proc/self/oom_score_adj").read())
    # Stands in for Codex: a tool it starts reports the score it inherited.
    code = (
        "import subprocess,sys; subprocess.run([sys.executable,'-c',"
        "\"import pathlib; pathlib.Path('score').write_text(open('/proc/self/oom_score_adj').read())\"],check=True)"
    )
    process = launch(tmp_path, code, raise_oom_score_adj=raise_score)
    summary = result(tmp_path, process)
    assert summary["return_code"] == 0
    assert int((tmp_path / "score").read_text()) == ((0 if start < 0 else 1000) if raise_score else start)


def test_spawn_error_is_not_success(tmp_path):
    process = launch(tmp_path, "raise RuntimeError('failed')")
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] != 0


def test_stop_marker_prevents_process_launch(tmp_path):
    (tmp_path / "stop.request").touch()
    process = launch(tmp_path, "open('should-not-exist', 'w').close()")
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] != 0
    assert not (tmp_path / "should-not-exist").exists()


def test_runtime_environment_does_not_inherit_provider_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_PROVIDER_API_KEY", "must-not-inherit")
    task_environment = {"HOME": str(tmp_path / "task-home"), "XDG_DATA_HOME": str(tmp_path / "task-data")}
    for key, value in task_environment.items():
        monkeypatch.setenv(key, value)
    process = launch(
        tmp_path,
        "import os,json; print(json.dumps({key:os.environ.get(key) "
        "for key in ('SANDBOX_PROVIDER_API_KEY', 'HOME', 'XDG_DATA_HOME')}))",
    )
    assert result(tmp_path, process)["cleanup_confirmed"]
    _, event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event == {"SANDBOX_PROVIDER_API_KEY": None, **task_environment}


def test_inherited_environment_is_opt_in_and_keeps_explicit_values(tmp_path, monkeypatch):
    monkeypatch.setenv("TASK_VARIABLE", "from-task-image")
    monkeypatch.setenv("HOME", "/root")
    code = (
        "import os,json; print(json.dumps({key: os.environ.get(key) "
        "for key in ('TASK_VARIABLE', 'HOME', 'CODEX_HOME', 'PATH')}))"
    )
    process = launch(tmp_path, code, inherit_env=True, env={"CODEX_HOME": "/tmp/session/.codex"})
    assert result(tmp_path, process)["cleanup_confirmed"]
    _, event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event == {
        "TASK_VARIABLE": "from-task-image",
        "HOME": "/root",
        "CODEX_HOME": "/tmp/session/.codex",
        "PATH": os.environ["PATH"],
    }


@pytest.mark.skipif(shutil.which("python3.8") is None, reason="Python 3.8 is not installed")
def test_python38_worker_captures_output_and_preserves_task_children(tmp_path):
    try:
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
        pid = int((tmp_path / "child.pid").read_text())
        assert Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    finally:
        if (tmp_path / "child.pid").exists():
            try:
                os.kill(int((tmp_path / "child.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_snapshot_retains_native_session_before_cleanup(tmp_path):
    sessions = tmp_path / "home/.codex/sessions/2026/10/19"
    sessions.mkdir(parents=True)
    records = '{"type":"response_item","payload":{"call_id":"native-call"}}\n'
    (sessions / "session.jsonl").write_text(records)
    (sessions / "second.jsonl").write_text(records)
    subprocess.run(
        [sys.executable, "-I", sandbox_runner.__file__, "--snapshot", str(tmp_path)],
        check=True,
        capture_output=True,
        timeout=10,
    )
    assert (tmp_path / "native_session.jsonl").read_text().splitlines() == records.splitlines() * 2


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("ending", ["success", "failure", "spawn_error"])
def test_history_transport_wiring_and_cleanup(tmp_path, monkeypatch, enabled, ending):
    gym_url = "http://gym/ng-rollout/test/v1"
    helper_url = "http://127.0.0.1:19001/v1"
    observation_url = "http://127.0.0.1:19002/v1"
    lifecycle = []

    @contextmanager
    def history_transport(base_url, *, timeout):
        lifecycle.append(("history_enter", base_url, timeout))
        try:
            yield helper_url
        finally:
            lifecycle.append(("history_exit",))

    @contextmanager
    def observation_proxy(base_url, invocation_id, output, *, timeout):
        lifecycle.append(("observation_enter", base_url, invocation_id, output, timeout))
        try:
            yield observation_url
        finally:
            lifecycle.append(("observation_exit",))

    def load_runner_file(path):
        name = Path(path).name
        if name == "model_history_transport.py":
            return {"HistoryTransport": history_transport}
        assert name == "sandbox_transport.py"
        return {"rate_limit_proxy": observation_proxy}

    load = Mock(side_effect=load_runner_file)
    monkeypatch.setattr(runpy, "run_path", load)
    monkeypatch.setattr(sandbox_runner.signal, "signal", Mock())
    command = ["node", "codex.js", "exec", "--json", "--", "-"]
    params = {
        "directory": str(tmp_path),
        "command": command.copy(),
        "model_base_url": gym_url,
        "invocation_id": "invocation",
        "timeout": 5,
        "prompt": "task input",
        "cwd": str(tmp_path),
        "env": {"TASK_SETTING": "preserved"},
    }
    if enabled:
        params["history_transport"] = {"base_url": gym_url, "timeout": 5}
    return_code = 0 if ending == "success" else 1
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b'{"type":"test"}\n')
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as stdout:
        child = SimpleNamespace(stdout=stdout, poll=lambda: return_code, wait=lambda: return_code)
        spawn = Mock(return_value=child)
        if ending == "spawn_error":
            spawn.side_effect = OSError("cannot start CLI")
        monkeypatch.setattr(sandbox_runner.subprocess, "Popen", spawn)
        if ending == "spawn_error":
            with pytest.raises(OSError, match="cannot start CLI"):
                sandbox_runner.run(params)
        else:
            assert sandbox_runner.run(params) == return_code
            _, event = json.loads((tmp_path / "events.jsonl").read_text())
            assert event == {"type": "test"}

    expected = []
    if enabled:
        expected.append(("history_enter", gym_url, 5))
    expected.extend(
        [
            (
                "observation_enter",
                helper_url if enabled else gym_url,
                "invocation",
                tmp_path / "model_attempts.jsonl",
                5,
            ),
            ("observation_exit",),
        ]
    )
    if enabled:
        expected.append(("history_exit",))
    assert lifecycle == expected
    assert [Path(call.args[0]).name for call in load.call_args_list] == (
        ["model_history_transport.py", "sandbox_transport.py"] if enabled else ["sandbox_transport.py"]
    )
    assert spawn.call_args.args[0] == [
        *command[:2],
        "-c",
        "model_providers.gym.base_url=" + json.dumps(observation_url),
        *command[2:],
    ]
    assert spawn.call_args.kwargs["env"]["TASK_SETTING"] == "preserved"
    assert spawn.call_args.kwargs["cwd"] == str(tmp_path)
    assert params["command"] == command
