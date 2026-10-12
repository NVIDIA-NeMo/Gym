# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test worker I/O separately from the shared Linux supervisor's cleanup contract."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from nemo_gym.agent_utils import process_supervisor
from responses_api_agents.opencode_agent import sandbox_runner


linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and /proc required")


def launch(tmp_path, code, *, timeout=3, python=sys.executable, env=None, supervised=True):
    params = {
        "directory": str(tmp_path),
        "command": [python, "-c", code],
        "cwd": str(tmp_path),
        "env": env or {},
        "prompt": "task input",
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(params))
    command = [python, "-I", sandbox_runner.__file__, str(path)]
    if supervised:
        command = [
            python,
            "-I",
            process_supervisor.__file__,
            "--timeout",
            str(timeout),
            "--cleanup-timeout",
            "0.5",
            "--receipt",
            str(tmp_path / "cleanup.json"),
            "--stop-file",
            str(tmp_path / "stop.request"),
            "--",
            *command,
        ]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def result(tmp_path, process):
    try:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
        return json.loads((tmp_path / "cleanup.json").read_text())
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize(
    "python",
    [
        sys.executable,
        pytest.param(
            shutil.which("python3.8"),
            marks=pytest.mark.skipif(shutil.which("python3.8") is None, reason="Python 3.8 unavailable"),
            id="python38",
        ),
    ],
)
def test_worker_io_and_runtime_are_independent_of_cleanup(tmp_path, monkeypatch, python):
    monkeypatch.setenv("HARNESS_INHERITED", "kept")
    monkeypatch.setenv("HARNESS_OVERRIDE", "old")
    process = launch(
        tmp_path,
        "import json,os,sys,pathlib; print(json.dumps(["
        "sys.stdin.read(),os.environ['HARNESS_INHERITED'],os.environ['HARNESS_OVERRIDE']]))",
        python=python,
        env={"HARNESS_OVERRIDE": "new"},
        supervised=False,
    )
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, (stdout, stderr)
    assert json.loads((tmp_path / "stdout.jsonl").read_text()) == ["task input", "kept", "new"]
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["hostname"] == os.uname().nodename
    assert runtime["pid"] == process.pid
    assert not (tmp_path / "cleanup.json").exists()
    assert not (tmp_path / "result.json").exists()


@linux_only
@pytest.mark.parametrize("ending", ["natural", "timeout", "cancel"])
def test_detached_descendants_are_gone_before_receipt(tmp_path, ending):
    code = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        "pathlib.Path('child.pid').write_text(str(p.pid)); " + ("time.sleep(60)" if ending != "natural" else "pass")
    )
    process = launch(tmp_path, code)
    if ending == "cancel":
        for _ in range(200):
            if (tmp_path / "child.pid").exists():
                break
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["timed_out"] is (ending == "timeout")
    assert set(summary) == {"return_code", "timed_out", "cleanup_confirmed", "error"}
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "child.pid").read_text()), 0)


@linux_only
def test_spawn_error_does_not_prevent_cleanup(tmp_path):
    summary = result(tmp_path, launch(tmp_path, "raise RuntimeError('failed')"))
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] != 0


@linux_only
def test_stop_marker_fences_launch_without_inventing_worker_identity(tmp_path):
    (tmp_path / "stop.request").touch()
    summary = result(tmp_path, launch(tmp_path, "open('unexpected', 'w').close()"))
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] is None
    assert not (tmp_path / "runtime.json").exists()
    assert not (tmp_path / "unexpected").exists()


def test_snapshot_keeps_root_output_and_child_usage(tmp_path):
    import sqlite3

    database = tmp_path / "data/opencode/opencode.db"
    database.parent.mkdir(parents=True)
    con = sqlite3.connect(database)
    con.execute("pragma journal_mode=wal")
    con.executescript("""
        create table session(id text, parent_id text, time_created integer);
        create table message(id text, session_id text, data text, time_created integer);
        create table part(id text, message_id text, session_id text, data text, time_created integer);
        insert into session values('root', null, 0), ('child', 'root', 1);
    """)
    for index, session_id in enumerate(("root", "child")):
        con.execute(
            "insert into message values(?, ?, ?, ?)",
            (str(index), session_id, json.dumps({"role": "assistant", "tokens": {"input": index + 1}}), index),
        )
        con.execute(
            "insert into part values(?, ?, ?, ?, ?)",
            (str(index), str(index), session_id, json.dumps({"type": "text", "text": session_id}), index),
        )
    con.commit()
    sandbox_runner.snapshot(tmp_path)
    export = json.loads((tmp_path / "export.json").read_text())
    assert len(export["messages"]) == 1
    assert export["messages"][0]["parts"][0]["text"] == "root"
    assert [info["tokens"]["input"] for info in export["usage_messages"]] == [1, 2]
    with sqlite3.connect(tmp_path / "observations.db") as copy:
        assert copy.execute("select count(*) from session").fetchone()[0] == 2
    con.close()
