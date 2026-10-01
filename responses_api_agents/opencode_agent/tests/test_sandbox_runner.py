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

from responses_api_agents.opencode_agent import sandbox_runner


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


def test_capture_and_stdin(tmp_path):
    process = launch(tmp_path, "import json,sys; print(json.dumps({'type':'test', 'prompt':sys.stdin.read()}))")
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] == 0
    event = json.loads((tmp_path / "stdout.jsonl").read_text())
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


@pytest.mark.parametrize("interruption", ["signal", "exception"])
def test_interruption_during_spawn_does_not_leak_child(tmp_path, interruption):
    # Interrupt after the real child exists but before Popen returns to run().
    # Isolate signal handlers/subreaper state from pytest, and always reap the test child.
    driver = """
import json, os, runpy, signal, subprocess, sys
runner = runpy.run_path(sys.argv[1])
spawn = subprocess.Popen
children = []
def interrupted_spawn(*args, **kwargs):
    child = spawn(*args, **kwargs)
    children.append(child)
    if sys.argv[3] == 'exception':
        raise RuntimeError('lost launch handle')
    os.kill(os.getpid(), signal.SIGTERM)
    return child
subprocess.Popen = interrupted_spawn
try:
    summary = runner['run']({
        'directory': sys.argv[2], 'cwd': sys.argv[2], 'env': {}, 'prompt': 'task',
        'command': [sys.executable, '-c', 'import time; time.sleep(60)'],
        'timeout': 5, 'cleanup_timeout': 2,
    })
    summary['child_alive'] = False
    for child in children:
        try:
            os.kill(child.pid, 0)
            summary['child_alive'] = True
        except ProcessLookupError:
            pass
    print(json.dumps(summary))
finally:
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait()
"""
    completed = subprocess.run(
        [sys.executable, "-c", driver, sandbox_runner.__file__, str(tmp_path), interruption],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
        check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["timed_out"] is (interruption == "signal")
    if interruption == "exception":
        assert summary["error"] == "lost launch handle"
    assert summary["cleanup_confirmed"] is True
    assert summary["child_alive"] is False


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


@pytest.mark.parametrize(
    "python",
    [
        sys.executable,
        pytest.param(
            shutil.which("python3.8"),
            marks=pytest.mark.skipif(shutil.which("python3.8") is None, reason="Python 3.8 is not installed"),
            id="python38",
        ),
    ],
)
def test_snapshot_runs_after_detached_descendants_are_reaped(tmp_path, python):
    import sqlite3

    database = tmp_path / "data/opencode/opencode.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("pragma journal_mode=wal")
        connection.executescript("""
            create table session(id text, parent_id text, time_created integer);
            create table message(id text, session_id text, data text, time_created integer);
            create table part(id text, message_id text, session_id text, data text, time_created integer);
            insert into session values('root', null, 0);
            insert into message values('m1', 'root', '{"role":"assistant"}', 0);
            insert into part values('p1', 'm1', 'root', '{"type":"text","text":"saved"}', 0);
        """)
    # Assert the ordering at the actual snapshot boundary, not just after run().
    driver = """
import json, os, pathlib, runpy, sys
runner = runpy.run_path(sys.argv[1])
snapshot = runner["snapshot"]
def checked_snapshot(directory):
    pid = int((directory / "child.pid").read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("snapshot began before descendant cleanup")
    snapshot(directory)
runner["run"].__globals__["snapshot"] = checked_snapshot
code = (
    "import json,os,pathlib,subprocess,sys; "
    "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
    "pathlib.Path('child.pid').write_text(str(child.pid)); "
    "print(json.dumps([sys.stdin.read(),os.environ['OPENCODE_TEST_VALUE']]))"
)
result = runner["run"]({
    "directory": sys.argv[2], "cwd": sys.argv[2], "prompt": "task input",
    "command": [sys.executable, "-c", code], "env": {"OPENCODE_TEST_VALUE": "override"},
    "timeout": 3, "cleanup_timeout": 2,
})
print(json.dumps(result))
"""
    completed = subprocess.run(
        [python, "-I", "-c", driver, sandbox_runner.__file__, str(tmp_path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
        check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] == 0
    assert summary["error"] is None
    assert json.loads((tmp_path / "stdout.jsonl").read_text()) == ["task input", "override"]
    export = json.loads((tmp_path / "export.json").read_text())
    assert export["messages"][0]["parts"][0]["text"] == "saved"
    with sqlite3.connect(tmp_path / "observations.db") as copy:
        assert copy.execute("select count(*) from session").fetchone()[0] == 1
