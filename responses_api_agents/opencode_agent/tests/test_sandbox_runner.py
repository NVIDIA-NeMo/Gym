# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test worker I/O separately from the shared Linux supervisor's cleanup contract."""

import errno
import hashlib
import json
import os
import posixpath
import shutil
import signal
import sqlite3
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from nemo_gym.agent_utils import process_supervisor
from responses_api_agents.opencode_agent import sandbox_runner


linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision required")


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
        process = launch(tmp_path, code)
        if ending == "cancel":
            for _ in range(200):
                if (tmp_path / "ready").exists():
                    break
                time.sleep(0.01)
            process.send_signal(signal.SIGTERM)
        summary = result(tmp_path, process)
        assert summary["cleanup_confirmed"] is True
        assert summary["timed_out"] is (ending == "timeout")
        assert set(summary) == {"return_code", "timed_out", "cleanup_confirmed", "error"}
        pid = int((tmp_path / "child.pid").read_text())
        assert Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    finally:
        if (tmp_path / "child.pid").exists():
            try:
                os.kill(int((tmp_path / "child.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


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


SESSIONS = ["ses_root", "ses_child", "ses_grandchild"]


def export_fixture(tmp_path, *, activation_id=None, failing=(), sleep=0.0, extra_session=None, extra_root=None):
    """A stored session tree, the run's input and a stand-in CLI that logs how ``export`` is invoked."""
    import sqlite3

    session = tmp_path / "session"
    database = session / "data/opencode/opencode.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("create table session(id text, parent_id text, time_created integer)")
        rows = [("ses_root", None, 0), ("ses_child", "ses_root", 1), ("ses_grandchild", "ses_child", 2)]
        if extra_session is not None:
            rows.append((extra_session, "ses_root", 3))
        if extra_root is not None:
            rows.append((extra_root, None, 4))
        connection.executemany("insert into session values(?, ?, ?)", rows)
    workdir = tmp_path / "task"
    workdir.mkdir()
    calls = tmp_path / "calls.jsonl"
    cli = tmp_path / "opencode"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        f"with open({str(calls)!r}, 'a') as log:\n"
        "    log.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'home': os.environ.get('HOME'),\n"
        "        'data': os.environ.get('XDG_DATA_HOME'), 'config': 'OPENCODE_CONFIG_CONTENT' in os.environ,\n"
        "        'project_config': os.environ.get('OPENCODE_DISABLE_PROJECT_CONFIG')}) + '\\n')\n"
        f"if sys.argv[2] in {list(failing)!r}:\n"
        "    sys.stderr.write('Session not found: ' + sys.argv[2])\n"
        "    sys.exit(1)\n"
        f"time.sleep({sleep})\n"
        "print(json.dumps({'info': {'id': sys.argv[2]}, 'messages': []}, indent=2))\n"
    )
    cli.chmod(0o755)
    env = {
        "HOME": str(session / "home"),
        "XDG_DATA_HOME": str(session / "data"),
        "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
        "OPENCODE_CONFIG_CONTENT": json.dumps({"plugin": ["file:///owner/plugin.mjs"]}),
    }
    (session / "input.json").write_text(
        json.dumps(
            {
                "command": [str(cli), "run", "--format", "json"],
                "cwd": str(workdir),
                "env": env,
                "activation_id": activation_id,
            }
        )
    )
    (session / "stdout.jsonl").write_text('{"type": "text"}\n')
    return session, workdir, calls


def export_all(session, exports, *, timeout=30):
    return sandbox_runner.export_sessions(
        directory=session, session_directory=session, export_dir=exports, timeout=timeout
    )


def run_export(session, exports, *, timeout=30):
    """Run ``--export`` as the adapter does, returning the process and the status line from its stdout."""
    finished = subprocess.run(
        [sys.executable, "-I", sandbox_runner.__file__, "--export", str(session), str(session), str(exports)]
        + [str(timeout)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = finished.stdout.splitlines()
    assert len(lines) == 1, finished.stdout
    return finished, json.loads(lines[0])


def test_export_writes_every_session_and_the_event_log_without_reloading_plugins(tmp_path):
    session, workdir, calls = export_fixture(tmp_path)
    exports = tmp_path / "exports"
    status = export_all(session, exports)
    assert sandbox_runner.export_failures(status) == []
    assert status["sessions"] == {session_id: "ok" for session_id in SESSIONS}
    assert status["event_log"] == "ok" and status["error"] is None and not status["deadline_exhausted"]
    assert json.loads((exports / "sessions.json").read_text()) == [
        {"id": "ses_root", "parent_id": None},
        {"id": "ses_child", "parent_id": "ses_root"},
        {"id": "ses_grandchild", "parent_id": "ses_child"},
    ]
    for session_id in SESSIONS:
        assert json.loads((exports / f"{session_id}.json").read_text()) == {"info": {"id": session_id}, "messages": []}
    assert (exports / "ses_root.events.jsonl").read_text() == '{"type": "text"}\n'
    logged = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [call["argv"] for call in logged] == [["export", session_id] for session_id in SESSIONS]
    for call in logged:
        # Same private store and project as the run, but without its plugins.
        assert Path(call["cwd"]).resolve() == workdir.resolve()
        assert call["home"] == str(session / "home")
        assert call["data"] == str(session / "data")
        assert call["project_config"] == "true"
        assert call["config"] is False
    assert sorted(path.name for path in exports.iterdir()) == sorted(
        [
            "sessions.json",
            "ses_root.events.jsonl",
            *(f"{session_id}.json" for session_id in SESSIONS),
        ]
    )
    finished, line = run_export(session, exports)
    assert finished.returncode == 0, finished.stderr
    assert line["sessions"] == status["sessions"] and line["event_log"] == "ok"


def test_failed_session_export_is_reported_and_never_left_stale(tmp_path):
    session, _, _ = export_fixture(tmp_path, failing=["ses_child"])
    exports = tmp_path / "exports"
    exports.mkdir()
    # An earlier activation's files must not stand in for this capture.
    (exports / "ses_child.json").write_text('{"stale": true}')
    (exports / "sessions.json").write_text("[]")
    status = export_all(session, exports)
    assert sandbox_runner.export_failures(status) == ["ses_child: exit 1: Session not found: ses_child"]
    assert status["sessions"]["ses_root"] == "ok"
    assert not (exports / "ses_child.json").exists()
    assert {"ses_root.json", "ses_grandchild.json", "ses_root.events.jsonl"} <= {p.name for p in exports.iterdir()}
    assert len(json.loads((exports / "sessions.json").read_text())) == 3
    finished, line = run_export(session, exports)
    assert finished.returncode == 1
    assert "OpenCode session export failed: ses_child: exit 1" in finished.stderr
    # The outcome reaches stdout, whatever a provider keeps of stderr.
    assert line["sessions"]["ses_child"] == "exit 1: Session not found: ses_child"


def test_unreadable_store_leaves_no_inventory_and_reports_why(tmp_path):
    session, _, _ = export_fixture(tmp_path)
    (session / "data/opencode/opencode.db").unlink()
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "sessions.json").write_text("[]")
    status = export_all(session, exports)
    assert status["error"].startswith("OperationalError: unable to open database file")
    assert status["sessions"] == {} and status["event_log"] == "not attempted"
    assert not (exports / "sessions.json").exists()
    finished, line = run_export(session, exports)
    assert finished.returncode == 1
    assert "Traceback" not in finished.stderr
    assert line["error"].startswith("OperationalError")


def test_event_log_without_a_single_root_is_reported(tmp_path):
    session, _, _ = export_fixture(tmp_path, extra_root="ses_other_root")
    exports = tmp_path / "exports"
    status = export_all(session, exports)
    assert status["event_log"] == "no single root session to belong to: ['ses_root', 'ses_other_root']"
    assert sandbox_runner.export_failures(status) == [f"event log: {status['event_log']}"]
    assert not list(exports.glob("*.events.jsonl"))


def test_continuation_keeps_one_event_log_per_activation(tmp_path):
    session, _, _ = export_fixture(tmp_path, activation_id=2)
    exports = tmp_path / "exports"
    assert sandbox_runner.export_failures(export_all(session, exports)) == []
    assert (exports / "ses_root.activation-2.events.jsonl").read_text() == '{"type": "text"}\n'
    assert not (exports / "ses_root.events.jsonl").exists()


def test_unsafe_session_id_never_becomes_a_path(tmp_path):
    session, _, calls = export_fixture(tmp_path, extra_session="../escape")
    exports = tmp_path / "exports"
    assert sandbox_runner.export_failures(export_all(session, exports)) == ["../escape: unsafe session ID"]
    assert not (tmp_path / "escape.json").exists()
    assert all(json.loads(line)["argv"][1] in SESSIONS for line in calls.read_text().splitlines())


def test_export_deadline_stops_a_hanging_cli(tmp_path):
    session, _, _ = export_fixture(tmp_path, sleep=60)
    exports = tmp_path / "exports"
    started = time.monotonic()
    status = export_all(session, exports, timeout=0.5)
    assert time.monotonic() - started < 10
    failures = sandbox_runner.export_failures(status)
    assert failures[0].startswith("ses_root: Command")
    assert failures[1:] == [f"{session_id}: export deadline exceeded" for session_id in SESSIONS[1:]]
    assert status["deadline_exhausted"] is True
    assert status["budget_seconds"] == 0.5 and 0.5 <= status["elapsed_seconds"] < 10
    assert not (exports / "ses_root.json").exists()
    # The event log needs no CLI and is still kept.
    assert (exports / "ses_root.events.jsonl").exists()


# Like OpenCode on Bun: one write without waiting for a pipe to drain, then exit. A pipe keeps what fits at once
# (OpenCode's exports arrived cut to 8 KB); a regular file takes all of it.
EARLY_EXIT_EXPORT = """
import fcntl, json, os, stat, sys
regular = stat.S_ISREG(os.fstat(1).st_mode)
data = json.dumps({"info": {"id": sys.argv[2], "regular": regular}, "messages": [{"text": "x" * 300000}]})
fcntl.fcntl(1, fcntl.F_SETFL, fcntl.fcntl(1, fcntl.F_GETFL) | os.O_NONBLOCK)
try:
    os.write(1, (data + "\\n").encode())
except BlockingIOError:
    pass
"""


def test_large_export_is_complete_although_the_cli_exits_before_a_pipe_would_drain(tmp_path):
    session, _, _ = export_fixture(tmp_path)
    cli = tmp_path / "opencode"
    cli.write_text(f"#!{sys.executable}\n{EARLY_EXIT_EXPORT}")
    exports = tmp_path / "exports"
    status = export_all(session, exports)
    assert sandbox_runner.export_failures(status) == []
    for session_id in SESSIONS:
        exported = json.loads((exports / f"{session_id}.json").read_text())
        assert exported["info"] == {"id": session_id, "regular": True}
        assert len(exported["messages"][0]["text"]) == 300000
    assert not list(exports.glob(".*.tmp"))
    # Through a pipe the same CLI loses most of its output, so a runner reading a pipe would reject every export.
    piped = subprocess.run([str(cli), "export", "ses_root"], stdout=subprocess.PIPE, timeout=30)
    assert piped.returncode == 0 and len(piped.stdout) < 300000


def test_cut_or_invalid_export_names_its_size_and_leaves_no_partial_file(tmp_path):
    session, _, _ = export_fixture(tmp_path)
    cli = tmp_path / "opencode"
    cli.write_text(f'#!{sys.executable}\nimport sys\nsys.stdout.write(\'{{"info": {{"id": "cut\')\n')
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "ses_root.json").write_text('{"stale": true}')
    status = export_all(session, exports)
    assert status["sessions"]["ses_root"].startswith("export (20 bytes) is not a JSON object: Unterminated string")
    assert not (exports / "ses_root.json").exists()
    assert not list(exports.glob(".*.tmp"))


def test_cli_working_directory_overrides_an_inherited_pwd(tmp_path):
    """OpenCode reads PWD before its cwd; the runner must not pass on the launcher's directory."""
    workdir = tmp_path / "task"
    workdir.mkdir()
    params = {
        "directory": str(tmp_path),
        "command": [sys.executable, "-c", "import os, json; print(json.dumps([os.environ['PWD'], os.getcwd()]))"],
        "cwd": str(workdir),
        "env": {},
        "prompt": "",
    }
    (tmp_path / "input.json").write_text(json.dumps(params))
    finished = subprocess.run(
        [sys.executable, "-I", sandbox_runner.__file__, str(tmp_path / "input.json")],
        cwd=tmp_path,
        env=os.environ | {"PWD": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert finished.returncode == 0, finished.stderr
    pwd, cwd = json.loads((tmp_path / "stdout.jsonl").read_text())
    assert pwd == str(workdir)
    assert Path(cwd).resolve() == workdir.resolve()


@pytest.mark.parametrize("plugins", [True, False], ids=["run", "export or import"])
def test_path_prefix_goes_ahead_of_the_sandbox_path_for_every_cli_process(monkeypatch, plugins):
    """Run, export and import share one environment builder, so the configured prefix reaches all three."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    params = {"cwd": "/app", "env": {"HOME": "/session/home"}, "path_prefix": ["/opt/gym-runtime", "/opt/tools"]}
    env = sandbox_runner._child_environment(params, plugins=plugins)
    assert env["PATH"] == "/opt/gym-runtime:/opt/tools:/usr/bin:/bin"
    assert env["PWD"] == "/app" and env["HOME"] == "/session/home"
    assert sandbox_runner._child_environment({**params, "path_prefix": []}, plugins=plugins)["PATH"] == "/usr/bin:/bin"


def test_export_cli_gets_the_task_directory_as_pwd(tmp_path, monkeypatch):
    session, workdir, _ = export_fixture(tmp_path)
    probe = tmp_path / "pwd-probe"
    probe.write_text(
        f"#!{sys.executable}\nimport json, os\nprint(json.dumps({{'info': {{'pwd': os.environ['PWD']}}}}))\n"
    )
    probe.chmod(0o755)
    params = json.loads((session / "input.json").read_text())
    params["command"][0] = str(probe)
    (session / "input.json").write_text(json.dumps(params))
    monkeypatch.setenv("PWD", str(tmp_path))
    exports = tmp_path / "exports"
    assert sandbox_runner.export_failures(export_all(session, exports)) == []
    assert json.loads((exports / "ses_root.json").read_text())["info"]["pwd"] == str(workdir)


def pinned(path, data):
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_stage_executable_installs_a_checked_copy_that_runs(tmp_path):
    source = tmp_path / "mounted-rg"
    digest = pinned(source, b"#!/bin/sh\necho 'ripgrep 15.1.0'\n")
    source.chmod(0o444)
    target = tmp_path / "session/cache/opencode/bin/rg"
    sandbox_runner.stage_executable(source, digest, target)
    assert target.read_bytes() == source.read_bytes()
    assert os.access(target, os.X_OK)
    assert [path.name for path in target.parent.iterdir()] == ["rg"]


@pytest.mark.parametrize(
    ("script", "digest", "message"),
    [
        (b"#!/bin/sh\necho 'ripgrep 15.1.0'\n", "0" * 64, "SHA-256 mismatch"),
        (b"#!/bin/sh\nexit 4\n", None, "--version exited 4"),
        (b"\x7fELF not really a binary", None, "does not run"),
    ],
)
def test_stage_executable_rejects_wrong_or_broken_bytes_without_installing(tmp_path, script, digest, message):
    source = tmp_path / "rg"
    actual = pinned(source, script)
    target = tmp_path / "bin/rg"
    with pytest.raises(sandbox_runner.Rejected, match=message):
        sandbox_runner.stage_executable(source, digest or actual, target)
    assert list(target.parent.iterdir()) == []


def test_stage_modes_exit_3_when_rejected_and_1_when_the_source_is_missing(tmp_path):
    source = tmp_path / "rg"
    pinned(source, b"#!/bin/sh\necho ok\n")
    target = str(tmp_path / "bin/rg")
    for path, digest, code in ((source, "0" * 64, 3), (tmp_path / "absent", "0" * 64, 1)):
        finished = subprocess.run(
            [sys.executable, "-I", sandbox_runner.__file__, "--stage-executable", str(path), digest, target],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert finished.returncode == code, finished.stderr


def test_version_check_killed_by_a_signal_stays_retryable(tmp_path):
    source = tmp_path / "rg"
    digest = pinned(source, b"#!/bin/sh\nkill -KILL $$\n")
    target = tmp_path / "bin/rg"
    with pytest.raises(RuntimeError, match="--version was killed by signal 9"):
        sandbox_runner.stage_executable(source, digest, target)
    assert list(target.parent.iterdir()) == []
    finished = subprocess.run(
        [sys.executable, "-I", sandbox_runner.__file__, "--stage-executable", str(source), digest, str(target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert finished.returncode == 1, finished.stderr


def test_version_check_out_of_time_stays_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_runner, "_VERSION_CHECK_TIMEOUT", 0.5)
    source = tmp_path / "rg"
    digest = pinned(source, b"#!/bin/sh\nexec sleep 30\n")
    target = tmp_path / "bin/rg"
    with pytest.raises(subprocess.TimeoutExpired):
        sandbox_runner.stage_executable(source, digest, target)
    assert list(target.parent.iterdir()) == []


def test_missing_interpreter_stays_retryable(tmp_path):
    source = tmp_path / "rg"
    digest = pinned(source, b"#!/nonexistent/interpreter\n")
    with pytest.raises(FileNotFoundError):
        sandbox_runner.stage_executable(source, digest, tmp_path / "bin/rg")


@pytest.mark.parametrize(
    ("code", "rejected"),
    [
        (errno.ENOEXEC, True),
        (errno.EACCES, True),
        (errno.ENOENT, False),
        (errno.ENOMEM, False),
        (errno.ETXTBSY, False),
    ],
)
def test_only_bytes_the_kernel_refuses_to_run_are_rejected(tmp_path, monkeypatch, code, rejected):
    source = tmp_path / "rg"
    digest = pinned(source, b"#!/bin/sh\necho ok\n")

    def refused(*args, **kwargs):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(sandbox_runner.subprocess, "run", refused)
    with pytest.raises(sandbox_runner.Rejected if rejected else OSError):
        sandbox_runner.stage_executable(source, digest, tmp_path / "bin/rg")


def plugin_archive(tmp_path, *members):
    """A config-directory bundle as tar builds it from an online run: files, folders and relative links."""
    import io

    archive = tmp_path / "plugin-dependencies.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, kind, data in members:
            info = tarfile.TarInfo(name)
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                bundle.addfile(info)
            elif kind in ("symlink", "hardlink"):
                info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                info.linkname = data
                bundle.addfile(info)
            else:
                info.size = len(data)
                info.mode = 0o755
                bundle.addfile(info, io.BytesIO(data))
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def without_data_filter(monkeypatch):
    """Make tarfile look like a Python without PEP 706 (and no backport): no filter, everything trusted."""
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    if hasattr(tarfile, "fully_trusted_filter"):
        # Python 3.14 filters by default through the module's data_filter.
        monkeypatch.setattr(tarfile.TarFile, "extraction_filter", staticmethod(tarfile.fully_trusted_filter))


@pytest.fixture(params=["data filter", "no data filter"])
def extraction(request, monkeypatch):
    """Run ``stage_archive`` on this Python as it is, and as a Python without PEP 706 (no backport) sees it."""
    if request.param == "no data filter":
        without_data_filter(monkeypatch)
    elif not hasattr(tarfile, "data_filter"):
        pytest.skip("this Python has no tarfile data filter")
    return request.param


# The real archive's links (npm's node_modules/.bin), plus a scoped and a nested package.
NPM_LINKS = [
    ("node_modules/.bin/uuid", "../uuid/dist-node/bin/uuid"),
    ("node_modules/.bin/yaml", "../yaml/bin.mjs"),
    ("node_modules/.bin/node-which", "../which/bin/node-which"),
    ("node_modules/.bin/plugin-tool", "../@opencode-ai/plugin/bin/tool.js"),
    ("node_modules/zod/node_modules/.bin/semver", "../semver/bin/semver.js"),
]


def test_stage_archive_unpacks_the_plugin_dependencies_with_their_links(tmp_path, extraction):
    targets = [posixpath.normpath(posixpath.join(posixpath.dirname(name), link)) for name, link in NPM_LINKS]
    archive, digest = plugin_archive(
        tmp_path,
        ("package.json", "file", b'{"dependencies": {"@opencode-ai/plugin": "1.17.11"}}'),
        ("package-lock.json", "file", b'{"packages": {"": {"dependencies": {"@opencode-ai/plugin": "1.17.11"}}}}'),
        ("node_modules/@opencode-ai/plugin/package.json", "file", b'{"version": "1.17.11"}'),
        *((path, "file", b"#!/usr/bin/env node\n") for path in targets),
        ("node_modules/.bin", "dir", b""),
        *((name, "symlink", link) for name, link in NPM_LINKS),
        ("node_modules/.package-lock.json", "hardlink", "package-lock.json"),
    )
    target = tmp_path / "session/config/opencode"
    sandbox_runner.stage_archive(archive, digest, target)
    assert json.loads((target / "node_modules/@opencode-ai/plugin/package.json").read_text()) == {"version": "1.17.11"}
    for (name, _), linked in zip(NPM_LINKS, targets):
        assert (target / name).is_symlink()
        assert (target / name).resolve() == (target / linked).resolve()
    assert (target / "node_modules/.package-lock.json").read_bytes() == (target / "package-lock.json").read_bytes()
    assert not list((tmp_path / "session/config").glob(".pinned-*"))


@pytest.mark.parametrize(
    "member",
    [
        ("../outside.json", "file", b"{}"),
        ("/etc/opencode.json", "file", b"{}"),
        ("node_modules/escape", "symlink", "../../../../outside"),
        ("node_modules/absolute", "symlink", "/etc/passwd"),
        ("node_modules/hard", "hardlink", "../outside.json"),
    ],
)
def test_stage_archive_refuses_members_that_leave_the_config_directory(tmp_path, extraction, member):
    archive, digest = plugin_archive(tmp_path, ("package.json", "file", b"{}"), member)
    target = tmp_path / "session/config/opencode"
    with pytest.raises(sandbox_runner.Rejected, match="leaves the target directory"):
        sandbox_runner.stage_archive(archive, digest, target)
    assert not target.exists()


# Each stays inside the target lexically; through the links it is written to, or leads to, the target's parent.
LINK_CHAINS = {
    # The review's repro: d/l is the target itself, so d/l/l2 is its parent.
    "member through links": [
        ("d", "dir", b""),
        ("d/l", "symlink", ".."),
        ("d/l/l2", "symlink", ".."),
        ("d/l/l2/escaped.txt", "file", b"outside"),
    ],
    "dot-dot after a link": [("d", "dir", b""), ("d/l", "symlink", ".."), ("d/l/../escaped.txt", "file", b"outside")],
    # Checked before d/x is a link, y lexically leads to d/escaped.txt; writing y follows d/x first.
    "link target through a later link": [
        ("y", "symlink", "d/x/../escaped.txt"),
        ("d", "dir", b""),
        ("d/x", "symlink", ".."),
        ("y", "file", b"outside"),
    ],
    "hard link through a link": [
        ("d", "dir", b""),
        ("d/l", "symlink", ".."),
        ("h", "hardlink", "d/l/../escaped.txt"),
    ],
    # Extraction cannot turn the folder d/l into a link afterwards; refusing it up front makes it a verdict.
    "member under a later link": [("d/l/f.txt", "file", b"inside"), ("d/l", "symlink", ".")],
}


@pytest.mark.parametrize("chain", LINK_CHAINS.values(), ids=LINK_CHAINS.keys())
def test_without_the_data_filter_no_path_may_pass_through_a_link(tmp_path, monkeypatch, chain):
    without_data_filter(monkeypatch)
    archive, digest = plugin_archive(tmp_path, *chain)
    target = tmp_path / "session/config/opencode"
    with pytest.raises(sandbox_runner.Rejected, match="passes through the link 'd/"):
        sandbox_runner.stage_archive(archive, digest, target)
    assert not target.exists()
    assert not (target.parent / "escaped.txt").exists()


@pytest.mark.parametrize("chain", LINK_CHAINS.values(), ids=LINK_CHAINS.keys())
def test_link_chains_never_write_outside_the_target(tmp_path, extraction, chain):
    archive, digest = plugin_archive(tmp_path, *chain)
    target = tmp_path / "session/config/opencode"
    try:
        sandbox_runner.stage_archive(archive, digest, target)
    except Exception:
        pass  # Refused, or (with the filter) failed while unpacking: either way nothing may land outside.
    assert not (target.parent / "escaped.txt").exists()


def test_link_chain_without_the_data_filter_is_a_deterministic_rejection(tmp_path):
    """The CLI exits 3 (HTTP 422), run as an interpreter without PEP 706 runs it."""
    archive, digest = plugin_archive(tmp_path, *LINK_CHAINS["member through links"])
    target = tmp_path / "session/config/opencode"
    argv = [sandbox_runner.__file__, "--stage-archive", str(archive), digest, str(target)]
    without_filter = (
        "import runpy, sys, tarfile\n"
        "if hasattr(tarfile, 'data_filter'):\n"
        "    del tarfile.data_filter\n"
        f"sys.argv = {argv!r}\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    finished = subprocess.run([sys.executable, "-I", "-c", without_filter], capture_output=True, text=True, timeout=60)
    assert finished.returncode == 3, finished.stderr
    assert "passes through the link 'd/l'" in finished.stderr
    assert not (target.parent / "escaped.txt").exists()


def import_fixture(tmp_path, *, exit_code=0, roots=1, prelude=""):
    """A stand-in CLI whose ``import`` stores a prefix session, logging the directory it ran from."""
    session = tmp_path / "session"
    workdir = tmp_path / "task"
    workdir.mkdir()
    calls = tmp_path / "import-calls.jsonl"
    cli = tmp_path / "opencode"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sqlite3, sys\n"
        f"{prelude}\n"
        f"with open({str(calls)!r}, 'a') as log:\n"
        "    log.write(json.dumps({'argv': sys.argv[1:], 'pwd': os.environ.get('PWD'), 'cwd': os.getcwd(),\n"
        "        'config': 'OPENCODE_CONFIG_CONTENT' in os.environ}) + '\\n')\n"
        f"if {exit_code}:\n"
        "    sys.stderr.write('invalid export')\n"
        f"    sys.exit({exit_code})\n"
        "store = os.path.join(os.environ['XDG_DATA_HOME'], 'opencode')\n"
        "os.makedirs(store, exist_ok=True)\n"
        "with sqlite3.connect(os.path.join(store, 'opencode.db')) as db:\n"
        "    db.execute('create table session(id text, parent_id text, time_created integer)')\n"
        "    db.execute('create table message(id text, session_id text, data text, time_created integer)')\n"
        "    db.execute('create table part(id text, message_id text, session_id text, data text, time_created integer)')\n"
        f"    for index in range({roots}):\n"
        "        db.execute('insert into session values(?, null, ?)', (f'ses_prefix{index or \"\"}', index))\n"
        "    for index, role in enumerate(['user', 'assistant']):\n"
        "        db.execute('insert into message values(?, ?, ?, ?)', (f'prefix-{index}', 'ses_prefix',\n"
        "            json.dumps({'role': role, 'time': {'completed': 1}, 'finish': 'stop'}), index))\n"
        "        db.execute('insert into part values(?, ?, ?, ?, ?)', (f'part-{index}', f'prefix-{index}', 'ses_prefix',\n"
        "            json.dumps({'type': 'text', 'text': 'Earlier ' + role}), index))\n"
        "print('Imported session: ses_prefix')\n"
    )
    cli.chmod(0o755)
    env = {key: str(session / folder) for key, folder in (("HOME", "home"), ("XDG_DATA_HOME", "data"))}
    env |= {"XDG_CONFIG_HOME": str(session / "config"), "OPENCODE_CONFIG_CONTENT": "{}"}
    params = {"command": [str(cli)], "cwd": str(workdir), "env": env}
    export = tmp_path / "prefix-export.json"
    export.write_text('{"info": {"id": "ses_prefix"}, "messages": []}')
    return session, workdir, params, export, calls


def test_import_runs_from_the_task_directory_and_reports_the_prefix(tmp_path, monkeypatch):
    session, workdir, params, export, calls = import_fixture(tmp_path)
    monkeypatch.setenv("PWD", str(tmp_path))
    imported = sandbox_runner.import_session(params, export, timeout=30)
    assert imported == {"session_id": "ses_prefix", "message_ids": ["prefix-0", "prefix-1"]}
    (call,) = [json.loads(line) for line in calls.read_text().splitlines()]
    assert call["argv"] == ["import", str(export)]
    assert call["pwd"] == str(workdir)
    assert Path(call["cwd"]).resolve() == workdir.resolve()
    assert call["config"] is False
    assert (session / "home").is_dir() and (session / "config").is_dir()


def test_import_plus_continuation_keeps_the_prefix_out_of_the_activation(tmp_path):
    session, workdir, params, export, _ = import_fixture(tmp_path)
    imported = sandbox_runner.import_session(params, export, timeout=30)
    # The continued run appends this activation's turn to the imported session.
    with sqlite3.connect(session / "data/opencode/opencode.db") as db:
        for index, role in enumerate(["user", "assistant"], start=2):
            db.execute(
                "insert into message values(?, 'ses_prefix', ?, ?)",
                (f"activation-{index}", json.dumps({"role": role, "tokens": {"input": index}}), index),
            )
            db.execute(
                "insert into part values(?, ?, 'ses_prefix', ?, ?)",
                (
                    f"part-{index}",
                    f"activation-{index}",
                    json.dumps({"type": "text", "text": "Continued " + role}),
                    index,
                ),
            )
    activation = tmp_path / "activation"
    activation.mkdir()
    run_input = {"harness_session_id": imported["session_id"], "previous_message_ids": imported["message_ids"]}
    (activation / "input.json").write_text(json.dumps(run_input))
    sandbox_runner.snapshot(activation, session_directory=session)
    snapshot = json.loads((activation / "export.json").read_text())
    assert snapshot["session_id"] == "ses_prefix"
    assert snapshot["activation_message_ids"] == ["activation-2", "activation-3"]
    assert [part["text"] for message in snapshot["messages"] for part in message["parts"]] == [
        "Continued user",
        "Continued assistant",
    ]
    assert [info["tokens"]["input"] for info in snapshot["usage_messages"]] == [2, 3]


def test_absent_import_file_runs_nothing(tmp_path):
    _, _, params, export, calls = import_fixture(tmp_path)
    export.unlink()
    assert sandbox_runner.import_session(params, export, timeout=30) is None
    assert not calls.exists()


@pytest.mark.parametrize(("exit_code", "roots", "message"), [(1, 1, "invalid export"), (0, 2, "one imported root")])
def test_unusable_import_is_rejected(tmp_path, exit_code, roots, message):
    _, _, params, export, _ = import_fixture(tmp_path, exit_code=exit_code, roots=roots)
    with pytest.raises(sandbox_runner.Rejected, match=message):
        sandbox_runner.import_session(params, export, timeout=30)
    (tmp_path / "params.json").write_text(json.dumps(params))
    finished = subprocess.run(
        [sys.executable, "-I", sandbox_runner.__file__, "--import", str(tmp_path / "params.json"), str(export), "30"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert finished.returncode == 3, finished.stderr


@pytest.mark.parametrize(
    ("prelude", "error"),
    [("os.kill(os.getpid(), 9)", RuntimeError), ("import time; time.sleep(30)", subprocess.TimeoutExpired)],
    ids=["killed", "out of time"],
)
def test_import_cut_short_stays_retryable(tmp_path, prelude, error):
    _, _, params, export, _ = import_fixture(tmp_path, prelude=prelude)
    with pytest.raises(error):
        sandbox_runner.import_session(params, export, timeout=1)
    (tmp_path / "params.json").write_text(json.dumps(params))
    finished = subprocess.run(
        [sys.executable, "-I", sandbox_runner.__file__, "--import", str(tmp_path / "params.json"), str(export), "1"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert finished.returncode == 1, finished.stderr
