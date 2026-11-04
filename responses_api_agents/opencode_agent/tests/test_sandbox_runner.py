# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the staged OpenCode activation through the bash runner and supervisor, and export its SQLite store."""

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.agents import HARNESS_RUNNER_FILE, HARNESS_RUNNER_SCRIPT, SUPERVISOR_SCRIPT
from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxSession
from responses_api_agents.opencode_agent.snapshot import snapshot


linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision required")
PRIVATE_KEYS = ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")


class LocalSandbox:
    """File transfer onto this machine, where the staged runner then executes."""

    async def upload(self, source, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


async def stage(tmp_path: Path, command: list[str], *, env: dict[str, str] | None = None) -> list[str]:
    """Stage an activation with the adapter, then return its argv with the runner it uploaded at seed time."""
    shutil.copyfile(HARNESS_RUNNER_SCRIPT, tmp_path / HARNESS_RUNNER_FILE)
    state = SimpleNamespace(
        session=SimpleNamespace(sandbox=LocalSandbox(), session_dir=str(tmp_path)),
        persistent_directory=str(tmp_path),
        is_continuation=False,
    )
    payload = {
        "instructions": "",
        "cwd": str(tmp_path),
        "command": command,
        "prompt": "task input",
        "env": {key: str(tmp_path / "private" / key) for key in PRIVATE_KEYS} | (env or {}),
    }
    return (await OpenCodeSandboxSession.stage_activation(state, payload)).argv


def supervise(tmp_path: Path, argv: list[str], *, timeout: float = 3) -> subprocess.Popen:
    return subprocess.Popen(
        ["bash", str(SUPERVISOR_SCRIPT), "--timeout", str(timeout), "--cleanup-timeout", "0.5"]
        + ["--receipt", str(tmp_path / "cleanup.json"), "--stop-file", str(tmp_path / "stop.request"), "--", *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def result(tmp_path, process):
    try:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
        return json.loads((tmp_path / "cleanup.json").read_text())
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


async def test_runner_records_task_directories_before_private_overrides(tmp_path):
    task_home = tmp_path / 'task "home" \\ dir'
    (tmp_path / "task-environment.ready").write_text("ready\n")
    argv = await stage(
        tmp_path,
        ["bash", "-c", 'cat; printf "|%s|%s|%s" "$HOME" "$HARNESS_INHERITED" "$HARNESS_OVERRIDE"'],
        env={"HARNESS_OVERRIDE": "new"},
    )
    inherited = {key: value for key, value in os.environ.items() if not key.startswith("XDG_")}
    env = inherited | {
        "HOME": f"{task_home}/",
        "XDG_CONFIG_HOME": "/task/config",
        "XDG_CACHE_HOME": "",
        "HARNESS_INHERITED": "kept",
        "HARNESS_OVERRIDE": "old",
    }
    completed = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    private = tmp_path / "private"
    assert (tmp_path / "stdout.jsonl").read_text() == f"task input|{private / 'HOME'}|kept|new"
    assert json.loads((tmp_path / "task-env.json").read_text()) == {
        "HOME": f"{task_home}/",
        "XDG_DATA_HOME": f"{task_home}/.local/share",
        "XDG_CONFIG_HOME": "/task/config",
        "XDG_CACHE_HOME": f"{task_home}/.cache",
        "XDG_STATE_HOME": f"{task_home}/.local/state",
    }
    # Never accept a receipt left by an earlier invocation.
    assert not (tmp_path / "task-environment.ready").exists()
    assert all((private / key).is_dir() for key in PRIVATE_KEYS)
    assert set(json.loads((tmp_path / "runtime.json").read_text())) == {"hostname", "pid"}


async def test_unset_home_falls_back_to_the_account_home(tmp_path):
    argv = await stage(tmp_path, ["true"])
    env = {key: value for key, value in os.environ.items() if key != "HOME" and not key.startswith("XDG_")}
    completed = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    home = os.path.expanduser("~")
    assert json.loads((tmp_path / "task-env.json").read_text())["XDG_STATE_HOME"] == f"{home}/.local/state"


@linux_only
@pytest.mark.parametrize("ending", ["natural", "timeout", "cancel"])
async def test_task_descendants_survive_until_owner_teardown(tmp_path, ending):
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
        process = supervise(tmp_path, await stage(tmp_path, [sys.executable, "-c", code]))
        if ending == "cancel":
            for _ in range(200):
                if (tmp_path / "ready").exists():
                    break
                time.sleep(0.01)
            (tmp_path / "stop.request").touch()
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
async def test_cli_failure_is_reported_in_the_receipt(tmp_path):
    summary = result(
        tmp_path, supervise(tmp_path, await stage(tmp_path, [sys.executable, "-c", "raise SystemExit(3)"]))
    )
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] == 3


@linux_only
async def test_stop_marker_fences_launch_without_inventing_worker_identity(tmp_path):
    (tmp_path / "stop.request").touch()
    argv = await stage(tmp_path, [sys.executable, "-c", "open('unexpected', 'w').close()"])
    summary = result(tmp_path, supervise(tmp_path, argv))
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] is None
    assert not (tmp_path / "runtime.json").exists()
    assert not (tmp_path / "unexpected").exists()


def session_store(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
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
    return con


@pytest.mark.parametrize(
    "copied", [("opencode.db", "opencode.db-wal"), ("opencode.db", "opencode.db-wal", "opencode.db-shm")]
)
def test_snapshot_of_a_copied_store_keeps_root_output_and_child_usage(tmp_path, copied):
    sandbox = tmp_path / "sandbox"
    con = session_store(sandbox / "opencode.db")
    # The CLI was stopped without a checkpoint: its latest messages are only in the write-ahead log.
    assert (sandbox / "opencode.db-wal").stat().st_size > 0
    local = tmp_path / "local"
    local.mkdir()
    for name in copied:
        shutil.copyfile(sandbox / name, local / name)
    con.close()
    snapshot(local / "opencode.db", local, harness_session_id=None, previous_message_ids=[])
    export = json.loads((local / "export.json").read_text())
    assert export["session_id"] == "root"
    assert export["message_ids"] == export["activation_message_ids"] == ["0", "1"]
    assert len(export["messages"]) == 1
    assert export["messages"][0]["parts"][0]["text"] == "root"
    assert [info["tokens"]["input"] for info in export["usage_messages"]] == [1, 2]
    with sqlite3.connect(local / "observations.db") as copy:
        assert copy.execute("select count(*) from session").fetchone()[0] == 2


def test_failed_observation_copy_keeps_the_export(tmp_path):
    con = session_store(tmp_path / "opencode.db")
    (tmp_path / "observations.db").mkdir()  # SQLite cannot open the destination
    snapshot(tmp_path / "opencode.db", tmp_path, harness_session_id=None, previous_message_ids=[])
    con.close()
    assert json.loads((tmp_path / "export.json").read_text())["session_id"] == "root"
