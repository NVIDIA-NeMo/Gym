# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session export and import run their shell steps for real here: a host shell stands in for the sandbox's."""

import asyncio
import json
import os
import shutil
import signal
import sqlite3
import sys
from pathlib import Path
from time import monotonic

import pytest

from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.base_responses_api_agent import AgentSeedRejectedError
from nemo_gym.sandbox.providers.base import SandboxExecResult
from responses_api_agents.opencode_agent.app import _OPENCODE_ISOLATION, _private_directories
from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxSession, cli_environment
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import seed, setup  # noqa: F401


class HostSandbox:
    """Run sandbox commands with this host's ``sh``; files move by copying.

    ``streams`` mimics what a provider keeps of a command's output: both streams, stderr merged into stdout with only
    the exit status as stderr (OpenSandbox's background exec), or no stderr at all. A timeout ends the command's whole
    process group, as providers do.
    """

    def __init__(self, streams="separate"):
        self.streams = streams

    async def exec(self, command, *, cwd=None, timeout_s=30):
        stderr_to = asyncio.subprocess.STDOUT if self.streams == "merged" else asyncio.subprocess.PIPE
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr_to,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        except BaseException as error:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            if isinstance(error, TimeoutError):
                return SandboxExecResult(None, f"timed out after {timeout_s:g}s", -1, error_type="timeout")
            raise
        stderr = (stderr or b"").decode(errors="replace")
        if self.streams == "merged":
            stderr = f"exit status {process.returncode}" if process.returncode else ""
        elif self.streams == "dropped":
            stderr = ""
        return SandboxExecResult(stdout.decode(errors="replace"), stderr, process.returncode)

    async def upload(self, source, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    async def download(self, source, destination):
        shutil.copyfile(source, destination)


def opencode_cli(runtime, *, failing=(), hanging=(), output=None, import_code=0, imported_roots=1):
    """A stand-in for ``<runtime>/opencode`` that logs each call; ``export`` and ``import`` act on the private store."""
    calls = runtime / "calls.jsonl"
    cli = runtime / "opencode"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sqlite3, sys, time\n"
        f"with open({str(calls)!r}, 'a') as log:\n"
        "    keys = ('HOME', 'XDG_DATA_HOME', 'PWD', 'PATH', 'OPENCODE_CONFIG_CONTENT', 'OPENCODE_DISABLE_PROJECT_CONFIG')\n"
        "    log.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), **{k: os.environ.get(k) for k in keys}}))\n"
        "    log.write('\\n')\n"
        "if sys.argv[1] == 'export':\n"
        f"    if sys.argv[2] in {list(failing)!r}:\n"
        "        sys.exit('Session not found: ' + sys.argv[2])\n"
        f"    if sys.argv[2] in {list(hanging)!r}:\n"
        "        time.sleep(30)\n"
        f"    print({output!r} if {output is not None} else json.dumps({{'info': {{'id': sys.argv[2]}}, 'messages': []}}))\n"
        "elif sys.argv[1] == 'import':\n"
        f"    if {import_code}:\n"
        f"        sys.exit({import_code})\n"
        "    export = json.load(open(sys.argv[2]))\n"
        "    store = os.path.join(os.environ['XDG_DATA_HOME'], 'opencode/opencode.db')\n"
        "    os.makedirs(os.path.dirname(store), exist_ok=True)\n"
        "    with sqlite3.connect(store) as connection:\n"
        "        connection.execute('create table session(id text, parent_id text, time_created integer)')\n"
        "        connection.execute('create table message(id text, session_id text, data text, time_created integer)')\n"
        f"        for number in range({imported_roots}):\n"
        "            connection.execute('insert into session values (?, null, ?)', (export['info']['id'] + str(number or ''), number))\n"
        "        for number, message in enumerate(export['messages']):\n"
        "            connection.execute('insert into message values (?, ?, ?, ?)', (message['id'], export['info']['id'], '{}', number))\n"
    )
    cli.chmod(0o755)
    return calls


def logged(calls):
    return [json.loads(line) for line in calls.read_text().splitlines()]


SESSIONS = [("ses_root", None), ("ses_child", "ses_root"), ("ses_grandchild", "ses_child")]


def host_session(tmp_path, *, streams="separate", sessions=SESSIONS, activation_id=None, path_prefix=()):
    """A finished session on this host whose private store holds ``sessions`` and whose run left an event log."""
    conversation = tmp_path / "session"
    store = conversation / "data/opencode/opencode.db"
    store.parent.mkdir(parents=True)
    with sqlite3.connect(store) as connection:
        connection.execute("create table session(id text, parent_id text, time_created integer)")
        connection.executemany(
            "insert into session values(?, ?, ?)", [(id_, parent, n) for n, (id_, parent) in enumerate(sessions)]
        )
    workdir = tmp_path / "task"
    workdir.mkdir()
    (conversation / "stdout.jsonl").write_text('{"type": "step_start"}\n')
    session = SandboxSession(
        sandbox=HostSandbox(streams), session_dir=str(conversation), workdir=str(workdir), harness="OpenCode"
    )
    session.cleanup = {"return_code": 0, "timed_out": False, "cleanup_confirmed": True, "error": None}
    state = OpenCodeSandboxSession(
        request=seed(),
        session=session,
        runtime=str(tmp_path),
        export_dir=str(tmp_path / "owner exports"),
        activation_id=activation_id,
        opencode_environment=cli_environment(
            workdir=str(workdir),
            env={**_private_directories(str(conversation)), **_OPENCODE_ISOLATION},
            path_prefix=list(path_prefix),
        ),
    )
    return state, tmp_path / "owner exports"


async def export(state, tmp_path, *, budget=30):
    """Write the exports as the capture does; the store is copied out of the sandbox first, as it is there."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(exist_ok=True)
    await asyncio.wait_for(state.write_exports(snapshot, deadline=monotonic() + budget), budget + 10)
    return state.take_export_gaps()


async def test_export_writes_every_session_and_the_event_log_without_reloading_plugins(tmp_path, monkeypatch):
    # An inherited configuration must not load the run's plugins again.
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps({"plugin": ["file:///owner/plugin.mjs"]}))
    calls = opencode_cli(tmp_path)
    state, exports = host_session(tmp_path, path_prefix=["/opt/gym runtime/bin"])
    assert await export(state, tmp_path) == []
    assert json.loads((exports / "sessions.json").read_text()) == [
        {"id": "ses_root", "parent_id": None},
        {"id": "ses_child", "parent_id": "ses_root"},
        {"id": "ses_grandchild", "parent_id": "ses_child"},
    ]
    for session_id, _ in SESSIONS:
        assert json.loads((exports / f"{session_id}.json").read_text()) == {"info": {"id": session_id}, "messages": []}
    assert (exports / "ses_root.events.jsonl").read_text() == '{"type": "step_start"}\n'
    assert sorted(path.name for path in exports.iterdir()) == sorted(
        ["sessions.json", "ses_root.events.jsonl", *(f"{session_id}.json" for session_id, _ in SESSIONS)]
    )
    workdir, conversation = tmp_path / "task", tmp_path / "session"
    for call, (session_id, _) in zip(logged(calls), SESSIONS, strict=True):
        # The run's private store and project, but not its plugins.
        assert call["argv"] == ["export", session_id]
        assert Path(call["cwd"]).resolve() == workdir.resolve() and call["PWD"] == str(workdir)
        assert call["HOME"] == f"{conversation}/home" and call["XDG_DATA_HOME"] == f"{conversation}/data"
        assert call["OPENCODE_DISABLE_PROJECT_CONFIG"] == "true" and call["OPENCODE_CONFIG_CONTENT"] is None
        assert call["PATH"].startswith("/opt/gym runtime/bin:")


@pytest.mark.parametrize("streams", ["separate", "merged", "dropped"])
async def test_failed_session_reaches_the_gap_whatever_the_provider_keeps_of_stderr(tmp_path, streams):
    opencode_cli(tmp_path, failing=["ses_child"])
    state, exports = host_session(tmp_path, streams=streams)
    # A copy from an earlier activation must not outlive a failed export.
    exports.mkdir()
    (exports / "ses_child.json").write_text('{"stale": true}')
    (gap,) = await export(state, tmp_path)
    assert gap.code == "export_dir_write_failed"
    assert gap.detail.startswith("2 of 3 sessions exported in ")
    assert gap.detail.endswith("; ses_child: exit 1: Session not found: ses_child")
    assert not (exports / "ses_child.json").exists()
    assert sorted(path.name for path in exports.iterdir()) == [
        "ses_grandchild.json",
        "ses_root.events.jsonl",
        "ses_root.json",
        "sessions.json",
    ]


async def test_an_export_that_is_not_a_json_object_is_not_kept(tmp_path):
    opencode_cli(tmp_path, output="Session not found")
    state, exports = host_session(tmp_path, sessions=[("ses_root", None)])
    (gap,) = await export(state, tmp_path)
    assert "ses_root: export (18 bytes) is not a JSON object" in gap.detail
    assert sorted(path.name for path in exports.iterdir()) == ["ses_root.events.jsonl", "sessions.json"]


async def test_a_large_export_is_written_whole(tmp_path):
    output = json.dumps({"info": {"id": "ses_root"}, "messages": [{"text": "x" * 1000}] * 2000})
    opencode_cli(tmp_path, output=output)
    state, exports = host_session(tmp_path, sessions=[("ses_root", None)])
    assert await export(state, tmp_path) == []
    assert (exports / "ses_root.json").read_text() == output + "\n"


async def test_an_unsafe_session_id_never_becomes_a_path(tmp_path):
    calls = opencode_cli(tmp_path)
    state, exports = host_session(tmp_path, sessions=[("ses_root", None), ("../escape", "ses_root")])
    (gap,) = await export(state, tmp_path)
    assert gap.detail.endswith("; ../escape: unsafe session ID")
    assert [call["argv"] for call in logged(calls)] == [["export", "ses_root"]]
    assert not (tmp_path / "escape.json").exists()


async def test_an_event_log_without_a_single_root_is_reported(tmp_path):
    opencode_cli(tmp_path)
    state, exports = host_session(tmp_path, sessions=[("ses_a", None), ("ses_b", None)])
    (gap,) = await export(state, tmp_path)
    assert gap.detail.endswith("; event log: no single root session to belong to: ['ses_a', 'ses_b']")
    assert (exports / "ses_a.json").is_file() and (exports / "ses_b.json").is_file()


async def test_continuation_keeps_one_event_log_per_activation(tmp_path):
    opencode_cli(tmp_path)
    state, exports = host_session(tmp_path, sessions=[("ses_root", None)], activation_id=2)
    assert await export(state, tmp_path) == []
    assert (exports / "ses_root.activation-2.events.jsonl").read_text() == '{"type": "step_start"}\n'


async def test_a_run_without_an_event_log_is_not_a_failure(tmp_path):
    opencode_cli(tmp_path)
    state, exports = host_session(tmp_path, sessions=[("ses_root", None)])
    (tmp_path / "session/stdout.jsonl").unlink()
    assert await export(state, tmp_path) == []
    assert sorted(path.name for path in exports.iterdir()) == ["ses_root.json", "sessions.json"]


async def test_the_deadline_stops_a_hanging_cli_and_reports_each_session(tmp_path):
    opencode_cli(tmp_path, hanging=["ses_child"])
    state, exports = host_session(tmp_path)
    started = monotonic()
    (gap,) = await export(state, tmp_path, budget=3)
    assert monotonic() - started < 10
    assert gap.detail.startswith("1 of 3 sessions exported in ") and ", deadline exhausted; " in gap.detail
    assert "ses_child: export deadline exceeded" in gap.detail
    assert "ses_grandchild: export deadline exceeded" in gap.detail
    assert not any(path.name.startswith(".") for path in exports.iterdir())


async def test_an_unreadable_store_leaves_no_inventory_and_reports_why(tmp_path):
    opencode_cli(tmp_path)
    state, exports = host_session(tmp_path)
    exports.mkdir()
    (exports / "sessions.json").write_text("[]")
    (tmp_path / "session/data/opencode/opencode.db").write_bytes(b"not a database")
    (gap,) = await export(state, tmp_path)
    assert "DatabaseError: file is not a database" in gap.detail
    assert gap.detail.startswith("0 of 0 sessions exported in ")
    assert not (exports / "sessions.json").exists()


async def test_no_export_runs_before_the_cli_is_known_to_have_stopped(tmp_path):
    calls = opencode_cli(tmp_path)
    state, exports = host_session(tmp_path)
    state.session.cleanup = {"return_code": None, "timed_out": True, "cleanup_confirmed": False, "error": "lost"}
    (gap,) = await export(state, tmp_path)
    assert "requires confirmed cleanup" in gap.detail
    assert not calls.exists() and not exports.exists()


IMPORT_EXPORT = {"info": {"id": "ses_prefix"}, "messages": [{"id": "prefix-user"}, {"id": "prefix-reply"}]}


def import_session(tmp_path, agent, **cli):
    """A seeded session on this host whose owner staged ``IMPORT_EXPORT`` for import."""
    calls = opencode_cli(tmp_path, **cli)
    state, _ = host_session(tmp_path, sessions=[])
    (tmp_path / "session/data/opencode/opencode.db").unlink()
    staged = tmp_path / "prefix-export.json"
    staged.write_text(json.dumps(IMPORT_EXPORT))
    agent.config.session_import_path = str(staged)
    return state, calls


async def test_import_runs_from_the_task_directory_and_reports_the_prefix(tmp_path, monkeypatch, setup):
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps({"plugin": ["file:///owner/plugin.mjs"]}))
    agent, _ = setup
    state, calls = import_session(tmp_path, agent)
    await agent._import_session(state)
    assert state.harness_session_id == "ses_prefix"
    assert state.message_ids == ["prefix-user", "prefix-reply"]
    (call,) = logged(calls)
    workdir, conversation = tmp_path / "task", tmp_path / "session"
    assert call["argv"] == ["import", str(tmp_path / "prefix-export.json")]
    # OpenCode binds an imported session to the importing process's directory.
    assert Path(call["cwd"]).resolve() == workdir.resolve() and call["PWD"] == str(workdir)
    assert call["XDG_DATA_HOME"] == f"{conversation}/data" and call["OPENCODE_CONFIG_CONTENT"] is None


async def test_an_absent_import_file_runs_nothing(tmp_path, setup):
    agent, _ = setup
    state, calls = import_session(tmp_path, agent)
    (tmp_path / "prefix-export.json").unlink()
    await agent._import_session(state)
    assert state.harness_session_id is None and state.message_ids == []
    assert not calls.exists()


@pytest.mark.parametrize(
    ("cli", "message"),
    [({"import_code": 1}, r"could not import session_import_path \(exit 1\)"), ({"imported_roots": 2}, "found 2")],
)
async def test_an_unusable_import_refuses_the_seed(tmp_path, setup, cli, message):
    agent, _ = setup
    state, _ = import_session(tmp_path, agent, **cli)
    with pytest.raises(AgentSeedRejectedError, match=message):
        await agent._import_session(state)


@pytest.mark.parametrize("code", [125, 126, 127, 137])
async def test_an_import_that_never_ran_or_was_killed_stays_retryable(tmp_path, setup, code):
    agent, _ = setup
    state, _ = import_session(tmp_path, agent, import_code=code)
    with pytest.raises(RuntimeError, match=f"exit={code}") as raised:
        await agent._import_session(state)
    assert not isinstance(raised.value, AgentSeedRejectedError)
