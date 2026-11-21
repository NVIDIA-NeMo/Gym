# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode sessions with partial-rollout checkpoints: pause in place, the stop fallback, the reprompt after a crash.

The harness conformance kit drives the real agent, the real process supervisor and the real ``reprompt_opencode.sh``
on this machine, with a fake ``opencode`` executable in place of the CLI. The fake keeps a conversation in an
OpenCode-shaped SQLite database, ``run`` answers once and then waits, and ``serve`` answers the reprompt request
``reprompt_opencode.sh`` sends, after asking for a permission when a test wants one.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

from nemo_gym.agents import (
    AGENT_ACTIVATION_HEADER,
    AGENT_SESSION_COOKIE_KEY,
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    AgentSessionConflictError,
    HarnessFixture,
    HarnessPauseResult,
    LaunchFinished,
    SandboxSession,
    check_harness_checkpointing,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    EpisodeId,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NgFailureEvent,
    TaskId,
)
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_runtime import RestoredAgentSession
from nemo_gym.trajectory import (
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    TrajectoryRecord,
    TrajectoryTurn,
)
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig, _PinnedFile, _request_digest
from responses_api_agents.opencode_agent.reprompt import interrupted_tool_calls, reprompt_decision, reprompt_payload
from responses_api_agents.opencode_agent.runtime import OPENCODE_VERSION
from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxSession, UnparsableConversationError


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision required")

REPROMPT_SCRIPT = Path(__file__).parents[1] / "reprompt_opencode.sh"

# The fake CLI. ``run`` saves the task's user turn and a first answer, then waits for the finish file and answers
# again; ``serve`` answers one reprompt request the same way. Both record their launch and their pid. With the control
# file ``ask_permission``, ``serve`` first holds the reprompt request until its permission request is replied to.
REFUSED_PROMPT = "fake opencode: refusing the prompt request"
PERMISSION = {
    "id": "per_test1",
    "sessionID": "ses_root",
    "permission": "read",
    "patterns": ["/etc/hostname"],
    "metadata": {},
    "always": [],
}
FAKE_OPENCODE = r"""#!{python}
import base64, http.server, json, os, pathlib, signal, sqlite3, sys, threading, time, urllib.parse

CONTROL = pathlib.Path({control!r})
VERSION = {version!r}
PERMISSION = {permission!r}
REFUSED_PROMPT = {refused!r}


def database():
    path = pathlib.Path(os.environ["XDG_DATA_HOME"]) / "opencode" / "opencode.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        "create table if not exists session(id text, parent_id text, time_created integer);"
        "create table if not exists message(id text, session_id text, data text, time_created integer);"
        "create table if not exists part(id text, message_id text, session_id text, data text, time_created integer);"
    )
    return connection


def add(connection, message_id, info, parts):
    info = {{"id": message_id, **info}}
    now = len(connection.execute("select id from message").fetchall()) + 1
    connection.execute("insert into message values(?, 'ses_root', ?, ?)", (message_id, json.dumps(info), now))
    for index, part in enumerate(parts):
        connection.execute(
            "insert into part values(?, ?, 'ses_root', ?, ?)", (f"{{message_id}}-{{index}}", message_id, json.dumps(part), now)
        )
    connection.commit()


def answer(connection, message_id, text, finish):
    add(
        connection,
        message_id,
        {{"role": "assistant", "time": {{"created": 1, "completed": 2}}, "finish": finish,
          "tokens": {{"input": 1, "output": 1, "reasoning": 0, "cache": {{"read": 0, "write": 0}}}}}},
        [{{"type": "text", "text": text}}],
    )


def plugin_ready():
    config = json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT", "{{}}"))
    for plugin in config.get("plugin", []):
        if plugin.endswith("task_environment.mjs"):
            path = pathlib.Path(urllib.parse.urlparse(plugin).path)
            path.with_name("task-environment.ready").write_text("ready\n")


def register():
    (CONTROL / "pids").mkdir(exist_ok=True)
    (CONTROL / "pids" / str(os.getpid())).touch()


def started(kind):
    with (CONTROL / "launches").open("a") as launches:
        launches.write(kind + "\n")


def wait_for_finish():
    while not (CONTROL / "finish").exists():
        time.sleep(0.05)


def run():
    prompt = sys.stdin.read()
    register()
    plugin_ready()
    connection = database()
    if not connection.execute("select id from session").fetchall():
        connection.execute("insert into session values('ses_root', null, 0)")
    add(connection, "msg_001", {{"role": "user", "agent": "build", "model": {{"providerID": "nemo_gym", "modelID": "dummy_model"}}}},
        [{{"type": "text", "text": prompt}}])
    answer(connection, "msg_002", "working", "tool-calls")
    print(json.dumps({{"type": "step_finish"}}), flush=True)
    # Counted as started only now: a checkpoint then finds a conversation the model has answered.
    started("run")
    wait_for_finish()
    answer(connection, "msg_003", "done", "stop")


def serve():
    register()
    started("serve")
    plugin_ready()
    password = os.environ["OPENCODE_SERVER_PASSWORD"]
    expected = "Basic " + base64.b64encode(f"opencode:{{password}}".encode()).decode()
    asked, replied = threading.Event(), threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def reply(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            if self.path.split("?")[0] != "/permission" or self.headers.get("Authorization") != expected:
                self.reply(404, {{}})
                return
            self.reply(200, [PERMISSION] if asked.is_set() and not replied.is_set() else [])

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            authorized = self.headers.get("Authorization") == expected
            if self.path.split("?")[0] == "/permission/" + PERMISSION["id"] + "/reply":
                record = {{"path": self.path, "body": body, "authorized": authorized}}
                (CONTROL / "permission_replies").open("a").write(json.dumps(record) + "\n")
                self.reply(200, True)
                replied.set()
                return
            ok = authorized and body.get("parts") == [] and "/session/ses_root/message" in self.path
            (CONTROL / "resume_requests").open("a").write(json.dumps({{"path": self.path, "body": body}}) + "\n")
            if ok:
                if (CONTROL / "ask_permission").exists():
                    asked.set()
                    replied.wait()
                    # The turn goes on for a moment after a rejection, so the helper has reported it by the end.
                    time.sleep(0.5)
                wait_for_finish()
                answer(database(), "msg_004", "reprompted and done", "stop")
            else:
                print(REFUSED_PROMPT, flush=True)
            self.reply(200 if ok else 400, {{"info": {{"id": "msg_004"}}}})

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(f"opencode server listening on http://127.0.0.1:{{server.server_address[1]}}", flush=True)
    server.serve_forever()


signal.signal(signal.SIGTERM, lambda *_: os._exit(143))
if sys.argv[1:] == ["--version"]:
    print(VERSION)
elif sys.argv[1:2] == ["run"]:
    run()
elif sys.argv[1:2] == ["serve"]:
    serve()
else:
    sys.exit(f"unexpected arguments {{sys.argv[1:]}}")
"""


def _fake_opencode(control: Path) -> str:
    return FAKE_OPENCODE.format(
        python=sys.executable,
        control=str(control),
        version=OPENCODE_VERSION,
        permission=PERMISSION,
        refused=REFUSED_PROMPT,
    )


class OpenCodeFixture(HarnessFixture):
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.control = tmp_path / "control"
        self.control.mkdir()
        (self.control / "launches").write_text("")
        self.workdir = tmp_path / "task"
        self.workdir.mkdir()
        self.binary = tmp_path / "opencode"
        self.binary.write_text(_fake_opencode(self.control))
        self.binary.chmod(0o755)
        # Optional tools the runtime installer would otherwise fetch with a package manager.
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        (self.bin / "rg").write_text("#!/bin/sh\nexit 0\n")
        (self.bin / "rg").chmod(0o755)
        # The sandbox is this machine: without this every run on the node would install its fake at one /tmp path.
        monkeypatch.setattr(
            "responses_api_agents.opencode_agent.app.SANDBOX_RUNTIME_PREFIX", str(tmp_path / "runtime-")
        )
        for name, value in (
            ("create_provider", lambda *_: self.provider),
            ("resolve_provider_config", lambda *_: {}),
            ("get_global_config_dict", lambda: {}),
        ):
            monkeypatch.setattr(f"responses_api_agents.opencode_agent.app.{name}", value)

    def make_agent(self, global_config: dict[str, Any]) -> OpenCodeAgent:
        client = MagicMock(spec=ServerClient)
        client.global_config_dict = {
            **global_config,
            "policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.1", "port": 1}}},
        }
        client._build_server_base_url = MagicMock(return_value="http://127.0.0.1:1")
        return OpenCodeAgent(
            config=OpenCodeAgentConfig(
                host="",
                port=0,
                entrypoint="",
                name="opencode",
                model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
                sandbox_executable_path=str(self.binary),
                sandbox_executable_sha256=hashlib.sha256(self.binary.read_bytes()).hexdigest(),
                sandbox_path_prefix=[str(self.bin)],
                sandbox_session_root=str(self.tmp_path / "sessions"),
                session_close_timeout_seconds=60,
                timeout=300,
            ),
            server_client=client,
        )

    def seed(self, agent_session_id: str, attempt: int) -> dict[str, Any]:
        return AgentSeedSessionRequest(
            agent_session_id=agent_session_id,
            episode_id=EpisodeId(rollout_id="r", attempt=attempt),
            task_id=TaskId(taskset="swebench", task_id="task"),
            sandbox_access={
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "sandbox",
                    "descriptor": {"sandbox_id": "local"},
                },
                "workdir": str(self.workdir),
            },
        ).model_dump(mode="json")

    def activation(self) -> dict[str, Any]:
        return {"input": "task"}

    def started(self) -> int:
        return len((self.control / "launches").read_text().splitlines())

    def harness_pids(self) -> list[int]:
        pids = []
        for path in (self.control / "pids").glob("*") if (self.control / "pids").exists() else ():
            try:
                state = Path(f"/proc/{path.name}/stat").read_text().rsplit(")", 1)[1].split()[0]
            except OSError:
                continue
            if state not in ("Z", "X"):
                pids.append(int(path.name))
        return pids

    def finish(self) -> None:
        (self.control / "finish").touch()

    def hold(self) -> None:
        (self.control / "finish").unlink(missing_ok=True)

    def break_next_pause(self, agent: Any) -> None:
        original = SandboxSession.pause_harness

        async def unconfirmed(session: SandboxSession, **kwargs: Any) -> HarnessPauseResult:
            self.monkeypatch.setattr(SandboxSession, "pause_harness", original)
            return HarnessPauseResult(status="unconfirmed", error="injected: a process stuck in disk I/O")

        self.monkeypatch.setattr(SandboxSession, "pause_harness", unconfirmed)

    def crash_recovery_assertion(self) -> Callable[[dict[str, Any]], None]:
        # The harness alive now is the paused ``run`` of the session about to crash: its data directory holds that
        # session's conversation.
        [data_home] = {
            entry.partition("=")[2]
            for pid in self.harness_pids()
            for entry in Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0")
            if entry.startswith("XDG_DATA_HOME=")
        }
        database = Path(data_home) / "opencode" / "opencode.db"
        store = database.stat().st_ino
        sessions, messages = self._conversation(database)
        users = [message_id for message_id, role in messages if role == "user"]
        assert len(sessions) == 1 and users, f"the checkpoint holds one conversation with a user turn: {messages}"
        launches = len(self._launches())
        requests = len(self._resume_requests())

        def continued(reply: dict[str, Any]) -> None:
            # Recovery reprompted the saved conversation through the server; ``run`` would start the task again.
            assert self._launches()[launches:] == ["serve"]
            [request] = self._resume_requests()[requests:]
            assert f"/session/{sessions[0]}/message" in request["path"]
            # It re-posted the saved user turn with no new parts, so no turn was added to the conversation.
            assert request["body"] == {
                "messageID": users[-1],
                "parts": [],
                "agent": "build",
                "model": {"providerID": "nemo_gym", "modelID": "dummy_model"},
            }
            # The conversation store was not set aside, and holds the same conversation with more answers.
            assert database.stat().st_ino == store
            sessions_now, messages_now = self._conversation(database)
            assert sessions_now == sessions and messages_now[: len(messages)] == messages
            added = messages_now[len(messages) :]
            assert added and all(role == "assistant" for _, role in added), f"progress after recovery: {added}"

        return continued

    def _launches(self) -> list[str]:
        return (self.control / "launches").read_text().splitlines()

    def _resume_requests(self) -> list[dict[str, Any]]:
        path = self.control / "resume_requests"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    @staticmethod
    def _conversation(database: Path) -> tuple[list[str], list[tuple[str, str]]]:
        """The session ids of the fake's store, and each message's id and role in the order they were saved."""
        with closing(sqlite3.connect(database)) as connection:
            sessions = [session_id for (session_id,) in connection.execute("select id from session")]
            rows = connection.execute("select id, data from message order by time_created, rowid").fetchall()
        return sessions, [(message_id, json.loads(data)["role"]) for message_id, data in rows]


async def test_opencode_conforms_to_the_harness_checkpoint_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = OpenCodeFixture(tmp_path, monkeypatch)
    try:
        await check_harness_checkpointing(fixture, tmp_path / "state")
    finally:
        fixture.finish()
        # A supervisor still waiting for a continue ends on its stop file.
        for supervisor in (tmp_path / "sessions").rglob("process_supervisor.sh"):
            (supervisor.parent / "stop.request").touch()
        for pid in fixture.harness_pids():
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
    # Every resume, after the stop fallback and after the crash, re-posted the saved user turn with no new parts.
    requests = fixture._resume_requests()
    assert requests and all(
        request["body"]
        == {
            "messageID": "msg_001",
            "parts": [],
            "agent": "build",
            "model": {"providerID": "nemo_gym", "modelID": "dummy_model"},
        }
        for request in requests
    )


# -- the reprompt decision -------------------------------------------------------------------------------------


def _export(*messages: dict) -> dict:
    return {"session_id": "ses_root", "messages": list(messages)}


USER = {
    "info": {
        "id": "msg_1",
        "role": "user",
        "agent": "build",
        "model": {"providerID": "p", "modelID": "m", "variant": "v"},
    },
    "parts": [{"type": "text", "text": "fix the bug\n"}],
}
ANSWER = {"info": {"id": "msg_2", "role": "assistant", "time": {"completed": 5}}, "parts": []}
PENDING = {"info": {"id": "msg_3", "role": "assistant", "time": {}}, "parts": []}
# A user turn whose model has no ID: the request body to re-post cannot be built from it.
INCOMPLETE_USER = {"info": {**USER["info"], "model": {"providerID": "p"}}, "parts": USER["parts"]}


def _tool(status: str) -> dict:
    return {"type": "tool", "tool": "bash", "callID": f"call_{status}", "state": {"status": status}}


# The assistant message a stopped launch left: two tool calls done, two that never finished.
INTERRUPTED = {
    "info": {"id": "msg_3", "role": "assistant", "time": {}},
    "parts": [
        _tool("completed"),
        _tool("running"),
        _tool("error"),
        _tool("pending"),
        {"type": "text", "text": "running the tests"},
    ],
}


def test_a_saved_conversation_of_this_task_that_the_model_answered_is_reprompted_without_a_new_turn() -> None:
    decision = reprompt_decision(_export(USER, ANSWER, PENDING), "fix the bug")

    assert decision.reprompt and decision.session_id == "ses_root"
    assert decision.reason == "reprompting the saved conversation"
    assert decision.payload == {
        "messageID": "msg_1",
        "parts": [],
        "agent": "build",
        "model": {"providerID": "p", "modelID": "m"},
        "variant": "v",
    }
    assert decision.interrupted_tool_calls == ()


@pytest.mark.parametrize(
    "export, reason",
    [
        (None, "no saved OpenCode conversation"),
        (_export(ANSWER), "the saved conversation has no user turn"),
        (_export(USER, ANSWER), "the saved conversation does not start with this task's prompt"),
        (_export(USER, PENDING), "the model had not answered the saved conversation yet"),
        (_export(INCOMPLETE_USER, ANSWER), "the saved conversation cannot be reprompted: 'modelID'"),
    ],
    ids=["none", "no_user", "another_task", "unanswered", "incomplete_user"],
)
def test_an_unusable_saved_conversation_starts_the_task_afresh(export: Any, reason: str) -> None:
    prompt = "another task" if "does not start" in reason else "fix the bug"
    decision = reprompt_decision(export, prompt)

    assert not decision.reprompt and decision.reason == reason
    assert decision.session_id is None and decision.payload is None and decision.interrupted_tool_calls == ()


def test_the_reprompt_payload_re_enters_the_latest_user_turn() -> None:
    later = {"info": {"id": "msg_9", "role": "user"}, "parts": []}
    assert reprompt_payload([USER, ANSWER, later]) == {"messageID": "msg_9", "parts": []}
    with pytest.raises(ValueError, match="cannot reprompt an OpenCode session without a saved user turn"):
        reprompt_payload([ANSWER])


def test_a_conversation_exported_from_opencode_s_database_is_reprompted() -> None:
    """snapshot.py keeps each message's id beside its info: OpenCode stores the id on the row, not in the data."""

    def as_exported(message: dict) -> dict:
        info = {key: value for key, value in message["info"].items() if key != "id"}
        return {"id": message["info"]["id"], "info": info, "parts": message["parts"]}

    decision = reprompt_decision(_export(*map(as_exported, (USER, ANSWER, PENDING))), "fix the bug")

    assert decision.reprompt, decision.reason
    assert decision.payload["messageID"] == "msg_1"


def test_the_reprompt_payload_keeps_the_user_turn_s_settings_and_nothing_else() -> None:
    user = {
        "id": "msg_5",
        "info": {
            "role": "user",
            "agent": "plan",
            "system": "be brief",
            "format": {"type": "text"},
            "tools": {"bash": False},
            "model": {"providerID": "p", "modelID": "m"},
            "time": {"created": 1},
        },
        "parts": [
            {"type": "text", "text": "fix "},
            {"type": "file", "url": "x"},
            {"type": "text", "text": "the bug\n"},
        ],
    }
    expected = {
        "messageID": "msg_5",
        "parts": [],
        "agent": "plan",
        "system": "be brief",
        "format": {"type": "text"},
        "tools": {"bash": False},
        "model": {"providerID": "p", "modelID": "m"},
    }
    assert reprompt_payload([user]) == expected
    # The prompt is the user turn's text parts joined; a file part between them is not text.
    decision = reprompt_decision(_export(user, ANSWER), "fix the bug")
    assert decision.reprompt and decision.payload == expected


def test_text_that_opencode_added_to_the_user_turn_is_not_part_of_the_prompt() -> None:
    attached = {"type": "text", "text": "Called the Read tool with the following input: ...", "synthetic": True}
    user = {"info": USER["info"], "parts": [*USER["parts"], attached]}

    assert reprompt_decision(_export(user, ANSWER), "fix the bug").reprompt


def test_a_saved_conversation_with_unfinished_tool_calls_is_still_reprompted_and_counts_them() -> None:
    decision = reprompt_decision(_export(USER, ANSWER, INTERRUPTED), "fix the bug")

    assert decision.reprompt, decision.reason
    assert decision.payload["messageID"] == "msg_1"
    # The running and the pending call; OpenCode recorded an end for the completed and the failed one.
    assert len(decision.interrupted_tool_calls) == 2


@pytest.mark.parametrize(
    "messages, count",
    [
        ([], 0),
        ([USER, ANSWER, PENDING], 0),
        ([{"info": {"role": "assistant"}}], 0),
        ([{"info": {"role": "assistant"}, "parts": [_tool("completed"), _tool("error")]}], 0),
        ([{"info": {"role": "assistant"}, "parts": [{"type": "tool"}, {"type": "tool", "state": None}]}], 0),
        ([{"info": {"role": "assistant"}, "parts": [{"type": "text", "state": {"status": "running"}}]}], 0),
        ([{"info": {"role": "assistant"}, "parts": [_tool("pending")]}], 1),
        ([INTERRUPTED], 2),
        ([INTERRUPTED, ANSWER, {"info": {"role": "assistant"}, "parts": [_tool("running")]}], 3),
    ],
    ids=[
        "no_messages",
        "no_tools",
        "no_parts",
        "finished_tools",
        "tools_without_a_state",
        "not_a_tool",
        "pending",
        "running_and_pending",
        "several_messages",
    ],
)
def test_only_pending_and_running_tool_calls_count_as_interrupted(messages: list[dict], count: int) -> None:
    assert len(interrupted_tool_calls(messages)) == count


# -- reprompt_opencode.sh --------------------------------------------------------------------------------------------

TRANSPORTS = [
    pytest.param("curl", marks=pytest.mark.skipif(shutil.which("curl") is None, reason="curl is required")),
    "dev_tcp",
]
REPROMPT_PAYLOAD = {"messageID": "msg_001", "parts": []}


def _helper(tmp_path: Path, transport: str = "any", *, payload: dict = REPROMPT_PAYLOAD) -> SimpleNamespace:
    """A fake OpenCode, a payload in a launch directory, and the command and environment of one run of the helper.

    With ``dev_tcp`` the helper finds only the tools it needs, and no curl: it speaks HTTP over bash's /dev/tcp.
    """
    control = tmp_path / "control"
    control.mkdir()
    (control / "launches").write_text("")
    binary = tmp_path / "opencode"
    binary.write_text(_fake_opencode(control))
    binary.chmod(0o755)
    launch = tmp_path / "launch-2"
    launch.mkdir()
    (launch / "reprompt.json").write_text(json.dumps(payload))
    workdir = tmp_path / "task dir"
    workdir.mkdir()
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    env = {
        **os.environ,
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "OPENCODE_CONFIG_CONTENT": "{}",
        "TMPDIR": str(scratch),
    }
    if transport == "dev_tcp":
        tools = tmp_path / "tools"
        tools.mkdir()
        required = ("bash", "seq", "sleep", "base64", "tr", "wc", "cat", "tail")
        for tool in (*required, "head", "od"):
            found = shutil.which(tool)
            if found:
                (tools / tool).symlink_to(found)
        assert all((tools / tool).exists() for tool in required)
        (tools / "python3").symlink_to(sys.executable)
        env["PATH"] = str(tools)
    return SimpleNamespace(
        control=control,
        command=["bash", str(REPROMPT_SCRIPT), str(binary), "ses_root", str(launch / "reprompt.json")],
        launch=launch,
        workdir=workdir,
        scratch=scratch,
        env=env,
    )


def _run_helper(helper: SimpleNamespace) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        helper.command, cwd=helper.workdir, env=helper.env, capture_output=True, text=True, timeout=60
    )


def _recorded(helper: SimpleNamespace, name: str) -> list[dict]:
    """What the fake server recorded in its control file ``name``, one entry per request."""
    path = helper.control / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _directory_query(helper: SimpleNamespace) -> str:
    # The server is asked for the session in the working directory the CLI would use.
    return "directory=" + str(helper.workdir).replace("/", "%2F").replace(" ", "%20")


def _temporary_files(helper: SimpleNamespace) -> list[Path]:
    return [*helper.scratch.glob("opencode-resume.*"), *helper.scratch.glob("opencode-reprompt.*")]


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_the_reprompt_helper_re_posts_the_saved_user_turn_through_the_opencode_server(
    tmp_path: Path, transport: str
) -> None:
    helper = _helper(tmp_path, transport)
    (helper.control / "finish").touch()

    completed = _run_helper(helper)

    assert completed.returncode == 0, completed.stderr
    assert "OpenCode ran the existing user turn again without adding a prompt" in completed.stdout
    [request] = _recorded(helper, "resume_requests")
    assert request["body"] == {"messageID": "msg_001", "parts": []}
    assert _directory_query(helper) in request["path"]
    assert (helper.control / "launches").read_text().splitlines() == ["serve"]
    # OpenCode asked for no permission: nothing was rejected.
    assert _recorded(helper, "permission_replies") == []
    assert "asked for a permission" not in completed.stderr
    # The server's log and the request's status stay beside the payload, in the launch directory.
    assert "opencode server listening" in (helper.launch / "reprompt-server.log").read_text()
    assert (helper.launch / "reprompt-status").read_text() == "200"
    assert _temporary_files(helper) == []


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_the_reprompt_helper_rejects_a_permission_request_as_opencode_run_does(tmp_path: Path, transport: str) -> None:
    helper = _helper(tmp_path, transport)
    (helper.control / "finish").touch()
    # The server holds the reprompt request until its permission request is replied to; nothing else replies.
    (helper.control / "ask_permission").touch()

    completed = _run_helper(helper)

    assert completed.returncode == 0, completed.stderr
    [reply] = _recorded(helper, "permission_replies")
    assert reply["path"].startswith("/permission/per_test1/reply?") and _directory_query(helper) in reply["path"]
    assert reply["body"] == {"reply": "reject"}
    assert reply["authorized"], "the reply carries the server's basic authentication"
    assert "OpenCode asked for a permission (per_test1); rejected" in completed.stderr
    # The turn itself was posted once and finished.
    [request] = _recorded(helper, "resume_requests")
    assert request["body"] == {"messageID": "msg_001", "parts": []}
    assert (helper.launch / "reprompt-status").read_text() == "200"


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_the_reprompt_helper_reports_a_refused_request_with_the_end_of_the_server_log(
    tmp_path: Path, transport: str
) -> None:
    # The fake refuses a prompt request that carries parts, with HTTP 400 and a line in its output.
    helper = _helper(tmp_path, transport, payload={"messageID": "msg_001", "parts": [{"type": "text", "text": "x"}]})
    (helper.control / "finish").touch()

    completed = _run_helper(helper)

    assert completed.returncode == 4, completed.stderr
    assert "OpenCode reprompt request failed with HTTP status 400; the server's log ends:" in completed.stderr
    assert REFUSED_PROMPT in completed.stderr
    assert "without adding a prompt" not in completed.stdout
    # The log and the status are kept in the directory of the payload, and nothing is left in TMPDIR.
    assert REFUSED_PROMPT in (helper.launch / "reprompt-server.log").read_text()
    assert (helper.launch / "reprompt-status").read_text() == "400"
    assert _temporary_files(helper) == []


def test_the_reprompt_helper_reports_a_server_that_never_starts(tmp_path: Path) -> None:
    broken = tmp_path / "opencode"
    broken.write_text("#!/bin/sh\necho 'fake opencode: cannot start'\nexit 1\n")
    broken.chmod(0o755)
    payload = tmp_path / "reprompt.json"
    payload.write_text("{}")

    completed = subprocess.run(
        ["bash", str(REPROMPT_SCRIPT), str(broken), "ses", str(payload)], capture_output=True, text=True, timeout=120
    )

    assert completed.returncode == 3
    assert "OpenCode server did not become ready for the reprompt:" in completed.stderr
    assert "fake opencode: cannot start" in completed.stderr
    assert "fake opencode: cannot start" in (tmp_path / "reprompt-server.log").read_text()


def test_a_stop_ends_the_reprompt_helper_and_its_server_while_it_rejects_permissions_for_the_running_turn(
    tmp_path: Path,
) -> None:
    helper = _helper(tmp_path)
    # No finish file: the server holds the request, as it does for the whole of a reprompted turn. The helper has
    # rejected one permission by the time it is stopped and goes on asking for more.
    (helper.control / "ask_permission").touch()
    process = subprocess.Popen(
        helper.command, cwd=helper.workdir, env=helper.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    def running(pid: str) -> bool:
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
        except OSError:
            return False

    def wait_for(condition: Any) -> None:
        deadline = time.monotonic() + 30
        while not condition():
            assert time.monotonic() < deadline, "timed out waiting"
            time.sleep(0.05)

    try:
        wait_for(lambda: _recorded(helper, "permission_replies"))
        assert len(_recorded(helper, "resume_requests")) == 1
        servers = [path.name for path in (helper.control / "pids").iterdir()]
        process.terminate()
        # The supervisor's TERM is acted on at once, not when the turn ends.
        assert process.wait(timeout=10) == 143
        wait_for(lambda: not any(running(pid) for pid in servers))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        for path in (helper.control / "pids").iterdir() if (helper.control / "pids").exists() else ():
            try:
                os.kill(int(path.name), 9)
            except ProcessLookupError:
                pass
    # A stop is no failed request: the turn was posted once and its one permission rejected once.
    assert [reply["body"] for reply in _recorded(helper, "permission_replies")] == [{"reply": "reject"}]


def test_the_reprompt_helper_is_plain_bash() -> None:
    assert REPROMPT_SCRIPT.read_text().startswith("#!/usr/bin/env bash")
    subprocess.run(["bash", "-n", str(REPROMPT_SCRIPT)], check=True)
    assert "python" not in textwrap.dedent(REPROMPT_SCRIPT.read_text()).split("set +o errexit", 1)[1]


# -- relaunch ----------------------------------------------------------------------------------------------------


async def _relaunch_after_a_stop(monkeypatch: pytest.MonkeyPatch, conversation: dict | None) -> tuple[Any, list[dict]]:
    """Run ``_run_launches`` for a session whose first launch was stopped with ``conversation`` saved.

    Returns the session and the payload of every launch that was executed.
    """
    agent = OpenCodeAgent.model_construct(
        config=OpenCodeAgentConfig.model_construct(
            session_close_timeout_seconds=1, checkpoint_pause_allowance_seconds=0
        )
    )
    launched: list[dict] = []

    async def execute(payload: dict, **_: Any) -> None:
        launched.append(payload)

    session = SimpleNamespace(cleanup=None, session_dir="/s/launch-2")
    state = SimpleNamespace(
        launch_count=1,
        harness_session_id="ses_old",
        message_ids=["m"],
        observations=object(),
        trajectory=object(),
        launch_records=[],
        interrupted_tool_calls=0,
        interrupted_tool_call_ids=[],
        reprompted_conversation=False,
        session=session,
        persistent_directory="/s",
        runtime="/rt",
        request=SimpleNamespace(agent_session_id="a"),
        require_session=lambda: session,
        prepare_launch=AsyncMock(),
        set_conversation_aside=AsyncMock(),
        execute=execute,
        inspect_conversation=AsyncMock(return_value=conversation),
    )

    async def run(
        session: Any, activation: Any, *, launch: Any, on_parked: Any, stop_timeout: float
    ) -> LaunchFinished:
        await launch()
        return LaunchFinished(None)

    monkeypatch.setattr("responses_api_agents.opencode_agent.app.run_with_checkpoint_pauses", run)

    def payload() -> dict:
        session_arguments = ["--session", state.harness_session_id] if state.harness_session_id else []
        return {"command": ["opencode", "run", *session_arguments]}

    await agent._run_launches(state, MagicMock(), payload, prompt="fix the bug", timeout=10, deadline=None)
    return state, launched


async def test_a_relaunch_that_starts_afresh_does_not_name_the_conversation_it_set_aside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The model had not answered when the first launch was stopped.
    state, launched = await _relaunch_after_a_stop(monkeypatch, _export(USER, PENDING))

    state.set_conversation_aside.assert_awaited_once()
    assert launched == [{"command": ["opencode", "run"]}]
    assert state.harness_session_id is None and state.reprompted_conversation is False
    # Evidence of the conversation that was set aside is not this launch's.
    assert state.observations is None and state.trajectory is None
    assert state.launch_records == [
        {
            "launch": 2,
            "reprompted": False,
            "interrupted_tool_calls": 0,
            "outcome": "finished",
            "return_code": None,
            "paused_s": 0.0,
        }
    ]


async def test_a_relaunch_that_reprompts_records_the_tool_calls_the_stopped_launch_left_unfinished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, launched = await _relaunch_after_a_stop(monkeypatch, _export(USER, ANSWER, INTERRUPTED))

    state.set_conversation_aside.assert_not_awaited()
    [launch] = launched
    assert launch["command"] == [
        "bash",
        "/s/reprompt_opencode.sh",
        "/rt/opencode",
        "ses_root",
        "/s/launch-2/reprompt.json",
    ]
    assert launch["reprompt"]["messageID"] == "msg_1" and launch["reprompt"]["parts"] == []
    assert state.harness_session_id == "ses_root" and state.reprompted_conversation is True
    assert state.observations is not None and state.trajectory is not None
    assert state.interrupted_tool_calls == 2
    assert state.launch_records == [
        {
            "launch": 2,
            "reprompted": True,
            "interrupted_tool_calls": 2,
            "outcome": "finished",
            "return_code": None,
            "paused_s": 0.0,
        }
    ]


def _borrowed_seed(attempt: int = 1) -> AgentSeedSessionRequest:
    """The seed of session ``s``, attempt ``attempt`` of rollout ``r``, in a sandbox it borrows."""
    return AgentSeedSessionRequest(
        agent_session_id="s",
        episode_id=EpisodeId(rollout_id="r", attempt=attempt),
        task_id=TaskId(taskset="t", task_id="0"),
        sandbox_access={
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "fork"}},
            "workdir": "/app",
        },
    )


def _write_store(directory: Path, *messages: dict) -> None:
    """An OpenCode 1 store with one root session holding ``messages``, as a download of the sandbox's would be."""
    with closing(sqlite3.connect(directory / "opencode.db")) as connection:
        connection.executescript(
            "create table session(id text, parent_id text, time_created integer);"
            "create table message(id text, session_id text, data text, time_created integer);"
            "create table part(id text, message_id text, session_id text, data text, time_created integer);"
            "insert into session values('ses_root', null, 0);"
        )
        for index, message in enumerate(messages):
            message_id = message["info"]["id"]
            info = {key: value for key, value in message["info"].items() if key != "id"}
            connection.execute(
                "insert into message values(?, 'ses_root', ?, ?)", (message_id, json.dumps(info), index)
            )
            for number, part in enumerate(message["parts"]):
                connection.execute(
                    "insert into part values(?, ?, 'ses_root', ?, ?)",
                    (f"{message_id}-{number}", message_id, json.dumps(part), index),
                )
        connection.commit()


def _stopped_session(
    monkeypatch: pytest.MonkeyPatch, *, names: tuple[str, ...] = ("opencode.db",), download: Any = None
) -> tuple[OpenCodeAgent, OpenCodeSandboxSession, SimpleNamespace]:
    """A session whose first launch was stopped, and the reads of the store it left in the sandbox.

    The sandbox lists ``names`` as the store's files, and ``download(names, directory)`` copies them or raises.
    """
    agent = OpenCodeAgent.model_construct(config=OpenCodeAgentConfig.model_construct(session_close_timeout_seconds=30))
    session = SimpleNamespace(root_dir="/sessions/s", session_dir="/sessions/s/launch-2")
    state = OpenCodeSandboxSession(
        request=_borrowed_seed(),
        session=session,
        runtime="/rt",
        checkpointed=True,
        launch_count=1,
        harness_session_id="ses_old",
        message_ids=["m"],
    )
    store = SimpleNamespace(
        list=AsyncMock(return_value=list(names)), download=AsyncMock(side_effect=download), aside=AsyncMock()
    )
    monkeypatch.setattr(OpenCodeSandboxSession, "list_store", store.list)
    monkeypatch.setattr(OpenCodeSandboxSession, "download_store", store.download)
    monkeypatch.setattr(OpenCodeSandboxSession, "set_conversation_aside", store.aside)
    monkeypatch.setattr("responses_api_agents.opencode_agent.sandbox._CONVERSATION_READ_RETRY_DELAY_S", 0)
    return agent, state, store


def _fresh_payload() -> dict:
    return {"command": ["opencode", "run"]}


async def test_a_saved_conversation_is_read_again_when_the_sandbox_refuses_its_first_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    async def download(names: list[str], directory: Path) -> None:
        attempts.append(len(attempts) + 1)
        if len(attempts) < 3:
            raise ConnectionError("502 Bad Gateway")
        _write_store(directory, USER, ANSWER)

    agent, state, store = _stopped_session(monkeypatch, download=download)

    conversation = await state.inspect_conversation(timeout=30)

    assert attempts == [1, 2, 3] and store.list.await_count == 3
    assert conversation["session_id"] == "ses_root"
    assert [message["id"] for message in conversation["messages"]] == ["msg_1", "msg_2"]
    # The conversation that was read is the one the relaunch reprompts.
    launch, interrupted = await agent._relaunch_payload(state, _fresh_payload, prompt="fix the bug")
    assert launch["reprompt"]["messageID"] == "msg_1" and interrupted == 0
    store.aside.assert_not_awaited()


async def test_a_saved_conversation_that_cannot_be_downloaded_fails_the_relaunch_and_is_not_set_aside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def download(names: list[str], directory: Path) -> None:
        raise ConnectionError("502 Bad Gateway")

    agent, state, store = _stopped_session(monkeypatch, download=download)
    payload = MagicMock()

    with pytest.raises(RuntimeError, match="Cannot read the saved OpenCode conversation: 502 Bad Gateway"):
        await agent._relaunch_payload(state, payload, prompt="fix the bug")

    # Three reads, and then the conversation, which may be intact, stays where it is.
    assert store.download.await_count == 3
    store.aside.assert_not_awaited()
    payload.assert_not_called()
    assert state.harness_session_id == "ses_old" and state.message_ids == ["m"]


async def test_a_saved_conversation_that_cannot_be_parsed_is_not_read_again(monkeypatch: pytest.MonkeyPatch) -> None:
    async def download(names: list[str], directory: Path) -> None:
        (directory / "opencode.db").write_bytes(b"this is not an SQLite database\n" * 64)

    _, state, store = _stopped_session(monkeypatch, download=download)

    with pytest.raises(UnparsableConversationError):
        await state.inspect_conversation(timeout=30)

    assert store.download.await_count == 1


async def test_a_session_that_saved_no_store_has_no_conversation_to_inspect(monkeypatch: pytest.MonkeyPatch) -> None:
    _, state, store = _stopped_session(monkeypatch, names=())

    assert await state.inspect_conversation(timeout=30) is None

    store.list.assert_awaited_once()
    store.download.assert_not_awaited()


@pytest.mark.parametrize(
    "saved, reason",
    [("unparsable", "the saved conversation cannot be parsed"), ("missing", "no saved OpenCode conversation")],
)
async def test_a_relaunch_without_a_usable_conversation_starts_afresh_and_logs_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, saved: str, reason: str
) -> None:
    async def download(names: list[str], directory: Path) -> None:
        (directory / "opencode.db").write_bytes(b"this is not an SQLite database\n" * 64)

    names = ("opencode.db",) if saved == "unparsable" else ()
    agent, state, store = _stopped_session(monkeypatch, names=names, download=download)
    state.interrupted_tool_calls = 2

    with caplog.at_level(logging.WARNING, logger="responses_api_agents.opencode_agent.app"):
        launch, interrupted = await agent._relaunch_payload(state, _fresh_payload, prompt="fix the bug")

    assert launch == {"command": ["opencode", "run"]} and interrupted == 0
    store.aside.assert_awaited_once()
    assert state.harness_session_id is None and state.message_ids == []
    assert state.interrupted_tool_calls == 0
    [logged] = [record.getMessage() for record in caplog.records if "starts the task afresh" in record.getMessage()]
    assert reason in logged
    other = {"the saved conversation cannot be parsed", "no saved OpenCode conversation"} - {reason}
    assert not any(text in logged for text in other)


async def test_relaunches_count_each_tool_call_they_interrupted_once_until_one_starts_afresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def download(names: list[str], directory: Path) -> None:
        _write_store(directory, USER, ANSWER, INTERRUPTED)

    agent, state, store = _stopped_session(monkeypatch, download=download)

    launch, interrupted = await agent._relaunch_payload(state, _fresh_payload, prompt="fix the bug")

    assert launch == {
        "command": [
            "bash",
            "/sessions/s/reprompt_opencode.sh",
            "/rt/opencode",
            "ses_root",
            "/sessions/s/launch-2/reprompt.json",
        ],
        "reprompt": {
            "messageID": "msg_1",
            "parts": [],
            "agent": "build",
            "model": {"providerID": "p", "modelID": "m"},
            "variant": "v",
        },
    }
    assert interrupted == 2 and state.interrupted_tool_calls == 2
    assert state.harness_session_id == "ses_root"

    # Another relaunch finds the same two calls still unfinished in the store: each is counted once.
    _, interrupted = await agent._relaunch_payload(state, _fresh_payload, prompt="fix the bug")
    assert interrupted == 2 and state.interrupted_tool_calls == 2
    store.aside.assert_not_awaited()

    # A relaunch for another prompt starts afresh: nothing of the conversation it sets aside is this launch's.
    launch, interrupted = await agent._relaunch_payload(state, _fresh_payload, prompt="another task")
    assert launch == {"command": ["opencode", "run"]} and interrupted == 0
    assert state.interrupted_tool_calls == 0
    store.aside.assert_awaited_once()


@pytest.mark.parametrize(
    "reprompted, return_code, error, reported",
    [
        (False, 1, "APIError", True),
        (False, 1, "ContextOverflowError", True),
        # OpenCode 1.15 exits 0 when compaction itself overflows the context.
        (False, 0, "ContextOverflowError", True),
        (False, 0, "APIError", False),
        # reprompt_opencode.sh exits 0 once the server answered, whatever the turn ended with.
        (True, 0, "APIError", True),
        (True, 0, "ContextOverflowError", True),
        (True, 1, "APIError", False),
        (True, 1, "ContextOverflowError", False),
    ],
)
def test_a_model_error_is_read_with_the_exit_code_of_the_command_the_launch_ran(
    reprompted: bool, return_code: int, error: str, reported: bool
) -> None:
    native = {"error": {"name": error}, "time": {"completed": 1}}
    if error == "ContextOverflowError":
        native["finish"] = "error"
    cleanup = {"return_code": return_code, "timed_out": False, "error": None}

    assert OpenCodeAgent._reported_model_error(native, cleanup, reprompted=reprompted) is reported


# -- re-bind -----------------------------------------------------------------------------------------------------


def _restored(
    monkeypatch: pytest.MonkeyPatch, sandbox: Any, **config: Any
) -> tuple[OpenCodeAgent, OpenCodeSandboxSession, Any]:
    agent = OpenCodeAgent.model_construct(
        config=OpenCodeAgentConfig.model_construct(
            # Long enough for a retried command's wait: a command's timeout is the budget of all its attempts.
            session_close_timeout_seconds=30,
            sandbox_install_timeout_seconds=30,
            export_dir="/exports",
            sandbox_path_prefix=[],
            **config,
        ),
        server_client=SimpleNamespace(global_config_dict={}),
    )
    body = _borrowed_seed()
    state = OpenCodeSandboxSession(request=body, session=None, runtime="/rt", checkpointed=True)
    state.restored = {"session": {}}
    provider = SimpleNamespace(aclose=AsyncMock())
    for name, value in (
        ("create_provider", lambda *_: provider),
        ("resolve_provider_config", lambda *_: {}),
        ("get_global_config_dict", lambda: {}),
        ("AsyncSandbox", SimpleNamespace(connect=AsyncMock(return_value=sandbox))),
    ):
        monkeypatch.setattr(f"responses_api_agents.opencode_agent.app.{name}", value)
    return agent, state, body


async def test_a_re_bind_retries_the_first_command_a_forked_sandbox_refused_and_restores_what_the_seed_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = MagicMock(sandbox_id="fork")
    present = SandboxExecResult(stdout="", stderr="", return_code=0)
    sandbox.exec = AsyncMock(side_effect=[ConnectionError("502 Bad Gateway"), present])
    sandbox.disconnect = AsyncMock()
    agent, state, body = _restored(monkeypatch, sandbox)
    session = SimpleNamespace(root_dir="/sessions/s")
    monkeypatch.setattr(SandboxSession, "rebind", staticmethod(lambda *_, **__: session))

    assert await agent._rebind_restored_agent_session_state(state, body) is state

    # The runtime and the session's files are in the fork: nothing is installed or uploaded again.
    assert sandbox.exec.await_count == 2 and state.session is session and state.restored is None
    sandbox.upload.assert_not_called()
    # A checkpoint does not carry these; without them a restored episode writes no exports.
    assert state.export_dir == "/exports" and "/sessions/s" in state.opencode_environment
    sandbox.disconnect.assert_not_awaited()


async def test_a_re_bind_into_a_sandbox_without_the_runtime_installs_it_from_the_pinned_files_of_the_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = MagicMock(sandbox_id="fork")

    async def run(command: str, **_: Any) -> SandboxExecResult:
        # The fork holds the session root but not the runtime: the check of the restored session's files fails.
        missing = command.startswith("[ -e /rt/opencode ]")
        return SandboxExecResult(stdout="", stderr="", return_code=1 if missing else 0)

    sandbox.exec = AsyncMock(side_effect=run)
    sandbox.upload = AsyncMock()
    sandbox.download = AsyncMock()
    sandbox.disconnect = AsyncMock()
    agent, state, body = _restored(
        monkeypatch,
        sandbox,
        sandbox_ripgrep_path="/pins/rg",
        sandbox_ripgrep_sha256="A" * 64,
        sandbox_plugin_dependencies_path="/pins/plugin-dependencies.tar",
        sandbox_plugin_dependencies_sha256="b" * 64,
    )
    session = SimpleNamespace(root_dir="/sessions/s")
    monkeypatch.setattr(SandboxSession, "rebind", staticmethod(lambda *_, **__: session))
    # The archive's members are checked on a download of it (see test_pinned.py); this sandbox holds no archive.
    monkeypatch.setattr("responses_api_agents.opencode_agent.app.check_archive", lambda *_: None)
    install = OpenCodeAgent._install_session_files
    installs: list[tuple[Any, ...]] = []

    async def installing(self: OpenCodeAgent, *arguments: Any, **options: Any) -> None:
        installs.append((*arguments, state.session, options))
        await install(self, *arguments, **options)

    monkeypatch.setattr(OpenCodeAgent, "_install_session_files", installing)

    assert await agent._rebind_restored_agent_session_state(state, body) is state

    ripgrep = _PinnedFile(digest="a" * 64, sandbox_path="/pins/rg")
    dependencies = _PinnedFile(digest="b" * 64, sandbox_path="/pins/plugin-dependencies.tar")
    assert (agent._pinned_file("ripgrep"), agent._pinned_file("plugin_dependencies")) == (ripgrep, dependencies)
    # Installed through the connection of the re-bind, while the session had none yet.
    assert installs == [
        (
            sandbox,
            state,
            "/sessions/s",
            "/app",
            "/rt",
            None,
            {"checkpointed": True, "ripgrep": ripgrep, "plugin_dependencies": dependencies},
        )
    ]
    commands = [call.args[0] for call in sandbox.exec.await_args_list]

    def ran(start: str) -> int:
        return sum(command.startswith(start) for command in commands)

    # The pinned rg replaces the installer's package install of ripgrep, and is staged in the session's cache.
    assert ran("NG_OPENCODE_SKIP_RIPGREP=1 bash ") == 1
    cache = "/sessions/s/cache/opencode/bin"
    assert ran(f"set -- /pins/rg {'a' * 64} {cache}/.pinned-rg {cache}/rg\n") == 1
    # The plugin dependencies are copied, checked on a download of the copy, and unpacked into the session's config.
    copy = "/sessions/s/config/.pinned-opencode"
    assert ran(f"set -- /pins/plugin-dependencies.tar {'b' * 64} {copy}\n") == 1
    assert ran(f"set -- {copy} /sessions/s/config/opencode\n") == 1
    sandbox.download.assert_awaited_once()
    assert sandbox.download.await_args.args[0] == copy
    assert "/sessions/s/reprompt_opencode.sh" in [call.args[1] for call in sandbox.upload.await_args_list]
    assert state.session is session and state.restored is None
    sandbox.disconnect.assert_not_awaited()


async def test_a_re_bind_that_fails_releases_its_sandbox_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = MagicMock(sandbox_id="fork")
    sandbox.disconnect = AsyncMock()
    agent, state, body = _restored(monkeypatch, sandbox)

    def broken(*_: Any, **__: Any) -> None:
        raise RuntimeError("the fork holds no session")

    monkeypatch.setattr(SandboxSession, "rebind", staticmethod(broken))

    with pytest.raises(RuntimeError, match="holds no session"):
        await agent._rebind_restored_agent_session_state(state, body)

    # The seed that re-binds is retried: each attempt must not leave a connection behind.
    sandbox.disconnect.assert_awaited_once()
    assert state.session is None and state.restored == {"session": {}}


# -- the saved reply and the evidence of a finished session ------------------------------------------------------

# The activation of session ``s``; each test changes one field of it.
ACTIVATION = {"input": "fix the bug", "instructions": "be brief", "temperature": 0.5}


def _body(**changes: Any) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(**{**ACTIVATION, **changes})


class _Participant:
    """A checkpoint participant with no checkpoint in progress, for which every session is open."""

    def __init__(self) -> None:
        self.saved_replies: list[str] = []

    def check_session(self, session_key: str, episode_id: EpisodeId) -> None:
        pass

    def answered_from_saved_reply(self, session_key: str, episode_id: EpisodeId) -> None:
        self.saved_replies.append(session_key)

    @asynccontextmanager
    async def activation(self, session_key: str, episode_id: EpisodeId, *, index: int | None = None) -> Any:
        yield MagicMock()

    async def wait_open(self) -> None:
        pass

    async def close_session(self, session_key: str) -> bool:
        return True


def _agent(fixture: OpenCodeFixture, *, checkpointing: bool = True) -> OpenCodeAgent:
    agent = fixture.make_agent({})
    if checkpointing:
        agent._checkpoint_participant = _Participant()
    return agent


def _launch_nothing(monkeypatch: pytest.MonkeyPatch, hold: asyncio.Event | None = None) -> list[str]:
    """Answer every activation that would launch OpenCode with an empty reply, once ``hold`` is set if given.

    Returns the prompts of the activations that started, one per launch a real session would have made.
    """
    launched: list[str] = []

    async def respond(self: OpenCodeAgent, state: Any, body: Any, *, prompt: str, **_: Any) -> NeMoGymResponse:
        launched.append(prompt)
        if hold is not None:
            await hold.wait()
        return NeMoGymResponse(
            id=f"resp_{len(launched)}",
            created_at=0,
            model="policy",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )

    monkeypatch.setattr(OpenCodeAgent, "_session_response", respond)
    return launched


def _open_session(agent: OpenCodeAgent) -> OpenCodeSandboxSession:
    """Install session ``s`` as its seed would have, with a sandbox session that only knows where its files are."""
    files = {"session_root": "/sessions/s", "session_dir": "/sessions/s", "launch_started": True}
    session = SimpleNamespace(root_dir="/sessions/s", checkpoint_state=lambda: files)
    state = OpenCodeSandboxSession(
        request=_borrowed_seed(), session=session, runtime="/rt", sandbox_id="fork", checkpointed=True
    )
    agent._install_restored_session("s", state)
    agent._session_records["s"].restored = False
    return state


def _activation(state: OpenCodeSandboxSession) -> Request:
    """A request of the session's first activation, on the rollout route of its episode."""
    return Request(
        {
            "type": "http",
            "headers": [(AGENT_ACTIVATION_HEADER.encode(), b"1")],
            "session": {AGENT_SESSION_COOKIE_KEY: state.request.agent_session_id},
            "path_params": {"rollout_id": state.request.episode_id.capture_key},
        }
    )


async def _checkpoint(agent: OpenCodeAgent) -> dict:
    """What a checkpoint keeps of session ``s``, as read back from storage."""
    exported = await agent.export_agent_sessions(["s"], checkpoint_id="c1")
    return json.loads(json.dumps(exported["s"]))


async def _restore(agent: OpenCodeAgent, saved: dict, *, attempt: int = 2) -> OpenCodeSandboxSession:
    """Install ``saved`` in ``agent`` for the replacement ``attempt`` of the rollout; it has no sandbox yet."""
    episode_id = EpisodeId(rollout_id="r", attempt=attempt)
    await agent.restore_agent_sessions([RestoredAgentSession(session_key="s", episode_id=episode_id, session=saved)])
    return agent._session_records["s"].state


async def _check_the_reply_is_given_again_to_its_request_only(
    agent: OpenCodeAgent, state: OpenCodeSandboxSession, reply_id: str, change: dict
) -> None:
    participant = agent.checkpoint_participant
    again = await agent.responses(_activation(state), _body())
    assert again.id == reply_id and participant.saved_replies == ["s"]
    with pytest.raises(AgentSessionConflictError, match="retry the same request") as conflict:
        await agent.responses(_activation(state), _body(**change))
    assert conflict.value.status_code == 409
    # The refused request did not claim the session's reply.
    assert participant.saved_replies == ["s"]


@pytest.mark.parametrize(
    "change",
    [{"input": "fix another bug"}, {"temperature": 0.7}, {"instructions": "be thorough"}],
    ids=["prompt", "temperature", "instructions"],
)
async def test_a_saved_reply_is_given_again_only_to_the_request_that_produced_it_also_after_a_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict
) -> None:
    fixture = OpenCodeFixture(tmp_path, monkeypatch)
    launched = _launch_nothing(monkeypatch)
    agent = _agent(fixture)
    state = _open_session(agent)

    reply = await agent.responses(_activation(state), _body())

    assert launched == ["fix the bug"]
    assert state.completed_activation == 1 and state.completed_reply["id"] == reply.id
    assert state.completed_request_digest == _request_digest(_body()) != _request_digest(_body(**change))
    await _check_the_reply_is_given_again_to_its_request_only(agent, state, reply.id, change)

    # The digest of the request travels in the checkpoint with the reply.
    saved = await _checkpoint(agent)
    assert saved["completed_request_digest"] == state.completed_request_digest
    restored_agent = _agent(fixture)
    restored = await _restore(restored_agent, saved)
    assert restored.session is None and restored.request.episode_id.capture_key == "r-a2"
    await _check_the_reply_is_given_again_to_its_request_only(restored_agent, restored, reply.id, change)
    assert launched == ["fix the bug"]


async def test_a_reply_restored_without_the_digest_of_its_request_is_not_given_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = OpenCodeFixture(tmp_path, monkeypatch)
    launched = _launch_nothing(monkeypatch)
    agent = _agent(fixture)
    await agent.responses(_activation(_open_session(agent)), _body())
    # A checkpoint with the reply and no digest of its request.
    saved = await _checkpoint(agent)
    del saved["completed_request_digest"]
    restored_agent = _agent(fixture)

    restored = await _restore(restored_agent, saved)

    assert restored.completed_reply is not None and restored.completed_request_digest is None
    with pytest.raises(AgentSessionConflictError, match="retry the same request") as conflict:
        await restored_agent.responses(_activation(restored), _body())
    assert conflict.value.status_code == 409
    assert restored_agent.checkpoint_participant.saved_replies == [] and launched == ["fix the bug"]


@pytest.mark.parametrize("checkpointing", [False, True])
async def test_a_different_request_is_refused_while_the_activation_is_still_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpointing: bool
) -> None:
    finish = asyncio.Event()
    launched = _launch_nothing(monkeypatch, hold=finish)
    agent = _agent(OpenCodeFixture(tmp_path, monkeypatch), checkpointing=checkpointing)
    state = _open_session(agent)
    running = asyncio.create_task(agent.responses(_activation(state), _body()))
    try:
        async with asyncio.timeout(30):
            while not launched:
                await asyncio.sleep(0.01)
        assert not state.task.done() and state.completed_reply is None

        with pytest.raises(AgentSessionConflictError, match="retry the same request") as conflict:
            # Bounded: a request that is not refused waits for the running activation's reply.
            await asyncio.wait_for(agent.responses(_activation(state), _body(input="fix another bug")), 5)
        assert conflict.value.status_code == 409

        # The same request again waits for the one activation and gets its reply.
        same = asyncio.create_task(agent.responses(_activation(state), _body()))
        await asyncio.sleep(0)
        finish.set()
        first, second = await asyncio.wait_for(asyncio.gather(running, same), 30)
    finally:
        finish.set()
        running.cancel()
    assert first.id == second.id == "resp_1" and launched == ["fix the bug"]


def _finished_evidence() -> tuple[TrajectoryRecord, AgentObservationBundle]:
    """The evidence of a finished activation of attempt 1: two decision turns, and a compaction between them.

    OpenCode's compaction is an observation, not a turn: the turns leave out the message that holds its summary.
    """

    def turn(number: int, message_id: str) -> TrajectoryTurn:
        return TrajectoryTurn(
            invocation_id="r-a1",
            source_message_id=message_id,
            task_id="0",
            rollout_id="r-a1",
            turn_no=number,
            timestamp=float(number),
            answer=f"answer {number}",
            step_count=number,
            model_calls=[ModelCallRef(model_call_id=f"call-{number}")],
        )

    trajectory = TrajectoryRecord(task_id="0", rollout_id="r-a1", turns=[turn(1, "m1"), turn(2, "m4")])
    compaction = ContextCompactionObservation(
        invocation_id="r-a1", source_message_ids=["m3"], trigger="auto", outcome="completed", summary="summary"
    )
    return trajectory, AgentObservationBundle(schema_version="2.0", source="opencode", records=[compaction])


def _turns(trajectory: TrajectoryRecord) -> list[tuple[Any, ...]]:
    return [
        (
            turn.turn_no,
            turn.source_message_id,
            turn.step_count,
            turn.invocation_id,
            turn.task_id,
            turn.timestamp,
            turn.answer,
            [call.model_call_id for call in turn.model_calls],
        )
        for turn in trajectory.turns
    ]


@pytest.mark.parametrize("returned", [True, False], ids=["trajectory_returned", "trajectory_not_returned"])
@pytest.mark.parametrize("rebound", [True, False], ids=["closed_after_the_re_bind", "closed_before_the_re_bind"])
async def test_a_session_that_finished_before_its_checkpoint_closes_with_its_trajectory_after_a_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rebound: bool, returned: bool
) -> None:
    fixture = OpenCodeFixture(tmp_path, monkeypatch)
    launched = _launch_nothing(monkeypatch)
    agent = _agent(fixture)
    state = _open_session(agent)
    reply = await agent.responses(_activation(state), _body())
    # What the capture of the finished launch left on the session.
    state.trajectory, state.observations = _finished_evidence()
    turns = _turns(state.trajectory)
    assert [turn[1:4] for turn in turns] == [("m1", 1, "r-a1"), ("m4", 2, "r-a1")]

    saved = await _checkpoint(agent)
    restored_agent = _agent(fixture)
    restored_agent.config.return_session_trajectory = returned
    restored = await _restore(restored_agent, saved, attempt=2)

    # Only the rollout ID is the replacement attempt's; the turns are the ones the harness reported.
    assert restored.trajectory.rollout_id == "r-a2" and restored.trajectory.task_id == "0"
    assert [turn.rollout_id for turn in restored.trajectory.turns] == ["r-a2", "r-a2"]
    assert _turns(restored.trajectory) == turns
    sandbox = MagicMock(sandbox_id="fork")
    sandbox.exec = AsyncMock(return_value=SandboxExecResult(stdout="", stderr="", return_code=0))
    release = AsyncMock()
    monkeypatch.setattr(OpenCodeSandboxSession, "close", release)
    if rebound:
        connect = SimpleNamespace(connect=AsyncMock(return_value=sandbox))
        monkeypatch.setattr("responses_api_agents.opencode_agent.app.AsyncSandbox", connect)
        session = SimpleNamespace(root_dir="/sessions/s")
        monkeypatch.setattr(SandboxSession, "rebind", staticmethod(lambda *_, **__: session))
        await restored_agent._rebind_restored_agent_session_state(restored, _borrowed_seed(attempt=2))
        assert restored.session is session
    again = await restored_agent.responses(_activation(restored), _body())
    assert again.id == reply.id

    closed = await restored_agent.close_agent_session(
        _activation(restored), AgentCloseSessionRequest(agent_session_id="s", episode_id=restored.request.episode_id)
    )

    # Nothing was launched after the restore: the reply and the evidence are the checkpoint's.
    assert launched == ["fix the bug"]
    assert sandbox.exec.await_count == (1 if rebound else 0)
    assert release.await_count == (1 if rebound else 0)
    [compaction] = closed.agent_observations.records
    assert (compaction.kind, compaction.invocation_id, compaction.source_message_ids) == (
        "context_compaction",
        "r-a1",
        ["m3"],
    )
    if not returned:
        assert closed.trajectory is None
        return
    assert closed.trajectory.rollout_id == "r-a2"
    assert [turn.rollout_id for turn in closed.trajectory.turns] == ["r-a2", "r-a2"]
    assert _turns(closed.trajectory) == turns


# -- checkpoint declarations -------------------------------------------------------------------------------------


def test_sessions_the_agent_cannot_capture_restart() -> None:
    borrowed = AgentSeedSessionRequest(
        agent_session_id="s",
        episode_id=EpisodeId(rollout_id="r"),
        task_id=TaskId(taskset="t", task_id="0"),
        sandbox_access={
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "x"}},
            "workdir": "/app",
        },
    )
    owned = borrowed.model_copy(update={"sandbox_access": None})
    continued = AgentSeedSessionRequest.model_validate(borrowed.model_dump() | {"continuation": {}})
    restarts = OpenCodeAgent._seed_restarts
    agent = SimpleNamespace(config=SimpleNamespace(session_import_path=None), _opencode2=False)
    importing = SimpleNamespace(config=SimpleNamespace(session_import_path="/imports/session.json"), _opencode2=False)
    opencode2 = SimpleNamespace(config=SimpleNamespace(session_import_path=None), _opencode2=True)

    assert OpenCodeAgent.checkpoint_mode == "exported" and OpenCodeAgent.rebinds_tool_accesses
    assert not restarts(agent, borrowed)
    assert restarts(agent, owned) and restarts(agent, continued)
    # An imported conversation is not this task's own: a relaunch could not tell it from a foreign one.
    assert restarts(importing, borrowed)
    # The relaunch that reprompts a saved conversation speaks OpenCode 1's server API and reads its store.
    assert restarts(opencode2, borrowed)


# -- review of the recovery paths ------------------------------------------------------------------------------------


@pytest.mark.parametrize("failures", [1, 3])
async def test_a_re_bind_does_not_install_when_the_check_of_its_files_could_not_run(
    monkeypatch: pytest.MonkeyPatch, failures: int
) -> None:
    monkeypatch.setattr("responses_api_agents.opencode_agent.app._RESTORED_CHECK_RETRY_DELAY_S", 0)
    sandbox = MagicMock(sandbox_id="fork")
    not_run = SandboxExecResult(stdout=None, stderr="stream closed", return_code=-1, error_type="transport")
    present = SandboxExecResult(stdout="", stderr="", return_code=0)
    sandbox.exec = AsyncMock(side_effect=[not_run] * failures + [present])
    sandbox.disconnect = AsyncMock()
    agent, state, body = _restored(monkeypatch, sandbox)
    session = SimpleNamespace(root_dir="/sessions/s")
    monkeypatch.setattr(SandboxSession, "rebind", staticmethod(lambda *_, **__: session))

    if failures < 3:
        # The check ran on a later attempt and found the files: nothing is installed.
        assert await agent._rebind_restored_agent_session_state(state, body) is state
        assert sandbox.exec.await_count == failures + 1 and state.session is session
        sandbox.disconnect.assert_not_awaited()
    else:
        # A check that never ran is not a missing runtime: the re-bind fails and its seed is tried again.
        with pytest.raises(NgFailureEvent):
            await agent._rebind_restored_agent_session_state(state, body)
        assert sandbox.exec.await_count == 3 and state.session is None
        sandbox.disconnect.assert_awaited_once()
    sandbox.upload.assert_not_called()


async def test_a_restored_session_closed_before_its_re_bind_without_evidence_reports_a_2_0_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, state, _ = _restored(monkeypatch, MagicMock())

    closed = await agent._close_agent_session_state(state)

    assert closed.agent_observations.schema_version == "2.0"
    assert [gap.code for gap in closed.agent_observations.gaps] == ["agent_activation_interrupted"]


async def test_a_launch_directory_is_prepared_again_when_the_sandbox_refuses_the_first_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nemo_gym.agents.utils.setup_retry.asyncio", SimpleNamespace(sleep=AsyncMock()))
    _, state, _ = _stopped_session(monkeypatch)
    made = SandboxExecResult(stdout="", stderr="", return_code=0)
    state.session.begin_activation = AsyncMock()
    state.session.sandbox = SimpleNamespace(exec=AsyncMock(side_effect=[ConnectionError("502 Bad Gateway"), made]))

    await state.prepare_launch(2, timeout=30)

    state.session.begin_activation.assert_awaited_once()
    calls = state.session.sandbox.exec.await_args_list
    assert [call.args for call in calls] == [("mkdir -p -- /sessions/s/launch-2",)] * 2


async def test_a_stopped_launch_with_no_budget_left_is_reported_as_timed_out_and_not_relaunched() -> None:
    agent = OpenCodeAgent.model_construct(
        config=OpenCodeAgentConfig.model_construct(
            session_close_timeout_seconds=1, checkpoint_pause_allowance_seconds=0
        )
    )
    stopped = {"cleanup_confirmed": True, "return_code": 143, "error": None, "timed_out": False, "stopped": True}
    session = SimpleNamespace(cleanup=stopped, artifacts='{"messages": []}', session_dir="/s/launch-1")
    state = SimpleNamespace(
        launch_count=1,
        launch_records=[],
        session=session,
        request=SimpleNamespace(agent_session_id="a"),
        prepare_launch=AsyncMock(),
        inspect_conversation=AsyncMock(),
        execute=AsyncMock(),
    )

    # The activation's deadline passed while the launch was being stopped.
    await agent._run_launches(
        state, MagicMock(), _fresh_payload, prompt="fix the bug", timeout=10, deadline=time.monotonic() - 1
    )

    # Preparing a relaunch would discard the stopped launch's output: it stays, under a receipt that says timeout.
    state.prepare_launch.assert_not_awaited()
    state.inspect_conversation.assert_not_awaited()
    state.execute.assert_not_awaited()
    assert session.cleanup == {**stopped, "timed_out": True}
    assert session.artifacts == '{"messages": []}' and state.launch_count == 1
