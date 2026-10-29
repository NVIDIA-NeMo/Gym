# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode sessions with partial-rollout checkpoints: pause in place, the stop fallback, and resume after a crash.

The harness conformance kit drives the real agent, the real process supervisor and the real ``resume_opencode.sh``
on this machine, with a fake ``opencode`` executable in place of the CLI. The fake keeps a conversation in an
OpenCode-shaped SQLite database, ``run`` answers once and then waits, and ``serve`` answers the resume request
``resume_opencode.sh`` sends.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_gym.agent_utils.sandbox_session import HarnessPauseResult, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.checkpoint_conformance import HarnessFixture, LocalExecProvider, check_harness_adapter
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig
from responses_api_agents.opencode_agent.resume import resume_decision, resume_payload
from responses_api_agents.opencode_agent.runtime import OPENCODE_VERSION


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision required")

RESUME_SCRIPT = Path(__file__).parents[1] / "resume_opencode.sh"

# The fake CLI. ``run`` saves the task's user turn and a first answer, then waits for the finish file and answers
# again; ``serve`` answers one resume request the same way. Both record their launch and their pid.
FAKE_OPENCODE = r"""#!{python}
import base64, http.server, json, os, pathlib, signal, sqlite3, sys, threading, time, urllib.parse

CONTROL = pathlib.Path({control!r})
VERSION = {version!r}


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


def started(kind):
    with (CONTROL / "launches").open("a") as launches:
        launches.write(kind + "\n")
    (CONTROL / "pids").mkdir(exist_ok=True)
    (CONTROL / "pids" / str(os.getpid())).touch()


def wait_for_finish():
    while not (CONTROL / "finish").exists():
        time.sleep(0.05)


def run():
    prompt = sys.stdin.read()
    started("run")
    plugin_ready()
    connection = database()
    if not connection.execute("select id from session").fetchall():
        connection.execute("insert into session values('ses_root', null, 0)")
    add(connection, "msg_001", {{"role": "user", "agent": "build", "model": {{"providerID": "nemo_gym", "modelID": "dummy_model"}}}},
        [{{"type": "text", "text": prompt}}])
    answer(connection, "msg_002", "working", "tool-calls")
    print(json.dumps({{"type": "step_finish"}}), flush=True)
    wait_for_finish()
    answer(connection, "msg_003", "done", "stop")


def serve():
    started("serve")
    plugin_ready()
    password = os.environ["OPENCODE_SERVER_PASSWORD"]
    expected = "Basic " + base64.b64encode(f"opencode:{{password}}".encode()).decode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ok = self.headers.get("Authorization") == expected and body.get("parts") == [] and "/session/ses_root/message" in self.path
            (CONTROL / "resume_requests").open("a").write(json.dumps({{"path": self.path, "body": body}}) + "\n")
            if ok:
                wait_for_finish()
                answer(database(), "msg_004", "resumed and done", "stop")
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({{"info": {{"id": "msg_004"}}}}).encode())

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(f"opencode server listening on http://127.0.0.1:{{server.server_address[1]}}", flush=True)
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
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
        self.binary.write_text(
            FAKE_OPENCODE.format(python=sys.executable, control=str(self.control), version=OPENCODE_VERSION)
        )
        self.binary.chmod(0o755)
        # Optional tools the runtime installer would otherwise fetch with a package manager.
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        (self.bin / "rg").write_text("#!/bin/sh\nexit 0\n")
        (self.bin / "rg").chmod(0o755)
        self.provider = LocalExecProvider()
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
                session_close_timeout_seconds=15,
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


async def test_opencode_conforms_to_the_harness_checkpoint_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = OpenCodeFixture(tmp_path, monkeypatch)
    try:
        await check_harness_adapter(fixture, tmp_path / "state")
    finally:
        fixture.finish()
        for pid in fixture.harness_pids():
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
    # Every resume re-posted the saved user turn with no new parts, so no turn was added to the conversation.
    requests = [json.loads(line) for line in (fixture.control / "resume_requests").read_text().splitlines()]
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


# -- the resume decision ---------------------------------------------------------------------------------------


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


def test_a_saved_conversation_of_this_task_that_the_model_answered_is_resumed_without_a_new_turn() -> None:
    decision = resume_decision(_export(USER, ANSWER, PENDING), "fix the bug")

    assert decision.resume and decision.session_id == "ses_root"
    assert decision.payload == {
        "messageID": "msg_1",
        "parts": [],
        "agent": "build",
        "model": {"providerID": "p", "modelID": "m"},
        "variant": "v",
    }


@pytest.mark.parametrize(
    "export, reason",
    [
        (None, "no saved OpenCode conversation"),
        (_export(ANSWER), "no user turn"),
        (_export(USER, ANSWER), "does not start with this task's prompt"),
        (_export(USER, PENDING), "had not answered"),
    ],
    ids=["none", "no_user", "another_task", "unanswered"],
)
def test_an_unusable_saved_conversation_starts_the_task_afresh(export: Any, reason: str) -> None:
    prompt = "another task" if reason.startswith("does not") else "fix the bug"
    decision = resume_decision(export, prompt)

    assert not decision.resume and reason in decision.reason


def test_the_resume_payload_re_enters_the_latest_user_turn() -> None:
    later = {"info": {"id": "msg_9", "role": "user"}, "parts": []}
    assert resume_payload([USER, ANSWER, later]) == {"messageID": "msg_9", "parts": []}
    with pytest.raises(ValueError):
        resume_payload([ANSWER])


# -- resume_opencode.sh --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("transport", ["curl", "dev_tcp"])
def test_the_resume_helper_re_posts_the_saved_user_turn_through_the_opencode_server(
    tmp_path: Path, transport: str
) -> None:
    control = tmp_path / "control"
    control.mkdir()
    (control / "launches").write_text("")
    (control / "finish").touch()
    binary = tmp_path / "opencode"
    binary.write_text(FAKE_OPENCODE.format(python=sys.executable, control=str(control), version=OPENCODE_VERSION))
    binary.chmod(0o755)
    data = tmp_path / "data"
    payload = tmp_path / "resume.json"
    payload.write_text(json.dumps({"messageID": "msg_001", "parts": []}))
    workdir = tmp_path / "task dir"
    workdir.mkdir()
    env = {**os.environ, "XDG_DATA_HOME": str(data), "OPENCODE_CONFIG_CONTENT": "{}"}
    if transport == "dev_tcp":
        # Only the tools the helper needs, and no curl: it speaks HTTP over bash's /dev/tcp.
        tools = tmp_path / "tools"
        tools.mkdir()
        for tool in ("bash", "mktemp", "seq", "sleep", "base64", "tr", "wc", "cat", "rm", "head", "od"):
            found = shutil.which(tool)
            if found:
                (tools / tool).symlink_to(found)
        (tools / "python3").symlink_to(sys.executable)
        env["PATH"] = str(tools)

    completed = subprocess.run(
        ["bash", str(RESUME_SCRIPT), str(binary), "ses_root", str(payload)],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    [request] = [json.loads(line) for line in (control / "resume_requests").read_text().splitlines()]
    assert request["body"] == {"messageID": "msg_001", "parts": []}
    # The server is asked for the session in the working directory the CLI would use.
    assert "directory=" + str(workdir).replace("/", "%2F").replace(" ", "%20") in request["path"]
    assert (control / "launches").read_text().splitlines() == ["serve"]


def test_the_resume_helper_reports_a_server_that_never_starts(tmp_path: Path) -> None:
    broken = tmp_path / "opencode"
    broken.write_text("#!/bin/sh\nexit 1\n")
    broken.chmod(0o755)
    payload = tmp_path / "resume.json"
    payload.write_text("{}")

    completed = subprocess.run(
        ["bash", str(RESUME_SCRIPT), str(broken), "ses", str(payload)], capture_output=True, text=True, timeout=120
    )

    assert completed.returncode == 3


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

    assert OpenCodeAgent.checkpoint_mode == "exported" and OpenCodeAgent.rebinds_tool_accesses
    assert not restarts(None, borrowed)
    assert restarts(None, owned) and restarts(None, continued)


def test_the_resume_helper_is_plain_bash() -> None:
    assert RESUME_SCRIPT.read_text().startswith("#!/usr/bin/env bash")
    subprocess.run(["bash", "-n", str(RESUME_SCRIPT)], check=True)
    assert "python" not in textwrap.dedent(RESUME_SCRIPT.read_text()).split("set +o errexit", 1)[1]
