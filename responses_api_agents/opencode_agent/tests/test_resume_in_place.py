# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode on a sandbox whose snapshots keep processes: hold at checkpoints, resume in place after a crash.

The sandbox is this machine, as in ``test_checkpoint``, with the real process supervisor and the real model relay. The
fake ``opencode`` makes one model call through the URL its config names after its first answer. A crash is the agent
server's death alone: its harness, supervisor and relay keep running, as a restore of a memory snapshot brings them
back. The model servers are local HTTP servers: the crashed attempt's holds the call, the replacement's answers it.
"""

import asyncio
import hashlib
import http.server
import json
import logging
import os
import shutil
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_gym.agents import AgentSeedSessionRequest, LaunchFinished, NotResumable, RestoredLaunch, SandboxSession
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, NgFailureEvent, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxCapabilities
from responses_api_agents.opencode_agent import app as opencode_app
from responses_api_agents.opencode_agent import runtime as opencode_runtime
from responses_api_agents.opencode_agent.sandbox import RELAY_ENDPOINT_FILE, OpenCodeSandboxSession
from responses_api_agents.opencode_agent.tests.test_checkpoint import (
    ANSWER,
    FAKE_OPENCODE,
    PERMISSION,
    REFUSED_PROMPT,
    USER,
    OpenCodeFixture,
    _export,
)


TOKEN = "resume-in-place-token"
_AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _control(checkpoint_id: str, **extra: Any) -> dict[str, Any]:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 120, **extra}


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision required")
JS_RUNTIME = os.environ.get("NEMO_GYM_TEST_JS_RUNTIME") or shutil.which("node") or shutil.which("bun")
# The relay is model_relay.mjs: a launch that resumes in place needs a JavaScript runtime to run it.
needs_relay_runtime = pytest.mark.skipif(
    JS_RUNTIME is None, reason="no JavaScript runtime (node or bun) for the relay"
)

# What the model-calling fake adds to test_checkpoint's, inserted before the fake dispatches on its arguments, so no
# line of the fake's own functions is patched. ``run`` counts itself as started once it has saved its first answer:
# the model call follows that, through the URL in its config, and its reply is kept. Like OpenCode's binary with
# ``BUN_BE_BUN=1``, the fake runs a script instead (the model relay), here on this machine's JavaScript runtime.
_MODEL_CALL = """if os.environ.get("BUN_BE_BUN") == "1":
    os.execv({js_runtime!r}, [{js_runtime!r}, *sys.argv[1:]])


def call_model():
    import urllib.request

    config = json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])
    base = config["provider"]["nemo_gym"]["options"]["baseURL"]
    body = json.dumps({{"model": "m", "messages": [{{"role": "user", "content": "one step"}}]}}).encode()
    request = urllib.request.Request(base + "/chat/completions", data=body, headers={{"Content-Type": "application/json"}})
    with urllib.request.urlopen(request, timeout=600) as response:
        (CONTROL / "model_reply").write_bytes(response.read())


_count_started = started


def started(kind):
    _count_started(kind)
    if kind == "run":
        call_model()


"""
_DISPATCH = "signal.signal(signal.SIGTERM, lambda *_: os._exit(143))\n"
assert FAKE_OPENCODE.count(_DISPATCH) == 1, "test_checkpoint's fake OpenCode lacks the line the model call goes before"
MODEL_CALLING_OPENCODE = FAKE_OPENCODE.replace(_DISPATCH, _MODEL_CALL + _DISPATCH)


class ModelServer:
    """A model server on a free local port that records each call and either answers it or holds it."""

    def __init__(self, *, answer: bool) -> None:
        self.calls: list[dict[str, Any]] = []
        self.release = threading.Event()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                server.calls.append({"path": self.path, "sha256": hashlib.sha256(body).hexdigest()})
                if not answer:
                    # Held until the test stops this server: a call dropped earlier is one the relay sends again.
                    server.release.wait()
                    return
                reply = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


class InPlaceFixture(OpenCodeFixture):
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__(tmp_path, monkeypatch)
        self.binary.write_text(
            MODEL_CALLING_OPENCODE.format(
                python=sys.executable,
                control=str(self.control),
                version=opencode_runtime.OPENCODE_VERSION,
                permission=PERMISSION,
                refused=REFUSED_PROMPT,
                js_runtime=JS_RUNTIME or "",
            )
        )
        # This machine as a sandbox whose snapshots keep processes.
        self.provider.capabilities = SandboxCapabilities(snapshot_keeps=frozenset({"files", "processes"}))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.relay_port = probe.getsockname()[1]

    def make_agent(self, global_config: dict[str, Any], *, model_url: str) -> Any:
        agent = super().make_agent(global_config)
        agent.config.sandbox_model_relay_port = self.relay_port
        agent.server_client._build_server_base_url = MagicMock(return_value=model_url)
        return agent

    def session_dirs(self) -> list[Path]:
        return sorted(path.parent for path in (self.tmp_path / "sessions").rglob("params.sh"))


def _state(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return None


async def _until(condition: Any, timeout: float = 120) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.05)


async def _crash_after_a_checkpoint(fixture: InPlaceFixture, state_dir: Path, held: ModelServer) -> Any:
    """Run a session until its model call is held, checkpoint it, and leave its agent server for dead."""
    import httpx

    config = {"checkpoint": {"enabled": True, "control_auth_token": TOKEN, "state_dir": str(state_dir)}}
    agent = fixture.make_agent(config, model_url=held.url)
    app = agent.setup_webserver()
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent", timeout=120)
    seeded = await http.post("/v1/agent_sessions", json=fixture.seed("kept", 0))
    assert seeded.status_code == 200, seeded.text
    activation = asyncio.create_task(
        http.post("/ng-rollout/r/v1/responses", json=fixture.activation(), headers={"x-ng-agent-activation": "1"})
    )
    await _until(lambda: held.calls)
    # The launch must be calling through its relay: a relay that did not come up falls back to a direct call, and
    # the restore would then relaunch.
    [launch_dir] = fixture.session_dirs()
    relay_log = launch_dir / "relay.log"
    assert not (launch_dir / "unavailable").exists(), (
        f"the model relay did not start on port {fixture.relay_port}: "
        f"{relay_log.read_text() if relay_log.exists() else 'no relay.log'}"
    )
    await _until(lambda: (launch_dir / "relay-status.json").exists(), 10)
    for operation, extra in (("prepare", {}), ("commit", {"checkpoint_dir": str(state_dir / "ckpt")})):
        reply = await http.post(f"/ng-control/v1/checkpoint/{operation}", json=_control("c1", **extra), headers=_AUTH)
        assert reply.status_code == 200, f"{operation}: {reply.text}"
    # Holding, not stopping: no harness process was signalled.
    assert all(_state(pid) not in ("T", "t") for pid in fixture.harness_pids())
    # The agent server dies here; the sandbox's processes do not. Its tasks are left as a dead process leaves them.
    return config, (agent, http, activation)


async def _replacement(
    fixture: InPlaceFixture,
    config: dict[str, Any],
    state_dir: Path,
    model_url: str,
    *,
    checkpoint: str = "ckpt",
    restore_id: str = "r1",
    attempt: int = 1,
) -> Any:
    """A new agent server that restores ``checkpoint`` and is seeded for ``attempt``, which continues the one before."""
    import httpx

    agent = fixture.make_agent(config, model_url=model_url)
    app = agent.setup_webserver()
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent", timeout=120)
    # A restore names the attempts it continues: the checkpoint's record is the replaced attempt's.
    replaced = {"rollout_id": "r", "attempt": attempt - 1}
    for operation, extra in (
        ("restore", {"checkpoint_dir": str(state_dir / checkpoint), "episode_ids": [replaced]}),
        ("resume", {}),
    ):
        reply = await http.post(
            f"/ng-control/v1/checkpoint/{operation}", json=_control(restore_id, **extra), headers=_AUTH
        )
        assert reply.status_code == 200, f"{operation}: {reply.text}"
    seeded = await http.post("/v1/agent_sessions", json=fixture.seed("kept", attempt))
    assert seeded.status_code == 200, seeded.text
    return agent, http


async def _clean_up(fixture: InPlaceFixture, *servers: ModelServer, crashed: Any = None) -> None:
    """Leave nothing of a test behind, whether or not it passed."""
    fixture.finish()
    for server in servers:
        server.stop()
    # A supervisor still holding a pause ends on its stop file; a harness in hold mode is never stopped for it.
    for launch_dir in fixture.session_dirs():
        (launch_dir / "stop.request").touch()
    for pid in fixture.harness_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if crashed is not None:
        _, http, activation = crashed
        activation.cancel()
        await asyncio.gather(activation, return_exceptions=True)
        await http.aclose()


@needs_relay_runtime
async def test_a_restore_that_kept_the_processes_continues_the_same_launch_against_the_new_model_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = InPlaceFixture(tmp_path, monkeypatch)
    crashed_server, new_server = ModelServer(answer=False), ModelServer(answer=True)
    endpoints_at_continue: list[dict[str, Any]] = []
    original_adopt = OpenCodeSandboxSession.adopt
    crashed = None

    async def recording_adopt(state: OpenCodeSandboxSession, **kwargs: Any) -> Any:
        # The take-over is what continues the launch: this is the endpoint its relay reads when it does.
        endpoint = Path(state.require_session().session_dir) / RELAY_ENDPOINT_FILE
        endpoints_at_continue.append(json.loads(endpoint.read_text()))
        return await original_adopt(state, **kwargs)

    try:
        config, crashed = await _crash_after_a_checkpoint(fixture, tmp_path / "state", crashed_server)
        harness = fixture.harness_pids()
        monkeypatch.setattr(OpenCodeSandboxSession, "adopt", recording_adopt)
        caplog.set_level(logging.WARNING)
        _, http = await _replacement(fixture, config, tmp_path / "state", new_server.url)
        fixture.finish()
        reply = await http.post(
            "/ng-rollout/r-a1/v1/responses", json=fixture.activation(), headers={"x-ng-agent-activation": "1"}
        )

        assert reply.status_code == 200, reply.text
        assert fixture.started() == 1, "the launch that ran through the restore finished; nothing was relaunched"
        assert reply.json()["metadata"]["harness_launches"] == "1"
        assert harness and not fixture.harness_pids(), "the same harness processes ran on to the end"
        # The relay re-sent the call the checkpoint held, byte for byte, to the replacement attempt's model URL.
        [held], [resent] = crashed_server.calls, new_server.calls
        assert resent["sha256"] == held["sha256"]
        assert "/r-a1" in resent["path"] and "/r-a1" not in held["path"]
        assert json.loads((fixture.control / "model_reply").read_text())["choices"][0]["message"]["content"] == "ok"
        # The relay was pointed at the replacement attempt before the launch continued.
        assert endpoints_at_continue[0]["gen"] == 2 and "/r-a1" in endpoints_at_continue[0]["upstream"]
        assert "OpenCode resumed in place: session kept, launch 1, pause 1, relay generation 2" in caplog.text
    finally:
        await _clean_up(fixture, crashed_server, new_server, crashed=crashed)


@needs_relay_runtime
async def test_a_launch_taken_over_once_is_checkpointed_and_taken_over_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = InPlaceFixture(tmp_path, monkeypatch)
    first, second, third = ModelServer(answer=False), ModelServer(answer=False), ModelServer(answer=True)
    state_dir = tmp_path / "state"
    activation = {"x-ng-agent-activation": "1"}
    crashed = taken_over = None
    try:
        config, crashed = await _crash_after_a_checkpoint(fixture, state_dir, first)
        agent, http = await _replacement(fixture, config, state_dir, second.url)
        # The first replacement takes the launch over; its model server holds the re-sent call too.
        running = asyncio.create_task(
            http.post("/ng-rollout/r-a1/v1/responses", json=fixture.activation(), headers=activation)
        )
        taken_over = (agent, http, running)
        await _until(lambda: second.calls)
        for operation, extra in (("prepare", {}), ("commit", {"checkpoint_dir": str(state_dir / "ckpt2")})):
            reply = await http.post(
                f"/ng-control/v1/checkpoint/{operation}", json=_control("c2", **extra), headers=_AUTH
            )
            assert reply.status_code == 200, f"{operation}: {reply.text}"
        # The second crash: again the agent server alone.
        caplog.set_level(logging.WARNING)
        _, http = await _replacement(
            fixture, config, state_dir, third.url, checkpoint="ckpt2", restore_id="r2", attempt=2
        )
        fixture.finish()
        reply = await http.post("/ng-rollout/r-a2/v1/responses", json=fixture.activation(), headers=activation)

        assert reply.status_code == 200, reply.text
        assert fixture.started() == 1, "the same launch ran through both restores"
        assert reply.json()["metadata"]["harness_launches"] == "1"
        [held], [again], [resent] = first.calls, second.calls, third.calls
        assert resent["sha256"] == again["sha256"] == held["sha256"]
        assert "/r-a1" in again["path"] and "/r-a2" in resent["path"]
        assert "OpenCode resumed in place: session kept, launch 1, pause 2, relay generation 3" in caplog.text
        assert "cannot resume in place" not in caplog.text
    finally:
        await _clean_up(fixture, first, second, third, crashed=crashed)
        await _clean_up(fixture, crashed=taken_over)


@needs_relay_runtime
async def test_a_restored_launch_whose_supervisor_died_is_abandoned_and_relaunched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = InPlaceFixture(tmp_path, monkeypatch)
    crashed_server, new_server = ModelServer(answer=False), ModelServer(answer=True)
    crashed = None
    try:
        config, crashed = await _crash_after_a_checkpoint(fixture, tmp_path / "state", crashed_server)
        [launch_dir] = fixture.session_dirs()
        identity = json.loads((launch_dir / "paused.json").read_text())["identity"]
        os.kill(int(identity.split(".")[0]), signal.SIGKILL)
        survivors = fixture.harness_pids()
        assert survivors, "the harness outlives its supervisor"
        caplog.set_level(logging.WARNING)
        _, http = await _replacement(fixture, config, tmp_path / "state", new_server.url)

        # The seed ended the launch it could not take over: nothing of it runs beside the relaunch.
        assert not set(survivors) & set(fixture.harness_pids())
        assert "OpenCode cannot resume in place: session kept, reason supervisor_gone" in caplog.text
        assert "OpenCode abandoned a restored launch: session kept, reason supervisor_gone" in caplog.text
        fixture.finish()
        reply = await http.post(
            "/ng-rollout/r-a1/v1/responses", json=fixture.activation(), headers={"x-ng-agent-activation": "1"}
        )
        assert reply.status_code == 200, reply.text
        assert fixture.started() == 2, "the abandoned launch is relaunched once, resuming the saved conversation"
        assert reply.json()["metadata"]["harness_launches"] == "2"
    finally:
        await _clean_up(fixture, crashed_server, new_server, crashed=crashed)


# -- the decision after a restore -----------------------------------------------------------------------------------


def _restored_state(agent: Any) -> OpenCodeSandboxSession:
    session = MagicMock()
    session.session_dir = "/sessions/s/launch-1"
    request = AgentSeedSessionRequest(
        agent_session_id="s", episode_id=EpisodeId(rollout_id="r", attempt=1), task_id=TaskId(taskset="t", task_id="0")
    )
    return OpenCodeSandboxSession(
        request=request, session=session, runtime="/rt", checkpointed=True, keeps_processes=True, relay_gen=1
    )


@pytest.mark.parametrize(
    "verdict, adopted, abandoned",
    [
        (RestoredLaunch(pause_seq=3, identity="10.20"), True, False),
        (NotResumable("launch_finished"), True, False),
        (NotResumable("relay_gone"), False, True),
        (NotResumable("pause_mismatch", "a continue was written"), False, True),
    ],
    ids=["resumable", "finished", "relay_gone", "pause_mismatch"],
)
async def test_a_restored_launch_is_taken_over_or_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: Any, adopted: bool, abandoned: bool
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _restored_state(agent)
    checks: list[dict[str, Any]] = []
    ended: list[str] = []

    async def check(sandbox: Any, session_dir: str, **kwargs: Any) -> Any:
        checks.append({"session_dir": session_dir, **kwargs})
        return verdict

    async def abandon(sandbox: Any, session_dir: str, *, identity: str, **kwargs: Any) -> dict:
        ended.append(identity)
        return {"stopped": True, "abandoned": True, "cleanup_confirmed": True, "return_code": None}

    monkeypatch.setattr(opencode_app, "check_restored", check)
    monkeypatch.setattr(opencode_app, "abandon_launch", abandon)

    await agent._check_restored_launch(state, {"pause_seq": 3, "supervisor_identity": "10.20"})

    assert checks[0]["pause_seq"] == 3 and checks[0]["relay_status_path"].endswith("/launch-1/relay-status.json")
    assert (state.adoption == RestoredLaunch(pause_seq=3, identity="10.20")) is adopted
    # A launch that had ended is only collected: its take-over leaves the relay alone.
    assert state.adoption_ended is (adopted and isinstance(verdict, NotResumable))
    assert ended == (["10.20"] if abandoned else [])


async def test_a_restored_session_without_a_held_pause_has_nothing_to_take_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _restored_state(agent)
    monkeypatch.setattr(opencode_app, "check_restored", MagicMock(side_effect=AssertionError("not checked")))

    await agent._check_restored_launch(state, {"session_root": "/s", "session_dir": "/s", "launch_started": True})

    assert state.adoption is None


async def test_a_restored_launch_without_a_held_pause_is_stopped_before_its_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _restored_state(agent)
    state.launch_count = 1
    stops: list[dict[str, Any]] = []

    async def stop(sandbox: Any, **kwargs: Any) -> dict:
        stops.append(kwargs)
        if len(stops) == 2:
            raise RuntimeError("OpenCode cleanup receipt missing")
        return {"stopped": True, "cleanup_confirmed": True, "return_code": None}

    monkeypatch.setattr(opencode_app, "stop_and_confirm_cleanup", stop)
    monkeypatch.setattr(opencode_app, "check_restored", MagicMock(side_effect=AssertionError("not checked")))
    saved = {"session_root": "/s", "session_dir": "/s", "launch_started": True}

    await agent._check_restored_launch(state, saved)

    assert [call["session_dir"] for call in stops] == ["/sessions/s/launch-1"] and stops[0]["harness"] == "OpenCode"
    assert state.adoption is None
    # A stop that fails is reported and the session relaunches all the same.
    await agent._check_restored_launch(state, saved)
    assert len(stops) == 2 and "could not stop the restored launch of session s" in caplog.text


async def test_a_restored_launch_check_that_could_not_run_is_made_once_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _restored_state(agent)
    verdicts = [NotResumable("check_failed", "exit 255"), RestoredLaunch(pause_seq=3, identity="10.20")]

    async def check(sandbox: Any, session_dir: str, **kwargs: Any) -> Any:
        return verdicts.pop(0)

    monkeypatch.setattr(opencode_app, "check_restored", check)
    monkeypatch.setattr(opencode_app, "abandon_launch", MagicMock(side_effect=AssertionError("not abandoned")))
    caplog.set_level(logging.WARNING)

    await agent._check_restored_launch(state, {"pause_seq": 3, "supervisor_identity": "10.20"})

    assert verdicts == [] and state.adoption == RestoredLaunch(pause_seq=3, identity="10.20")
    assert "restored-launch check failed for session s (exit 255); checking once more" in caplog.text


async def test_a_restored_launch_check_that_cannot_run_twice_fails_the_re_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _restored_state(agent)
    checks: list[str] = []

    async def check(sandbox: Any, session_dir: str, **kwargs: Any) -> Any:
        checks.append(session_dir)
        return NotResumable("check_failed", "exit 255")

    monkeypatch.setattr(opencode_app, "check_restored", check)
    monkeypatch.setattr(opencode_app, "abandon_launch", MagicMock(side_effect=AssertionError("not abandoned")))

    with pytest.raises(NgFailureEvent, match="could not check the restored launch of session s: exit 255"):
        await agent._check_restored_launch(state, {"pause_seq": 3, "supervisor_identity": "10.20"})

    # Nothing is known about the launch: it is neither taken over nor ended, and the retried seed checks again.
    assert len(checks) == 2 and state.adoption is None and not state.adoption_ended


async def test_a_restored_launch_that_cannot_be_ended_is_reported_and_not_taken_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _restored_state(agent)
    attempts: list[str] = []

    async def check(sandbox: Any, session_dir: str, **kwargs: Any) -> Any:
        return NotResumable("supervisor_gone")

    async def abandon(sandbox: Any, session_dir: str, *, identity: str, **kwargs: Any) -> dict:
        attempts.append(identity)
        raise RuntimeError("OpenCode abandoned launch left no receipt")

    monkeypatch.setattr(opencode_app, "check_restored", check)
    monkeypatch.setattr(opencode_app, "abandon_launch", abandon)
    caplog.set_level(logging.WARNING)

    await agent._check_restored_launch(state, {"pause_seq": 3, "supervisor_identity": "10.20"})

    # The abandon kills and is safe to repeat: it is tried twice, then reported, and the seed goes on.
    assert attempts == ["10.20", "10.20"] and state.adoption is None
    assert "abandon of the restored launch of session s failed" in caplog.text
    assert "could not abandon the restored launch of session s (reason supervisor_gone)" in caplog.text


async def test_a_rebound_session_not_yet_taken_over_exports_the_pause_it_still_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _session_state(agent, tmp_path, keeps_processes=True)
    state.launch_count = 1
    state.reprompted_conversation = True
    state.adoption = RestoredLaunch(pause_seq=3, identity="10.20")
    monkeypatch.setattr(type(agent), "_session_state", lambda self, key: state)

    exported = (await agent.export_agent_sessions(["s"], checkpoint_id="c"))["s"]

    # The session itself launched nothing, yet the launch the restore brought back still holds this pause.
    assert exported["session"] == {
        "session_root": str(tmp_path / "s"),
        "session_dir": str(tmp_path / "s"),
        "launch_started": True,
        "pause_seq": 3,
        "supervisor_identity": "10.20",
    }
    assert exported["reprompted_conversation"] is True and exported["relay_gen"] == 1
    # Taken over or ended, the session exports what it holds itself.
    state.adoption = None
    exported = (await agent.export_agent_sessions(["s"], checkpoint_id="c"))["s"]
    assert exported["session"]["launch_started"] is False and "pause_seq" not in exported["session"]


def _relaunching(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    """Stand in for the sandbox side of a relaunch and of an abandon, recording each in ``events``."""

    async def abandon(sandbox: Any, session_dir: str, *, identity: str, **kwargs: Any) -> dict:
        events.append(f"abandon {identity}")
        return {"stopped": True, "abandoned": True, "cleanup_confirmed": True, "return_code": None}

    async def prepare_launch(self: OpenCodeSandboxSession, launch: int, *, timeout: float) -> None:
        events.append(f"prepare {launch}")

    async def inspect_conversation(self: OpenCodeSandboxSession, *, timeout: float) -> dict:
        return _export(USER, ANSWER)

    async def execute(self: OpenCodeSandboxSession, payload: dict, **kwargs: Any) -> str:
        events.append("reprompt" if payload.get("reprompt") is not None else "fresh")
        return ""

    async def run(
        session: Any, activation: Any, *, launch: Any, on_parked: Any, stop_timeout: float
    ) -> LaunchFinished:
        return LaunchFinished(await launch())

    for name, value in (
        ("prepare_launch", prepare_launch),
        ("inspect_conversation", inspect_conversation),
        ("execute", execute),
    ):
        monkeypatch.setattr(OpenCodeSandboxSession, name, value)
    monkeypatch.setattr(opencode_app, "abandon_launch", abandon)
    monkeypatch.setattr(opencode_app, "run_with_checkpoint_pauses", run)


@pytest.mark.parametrize("failing", ["point_relay", "adopt"])
async def test_a_take_over_that_fails_ends_the_restored_launch_before_its_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failing: str
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _session_state(agent, tmp_path, keeps_processes=True)
    state.launch_count = 1
    state.adoption = RestoredLaunch(pause_seq=1, identity="10.20")
    events: list[str] = []
    _relaunching(monkeypatch, events)

    async def point_relay(self: OpenCodeSandboxSession, upstream: str) -> None:
        if failing == "point_relay":
            raise ConnectionError("502 Bad Gateway")

    async def adopt(self: OpenCodeSandboxSession, **kwargs: Any) -> str:
        # As SandboxSession.adopt leaves a launch whose continue was not confirmed and whose stop then failed.
        self.session.launch_started = True
        raise RuntimeError("OpenCode restored launch did not continue")

    monkeypatch.setattr(OpenCodeSandboxSession, "point_relay", point_relay)
    monkeypatch.setattr(OpenCodeSandboxSession, "adopt", adopt)
    caplog.set_level(logging.WARNING)

    await agent._run_launches(
        state, MagicMock(), lambda: {"command": ["opencode", "run"]}, prompt="fix the bug", timeout=10, deadline=None
    )

    # Nothing of the restored launch may run beside its relaunch, whichever step of the take-over failed.
    assert events == ["abandon 10.20", "prepare 2", "reprompt"]
    assert state.adoption is None and state.launch_count == 2
    [record] = state.launch_records
    assert record["launch"] == 2 and record["reprompted"] is True and "resumed_in_place" not in record
    assert "reason adopt_failed" in caplog.text and "resumed in place" not in caplog.text


@pytest.mark.parametrize(
    "receipt, relaunched",
    [
        ({"stopped": True, "abandoned": True, "cleanup_confirmed": True, "return_code": None}, True),
        # The held pause outlasted --max-pause before the take-over: the supervisor ended OpenCode at its limit.
        (
            {
                "stopped": False,
                "timed_out": True,
                "pause_expired": True,
                "cleanup_confirmed": True,
                "return_code": 143,
            },
            True,
        ),
        ({"stopped": False, "cleanup_confirmed": True, "return_code": 0}, False),
    ],
    ids=["stopped", "pause_expired", "ended_by_itself"],
)
async def test_a_launch_taken_over_after_it_was_stopped_is_relaunched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, receipt: dict, relaunched: bool
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _session_state(agent, tmp_path, keeps_processes=True)
    state.launch_count = 1
    state.adoption = RestoredLaunch(pause_seq=1, identity="10.20")
    events: list[str] = []
    _relaunching(monkeypatch, events)

    async def point_relay(self: OpenCodeSandboxSession, upstream: str) -> None:
        pass

    async def adopt(self: OpenCodeSandboxSession, **kwargs: Any) -> str:
        # The restored launch had ended before the take-over: only its receipt is left to read.
        self.session.launch_started = True
        self.session.cleanup = receipt
        return ""

    monkeypatch.setattr(OpenCodeSandboxSession, "point_relay", point_relay)
    monkeypatch.setattr(OpenCodeSandboxSession, "adopt", adopt)
    caplog.set_level(logging.WARNING)

    await agent._run_launches(
        state, MagicMock(), lambda: {"command": ["opencode", "run"]}, prompt="fix the bug", timeout=10, deadline=None
    )

    # A stop, an abandon or the pause's limit ended it, not OpenCode: its conversation is not finished, and a relaunch
    # resumes it. Only a launch that ran on after the take-over resumed in place.
    assert events == (["prepare 2", "reprompt"] if relaunched else [])
    resumed = "OpenCode resumed in place: session s, launch 1, pause 1, relay generation 1" in caplog.text
    assert resumed is not relaunched
    outcomes = [
        (record["launch"], record.get("resumed_in_place", False), record["reprompted"], record["outcome"])
        for record in state.launch_records
    ]
    assert outcomes == (
        [(1, True, False, "stopped"), (2, False, True, "finished")] if relaunched else [(1, True, False, "finished")]
    )


async def test_a_take_over_of_a_launch_that_ended_by_itself_fails_the_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _session_state(agent, tmp_path, keeps_processes=True)
    state.launch_count = 1
    state.adoption = RestoredLaunch(pause_seq=1, identity="10.20")
    events: list[str] = []
    _relaunching(monkeypatch, events)

    async def point_relay(self: OpenCodeSandboxSession, upstream: str) -> None:
        pass

    async def adopt(self: OpenCodeSandboxSession, **kwargs: Any) -> str:
        self.session.launch_started = True
        self.session.cleanup = {"stopped": False, "cleanup_confirmed": True, "return_code": 1}
        raise RuntimeError("OpenCode exited with code 1")

    monkeypatch.setattr(OpenCodeSandboxSession, "point_relay", point_relay)
    monkeypatch.setattr(OpenCodeSandboxSession, "adopt", adopt)

    # As after `execute`: the launch ran to its own end, so its failure is not a reason to run it again.
    with pytest.raises(RuntimeError, match="exited with code 1"):
        await agent._run_launches(
            state,
            MagicMock(),
            lambda: {"command": ["opencode", "run"]},
            prompt="fix the bug",
            timeout=10,
            deadline=None,
        )

    assert events == []


async def test_a_failed_take_over_of_a_launch_whose_held_pause_expired_relaunches_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _session_state(agent, tmp_path, keeps_processes=True)
    state.launch_count = 1
    state.adoption = RestoredLaunch(pause_seq=1, identity="10.20")
    events: list[str] = []
    _relaunching(monkeypatch, events)

    async def point_relay(self: OpenCodeSandboxSession, upstream: str) -> None:
        pass

    async def adopt(self: OpenCodeSandboxSession, **kwargs: Any) -> str:
        self.session.launch_started = True
        self.session.cleanup = {
            "stopped": False,
            "timed_out": True,
            "pause_expired": True,
            "cleanup_confirmed": True,
            "return_code": 143,
        }
        raise RuntimeError("OpenCode exited with code 143")

    monkeypatch.setattr(OpenCodeSandboxSession, "point_relay", point_relay)
    monkeypatch.setattr(OpenCodeSandboxSession, "adopt", adopt)

    await agent._run_launches(
        state, MagicMock(), lambda: {"command": ["opencode", "run"]}, prompt="fix the bug", timeout=10, deadline=None
    )

    # The supervisor ended it at the pause's limit, not OpenCode: nothing is left to abandon, and a relaunch resumes it.
    assert events == ["prepare 2", "reprompt"]


async def test_a_restored_launch_that_had_ended_is_collected_without_pointing_its_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _session_state(agent, tmp_path, keeps_processes=True)
    state.launch_count = 1
    state.adoption = RestoredLaunch(pause_seq=1, identity="10.20")
    state.adoption_ended = True
    events: list[str] = []
    _relaunching(monkeypatch, events)

    async def point_relay(self: OpenCodeSandboxSession, upstream: str) -> None:
        raise ConnectionError("502 Bad Gateway")

    async def adopt(self: OpenCodeSandboxSession, **kwargs: Any) -> str:
        self.session.launch_started = True
        self.session.cleanup = {"stopped": False, "cleanup_confirmed": True, "return_code": 0}
        return ""

    monkeypatch.setattr(OpenCodeSandboxSession, "point_relay", point_relay)
    monkeypatch.setattr(OpenCodeSandboxSession, "adopt", adopt)
    caplog.set_level(logging.INFO)

    await agent._run_launches(
        state, MagicMock(), lambda: {"command": ["opencode", "run"]}, prompt="fix the bug", timeout=10, deadline=None
    )

    # Its result is collected as it was left: no relay upload that could fail, no abandon, no relaunch.
    assert events == [] and not state.adoption_ended
    assert [(record["launch"], record["outcome"]) for record in state.launch_records] == [(1, "finished")]
    assert "collecting a restored launch that had ended: session s, launch 1, pause 1" in caplog.text
    assert "resumed in place" not in caplog.text


async def test_closing_a_session_whose_restored_launch_was_not_taken_over_ends_that_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _session_state(agent, tmp_path, keeps_processes=True)
    state.launch_count = 1
    state.adoption = RestoredLaunch(pause_seq=1, identity="10.20")
    events: list[str] = []
    _relaunching(monkeypatch, events)

    async def close(self: OpenCodeSandboxSession, timeout: float) -> None:
        events.append("close")

    monkeypatch.setattr(OpenCodeSandboxSession, "close", close)

    await agent._close_agent_session_state(state)

    assert events == ["abandon 10.20", "close"] and state.adoption is None


# -- the launch's model URL and relay -------------------------------------------------------------------------------


def _session_state(agent: Any, tmp_path: Path, *, keeps_processes: bool) -> OpenCodeSandboxSession:
    sandbox = MagicMock(spec=AsyncSandbox)
    session = SandboxSession(sandbox=sandbox, session_dir=str(tmp_path / "s"), workdir="/work", harness="OpenCode")
    request = AgentSeedSessionRequest(
        agent_session_id="s", episode_id=EpisodeId(rollout_id="r"), task_id=TaskId(taskset="t", task_id="0")
    )
    return OpenCodeSandboxSession(
        request=request,
        session=session,
        runtime="/rt",
        checkpointed=True,
        keeps_processes=keeps_processes,
        relay_gen=1 if keeps_processes else 0,
    )


def _base_url(payload: dict[str, Any]) -> str:
    return json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])["provider"]["nemo_gym"]["options"]["baseURL"]


def test_a_launch_on_a_process_keeping_sandbox_calls_the_model_through_the_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    direct = agent.resolve_model_base_url("policy_model", "r")

    payload = agent._launch_payload(
        _session_state(agent, tmp_path, keeps_processes=True),
        NeMoGymResponseCreateParamsNonStreaming(input="p"),
        prompt="p",
        system="",
    )

    assert _base_url(payload) == "http://127.0.0.1:18431/v1"
    assert payload["relay"]["port"] == 18431 and payload["relay"]["upstream"] == direct
    # Without a relay in the sandbox, the fallback environment points OpenCode at the model server directly.
    [fallback] = payload["relay"]["fallback_env"]
    name, value = fallback.split("=", 1)
    assert name == "OPENCODE_CONFIG_CONTENT"
    assert json.loads(value)["provider"]["nemo_gym"]["options"]["baseURL"] == direct


def test_a_launch_elsewhere_calls_the_model_server_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})

    payload = agent._launch_payload(
        _session_state(agent, tmp_path, keeps_processes=False),
        NeMoGymResponseCreateParamsNonStreaming(input="p"),
        prompt="p",
        system="",
    )

    assert "relay" not in payload and _base_url(payload) == agent.resolve_model_base_url("policy_model", "r")


async def test_staging_a_relayed_launch_writes_the_relay_endpoint_and_its_runner_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = OpenCodeFixture(tmp_path, monkeypatch)
    agent = fixture.make_agent({})
    sandbox = await AsyncSandbox.connect({"sandbox_id": "local"}, provider=fixture.provider)
    state = _session_state(agent, tmp_path, keeps_processes=True)
    state.session = SandboxSession(
        sandbox=sandbox, session_dir=str(tmp_path / "s"), workdir=str(tmp_path), harness="OpenCode"
    )
    payload = agent._launch_payload(state, NeMoGymResponseCreateParamsNonStreaming(input="p"), prompt="p", system="")

    await state.stage_activation(payload)

    endpoint = json.loads((tmp_path / "s" / RELAY_ENDPOINT_FILE).read_text())
    assert endpoint == {"gen": 1, "upstream": payload["relay"]["upstream"]}
    params = (tmp_path / "s" / "params.sh").read_text()
    for name in ("MODEL_RELAY_PORT", "MODEL_RELAY_ENDPOINT", "MODEL_RELAY_STATUS"):
        assert f"{name}=" in params
    assert "MODEL_RELAY_FALLBACK_ENV=(" in params
    # The relay runs on OpenCode's own binary as Bun, whatever the image holds; the switch reaches the relay alone.
    relay_command = f"MODEL_RELAY_COMMAND=({state.runtime}/opencode {state.persistent_directory}/model_relay.mjs)"
    assert relay_command in params
    assert "MODEL_RELAY_COMMAND_ENV=(BUN_BE_BUN=1)" in params
    assert "BUN_BE_BUN" not in payload["env"]


async def test_pointing_the_relay_raises_its_endpoint_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = OpenCodeFixture(tmp_path, monkeypatch)
    agent = fixture.make_agent({})
    sandbox = await AsyncSandbox.connect({"sandbox_id": "local"}, provider=fixture.provider)
    state = _session_state(agent, tmp_path, keeps_processes=True)
    state.session = SandboxSession(
        sandbox=sandbox, session_dir=str(tmp_path / "s"), workdir=str(tmp_path), harness="OpenCode"
    )

    await state.point_relay("http://gym-b/ng-rollout/r-a1/v1")

    assert state.relay_gen == 2
    endpoint = json.loads((tmp_path / "s" / RELAY_ENDPOINT_FILE).read_text())
    assert endpoint == {"gen": 2, "upstream": "http://gym-b/ng-rollout/r-a1/v1"}


# -- launches that cannot resume in place by design ------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict, level, detail",
    [
        (NotResumable("relay_unavailable"), "WARNING", "None"),
        (NotResumable("relay_gone"), "INFO", "None"),
        (NotResumable("pause_mismatch", "a continue was written"), "WARNING", "a continue was written"),
    ],
    ids=["relay_unavailable", "relay_gone", "pause_mismatch"],
)
async def test_only_expected_restore_fallbacks_are_logged_at_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    verdict: Any,
    level: str,
    detail: str,
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _restored_state(agent)

    async def check(sandbox: Any, session_dir: str, **kwargs: Any) -> Any:
        return verdict

    async def abandon(sandbox: Any, session_dir: str, *, identity: str, **kwargs: Any) -> dict:
        return {"stopped": True, "abandoned": True, "cleanup_confirmed": True, "return_code": None}

    monkeypatch.setattr(opencode_app, "check_restored", check)
    monkeypatch.setattr(opencode_app, "abandon_launch", abandon)
    caplog.set_level(logging.INFO)

    await agent._check_restored_launch(state, {"pause_seq": 3, "supervisor_identity": "10.20"})

    [cannot] = [r for r in caplog.records if "cannot resume in place" in r.getMessage()]
    [abandoned] = [r for r in caplog.records if "abandoned a restored launch" in r.getMessage()]
    assert cannot.levelname == abandoned.levelname == level
    assert f"detail {detail};" in cannot.getMessage()
