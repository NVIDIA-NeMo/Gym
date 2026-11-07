# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode on a sandbox whose snapshots keep processes: hold at checkpoints, resume in place after a crash.

The sandbox is this machine, as in ``test_checkpoint``, with the real process supervisor and the real model relay. The
fake ``opencode`` makes one model call through the URL its config names before it answers. A crash is the agent
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

from nemo_gym.agents import AgentSeedSessionRequest, NotResumable, RestoredLaunch, SandboxSession
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_runtime import LocalExecProvider
from responses_api_agents.opencode_agent import app as opencode_app
from responses_api_agents.opencode_agent.sandbox import RELAY_ENDPOINT_FILE, OpenCodeSandboxSession
from responses_api_agents.opencode_agent.tests.test_checkpoint import FAKE_OPENCODE, OpenCodeFixture


TOKEN = "resume-in-place-token"
_AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _control(checkpoint_id: str, **extra: Any) -> dict[str, Any]:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 30, **extra}


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision required")
JS_RUNTIME = os.environ.get("NEMO_GYM_TEST_JS_RUNTIME") or shutil.which("node") or shutil.which("bun")
# The relay is model_relay.mjs: a launch that resumes in place needs a JavaScript runtime to run it.
needs_relay_runtime = pytest.mark.skipif(
    JS_RUNTIME is None, reason="no JavaScript runtime (node or bun) for the relay"
)

# ``run`` first asks the model through the URL in its config, and keeps the reply. Like OpenCode's binary with
# ``BUN_BE_BUN=1``, it runs a script instead (the model relay), here on this machine's JavaScript runtime.
MODEL_CALLING_OPENCODE = (
    FAKE_OPENCODE.replace(
        "import base64, http.server, json, os, pathlib, signal, sqlite3, sys, threading, time, urllib.parse",
        "import base64, http.server, json, os, pathlib, signal, sqlite3, sys, threading, time, urllib.parse, "
        "urllib.request",
    )
    .replace(
        "CONTROL = pathlib.Path({control!r})",
        """if os.environ.get("BUN_BE_BUN") == "1":
    os.execv({js_runtime!r}, [{js_runtime!r}, *sys.argv[1:]])

CONTROL = pathlib.Path({control!r})""",
    )
    .replace(
        """def run():
    prompt = sys.stdin.read()
    started("run")
    plugin_ready()
""",
        """def call_model():
    config = json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])
    base = config["provider"]["nemo_gym"]["options"]["baseURL"]
    body = json.dumps({{"model": "m", "messages": [{{"role": "user", "content": "one step"}}]}}).encode()
    request = urllib.request.Request(base + "/chat/completions", data=body, headers={{"Content-Type": "application/json"}})
    with urllib.request.urlopen(request, timeout=600) as response:
        (CONTROL / "model_reply").write_bytes(response.read())


def run():
    prompt = sys.stdin.read()
    started("run")
    plugin_ready()
    call_model()
""",
    )
)
assert "call_model()" in MODEL_CALLING_OPENCODE


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
                    server.release.wait(120)
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


class KeepingProvider(LocalExecProvider):
    """This machine as a sandbox whose snapshots keep processes."""

    snapshot_keeps_processes = True


class InPlaceFixture(OpenCodeFixture):
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__(tmp_path, monkeypatch)
        self.binary.write_text(
            MODEL_CALLING_OPENCODE.format(
                python=sys.executable,
                control=str(self.control),
                version="1.17.11",
                js_runtime=JS_RUNTIME or "",
            )
        )
        self.provider = KeepingProvider()
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


async def _until(condition: Any, timeout: float = 60) -> None:
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
    for operation, extra in (("prepare", {}), ("commit", {"checkpoint_dir": str(state_dir / "ckpt")})):
        reply = await http.post(f"/ng-control/v1/checkpoint/{operation}", json=_control("c1", **extra), headers=_AUTH)
        assert reply.status_code == 200, f"{operation}: {reply.text}"
    # Holding, not stopping: no harness process was signalled.
    assert all(_state(pid) not in ("T", "t") for pid in fixture.harness_pids())
    # The agent server dies here; the sandbox's processes do not. Its tasks are left as a dead process leaves them.
    return config, (agent, http, activation)


async def _replacement(fixture: InPlaceFixture, config: dict[str, Any], state_dir: Path, model_url: str) -> Any:
    import httpx

    agent = fixture.make_agent(config, model_url=model_url)
    app = agent.setup_webserver()
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent", timeout=120)
    for operation, extra in (
        ("restore", {"checkpoint_dir": str(state_dir / "ckpt"), "episode_ids": [{"rollout_id": "r"}]}),
        ("resume", {}),
    ):
        reply = await http.post(f"/ng-control/v1/checkpoint/{operation}", json=_control("r1", **extra), headers=_AUTH)
        assert reply.status_code == 200, f"{operation}: {reply.text}"
    seeded = await http.post("/v1/agent_sessions", json=fixture.seed("kept", 1))
    assert seeded.status_code == 200, seeded.text
    return agent, http


@needs_relay_runtime
async def test_a_restore_that_kept_the_processes_continues_the_same_launch_against_the_new_model_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = InPlaceFixture(tmp_path, monkeypatch)
    crashed_server, new_server = ModelServer(answer=False), ModelServer(answer=True)
    endpoints_at_continue: list[dict[str, Any]] = []
    original_continue = SandboxSession.continue_harness

    async def recording_continue(session: SandboxSession, **kwargs: Any) -> Any:
        endpoint = Path(session.session_dir) / RELAY_ENDPOINT_FILE
        endpoints_at_continue.append(json.loads(endpoint.read_text()))
        return await original_continue(session, **kwargs)

    try:
        config, _crashed = await _crash_after_a_checkpoint(fixture, tmp_path / "state", crashed_server)
        harness = fixture.harness_pids()
        monkeypatch.setattr(SandboxSession, "continue_harness", recording_continue)
        caplog.set_level(logging.WARNING)
        _, http = await _replacement(fixture, config, tmp_path / "state", new_server.url)
        fixture.finish()
        reply = await http.post(
            "/ng-rollout/r-a1/v1/responses", json=fixture.activation(), headers={"x-ng-agent-activation": "1"}
        )

        assert reply.status_code == 200, reply.text
        assert fixture.started() == 1, "the launch that ran through the restore finished; nothing was relaunched"
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
        fixture.finish()
        crashed_server.stop()
        new_server.stop()
        for pid in fixture.harness_pids():
            os.kill(pid, signal.SIGKILL)


@needs_relay_runtime
async def test_a_restored_launch_whose_supervisor_died_is_abandoned_and_relaunched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = InPlaceFixture(tmp_path, monkeypatch)
    crashed_server, new_server = ModelServer(answer=False), ModelServer(answer=True)
    try:
        config, _crashed = await _crash_after_a_checkpoint(fixture, tmp_path / "state", crashed_server)
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
    finally:
        fixture.finish()
        crashed_server.stop()
        new_server.stop()
        for pid in fixture.harness_pids():
            os.kill(pid, signal.SIGKILL)


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
    assert ended == (["10.20"] if abandoned else [])


async def test_a_restored_session_without_a_held_pause_has_nothing_to_take_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    state = _restored_state(agent)
    monkeypatch.setattr(opencode_app, "check_restored", MagicMock(side_effect=AssertionError("not checked")))

    await agent._check_restored_launch(state, {"session_root": "/s", "session_dir": "/s", "launch_started": True})

    assert state.adoption is None


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
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    sandbox = await AsyncSandbox.connect({"sandbox_id": "local"}, provider=LocalExecProvider())
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
    assert "BUN_BE_BUN" not in payload["env"] and "MODEL_RELAY_SCRIPT" not in params


async def test_pointing_the_relay_raises_its_endpoint_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = OpenCodeFixture(tmp_path, monkeypatch).make_agent({})
    sandbox = await AsyncSandbox.connect({"sandbox_id": "local"}, provider=LocalExecProvider())
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
async def test_expected_restore_fallbacks_are_logged_at_info(
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
