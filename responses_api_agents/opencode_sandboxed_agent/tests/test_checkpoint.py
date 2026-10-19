# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenCode agent's legacy /run as an interruptible participant in partial-rollout checkpoints.

The fake sandbox's ``opencode run`` blocks until the agent's interrupt command arrives or the test lets it
finish, as the real one runs until its task is done or it is signalled. A launch that gets as far as OpenCode
saves its session in the data folder it names, as OpenCode's database does. The resources server is faked at
the HTTP seam.
"""

import asyncio
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import DictConfig

from nemo_gym._checkpoint.agent import AgentSessionRecord
from nemo_gym._checkpoint.control import CheckpointRequest
from nemo_gym._checkpoint.harness import LAUNCH_TAG
from nemo_gym._checkpoint.harness_run import LentSandbox
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, TrajectoryRecord
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent.app import (
    _REMOTE_RESUME_HELPER,
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
)


APP = "responses_api_agents.opencode_sandboxed_agent.app"
HARNESS = "nemo_gym._checkpoint.harness"
CLOSE = CheckpointRequest(checkpoint_id="c1", deadline_ts=time.time() + 3600)
EXPORT = {
    "messages": [
        {"info": {"role": "user"}, "parts": [{"type": "text", "text": "fix the bug"}]},
        {
            "info": {
                "role": "assistant",
                "time": {"created": 1, "completed": 2},
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "total": 15, "cache": {"read": 0, "write": 0}},
            },
            "parts": [{"type": "text", "text": "done"}],
        },
    ]
}
RESTORED_DATA_HOME = "/tmp/nemo-gym-opencode-fixed"


def done(stdout: str = "", return_code: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, return_code=return_code, error_type=None)


class FakeResponse:
    def __init__(
        self, payload: dict, *, cookies: dict | None = None, status: int = 200, headers: dict | None = None
    ) -> None:
        self._payload = payload
        self.cookies = cookies or {}
        self.status = status
        self.ok = status < 400
        self.headers: dict[str, str] = headers or {}
        self.content = SimpleNamespace(read=AsyncMock(return_value=json.dumps(payload).encode()))

    async def json(self) -> dict:
        return self._payload

    async def text(self) -> str:
        return json.dumps(self._payload)

    async def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status}: {self._payload}")


class FakeResourcesServer:
    """The resources server at the HTTP seam: seed, sandbox access, and verify.

    ``restart_only`` answers the seed with the header of a server that cannot capture its session;
    ``lends_sandbox=False`` is a server whose sessions hold no sandbox, so the agent creates its own;
    ``access_status`` makes ``/sandbox_access`` fail; ``hold_verify`` holds ``/verify`` until released.
    """

    def __init__(
        self,
        *,
        restart_only: bool = False,
        lends_sandbox: bool = True,
        access_status: int = 200,
        hold_verify: bool = False,
    ) -> None:
        self.calls: list[str] = []
        # The sandbox /sandbox_access names: after a restore, the fork the resources server rebuilt.
        self.sandbox_id = "sb-1"
        self.restart_only = restart_only
        self.lends_sandbox = lends_sandbox
        self.access_status = access_status
        self.verify_started = asyncio.Event()
        self.release_verify = asyncio.Event()
        if not hold_verify:
            self.release_verify.set()

    async def post(self, *, server_name: str = "", url_path: str, json: Any = None, cookies: Any = None, **_: Any):
        self.calls.append(url_path)
        if url_path == "/seed_session":
            seeded = {"sandbox_handle": "sb-1", "workdir": "/testbed"} if self.lends_sandbox else {}
            headers = {"x-ng-checkpoint-restart": "1"} if self.restart_only else {}
            return FakeResponse(seeded, cookies={"ng_session": "s"}, headers=headers)
        if url_path == "/sandbox_access":
            if self.restart_only or not self.lends_sandbox:
                return FakeResponse({"detail": "this session has no sandbox access"}, status=404)
            if self.access_status != 200:
                return FakeResponse({"detail": "the resources server failed"}, status=self.access_status)
            return FakeResponse(
                {
                    "connection": {
                        "kind": "direct",
                        "provider_config_ref": "sandbox",
                        "descriptor": {"sandbox_id": self.sandbox_id},
                    },
                    "workdir": "/testbed",
                }
            )
        if url_path == "/close_session":
            return FakeResponse({"closed": True})
        if url_path == "/verify":
            self.verify_started.set()
            await self.release_verify.wait()
            return FakeResponse({**json, "reward": 1.0})
        raise AssertionError(url_path)


class FakeSandbox:
    """OpenCode runs until interrupted or until the test lets it finish; everything else answers at once.

    ``finish_runs_after``: the n-th launch (1-based) and later finish on their own instead of blocking.
    ``saves_session_from``: launches from the n-th on get as far as OpenCode, which saves its session (the task
    first) in the launch's data folder; earlier ones are stopped while still installing.
    """

    def __init__(
        self, resources: FakeResourcesServer, *, finish_runs_after: int = 1, saves_session_from: int = 1
    ) -> None:
        self.resources = resources
        # What ``opencode export`` returns for a saved session.
        self.export = EXPORT
        self.commands: list[str] = []
        self.started = asyncio.Event()
        self.killed = asyncio.Event()
        self.finish = asyncio.Event()
        self.stopped = False
        self.disconnected = 0
        self.runs = 0
        self.finish_runs_after = finish_runs_after
        self.saves_session_from = saves_session_from
        self.launching = False
        # The data folders whose OpenCode database holds a session, as the sandbox's filesystem would.
        self.saved_sessions: set[Optional[str]] = set()
        self.upload = AsyncMock()

    def stop_launch(self) -> None:
        """An interrupt ends the launch that is running; the kill after the launch ended finds nothing."""
        self.killed.set()

    async def exec(self, command: str, timeout_s: float | None = None, env: Any = None, **_: Any) -> Any:
        self.commands.append(command)
        if is_interrupt(command):
            if self.launching:
                self.stop_launch()
            return done()
        # A fresh launch runs ``opencode run``; a resumed one runs the helper that re-enters the saved user turn.
        if "opencode run" in command or _REMOTE_RESUME_HELPER in command:
            self.runs += 1
            self.started.set()
            if self.runs >= self.saves_session_from:
                self.saved_sessions.add(data_home_of(command))
            if self.runs > self.finish_runs_after:
                return done("Shell: /bin/bash\nOpenCode run finished")
            self.launching = True
            killed = asyncio.ensure_future(self.killed.wait())
            finished = asyncio.ensure_future(self.finish.wait())
            try:
                await asyncio.wait({killed, finished}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                killed.cancel()
                finished.cancel()
                self.launching = False
            if self.finish.is_set():
                return done("Shell: /bin/bash\nOpenCode run finished")
            self.killed.clear()
            return done("Shell: /bin/bash\n", return_code=130, stderr="interrupted")
        data_home = (env or {}).get("XDG_DATA_HOME")
        if "session list" in command:
            return done('[{"id": "ses_1"}]' if data_home in self.saved_sessions else "")
        if "opencode export" in command:
            if data_home not in self.saved_sessions:
                return done(return_code=1, stderr="Session not found")
            # OpenCode reports progress on stderr, which OpenSandbox's background executions return inside stdout.
            progress = "" if "2>/dev/null" in command else "Exporting session: ses_1\n"
            return done(progress + json.dumps(self.export))
        if "opencode db path" in command:
            return done("/tmp/db")
        return done()

    async def download(self, remote: str, local: Path) -> None:
        if str(remote).endswith("export.json"):
            Path(local).write_text(json.dumps(EXPORT))

    async def stop(self) -> None:
        self.stopped = True

    async def disconnect(self) -> None:
        self.disconnected += 1


class IgnoresTheInterrupt(FakeSandbox):
    """OpenCode that a SIGINT does not stop: only ``ends_on`` ends the launch. ``None`` is a launch no signal
    reaches, as when the park request arrives while the launch is still installing OpenCode."""

    def __init__(self, resources: FakeResourcesServer, *, ends_on: str | None = "KILL", **kwargs: Any) -> None:
        super().__init__(resources, **kwargs)
        self.ends_on = ends_on
        self.signals: list[str] = []

    async def exec(self, command: str, timeout_s: float | None = None, env: Any = None, **kwargs: Any) -> Any:
        if is_interrupt(command):
            self.commands.append(command)
            signal_name = signal_of(command)
            self.signals.append(signal_name)
            if signal_name == self.ends_on and self.launching:
                self.stop_launch()
            return done()
        return await super().exec(command, timeout_s=timeout_s, env=env, **kwargs)


def make_agent(tmp_path: Path, resources: FakeResourcesServer) -> OpenCodeSandboxedAgent:
    config = OpenCodeSandboxedAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="opencode",
        resources_server=ResourcesServerRef(type="resources_servers", name="swebench"),
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        opencode_version="1.0",
        preinstalled_opencode=True,
        sandbox_provider="sandbox",
        sandbox_config={},
        sandbox_timeout=600,
        opencode_max_context_window=1000,
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    client.post = resources.post
    agent = OpenCodeSandboxedAgent(config=config, server_client=client)
    agent.setup_webserver()
    agent._create_opencode_config = AsyncMock(return_value={})
    return agent


class FakeRequest:
    """A request whose ``cookies`` follow ``_cookies``, as Starlette's do: the run swaps them for the seeded ones."""

    def __init__(self, session_id: str, body: dict) -> None:
        self._cookies: dict[str, str] = {}
        self.session = {SESSION_ID_KEY: session_id}
        self.state = SimpleNamespace()
        self.json = AsyncMock(return_value=body)

    @property
    def cookies(self) -> dict[str, str]:
        return self._cookies


def make_request(session_id: str, body: dict) -> FakeRequest:
    return FakeRequest(session_id, body)


def body_for(rollout_id: str, attempt: int = 0) -> OpenCodeSandboxedAgentRunRequest:
    key = EpisodeId(rollout_id=rollout_id, attempt=attempt).capture_key
    return OpenCodeSandboxedAgentRunRequest.model_validate(
        {"responses_create_params": {"input": [{"role": "user", "content": "fix the bug"}]}, "_ng_rollout_id": key}
    )


def is_interrupt(command: str) -> bool:
    return "kill -INT" in command or "kill -KILL" in command


def signal_of(interrupt: str) -> str:
    return "KILL" if "kill -KILL" in interrupt else "INT"


def launch_commands(sandbox: FakeSandbox) -> list[str]:
    return [c for c in sandbox.commands if "opencode run" in c or _REMOTE_RESUME_HELPER in c]


def resumes(command: str) -> bool:
    """A launch that re-enters the saved user turn instead of starting OpenCode with a prompt."""
    return _REMOTE_RESUME_HELPER in command and "opencode run" not in command


def interrupt_commands(sandbox: FakeSandbox) -> list[str]:
    return [c for c in sandbox.commands if is_interrupt(c)]


def launch_id(command: str) -> Optional[str]:
    """The harness launch a launch or stop command names (``NEMO_GYM_HARNESS_LAUNCH=<id>``)."""
    match = re.search(rf"{LAUNCH_TAG}=([0-9a-f]+)", command)
    return match.group(1) if match else None


def data_home_of(command: str) -> Optional[str]:
    return next((token.split("=", 1)[1] for token in command.split() if token.startswith("XDG_DATA_HOME=")), None)


async def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not met"
        await asyncio.sleep(0.01)


async def install_episode(agent: OpenCodeSandboxedAgent, episode: dict[str, Any]) -> None:
    """Install a restored legacy episode for rollout ``r``, as a restore after a crash does, and reopen."""
    participant = agent.checkpoint_participant
    await participant.install(
        [
            AgentSessionRecord(
                session_key="run:r", episode_id=EpisodeId(rollout_id="r"), session={}, boundary=None, episode=episode
            )
        ],
        [EpisodeId(rollout_id="r")],
    )
    await participant.open_admission()


async def park_and_resume(agent: OpenCodeSandboxedAgent, sandbox: FakeSandbox) -> AgentSessionRecord:
    """Close admission while OpenCode runs, wait for the run to park, export its boundary, and reopen."""
    participant = agent.checkpoint_participant
    await asyncio.wait_for(sandbox.started.wait(), timeout=5)
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready, timeout=30)
    [record] = await participant.export(None, checkpoint_id="c")
    await participant.open_admission()
    return record


def harness_bookmark(*, resume: bool) -> dict[str, Any]:
    """The harness bookmark a checkpoint took before a crash: the sandbox it names is the superseded one."""
    interrupted = {"resumed": False, "reason": None, "seconds": 1.0, "ended": "interrupted"}
    return {
        "v": 1,
        "next": "harness",
        "cookies": {"ng_session": "s"},
        "verify_mode": "wait",
        "sandbox": {"descriptor": {"sandbox_id": "sb-1"}, "workdir": "/testbed"},
        "state": {"data_home": RESTORED_DATA_HOME},
        "resume": resume,
        "launches": [interrupted] if resume else [],
    }


# -- interrupt, park, and continue ---------------------------------------------------------------------------


async def test_a_checkpoint_interrupts_opencode_and_the_resume_continues_the_session(tmp_path: Path) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    participant = agent.checkpoint_participant
    sandbox = FakeSandbox(resources)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await asyncio.wait_for(sandbox.started.wait(), timeout=5)
    assert participant.readiness().blockers == ["r"], "OpenCode is running: the run blocks prepare"

    # The commit snapshots the sandbox and leaves it running; the run is parked at its boundary meanwhile.
    record = await park_and_resume(agent, sandbox)
    result = await asyncio.wait_for(run, timeout=10)

    # The exported boundary continues the OpenCode step with the same data home and the sandbox handle.
    assert record.episode["next"] == "harness" and record.episode["resume"] is True
    assert record.episode["sandbox"] == {"descriptor": {"sandbox_id": "sb-1"}, "workdir": "/testbed"}
    assert record.episode["cookies"]["ng_session"] == "s"
    first, second = launch_commands(sandbox)
    assert not resumes(first) and resumes(second)
    assert "interrupted" not in second, "the resume adds no user turn"
    assert data_home_of(first) == data_home_of(second) == record.episode["state"]["data_home"]
    assert 'test -x "$HOME/.opencode/bin/opencode"' in second, "the install is skipped when the binary is present"
    # SIGINT ended the launch; the kill that follows stops whatever it left running before the sandbox is parked.
    assert [signal_of(c) for c in interrupt_commands(sandbox)] == ["INT", "KILL"]
    assert {launch_id(c) for c in interrupt_commands(sandbox)} == {launch_id(first)}, "they stop the first launch"
    # The continued launch really ran OpenCode again, in the sandbox the commit left running.
    assert sandbox.runs == 2
    assert result.opencode_failed is False, result.opencode_error_type
    assert result.opencode_finished is True and result.opencode_export_found is True
    # Before each launch the run asked the resources server which sandbox is current, and before the relaunch it
    # checked that the saved session carries the task.
    assert resources.calls == ["/seed_session", "/sandbox_access", "/sandbox_access", "/verify"]
    assert any("opencode export ses_1" in c and c.endswith("2>/dev/null") for c in sandbox.commands), "stdout only"
    assert sandbox.disconnected == 2, "the stale client was released before reconnecting, and the last at the end"
    # The owner ended the session at verify; a checkpointed run never stops the sandbox itself.
    assert result.reward == 1.0 and not sandbox.stopped
    launches = result.model_dump(mode="json")["ng_harness_launches"]
    assert [(launch["resumed"], launch["ended"]) for launch in launches] == [
        (False, "interrupted"),
        (True, "finished"),
    ]
    assert launches[1]["reason"] == "OpenCode's session ses_1 has completed model responses"


async def test_a_session_stopped_before_its_first_model_response_starts_over_with_the_task(tmp_path: Path) -> None:
    """OpenCode saved the task, then a checkpoint stopped it before the model answered. Re-entering that turn can
    end the rollout with no model call at all; starting over loses nothing, so the relaunch does that."""
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    sandbox = FakeSandbox(resources)
    sandbox.export = {"messages": [EXPORT["messages"][0], {"info": {"role": "assistant", "time": {"created": 1}}}]}
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await asyncio.wait_for(sandbox.started.wait(), timeout=5)
    await park_and_resume(agent, sandbox)
    result = await asyncio.wait_for(run, timeout=10)

    first, second = launch_commands(sandbox)
    assert not resumes(second) and "fix the bug" in second, "a fresh launch with the task"
    assert data_home_of(second) != data_home_of(first), "in a fresh data folder"
    launches = result.model_dump(mode="json")["ng_harness_launches"]
    assert launches[1]["resumed"] is False
    assert launches[1]["reason"] == "OpenCode's session ses_1 has no completed model response yet"
    assert result.reward == 1.0 and result.opencode_finished is True


async def test_opencode_that_ignores_the_interrupt_is_killed_after_the_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{HARNESS}._GRACEFUL_S", 0.05)
    monkeypatch.setattr(f"{HARNESS}._KILL_S", 10.0)
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    sandbox = IgnoresTheInterrupt(resources, ends_on="KILL")
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    record = await park_and_resume(agent, sandbox)
    result = await asyncio.wait_for(run, timeout=15)

    assert sandbox.signals == ["INT", "KILL"], "the grace expired, so the kill followed"
    assert record.episode["resume"] is True
    assert resumes(launch_commands(sandbox)[1])
    assert result.reward == 1.0


async def test_a_relaunch_without_a_saved_session_starts_over_with_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch that neither signal ends (the sandbox stopped answering) is abandoned after both graces, and
    OpenCode had saved nothing: a resume would find no user turn to re-enter, so the relaunch starts over with the
    task, in a fresh data folder."""
    monkeypatch.setattr(f"{HARNESS}._GRACEFUL_S", 0.05)
    monkeypatch.setattr(f"{HARNESS}._KILL_S", 0.05)
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    sandbox = IgnoresTheInterrupt(resources, ends_on=None, saves_session_from=2)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    record = await park_and_resume(agent, sandbox)
    result = await asyncio.wait_for(run, timeout=15)

    assert sandbox.signals == ["INT", "KILL"], "both signals were sent and neither reached a process"
    assert record.episode["resume"] is True, "the bookmark asks to continue; the relaunch checks first"
    first, second = launch_commands(sandbox)
    assert not resumes(second)
    assert "fix the bug" in second, "the relaunch carries the task"
    assert data_home_of(second) != data_home_of(first), "a fresh data folder: no empty session is ever continued"
    assert result.reward == 1.0 and result.opencode_finished is True


async def test_a_park_requested_before_the_launch_starts_nothing_until_the_resume(tmp_path: Path) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    participant = agent.checkpoint_participant
    sandbox = FakeSandbox(resources, finish_runs_after=0)
    closes = [CLOSE]

    async def connect(**_: Any) -> FakeSandbox:
        # A checkpoint closes admission after the run recorded its boundary and before it starts OpenCode.
        if closes:
            await participant.close_admission(closes.pop())
        return sandbox

    agent._start_sandbox = AsyncMock(side_effect=connect)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await wait_until(lambda: not closes and participant.readiness().ready)
    [record] = await participant.export(None, checkpoint_id="c")
    assert launch_commands(sandbox) == [] and interrupt_commands(sandbox) == [], "OpenCode was never started"
    await participant.open_admission()
    result = await asyncio.wait_for(run, timeout=10)

    assert record.episode["next"] == "harness" and record.episode["resume"] is False
    [launch] = launch_commands(sandbox)
    assert not resumes(launch) and "fix the bug" in launch
    assert result.reward == 1.0


# -- which sandbox a run continues in ------------------------------------------------------------------------


async def test_a_replacement_attempt_continues_from_the_opencode_boundary_without_reseeding(tmp_path: Path) -> None:
    resources = FakeResourcesServer()
    # The resources server restored the session by forking the checkpoint's snapshot under a new id.
    resources.sandbox_id = "sb-2"
    agent = make_agent(tmp_path, resources)
    sandbox = FakeSandbox(resources, finish_runs_after=0)
    # The fork holds OpenCode's database as the snapshot had it.
    sandbox.saved_sessions.add(RESTORED_DATA_HOME)
    agent._start_sandbox = AsyncMock(return_value=sandbox)
    await install_episode(agent, harness_bookmark(resume=True))

    result = await asyncio.wait_for(agent.run(make_request("agent-session-2", {}), body_for("r", attempt=1)), 10)

    [launch] = launch_commands(sandbox)
    assert resumes(launch) and data_home_of(launch) == RESTORED_DATA_HOME
    assert resources.calls == ["/sandbox_access", "/verify"], "no re-seed: the session is the checkpoint's"
    # The run continued in the fork the resources server named, not in the sandbox the boundary recorded.
    assert agent._start_sandbox.await_args.kwargs == {"sandbox_id": "sb-2", "workdir": "/testbed"}
    assert result.reward == 1.0 and not sandbox.stopped, "the owner ended the session at verify"


async def test_a_replacement_from_a_boundary_before_the_first_launch_asks_which_sandbox_is_current(
    tmp_path: Path,
) -> None:
    """A prepare that closes during the seed parks the run at its first OpenCode boundary, ``continue: false``.
    After a crash the restore forked the sandbox under a new id and stopped the old one, so this replacement must
    ask too: the sandbox a boundary names is only a hint."""
    resources = FakeResourcesServer()
    resources.sandbox_id = "sb-2"
    agent = make_agent(tmp_path, resources)
    sandbox = FakeSandbox(resources, finish_runs_after=0)
    agent._start_sandbox = AsyncMock(return_value=sandbox)
    await install_episode(agent, harness_bookmark(resume=False))

    result = await asyncio.wait_for(agent.run(make_request("agent-session-4", {}), body_for("r", attempt=1)), 10)

    [launch] = launch_commands(sandbox)
    assert not resumes(launch), "OpenCode never ran before the checkpoint: there is nothing to continue"
    assert resources.calls == ["/sandbox_access", "/verify"], "no re-seed, and the fork is asked for"
    assert agent._start_sandbox.await_args.kwargs == {"sandbox_id": "sb-2", "workdir": "/testbed"}
    assert result.reward == 1.0


async def test_a_failing_sandbox_access_fails_the_run_instead_of_using_the_known_sandbox(tmp_path: Path) -> None:
    """After a restore the sandbox a boundary names is the superseded one: a refresh that fails for any reason but
    404 fails the attempt, so the failure is not scored as the model's."""
    resources = FakeResourcesServer(access_status=500)
    agent = make_agent(tmp_path, resources)
    agent._start_sandbox = AsyncMock(side_effect=AssertionError("no sandbox may be connected"))
    await install_episode(agent, harness_bookmark(resume=True))

    with pytest.raises(RuntimeError, match="HTTP 500"):
        await asyncio.wait_for(agent.run(make_request("agent-session-5", {}), body_for("r", attempt=1)), 10)

    # The run leaves without a verdict, so it closes its session through the owner, which stops the sandbox.
    assert resources.calls == ["/sandbox_access", "/close_session"]
    agent._start_sandbox.assert_not_awaited()


async def test_a_404_from_sandbox_access_keeps_the_known_handle(tmp_path: Path) -> None:
    """A resources server that lends no sandbox answers 404; the run keeps the handle it has."""
    resources = FakeResourcesServer(lends_sandbox=False)
    agent = make_agent(tmp_path, resources)

    hint = LentSandbox(descriptor={"sandbox_id": "sb-1"}, workdir="/testbed")

    assert await agent.refresh_lent_sandbox({"ng_session": "s"}, hint) == hint
    assert resources.calls == ["/sandbox_access"]


# -- runs nothing can checkpoint -----------------------------------------------------------------------------


async def test_a_resources_server_that_cannot_capture_its_session_makes_the_run_a_restart(tmp_path: Path) -> None:
    resources = FakeResourcesServer(restart_only=True)
    agent = make_agent(tmp_path, resources)
    participant = agent.checkpoint_participant
    sandbox = FakeSandbox(resources)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await asyncio.wait_for(sandbox.started.wait(), timeout=5)
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)
    report = participant.readiness()
    await asyncio.sleep(0.05)
    assert interrupt_commands(sandbox) == [], "a restart is never asked to park"
    await participant.open_admission()
    sandbox.finish.set()
    result = await asyncio.wait_for(run, timeout=10)

    assert report.restarts == ["r"] and report.blockers == []
    [launch] = launch_commands(sandbox)
    assert not resumes(launch)
    assert result.reward == 1.0


async def test_a_run_in_a_sandbox_of_its_own_is_a_restart(tmp_path: Path) -> None:
    """A resources server that lends no sandbox leaves the agent to create one, and nothing snapshots it: the run
    is never interrupted, and after a crash it starts over."""
    resources = FakeResourcesServer(lends_sandbox=False)
    agent = make_agent(tmp_path, resources)
    participant = agent.checkpoint_participant
    sandbox = FakeSandbox(resources)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await asyncio.wait_for(sandbox.started.wait(), timeout=5)
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)
    report = participant.readiness()
    await asyncio.sleep(0.05)
    assert interrupt_commands(sandbox) == []
    await participant.open_admission()
    sandbox.finish.set()
    result = await asyncio.wait_for(run, timeout=10)

    assert report.restarts == ["r"] and report.blockers == []
    assert agent._start_sandbox.await_count == 1, "one sandbox, created by the agent"
    assert agent._start_sandbox.await_args.kwargs == {"sandbox_id": None, "workdir": None}
    assert result.reward == 1.0


# -- after the OpenCode step ---------------------------------------------------------------------------------


async def test_a_run_parked_after_verify_keeps_its_trajectory_in_the_checkpoint(tmp_path: Path) -> None:
    """Verify is a wait step, so a checkpoint that opens during it parks the run at its "return" boundary. A
    replacement continues from that boundary, so it carries the trajectory and observations with the result."""
    resources = FakeResourcesServer(hold_verify=True)
    agent = make_agent(tmp_path, resources)
    participant = agent.checkpoint_participant
    sandbox = FakeSandbox(resources, finish_runs_after=0)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await asyncio.wait_for(resources.verify_started.wait(), timeout=5)
    await participant.close_admission(CLOSE)
    resources.release_verify.set()
    await wait_until(lambda: participant.readiness().ready)
    [record] = await participant.export(None, checkpoint_id="c")
    await participant.open_admission()
    result = await asyncio.wait_for(run, timeout=10)

    assert record.episode["next"] == "return" and record.episode["result"]["reward"] == 1.0
    assert record.episode["outcome"]["trajectory"] is not None
    assert record.episode["outcome"]["observations"] is not None
    assert result.reward == 1.0


async def test_a_replacement_from_the_return_boundary_returns_the_stored_result_without_regrading(
    tmp_path: Path,
) -> None:
    """A run parked after verify (it replied during prepare, or was about to) is continued from "return"."""
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    agent._start_sandbox = AsyncMock(side_effect=AssertionError("no sandbox is touched after verify"))

    stored_result = {
        **body_for("r").model_dump(mode="json"),
        "response": {
            "id": "resp_x",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "output": [],
            "tool_choice": "auto",
            "tools": [],
            "parallel_tool_calls": False,
        },
        "reward": 1.0,
        "resolved": True,
    }
    run_result = {
        "opencode_failed": False,
        "opencode_exit_code": 0,
        "opencode_error_type": None,
        "opencode_results_fpath": "/x/export.json",
        "opencode_run_stdout": "Shell: /bin/bash\nOpenCode run finished",
        "opencode_run_stderr": "",
        "opencode_export_found": True,
        "opencode_finished": True,
    }
    observations = AgentObservationBundle(source="opencode", records=[AgentInvocation(invocation_id="r")])
    trajectory = TrajectoryRecord(task_id="", rollout_id="r")
    await install_episode(
        agent,
        {
            "v": 1,
            "next": "return",
            "result": stored_result,
            "outcome": {
                "response": stored_result["response"],
                "run_result": run_result,
                "observations": observations.model_dump(mode="json"),
                "trajectory": trajectory.model_dump(mode="json"),
            },
            "launches": [],
        },
    )

    result = await asyncio.wait_for(agent.run(make_request("agent-session-3", {}), body_for("r", attempt=1)), 10)

    assert result.reward == 1.0 and result.opencode_finished is True
    assert result.opencode_results_fpath == "/x/export.json"
    assert resources.calls == [], "no re-seed, no re-grade"
    # The trajectory and observations the run captured come back with the stored result.
    dumped = result.model_dump(mode="json")
    assert dumped.get("ng_trajectory") is not None
    assert "observation_capture_failed" not in [gap["code"] for gap in dumped["ng_agent_observations"]["gaps"]]


async def test_a_replacement_from_the_verify_boundary_regrades_without_relaunching_opencode(
    tmp_path: Path,
) -> None:
    """A run checkpointed after the OpenCode step continues at ``verify``: it grades the stored response and never
    touches a sandbox."""
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    agent._start_sandbox = AsyncMock(side_effect=AssertionError("no sandbox after the OpenCode step"))
    stored_response = {
        "id": "resp_x",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": [],
        "tool_choice": "auto",
        "tools": [],
        "parallel_tool_calls": False,
    }
    run_result = {
        "opencode_failed": False,
        "opencode_exit_code": 0,
        "opencode_error_type": None,
        "opencode_results_fpath": "/x/export.json",
        "opencode_run_stdout": "Shell: /bin/bash\nOpenCode run finished",
        "opencode_run_stderr": "",
        "opencode_export_found": True,
        "opencode_finished": True,
    }
    await install_episode(
        agent,
        {
            "v": 1,
            "next": "verify",
            "cookies": {"ng_session": "s"},
            "verify_mode": "wait",
            "outcome": {"response": stored_response, "run_result": run_result},
            "launches": [],
        },
    )

    result = await asyncio.wait_for(agent.run(make_request("agent-session-6", {}), body_for("r", attempt=1)), 10)

    assert resources.calls == ["/verify"], "no re-seed and no relaunch: only the grading is left"
    assert result.reward == 1.0 and result.opencode_finished is True
    assert result.opencode_results_fpath == "/x/export.json"


# -- cancellation, and checkpointing off ---------------------------------------------------------------------


async def test_a_run_cancelled_while_parked_leaves_the_checkpointed_sandbox_alone(tmp_path: Path) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    participant = agent.checkpoint_participant
    sandbox = FakeSandbox(resources)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await asyncio.wait_for(sandbox.started.wait(), timeout=5)
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)

    # The caller goes away while the run is parked (the collector stopped after its checkpoint).
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert not sandbox.stopped, "OpenCode is already stopped and the sandbox is the resources server's to keep"
    assert agent._sandbox_id_to_sandbox == {}


async def test_a_run_cancelled_while_opencode_runs_kills_the_launch_and_closes_the_session_through_the_owner(
    tmp_path: Path,
) -> None:
    """The owner may still export the session, so a checkpointed run never stops its sandbox: it kills its own
    launch, so OpenCode stops generating, and closes the session, which the owner ends by stopping the sandbox."""
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    sandbox = FakeSandbox(resources)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await asyncio.wait_for(sandbox.started.wait(), timeout=5)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    [kill] = interrupt_commands(sandbox)
    assert signal_of(kill) == "KILL" and launch_id(kill) == launch_id(launch_commands(sandbox)[0])
    assert resources.calls[-1] == "/close_session"
    assert not sandbox.stopped and sandbox.disconnected == 1


@pytest.mark.parametrize("lender", ["restart_only", "no_sandbox"])
async def test_a_restart_still_stops_its_sandbox_when_it_fails(tmp_path: Path, lender: str) -> None:
    """A lender that cannot capture the session (or a sandbox of the agent's own) makes the run a restart, and
    such a lender may have no empty-body close: the run stops the sandbox itself, as it does without
    checkpointing."""
    resources = FakeResourcesServer(restart_only=lender == "restart_only", lends_sandbox=lender != "no_sandbox")
    agent = make_agent(tmp_path, resources)
    sandbox = FakeSandbox(resources)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await asyncio.wait_for(sandbox.started.wait(), timeout=5)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert sandbox.stopped and "/close_session" not in resources.calls


async def test_without_checkpointing_the_run_is_unchanged(tmp_path: Path) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    agent._checkpoint_participant = None
    sandbox = FakeSandbox(resources, finish_runs_after=0)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    result = await asyncio.wait_for(agent.run(make_request("agent-session", {}), body_for("r")), 10)

    [launch] = launch_commands(sandbox)
    assert not resumes(launch)
    assert resources.calls == ["/seed_session", "/verify"]
    assert result.reward == 1.0 and sandbox.stopped
