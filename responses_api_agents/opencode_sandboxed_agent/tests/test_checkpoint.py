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
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import DictConfig

from nemo_gym._checkpoint.agent import AgentSessionRecord
from nemo_gym._checkpoint.control import CheckpointRequest
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, TrajectoryRecord
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent.app import (
    _CONTINUE_PROMPT,
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
    _interrupt_command,
)


APP = "responses_api_agents.opencode_sandboxed_agent.app"
CLOSE = CheckpointRequest(checkpoint_id="c1", deadline_ts=time.time() + 3600)
EXPORT = {
    "messages": [
        {"info": {"role": "user"}, "parts": [{"type": "text", "text": "fix the bug"}]},
        {
            "info": {
                "role": "assistant",
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
        self.commands: list[str] = []
        self.started = asyncio.Event()
        self.killed = asyncio.Event()
        self.finish = asyncio.Event()
        self.stopped = False
        self.disconnected = 0
        self.runs = 0
        self.finish_runs_after = finish_runs_after
        self.saves_session_from = saves_session_from
        # The data folders whose OpenCode database holds a session, as the sandbox's filesystem would.
        self.saved_sessions: set[Optional[str]] = set()
        self.upload = AsyncMock()

    async def exec(self, command: str, timeout_s: float | None = None, env: Any = None, **_: Any) -> Any:
        self.commands.append(command)
        if "[o]pencode run" in command:
            self.killed.set()
            return done()
        if "opencode run" in command:
            self.runs += 1
            self.started.set()
            if self.runs >= self.saves_session_from:
                self.saved_sessions.add(data_home_of(command))
            if self.runs > self.finish_runs_after:
                return done("Shell: /bin/bash\nOpenCode run finished")
            killed = asyncio.ensure_future(self.killed.wait())
            finished = asyncio.ensure_future(self.finish.wait())
            try:
                await asyncio.wait({killed, finished}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                killed.cancel()
                finished.cancel()
            if self.finish.is_set():
                return done("Shell: /bin/bash\nOpenCode run finished")
            self.killed.clear()
            return done("Shell: /bin/bash\n", return_code=130, stderr="interrupted")
        data_home = (env or {}).get("XDG_DATA_HOME")
        if "session list" in command:
            return done('[{"id": "ses_1"}]' if data_home in self.saved_sessions else "[]")
        if "opencode export" in command:
            if data_home not in self.saved_sessions:
                return done(return_code=1, stderr="Session not found")
            return done(json.dumps(EXPORT))
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
        if "[o]pencode run" in command:
            self.commands.append(command)
            signal_name = "KILL" if "-KILL" in command else "INT"
            self.signals.append(signal_name)
            if signal_name == self.ends_on:
                self.killed.set()
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


def launch_commands(sandbox: FakeSandbox) -> list[str]:
    return [c for c in sandbox.commands if "opencode run" in c and "[o]pencode" not in c]


def interrupt_commands(sandbox: FakeSandbox) -> list[str]:
    return [c for c in sandbox.commands if "[o]pencode run" in c]


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


def opencode_boundary(*, continue_session: bool) -> dict[str, Any]:
    """The OpenCode boundary a checkpoint took before a crash: the sandbox it names is the superseded one."""
    return {
        "next": "opencode",
        "cookies": {"ng_session": "s"},
        "verify_mode": "wait",
        "sandbox_handle": "sb-1",
        "workdir": "/testbed",
        "data_home": RESTORED_DATA_HOME,
        "continue": continue_session,
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
    assert record.episode["next"] == "opencode" and record.episode["continue"] is True
    assert record.episode["sandbox_handle"] == "sb-1" and record.episode["workdir"] == "/testbed"
    assert record.episode["cookies"]["ng_session"] == "s"
    first, second = launch_commands(sandbox)
    assert "--continue" not in first and "--continue" in second
    assert _CONTINUE_PROMPT in second
    assert data_home_of(first) == data_home_of(second) == record.episode["data_home"]
    assert 'test -x "$HOME/.opencode/bin/opencode"' in second, "the install is skipped when the binary is present"
    interrupts = interrupt_commands(sandbox)
    assert len(interrupts) == 1 and "-INT" in interrupts[0]
    # The continued launch really ran OpenCode again, in the sandbox the commit left running.
    assert sandbox.runs == 2
    assert result.opencode_failed is False, result.opencode_error_type
    assert result.opencode_finished is True and result.opencode_export_found is True
    # Before each launch the run asked the resources server which sandbox is current, and before the relaunch it
    # checked that the saved session carries the task.
    assert resources.calls == ["/seed_session", "/sandbox_access", "/sandbox_access", "/verify"]
    assert any("opencode export ses_1" in command and ">" not in command for command in sandbox.commands)
    assert sandbox.disconnected == 1, "the stale client was released before reconnecting"
    assert result.reward == 1.0 and sandbox.stopped, "the episode finished and the sandbox was stopped"


async def test_opencode_that_ignores_the_interrupt_is_killed_after_the_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{APP}._INTERRUPT_GRACE_S", 0.05)
    monkeypatch.setattr(f"{APP}._KILL_GRACE_S", 10.0)
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    sandbox = IgnoresTheInterrupt(resources, ends_on="KILL")
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    record = await park_and_resume(agent, sandbox)
    result = await asyncio.wait_for(run, timeout=15)

    assert sandbox.signals == ["INT", "KILL"], "the grace expired, so the kill followed"
    assert record.episode["continue"] is True
    assert "--continue" in launch_commands(sandbox)[1]
    assert result.reward == 1.0


async def test_a_relaunch_without_a_saved_session_starts_over_with_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A park request that arrives while the launch is still installing or booting OpenCode reaches no process,
    and OpenCode saved nothing: ``--continue`` would open an empty session, so the relaunch starts over with the
    task, in a fresh data folder."""
    monkeypatch.setattr(f"{APP}._INTERRUPT_GRACE_S", 0.05)
    monkeypatch.setattr(f"{APP}._KILL_GRACE_S", 0.05)
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    sandbox = IgnoresTheInterrupt(resources, ends_on=None, saves_session_from=2)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    record = await park_and_resume(agent, sandbox)
    result = await asyncio.wait_for(run, timeout=15)

    assert sandbox.signals == ["INT", "KILL"], "both signals were sent and neither reached a process"
    assert record.episode["continue"] is True, "the boundary asks to continue; the relaunch checks first"
    first, second = launch_commands(sandbox)
    assert "--continue" not in second and _CONTINUE_PROMPT not in second
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

    assert record.episode["next"] == "opencode" and record.episode["continue"] is False
    [launch] = launch_commands(sandbox)
    assert "--continue" not in launch and "fix the bug" in launch
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
    await install_episode(agent, opencode_boundary(continue_session=True))

    result = await asyncio.wait_for(agent.run(make_request("agent-session-2", {}), body_for("r", attempt=1)), 10)

    [launch] = launch_commands(sandbox)
    assert "--continue" in launch and data_home_of(launch) == RESTORED_DATA_HOME
    assert resources.calls == ["/sandbox_access", "/verify"], "no re-seed: the session is the checkpoint's"
    # The run continued in the fork the resources server named, not in the sandbox the boundary recorded.
    assert agent._start_sandbox.await_args.kwargs == {"sandbox_id": "sb-2", "workdir": "/testbed"}
    assert result.reward == 1.0 and sandbox.stopped


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
    await install_episode(agent, opencode_boundary(continue_session=False))

    result = await asyncio.wait_for(agent.run(make_request("agent-session-4", {}), body_for("r", attempt=1)), 10)

    [launch] = launch_commands(sandbox)
    assert "--continue" not in launch, "OpenCode never ran before the checkpoint: there is nothing to continue"
    assert resources.calls == ["/sandbox_access", "/verify"], "no re-seed, and the fork is asked for"
    assert agent._start_sandbox.await_args.kwargs == {"sandbox_id": "sb-2", "workdir": "/testbed"}
    assert result.reward == 1.0


async def test_a_failing_sandbox_access_fails_the_run_instead_of_using_the_known_sandbox(tmp_path: Path) -> None:
    """After a restore the sandbox a boundary names is the superseded one: a refresh that fails for any reason but
    404 fails the attempt, so the failure is not scored as the model's."""
    resources = FakeResourcesServer(access_status=500)
    agent = make_agent(tmp_path, resources)
    agent._start_sandbox = AsyncMock(side_effect=AssertionError("no sandbox may be connected"))
    await install_episode(agent, opencode_boundary(continue_session=True))

    with pytest.raises(RuntimeError, match="HTTP 500"):
        await asyncio.wait_for(agent.run(make_request("agent-session-5", {}), body_for("r", attempt=1)), 10)

    assert resources.calls == ["/sandbox_access"]
    agent._start_sandbox.assert_not_awaited()


async def test_a_404_from_sandbox_access_keeps_the_known_handle(tmp_path: Path) -> None:
    """A resources server that lends no sandbox answers 404; the run keeps the handle it has."""
    resources = FakeResourcesServer(lends_sandbox=False)
    agent = make_agent(tmp_path, resources)

    handle, workdir = await agent._refresh_sandbox_access(
        {"ng_session": "s"}, sandbox_handle="sb-1", workdir="/testbed"
    )

    assert (handle, workdir) == ("sb-1", "/testbed")
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
    assert "--continue" not in launch
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
    assert record.episode["trajectory"] is not None
    assert record.episode["observations"] is not None
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
            "next": "return",
            "result": stored_result,
            "run_result": run_result,
            "observations": observations.model_dump(mode="json"),
            "trajectory": trajectory.model_dump(mode="json"),
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
            "next": "verify",
            "cookies": {"ng_session": "s"},
            "verify_mode": "wait",
            "response": stored_response,
            "run_result": run_result,
            "observations": None,
            "trajectory": None,
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


async def test_a_run_cancelled_while_opencode_runs_still_stops_the_sandbox(tmp_path: Path) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    sandbox = FakeSandbox(resources)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await asyncio.wait_for(sandbox.started.wait(), timeout=5)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert sandbox.stopped, "a disconnected caller must not leave OpenCode generating in its pod"


async def test_without_checkpointing_the_run_is_unchanged(tmp_path: Path) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    agent._checkpoint_participant = None
    sandbox = FakeSandbox(resources, finish_runs_after=0)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    result = await asyncio.wait_for(agent.run(make_request("agent-session", {}), body_for("r")), 10)

    [launch] = launch_commands(sandbox)
    assert "--continue" not in launch
    assert resources.calls == ["/seed_session", "/verify"]
    assert result.reward == 1.0 and sandbox.stopped


# -- the interrupt command, run for real ---------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("bash") is None, reason="the interrupt command is a shell snippet")
@pytest.mark.skipif(
    os.environ.get("NEMO_GYM_RUN_SIGNAL_TESTS") != "1",
    reason="signals every process on the host whose command line names `opencode run`; opt in on an isolated host",
)
@pytest.mark.parametrize("without_pkill", [False, True], ids=["pkill", "proc_fallback"])
def test_the_interrupt_command_signals_opencode_and_its_launching_shell(tmp_path: Path, without_pkill: bool) -> None:
    """Run ``_interrupt_command`` under a real bash, both branches, against processes shaped like the ones it meets
    in the sandbox: the OpenCode binary (its argv names it) and the shell whose command line launched it (the one
    ``sandbox.exec`` runs). Both are signalled; the shell running the snippet does not match itself."""
    bash = shutil.which("bash")
    binary = subprocess.Popen([bash, "-c", 'exec -a "opencode run --title NG dummy title" sleep 300'])
    launcher = subprocess.Popen([bash, "-c", "sleep 300; true  # opencode run --title NG dummy title"])
    environment = dict(os.environ)
    if without_pkill:
        # No pkill on PATH, so the snippet takes its /proc fallback; it still needs tr, and kill is a builtin.
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        (fake_bin / "tr").symlink_to(shutil.which("tr"))
        environment["PATH"] = str(fake_bin)
    try:
        time.sleep(0.2)
        interrupting = subprocess.run(
            [bash, "-c", _interrupt_command("TERM")], env=environment, capture_output=True, text=True, timeout=60
        )

        assert interrupting.returncode == 0, interrupting.stderr
        assert binary.wait(timeout=10) is not None, "the OpenCode process, whose argv names it, was signalled"
        assert launcher.wait(timeout=10) is not None, "so was the shell whose command line launched it"
    finally:
        for process in (binary, launcher):
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
