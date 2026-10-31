# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP probes run under the real process supervisor on the local sandbox provider.

The probe command is a FIXTURE, not OpenClaw: a small Python script installed where OpenClaw's Node binary would be.
It records its PID and either answers like `openclaw mcp probe --json` or blocks. The tests check real processes.
"""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.agent_utils import sandbox_session as sandbox_session_module
from nemo_gym.agent_utils.sandbox_session import HARNESS_RUNNER_FILE, HARNESS_RUNNER_SCRIPT, SandboxSession
from nemo_gym.client import ServerClient
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from responses_api_agents.openclaw_agent.app import OpenClawAgent, OpenClawAgentConfig
from responses_api_agents.openclaw_agent.sandbox import OpenClawSandboxSession


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("python3") is None, reason="needs Linux process supervision and python3"
)

FIXTURE = """#!/usr/bin/env python3
# FIXTURE standing in for `node openclaw.mjs mcp probe <name> --json`; not OpenClaw.
import json, os, sys, time
name = sys.argv[4]
pid_dir = os.environ["PROBE_FIXTURE_PID_DIR"]
open(os.path.join(pid_dir, name + ".pid"), "w").write(str(os.getpid()))
if name in os.environ.get("PROBE_FIXTURE_BLOCK", "").split(","):
    time.sleep(3600)
print(json.dumps({"servers": {name: {"tools": 4}}, "tools": [], "diagnostics": []}))
"""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers signal 0; it has exited for these tests.
    with open(f"/proc/{pid}/stat") as stat:
        return stat.read().split(")")[-1].split()[0] != "Z"


@pytest.fixture
def probe_env(tmp_path):
    runtime = tmp_path / "runtime"
    node = runtime / "node" / "bin" / "node"
    node.parent.mkdir(parents=True)
    node.write_text(FIXTURE)
    node.chmod(0o755)
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    workdir = tmp_path / "task"
    workdir.mkdir()
    agent = OpenClawAgent(
        config=OpenClawAgentConfig(
            name="openclaw",
            host="localhost",
            port=8001,
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "policy"},
            model="test-model",
            openclaw_version="2026.6.11",
            session_close_timeout_seconds=15,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    return agent, runtime, pid_dir, workdir


async def borrowed_state(tmp_path, runtime, workdir):
    """The owner's sandbox, and an agent session over it whose release records what was alive at disconnect."""
    owner = AsyncSandbox(LocalProvider())
    await owner.start(SandboxSpec(workdir=str(workdir)))
    session_dir = tmp_path / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    shutil.copy(HARNESS_RUNNER_SCRIPT, session_dir / HARNESS_RUNNER_FILE)
    session = SandboxSession(sandbox=owner, session_dir=str(session_dir), workdir=str(workdir), harness="OpenClaw")
    state = OpenClawSandboxSession(request=MagicMock(), session=session, runtime=str(runtime))
    release = {"calls": 0, "alive_at_disconnect": None}
    return owner, state, release


def payload_for(runtime, pid_dir, block=""):
    return {
        "command": [f"{runtime}/node/bin/node", f"{runtime}/openclaw/node_modules/openclaw/openclaw.mjs"],
        "env": {"PROBE_FIXTURE_PID_DIR": str(pid_dir), "PROBE_FIXTURE_BLOCK": block},
    }


def pid_of(pid_dir, name) -> int:
    return int((pid_dir / f"{name}.pid").read_text())


async def test_optional_probe_timeout_is_stopped_before_the_agent_runs(probe_env, tmp_path):
    agent, runtime, pid_dir, workdir = probe_env
    owner, state, _ = await borrowed_state(tmp_path, runtime, workdir)
    servers = {"task_tools": {"url": "http://x/mcp"}, "extra": {"url": "http://y/mcp"}}
    try:
        gaps = await agent._probe_mcp_grants(
            state, payload_for(runtime, pid_dir, block="extra"), servers, {"task_tools"}, timeout=2
        )
        # When the probe step returns, the agent could launch; the blocked probe must already be gone.
        assert not alive(pid_of(pid_dir, "extra"))
        assert not alive(pid_of(pid_dir, "task_tools"))
        assert [gap.detail for gap in gaps] == ["extra: timeout: probe exceeded 2 s"]
        assert set(servers) == {"task_tools"} and getattr(state, "probe", None) is None
        receipt = json.loads((Path(state.session.session_dir) / "mcp-probe-1" / "cleanup.json").read_text())
        assert receipt["cleanup_confirmed"] is True and receipt["timed_out"] is True
    finally:
        await owner.stop()


async def test_required_probe_timeout_fails_and_leaves_no_probe(probe_env, tmp_path):
    agent, runtime, pid_dir, workdir = probe_env
    owner, state, _ = await borrowed_state(tmp_path, runtime, workdir)
    try:
        with pytest.raises(HTTPException) as failed:
            await agent._probe_mcp_grants(
                state,
                payload_for(runtime, pid_dir, block="task_tools"),
                {"task_tools": {"url": "http://x/mcp"}},
                {"task_tools"},
                timeout=2,
            )
        assert failed.value.status_code == 502
        assert not alive(pid_of(pid_dir, "task_tools")) and getattr(state, "probe", None) is None
    finally:
        await owner.stop()


async def test_close_during_a_probe_stops_it_before_release(probe_env, tmp_path):
    agent, runtime, pid_dir, workdir = probe_env
    owner, state, release = await borrowed_state(tmp_path, runtime, workdir)

    async def record_release(*args, **kwargs):
        release["calls"] += 1
        release["alive_at_disconnect"] = alive(pid_of(pid_dir, "task_tools"))

    state.session.sandbox.disconnect = record_release
    try:
        state.task = asyncio.create_task(
            agent._probe_mcp_grants(
                state,
                payload_for(runtime, pid_dir, block="task_tools"),
                {"task_tools": {"url": "http://x/mcp"}},
                {"task_tools"},
                timeout=60,
            )
        )
        for _ in range(200):
            if (pid_dir / "task_tools.pid").exists():
                break
            await asyncio.sleep(0.05)
        pid = pid_of(pid_dir, "task_tools")
        assert alive(pid)
        await state.close(timeout=15)
        assert not alive(pid)
        assert release == {"calls": 1, "alive_at_disconnect": False}
        assert getattr(state, "probe", None) is None and state.session.closed
        # The owner's sandbox is not stopped: it still runs commands in the task directory.
        result = await owner.exec("echo owner-alive", timeout_s=30)
        assert result.return_code == 0 and "owner-alive" in (result.stdout or "")
    finally:
        await owner.stop()


async def test_unconfirmed_probe_termination_fails_close(probe_env, tmp_path):
    agent, runtime, pid_dir, workdir = probe_env
    owner, state, release = await borrowed_state(tmp_path, runtime, workdir)

    async def record_release(*args, **kwargs):
        release["calls"] += 1

    state.session.sandbox.disconnect = record_release
    # A probe that launched and left no receipt: its directory is gone, so termination cannot be confirmed.
    probe = SandboxSession(
        sandbox=owner, session_dir=str(tmp_path / "missing-probe"), workdir=str(workdir), harness="OpenClaw MCP probe"
    )
    probe.launch_started = True
    state.probe = probe
    try:
        with pytest.raises(RuntimeError, match="cannot confirm termination"):
            await state.close(timeout=3)
        assert state.probe is probe and not state.session.closed and release["calls"] == 0
    finally:
        await owner.stop()


async def test_final_negative_probe_receipt_releases_unconfirmed(probe_env, tmp_path):
    from nemo_gym.agent_utils.supervisor_client import harness_termination_failure

    agent, runtime, pid_dir, workdir = probe_env
    owner, state, release = await borrowed_state(tmp_path, runtime, workdir)

    async def record_release(*args, **kwargs):
        release["calls"] += 1

    state.session.sandbox.disconnect = record_release
    probe = SandboxSession(
        sandbox=owner, session_dir=str(tmp_path / "probe"), workdir=str(workdir), harness="OpenClaw MCP probe"
    )
    probe.launch_started = True
    receipt = {"cleanup_confirmed": False, "return_code": None, "timed_out": False, "error": "group survived"}
    probe.termination_error = harness_termination_failure(harness="OpenClaw MCP probe", receipt=receipt)
    original_failure = probe.termination_error
    original_failure.__cause__ = RuntimeError("supervisor reported a surviving process group")
    original_record = original_failure.to_record()
    state.probe = probe
    try:
        await state.close(timeout=5)
        # Released for Resources teardown, as the agent run's session does, but never reported as cleaned up.
        assert release["calls"] == 1 and state.session.closed and not state.session.cleanup_confirmed
        assert state.probe is None
        assert state.session.termination_error is original_failure
        assert state.session.termination_error.to_record() == original_record
    finally:
        await owner.stop()


async def test_close_before_probe_registration_starts_no_probe(probe_env, tmp_path):
    """Close that begins while the activation still stages its MCP config must not let a probe start afterwards."""
    agent, runtime, pid_dir, workdir = probe_env
    owner, state, release = await borrowed_state(tmp_path, runtime, workdir)
    staged, resume_staging = asyncio.Event(), asyncio.Event()
    retiring, resume_retire = asyncio.Event(), asyncio.Event()
    upload_json, execute = state.upload_json, owner.exec

    async def delayed_config(name, data):
        staged.set()
        await resume_staging.wait()
        await upload_json(name, data)

    async def held_remove(command, **kwargs):
        # Hold the session-directory removal before it moves anything, so the runner files still exist.
        if command.startswith("if [ -d") and ".closed" in command:
            retiring.set()
            await resume_retire.wait()
        return await execute(command, **kwargs)

    async def record_release(*args, **kwargs):
        release["calls"] += 1

    state.upload_json = delayed_config
    owner.exec = held_remove
    state.session.sandbox.disconnect = record_release
    try:
        state.task = asyncio.create_task(
            agent._probe_mcp_grants(
                state,
                payload_for(runtime, pid_dir, block="task_tools"),
                {"task_tools": {"url": "http://x/mcp"}},
                {"task_tools"},
                timeout=60,
            )
        )
        await asyncio.wait_for(staged.wait(), timeout=5)
        closing = asyncio.create_task(state.close(timeout=15))
        await asyncio.wait_for(retiring.wait(), timeout=10)
        resume_staging.set()
        for _ in range(60):
            if (pid_dir / "task_tools.pid").exists() or state.task.done():
                break
            await asyncio.sleep(0.05)
        started_during_close = (pid_dir / "task_tools.pid").exists()
        resume_retire.set()
        await asyncio.wait_for(closing, timeout=30)
        assert not started_during_close, "a probe started after close began"
        assert state.session.closed and release["calls"] == 1 and state.task.done()
    finally:
        if (pid_dir / "task_tools.pid").exists() and alive(pid_of(pid_dir, "task_tools")):
            os.kill(pid_of(pid_dir, "task_tools"), 9)
        await owner.stop()


async def test_unconfirmed_first_cleanup_keeps_the_probe_for_close(probe_env, tmp_path, monkeypatch):
    """A probe whose first stop cannot be confirmed fails the activation and stays registered until close confirms."""
    agent, runtime, pid_dir, workdir = probe_env
    owner, state, release = await borrowed_state(tmp_path, runtime, workdir)
    confirm = sandbox_session_module.stop_and_confirm_cleanup
    calls = []

    async def first_stop_unconfirmed(*args, **kwargs):
        calls.append(kwargs["session_dir"])
        if len(calls) == 1:
            raise RuntimeError("probe launch outcome is unknown; cannot confirm termination")
        return await confirm(*args, **kwargs)

    async def record_release(*args, **kwargs):
        release["calls"] += 1

    monkeypatch.setattr(sandbox_session_module, "stop_and_confirm_cleanup", first_stop_unconfirmed)
    state.session.sandbox.disconnect = record_release
    try:
        with pytest.raises(RuntimeError, match="did not confirm termination"):
            await agent._probe_mcp_grants(
                state,
                payload_for(runtime, pid_dir, block="task_tools"),
                {"task_tools": {"url": "http://x/mcp"}},
                {"task_tools"},
                timeout=2,
            )
        # The result did not count, and the probe stays registered: close must confirm it before release.
        assert state.probe is not None and not state.probe.cleanup_confirmed and release["calls"] == 0
        await state.close(timeout=15)
        assert len(calls) == 2 and calls[0] == calls[1]
        assert state.probe is None and state.session.closed and release["calls"] == 1
        assert not alive(pid_of(pid_dir, "task_tools"))
    finally:
        await owner.stop()


async def test_probe_directory_belongs_to_the_session_user(probe_env, tmp_path):
    """An upload that creates missing parents the session user cannot write must not break the probe."""
    agent, runtime, pid_dir, workdir = probe_env
    owner, state, release = await borrowed_state(tmp_path, runtime, workdir)
    upload = owner.upload

    async def upload_creating_foreign_parents(source, destination):
        # Like a provider that creates missing parents as root: the session user cannot write into them.
        parent = Path(destination).parent
        if "/mcp-probe-" in str(parent) and not parent.exists():
            parent.mkdir(parents=True)
            await upload(source, destination)
            parent.chmod(0o555)
            return
        await upload(source, destination)

    async def record_release(*args, **kwargs):
        release["calls"] += 1

    owner.upload = upload_creating_foreign_parents
    state.session.sandbox.disconnect = record_release
    servers = {"task_tools": {"url": "http://x/mcp"}}
    try:
        gaps = await asyncio.wait_for(
            agent._probe_mcp_grants(state, payload_for(runtime, pid_dir), servers, {"task_tools"}, timeout=20),
            timeout=60,
        )
        assert gaps == [] and set(servers) == {"task_tools"} and state.probe is None
        await state.close(timeout=15)
        assert state.session.closed and release["calls"] == 1
    finally:
        for path in Path(state.session.session_dir).parent.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
        await owner.stop()


async def test_probe_that_fails_before_openclaw_starts_reports_why(probe_env, tmp_path):
    """A probe whose command cannot start (no Node) reports the error from output.log, not an empty `exit 1: `."""
    agent, runtime, pid_dir, workdir = probe_env
    owner, state, _ = await borrowed_state(tmp_path, runtime, workdir)
    (runtime / "node" / "bin" / "node").unlink()
    try:
        with pytest.raises(HTTPException) as failed:
            await agent._probe_mcp_grants(
                state,
                payload_for(runtime, pid_dir),
                {"task_tools": {"url": "http://x/mcp"}},
                {"task_tools"},
                timeout=20,
            )
        detail = failed.value.detail
        assert detail.startswith("Required Gym MCP tools could not be initialized: task_tools: exit ")
        assert detail.endswith(": executable not found"), detail
    finally:
        await owner.stop()


@pytest.mark.parametrize("kind", ["unknown", "model", "other_harness"])
async def test_probe_failure_outside_final_termination_is_preserved(kind):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nemo_gym.typed_exceptions import NgFailureEvent

    error = RuntimeError("cleanup evidence is not available")
    if kind != "unknown":
        error = NgFailureEvent(
            boundary="model" if kind == "model" else "harness",
            failure_kind="model_transport_error",
            failure_reason="not a final harness termination receipt",
        )
    probe = SimpleNamespace(cleanup_confirmed=False, stop_harness=AsyncMock(side_effect=error))
    session = SimpleNamespace(termination_error=None)
    state = OpenClawSandboxSession(request=MagicMock(), session=session, runtime="unused", probe=probe)

    with pytest.raises(type(error)) as raised:
        await state._stop_probe(timeout=1)

    assert raised.value is error
    assert state.probe is probe
    assert session.termination_error is None
    probe.stop_harness.assert_awaited_once_with(timeout=1)
