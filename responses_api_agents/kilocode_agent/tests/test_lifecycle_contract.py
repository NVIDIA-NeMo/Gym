# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cleanup evidence must not depend on parseable harness output, and capture must follow it."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.kilocode_agent.sandbox import KiloSandboxSession
from responses_api_agents.kilocode_agent.tests.test_sandbox import seed


DIRECTORY = "/tmp/nemo-gym-kilo-sessions/lifecycle"
OK = SandboxExecResult(stdout="", stderr="", return_code=0)


class Sandbox:
    def __init__(self):
        self.files = {}
        self.events = []
        self.result = {"return_code": 0, "timed_out": False, "cleanup_confirmed": True, "error": None}
        self.exec = AsyncMock(side_effect=self.run_exec)
        self.stop = AsyncMock()
        self.disconnect = AsyncMock()

    async def run_exec(self, command, **kwargs):
        if "--receipt" in command and "process_supervisor.sh" in command:
            self.events.append("launch")
            self.files[f"{DIRECTORY}/stdout.jsonl"] = '{"type":"text","part":{"text":"done"}}\n'
            self.files[f"{DIRECTORY}/stderr.log"] = ""
            self.files[f"{DIRECTORY}/cleanup.json"] = json.dumps(self.result)
        elif "kilo.db" in command:
            self.events.append("snapshot")
            return SandboxExecResult(stdout="", stderr="no database", return_code=1)
        return OK

    async def upload(self, source, destination):
        self.files[destination] = Path(source).read_text()

    async def download(self, source, destination):
        self.events.append(f"read {Path(source).name}")
        Path(destination).write_text(self.files[source])


def session() -> tuple[KiloSandboxSession, Sandbox]:
    sandbox = Sandbox()
    state = KiloSandboxSession(
        request=seed(),
        provider_name="docker",
        session=SandboxSession(sandbox=sandbox, session_dir=DIRECTORY, workdir="/repo", harness="KiloCode"),
    )
    return state, sandbox


async def execute(state: KiloSandboxSession):
    return await state.session.execute(
        stage_activation=lambda: state.stage_activation(command=["kilo", "run"], config="{}"),
        collect=state.collect_artifacts,
        timeout=5,
        close_timeout=1,
    )


async def test_capture_follows_confirmed_cleanup_and_keeps_output():
    state, sandbox = session()
    artifacts = await execute(state)
    assert artifacts.stdout and artifacts.exit_code == 0 and artifacts.observations is None
    assert state.session.cleanup == {"cleanup_confirmed": True, "return_code": 0, "error": None, "timed_out": False}
    assert sandbox.events.index("read cleanup.json") < sandbox.events.index("read stdout.jsonl")
    assert sandbox.events.index("read cleanup.json") < sandbox.events.index("snapshot")
    await state.close(timeout=1)
    assert state.session.closed
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize(
    "invalid", [{"cleanup_confirmed": 1}, {"cleanup_confirmed": "true"}, {"cleanup_confirmed": False}]
)
async def test_bad_cleanup_keeps_state_retryable_without_capture(invalid):
    state, sandbox = session()
    sandbox.result.update(invalid)
    with pytest.raises(RuntimeError):
        await execute(state)
    assert state.session.cleanup is None
    assert state.session.artifacts is None
    assert "read stdout.jsonl" not in sandbox.events and "snapshot" not in sandbox.events
    if invalid["cleanup_confirmed"] is False:
        # A negative receipt is final: close releases the borrowed session for Resources teardown.
        await state.close(timeout=1)
        assert state.session.closed
        assert state.session.termination_error is not None
        assert state.session.capture_error is state.session.termination_error
        assert state.session.artifacts is None
        sandbox.disconnect.assert_awaited_once()
        sandbox.stop.assert_not_awaited()
        return
    with pytest.raises(RuntimeError):
        await state.close(timeout=1)
    assert not state.session.closed
    sandbox.disconnect.assert_not_awaited()
    sandbox.files[f"{DIRECTORY}/cleanup.json"] = json.dumps({"cleanup_confirmed": True, "error": None})
    await state.close(timeout=1)
    assert state.session.closed
    assert state.session.artifacts is not None


async def test_output_failure_does_not_erase_cleanup_confirmation():
    state, sandbox = session()
    download = sandbox.download

    async def fail_output(source, destination):
        if source.endswith("/stdout.jsonl"):
            raise RuntimeError("output unreadable")
        return await download(source, destination)

    sandbox.download = fail_output
    with pytest.raises(RuntimeError, match="did not produce readable artifacts"):
        await execute(state)
    assert state.session.cleanup["cleanup_confirmed"] is True
    assert state.session.cleanup["error"] is None
    await state.close(timeout=1)
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("diagnostics", [{"return_code": "0"}, {"hostname": "extra"}, {"timed_out": 1}])
async def test_diagnostic_schema_drift_does_not_block_capture_or_release(diagnostics):
    state, sandbox = session()
    sandbox.result.update(diagnostics)
    artifacts = await execute(state)
    assert artifacts.stdout
    assert state.session.cleanup["cleanup_confirmed"] is True
    if "return_code" in diagnostics:
        assert state.session.cleanup["return_code"] is None
    await state.close(timeout=1)
    assert state.session.closed
    sandbox.disconnect.assert_awaited_once()
