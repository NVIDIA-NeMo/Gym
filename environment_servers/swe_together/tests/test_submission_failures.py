# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from environment_servers.swe_together import submission
from environment_servers.swe_together.evaluation import JudgeMeasurementError
from nemo_gym.sandbox import SandboxExecResult


KILLED = SandboxExecResult(stdout=None, stderr="signal: killed", return_code=-1, error_type=None)
PROBE = SandboxExecResult(stdout="oom_kill 0\nmemory.peak 1024", stderr=None, return_code=0, error_type=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["freeze", "restore"])
async def test_outer_timeout_keeps_failed_command_before_resource_probe(tmp_path, monkeypatch, action):
    clock = iter([1000.0, 1042.0])
    monkeypatch.setattr(submission, "monotonic", lambda: next(clock))
    failed = SandboxExecResult(stdout="worker output", stderr="signal: killed", return_code=-1, error_type=None)
    probing = asyncio.Event()

    async def execute(command, **kwargs):
        if command == action:
            return failed
        assert command == submission.RESOURCE_PROBE
        probing.set()
        await asyncio.Event().wait()

    sandbox = SimpleNamespace(exec=AsyncMock(side_effect=execute))
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await submission.run_transfer(sandbox, action, action=action, artifact_dir=tmp_path)
    assert probing.is_set()
    assert json.loads((tmp_path / f"{action}-command.json").read_text()) == {
        "return_code": -1,
        "error_type": None,
        "stdout": "worker output",
        "stderr": "signal: killed",
        "timeout_seconds": submission.TRANSFER_TIMEOUT_S,
        "elapsed_seconds": 42.0,
        "reached_timeout": False,
    }
    assert not (tmp_path / f"{action}-resources.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed", [submission.TRANSFER_TIMEOUT_S + 11.0, 42.0])
async def test_killed_freeze_is_not_called_a_timeout_and_keeps_evidence(tmp_path, monkeypatch, elapsed):
    # Observed shape: execd reports a kill at its command timeout and an OOM kill alike, as exit -1 only.
    clock = iter([1000.0, 1000.0 + elapsed])
    monkeypatch.setattr(submission, "monotonic", lambda: next(clock))
    sandbox = SimpleNamespace(exec=AsyncMock(side_effect=[KILLED, PROBE]))
    with pytest.raises(JudgeMeasurementError) as failure:
        await submission.run_transfer(sandbox, "freeze", action="freeze", artifact_dir=tmp_path)
    assert failure.value.cause == "submission_killed"
    assert f"elapsed_seconds={elapsed:.1f}, timeout_seconds={submission.TRANSFER_TIMEOUT_S:g}" in str(failure.value)
    record = json.loads((tmp_path / "freeze-command.json").read_text())
    assert record["return_code"] == -1 and record["stderr"] == "signal: killed"
    assert record["timeout_seconds"] == submission.TRANSFER_TIMEOUT_S and record["elapsed_seconds"] == elapsed
    assert record["reached_timeout"] is (elapsed >= submission.TRANSFER_TIMEOUT_S)
    assert json.loads((tmp_path / "freeze-resources.json").read_text())["stdout"] == PROBE.stdout


@pytest.mark.asyncio
async def test_provider_reported_timeout_is_a_submission_timeout(tmp_path):
    timed_out = SandboxExecResult(stdout=None, stderr="Command timed out", return_code=124, error_type="timeout")
    sandbox = SimpleNamespace(exec=AsyncMock(side_effect=[timed_out, PROBE]))
    with pytest.raises(JudgeMeasurementError) as failure:
        await submission.run_transfer(sandbox, "freeze", action="freeze", artifact_dir=tmp_path)
    assert failure.value.cause == "submission_timeout"


@pytest.mark.asyncio
async def test_successful_freeze_records_timing_without_resource_probe(tmp_path):
    ok = SandboxExecResult(stdout="{}", stderr=None, return_code=0, error_type=None)
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=ok))
    assert await submission.run_transfer(sandbox, "freeze", action="freeze", artifact_dir=tmp_path) is ok
    sandbox.exec.assert_awaited_once()
    assert (
        json.loads((tmp_path / "freeze-command.json").read_text())["timeout_seconds"] == submission.TRANSFER_TIMEOUT_S
    )
    assert not (tmp_path / "freeze-resources.json").exists()


def test_failure_without_timing_is_a_plain_submission_failure(tmp_path):
    result = SandboxExecResult(stdout="Traceback", stderr="exit status 1", return_code=1, error_type=None)
    with pytest.raises(JudgeMeasurementError) as failure:
        submission.record_command(result, action="restore", artifact_dir=tmp_path)
    assert failure.value.cause == "submission_failed"
