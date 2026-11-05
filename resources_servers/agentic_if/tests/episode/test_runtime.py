# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution in the borrowed sandbox keeps the task identity, its working directory and capped deadlines."""

import pytest

from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if.episode.runtime import BoundSandbox, private_staging


class SandboxTransport:
    """External provider double; execution bindings are the observable wire contract."""

    def __init__(self):
        self.calls = []

    async def exec(self, **kwargs):
        self.calls.append(kwargs)
        return SandboxExecResult(stdout="result", stderr="", return_code=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["agent", 1000, None, "root"])
async def test_commands_keep_bound_identity_and_workdir(user):
    transport = SandboxTransport()
    sandbox = BoundSandbox(transport, user=user, workdir="/app/project", timeout_s=60)
    result = await sandbox.exec(command="pwd", timeout_s=60)
    assert result.stdout == "result"
    assert transport.calls == [{"command": "pwd", "user": user, "cwd": "/app/project", "timeout_s": 60, "env": None}]


@pytest.mark.asyncio
async def test_timeout_is_bounded_without_replacing_an_explicit_cwd():
    transport = SandboxTransport()
    sandbox = BoundSandbox(transport, user="agent", workdir="/app", timeout_s=60)
    await sandbox.exec(command="pwd", cwd="/app/src", timeout_s=10)
    await sandbox.exec(command="pwd", timeout_s=999)
    assert [(call["timeout_s"], call["cwd"]) for call in transport.calls] == [(10, "/app/src"), (60, "/app")]


class FailedRemoval(SandboxTransport):
    """Removing the staging directory fails: with an exit status, without one, or by raising."""

    def __init__(self, removal):
        super().__init__()
        self.removal = removal

    async def exec(self, **kwargs):
        self.calls.append(kwargs)
        if not kwargs["command"].startswith("rm -rf "):
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        if isinstance(self.removal, Exception):
            raise self.removal
        return self.removal


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "removal",
    [
        SandboxExecResult(stdout=None, stderr=None, return_code=1),
        SandboxExecResult(stdout=None, stderr=None, return_code=-1, error_type="timeout"),
        ConnectionError("sandbox connection lost"),
    ],
    ids=["exit-status", "no-exit-status", "raises"],
)
async def test_a_failed_staging_removal_is_logged_and_keeps_the_original_error(removal, caplog):
    transport = FailedRemoval(removal)
    sandbox = BoundSandbox(transport, user="agent", workdir="/app", timeout_s=60)
    with pytest.raises(ValueError, match="recorded diff does not apply"):
        async with private_staging(sandbox, "replay") as staging:
            raise ValueError("recorded diff does not apply")
    assert f"could not remove {staging}" in caplog.text
    caplog.clear()
    async with private_staging(sandbox, "replay") as staging:
        pass
    assert f"could not remove {staging}" in caplog.text
    removals = [call for call in transport.calls if call["command"].startswith("rm -rf ")]
    assert [call["user"] for call in removals] == ["root", "root"]
