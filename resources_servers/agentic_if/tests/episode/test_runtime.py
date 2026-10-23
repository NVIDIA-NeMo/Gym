# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution in the borrowed sandbox keeps the task identity, its working directory and capped deadlines."""

import pytest

from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if.episode.runtime import BoundSandbox


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
    result = await sandbox.exec(command="pwd", env={"X": "literal"})
    assert result.stdout == "result"
    assert transport.calls == [
        {"command": "pwd", "user": user, "cwd": "/app/project", "timeout_s": 60, "env": {"X": "literal"}}
    ]


@pytest.mark.asyncio
async def test_timeout_is_bounded_without_replacing_an_explicit_cwd():
    transport = SandboxTransport()
    sandbox = BoundSandbox(transport, user="agent", workdir="/app", timeout_s=60)
    await sandbox.exec(command="pwd", cwd="/app/src", timeout_s=10)
    await sandbox.exec(command="pwd", timeout_s=999)
    assert [(call["timeout_s"], call["cwd"]) for call in transport.calls] == [(10, "/app/src"), (60, "/app")]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_execution_budget_rejected(timeout):
    with pytest.raises(ValueError, match="timeout"):
        BoundSandbox(SandboxTransport(), user=None, workdir="/app", timeout_s=timeout)
