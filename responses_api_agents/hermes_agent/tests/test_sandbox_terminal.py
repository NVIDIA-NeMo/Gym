# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox.providers.base import SandboxExecResult
from responses_api_agents.hermes_agent import sandbox_terminal


class _Sandbox:
    def __init__(self, name: str) -> None:
        self.name = name
        self.commands: list[tuple[str, str | None]] = []
        self.disconnected = False

    async def exec(self, command: str, *, cwd: str | None, timeout_s: int) -> SandboxExecResult:
        self.commands.append((command, cwd))
        return SandboxExecResult(stdout=f"{self.name}:{command}", stderr="", return_code=0)

    async def disconnect(self) -> None:
        self.disconnected = True


@pytest.fixture(autouse=True)
def _clear_bindings() -> None:
    sandbox_terminal._BINDINGS.clear()


async def test_terminal_routes_concurrent_sessions_to_distinct_sandboxes(monkeypatch) -> None:
    sandboxes = [_Sandbox("first"), _Sandbox("second")]
    connect = AsyncMock(side_effect=sandboxes)
    monkeypatch.setattr(sandbox_terminal, "create_provider", lambda config: AsyncMock())
    monkeypatch.setattr(sandbox_terminal.AsyncSandbox, "connect", connect)

    await sandbox_terminal.bind_sandbox(
        "session-1",
        descriptor={"sandbox_id": "first"},
        provider_config={"opensandbox": {}},
        workdir="/app",
    )
    await sandbox_terminal.bind_sandbox(
        "session-2",
        descriptor={"sandbox_id": "second"},
        provider_config={"opensandbox": {}},
        workdir="/workspace",
    )

    first, second = await asyncio.gather(
        asyncio.to_thread(
            sandbox_terminal._terminal_handler,
            {"command": "pwd"},
            task_id="session-1",
        ),
        asyncio.to_thread(
            sandbox_terminal._terminal_handler,
            {"command": "ls"},
            task_id="session-2",
        ),
    )

    assert json.loads(first)["output"] == "first:pwd"
    assert json.loads(second)["output"] == "second:ls"
    assert sandboxes[0].commands == [("pwd", "/app")]
    assert sandboxes[1].commands == [("ls", "/workspace")]


async def test_unbind_disconnects_without_stopping(monkeypatch) -> None:
    sandbox = _Sandbox("sandbox")
    monkeypatch.setattr(sandbox_terminal, "create_provider", lambda config: AsyncMock())
    monkeypatch.setattr(sandbox_terminal.AsyncSandbox, "connect", AsyncMock(return_value=sandbox))
    await sandbox_terminal.bind_sandbox(
        "session",
        descriptor={"sandbox_id": "sandbox"},
        provider_config={"opensandbox": {}},
        workdir="/app",
    )

    await sandbox_terminal.unbind_sandbox("session")

    assert sandbox.disconnected is True


def test_terminal_fails_closed_without_session_or_for_unsupported_arguments() -> None:
    with pytest.raises(RuntimeError, match="No borrowed sandbox"):
        sandbox_terminal._terminal_handler({"command": "pwd"}, task_id="unknown")
    with pytest.raises(ValueError, match="Unsupported"):
        sandbox_terminal._terminal_handler(
            {"command": "pwd", "background": True},
            task_id="unknown",
        )
