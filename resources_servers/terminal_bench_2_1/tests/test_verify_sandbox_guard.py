# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify must not grade a sandbox that has died (RL-1469).

The OpenSandbox server can route a dead sandbox's requests to a live sandbox
that reused its pod IP. Grading a dead sandbox would upload this task's tests
into, and run them inside, a stranger's sandbox. The grading commands therefore
ask for the sandbox status first (``require_running``).
"""

from pathlib import Path
from time import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxStatus
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_2_1.app import (
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
    TerminalBench21VerifyRequest,
)


class GradingProvider:
    """Minimal sandbox provider that records grading traffic and reports a fixed status."""

    name = "fake"

    def __init__(self, status: SandboxStatus) -> None:
        self.status_value = status
        self.exec_commands: list[str] = []
        self.uploads: list[str] = []
        self.downloads: list[str] = []
        self.closed = False

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        return SandboxHandle(sandbox_id="fake-1", provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **_kwargs: Any) -> SandboxExecResult:
        self.exec_commands.append(command)
        return SandboxExecResult(stdout="tests ran", stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        self.uploads.append(target_path)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        self.downloads.append(source_path)
        target_path.write_text("1.0", encoding="utf-8")

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return self.status_value

    async def close(self, handle: SandboxHandle) -> None:
        self.closed = True

    async def aclose(self) -> None:
        pass

    def graded(self) -> bool:
        return any("test.sh" in command for command in self.exec_commands)


def _server() -> TerminalBench21ResourcesServer:
    config = TerminalBench21ResourcesServerConfig(
        sandbox_provider="",
        sandbox_config=dict(),
        host="",
        port=0,
        entrypoint="",
        name="tb",
        evaluation_timeout=10,
    )
    return TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _verify_request(task_folder: Path) -> TerminalBench21VerifyRequest:
    response = NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=int(time()),
        model="m",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )
    return TerminalBench21VerifyRequest(
        task_name="task",
        docker_image="image",
        task_folder=str(task_folder),
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response,
    )


def _task_folder(tmp_path: Path) -> Path:
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text("mkdir -p /logs/verifier && echo 1 > /logs/verifier/reward.txt\n")
    return tmp_path / "task"


async def _verify(provider: GradingProvider, tmp_path: Path):
    sandbox = AsyncSandbox(provider)
    await sandbox.start(SandboxSpec(image="image"))
    server = _server()
    server._session_id_to_sandbox["session-1"] = sandbox
    request = SimpleNamespace(session={SESSION_ID_KEY: "session-1"})
    return await server.verify(request, _verify_request(_task_folder(tmp_path)))


async def test_verify_grades_a_running_sandbox(tmp_path: Path) -> None:
    provider = GradingProvider(SandboxStatus.RUNNING)

    result = await _verify(provider, tmp_path)

    assert result.evaluation_completed is True
    assert result.reward == 1.0
    assert result.failure_reason is None
    assert provider.graded()
    assert provider.uploads == ["/tests/test.sh"]
    assert provider.closed


async def test_verify_skips_grading_when_the_sandbox_died(tmp_path: Path) -> None:
    provider = GradingProvider(SandboxStatus.ERROR)

    result = await _verify(provider, tmp_path)

    assert result.evaluation_completed is False
    assert result.reward == 0.0
    assert result.failure_reason is not None and "error" in result.failure_reason
    assert not provider.graded()
    assert provider.uploads == []
    assert provider.downloads == []
    assert provider.closed
