# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify must only grade a sandbox that is confirmed running (RL-1469).

The OpenSandbox server can route a dead sandbox's requests to a live sandbox
that reused its pod IP. Grading a sandbox that died (OOM) would upload this
task's tests into, and run them inside, a stranger's sandbox. So verify asks
the control plane for the sandbox status first and grades only on RUNNING.
"""

from pathlib import Path
from time import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxEndedError
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxStatus
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_2_1.app import (
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
    TerminalBench21VerifyRequest,
)


class FakeSandbox:
    def __init__(
        self,
        status: SandboxStatus | BaseException,
        upload_error: BaseException | None = None,
    ) -> None:
        self._status = status
        self._upload_error = upload_error
        self.calls: list[tuple[Any, ...]] = []

    async def status(self) -> SandboxStatus:
        if isinstance(self._status, BaseException):
            raise self._status
        return self._status

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        self.calls.append(("upload", remote_path))
        if self._upload_error is not None:
            raise self._upload_error

    async def exec(self, command: str, **_kwargs: Any) -> SandboxExecResult:
        self.calls.append(("exec", command))
        return SandboxExecResult(stdout="tests ran", stderr="", return_code=0)

    async def download(self, remote_path: str, local_path: Path | str) -> None:
        self.calls.append(("download", remote_path))
        Path(local_path).write_text("1.0", encoding="utf-8")

    async def stop(self) -> None:
        self.calls.append(("stop",))

    def graded(self) -> bool:
        return any(call[0] == "exec" and "test.sh" in call[1] for call in self.calls)


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


async def _verify(sandbox: FakeSandbox, tmp_path: Path):
    server = _server()
    server._session_id_to_sandbox["session-1"] = sandbox
    request = SimpleNamespace(session={SESSION_ID_KEY: "session-1"})
    return await server.verify(request, _verify_request(_task_folder(tmp_path)))


async def test_verify_grades_a_running_sandbox(tmp_path: Path) -> None:
    sandbox = FakeSandbox(SandboxStatus.RUNNING)

    result = await _verify(sandbox, tmp_path)

    assert result.evaluation_completed is True
    assert result.reward == 1.0
    assert result.failure_reason is None
    assert sandbox.graded()
    assert ("stop",) in sandbox.calls


async def test_verify_skips_grading_when_the_sandbox_died(tmp_path: Path) -> None:
    sandbox = FakeSandbox(SandboxStatus.ERROR)

    result = await _verify(sandbox, tmp_path)

    assert result.evaluation_completed is False
    assert result.reward == 0.0
    assert result.failure_reason is not None and "error" in result.failure_reason
    assert [call for call in sandbox.calls if call[0] in {"upload", "exec", "download"}] == []
    assert ("stop",) in sandbox.calls


async def test_verify_skips_grading_when_the_sandbox_status_is_unavailable(tmp_path: Path) -> None:
    sandbox = FakeSandbox(RuntimeError("control plane unreachable"))

    result = await _verify(sandbox, tmp_path)

    assert result.evaluation_completed is False
    assert result.reward == 0.0
    assert result.failure_reason is not None and "control plane unreachable" in result.failure_reason
    assert not sandbox.graded()
    assert ("stop",) in sandbox.calls


async def test_verify_reports_a_sandbox_that_ends_mid_grading(tmp_path: Path) -> None:
    sandbox = FakeSandbox(SandboxStatus.RUNNING, upload_error=SandboxEndedError("Sandbox 'a' no longer exists"))

    result = await _verify(sandbox, tmp_path)

    assert result.evaluation_completed is False
    assert result.reward == 0.0
    assert result.failure_reason is not None and "no longer exists" in result.failure_reason
    assert not sandbox.graded()
    assert [call for call in sandbox.calls if call[0] == "download"] == []
    assert ("stop",) in sandbox.calls
