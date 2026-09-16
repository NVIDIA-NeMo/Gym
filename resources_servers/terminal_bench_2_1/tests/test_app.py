# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_2_1.app import (
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
    TerminalBench21VerifyRequest,
)


class TestApp:
    def test_sanity(self) -> None:
        config = TerminalBench21ResourcesServerConfig(
            sandbox_provider="",
            sandbox_config=dict(),
            host="",
            port=0,
            entrypoint="",
            name="",
        )
        TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_server() -> TerminalBench21ResourcesServer:
    config = TerminalBench21ResourcesServerConfig(
        sandbox_provider="",
        sandbox_config=dict(),
        host="",
        port=0,
        entrypoint="",
        name="",
    )
    return TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _verify_body(task_folder: str) -> TerminalBench21VerifyRequest:
    return TerminalBench21VerifyRequest(
        task_name="circuit-fibsqrt",
        docker_image="image",
        task_folder=task_folder,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="solve"),
        response=NeMoGymResponse(
            id="resp",
            created_at=0,
            model="",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ),
    )


class _DockerCpSandbox:
    """Emulates `docker cp` download semantics: the target file is REPLACED
    (new inode), so a pre-opened handle would read the now-empty original —
    exactly how a passing 3/3-test rollout once scored reward 0.0."""

    def __init__(self, *, reward: str | None = "1\n") -> None:
        self.reward = reward
        self.exec_calls: list[str] = []
        self.stopped = False

    async def exec(self, command: str, **kwargs: Any) -> Any:
        self.exec_calls.append(command)
        return SimpleNamespace(return_code=0, stdout="3 passed\n", stderr="")

    async def download(self, remote_path: str, local_path: str) -> None:
        if self.reward is None:
            raise FileNotFoundError(remote_path)
        staged = Path(str(local_path) + ".staged")
        staged.write_text(self.reward)
        os.replace(staged, local_path)

    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_verify_reads_reward_from_replaced_file(tmp_path: Path) -> None:
    """The reward must be read from the downloaded file itself, not a handle
    opened before the download replaced it (rerun #3 passed 3/3 tests but
    scored 0.0)."""
    server = _make_server()
    (tmp_path / "tests").mkdir()
    sandbox = _DockerCpSandbox(reward="1\n")
    server._session_id_to_sandbox["session"] = sandbox

    response = await server.verify(SimpleNamespace(session={SESSION_ID_KEY: "session"}), _verify_body(str(tmp_path)))

    assert response.reward == 1.0
    assert response.evaluation_completed is True
    assert "bash /tests/test.sh" in sandbox.exec_calls
    assert sandbox.stopped is True


@pytest.mark.asyncio
async def test_verify_reports_incomplete_when_reward_is_unavailable(tmp_path: Path) -> None:
    server = _make_server()
    (tmp_path / "tests").mkdir()
    sandbox = _DockerCpSandbox(reward=None)
    server._session_id_to_sandbox["session"] = sandbox

    response = await server.verify(SimpleNamespace(session={SESSION_ID_KEY: "session"}), _verify_body(str(tmp_path)))

    assert response.reward == 0.0
    assert response.evaluation_completed is False
