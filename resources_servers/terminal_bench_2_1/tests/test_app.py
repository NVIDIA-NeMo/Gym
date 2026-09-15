# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_2_1.app import (
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
    TerminalBench21VerifyRequest,
)


def _verify_request() -> TerminalBench21VerifyRequest:
    return TerminalBench21VerifyRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "response": {
                "output": [],
                "id": "",
                "created_at": 0,
                "model": "",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
            "task_name": "terminal-bench/hf-model-inference",
            "docker_image": "alexgshaw/hf-model-inference:20260430",
            "task_folder": "benchmarks/terminal_bench_2_1/terminal-bench-2-1/tasks/hf-model-inference",
        }
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

    async def test_golden_patch_runs_solve_sh_detached_and_test_sh_plain(self) -> None:
        server = TerminalBench21ResourcesServer(
            config=TerminalBench21ResourcesServerConfig(
                sandbox_provider="",
                sandbox_config=dict(),
                is_verifying_golden_patch=True,
                evaluation_timeout=1800,
                host="",
                port=0,
                entrypoint="",
                name="",
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        sandbox = MagicMock()
        sandbox.exec = AsyncMock(return_value=SimpleNamespace(stdout="/app\n", stderr=None, return_code=0))
        sandbox.exec_setsid = AsyncMock(return_value=SimpleNamespace(stdout="served\n", stderr=None, return_code=0))
        sandbox.download = AsyncMock(side_effect=lambda remote_path, local_path: Path(local_path).write_text("1"))
        sandbox.stop = AsyncMock()
        server._create_sandbox = AsyncMock(return_value=sandbox)
        server._upload_folder = AsyncMock()

        response = await server.verify(MagicMock(), _verify_request())

        sandbox.exec_setsid.assert_awaited_once_with("bash /app/solve.sh", timeout_s=1800)
        plain_commands = [call.args[0] for call in sandbox.exec.await_args_list]
        assert "bash /tests/test.sh" in plain_commands
        assert not any("solve.sh" in command for command in plain_commands)
        assert response.golden_patch_output == "served\n"
        assert response.reward == 1.0
        assert response.evaluation_completed
        sandbox.stop.assert_awaited_once()
