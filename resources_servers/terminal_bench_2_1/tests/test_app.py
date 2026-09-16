# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import resources_servers.terminal_bench_2_1.app as terminal_bench_app
from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_2_1.app import (
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
    TerminalBench21SeedSessionRequest,
    TerminalBench21VerifyRequest,
)


def _verify_request(task_folder: Path) -> TerminalBench21VerifyRequest:
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
            "task_name": "terminal-bench/background-server",
            "docker_image": "unused-test-image",
            "task_folder": str(task_folder),
        }
    )


class TestApp:
    @pytest.fixture
    def server(self) -> TerminalBench21ResourcesServer:
        config = TerminalBench21ResourcesServerConfig(
            sandbox_provider="",
            sandbox_config=dict(),
            host="",
            port=0,
            entrypoint="",
            name="",
        )
        return TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.parametrize("timeout", [1800, None])
    async def test_golden_patch_preserves_services_and_runs_verifier_separately(
        self, server: TerminalBench21ResourcesServer, tmp_path: Path, timeout: int | None
    ) -> None:
        server.config.is_verifying_golden_patch = True
        server.config.evaluation_timeout = timeout
        sandbox = MagicMock()
        sandbox.exec = AsyncMock(return_value=SimpleNamespace(stdout="/app\n", stderr=None, return_code=0))
        sandbox.exec_with_background_services = AsyncMock(
            return_value=SimpleNamespace(stdout="served\n", stderr=None, return_code=0)
        )
        sandbox.download = AsyncMock(side_effect=lambda remote_path, local_path: Path(local_path).write_text("1"))
        sandbox.stop = AsyncMock()
        server._create_sandbox = AsyncMock(return_value=sandbox)
        server._upload_folder = AsyncMock()

        # Container creation and uploads are mocked; no benchmark checkout or image is needed.
        await server.verify(MagicMock(), _verify_request(tmp_path))

        # The solution needs to leave its server running; the verifier only needs to check it.
        # Forward the configured timeout unchanged, including None (no client deadline).
        sandbox.exec_with_background_services.assert_awaited_once_with("bash /app/solve.sh", timeout_s=timeout)
        plain_commands = [call.args[0] for call in sandbox.exec.await_args_list]
        assert "bash /tests/test.sh" in plain_commands
        assert not any("solve.sh" in command for command in plain_commands)

    async def test_create_sandbox_uses_start_with_setup(self, monkeypatch, tmp_path: Path) -> None:
        sandbox = AsyncMock()
        sandbox.start_with_setup = AsyncMock(return_value=sandbox)
        monkeypatch.setattr(terminal_bench_app, "AsyncSandbox", lambda _provider: sandbox)
        monkeypatch.setattr(terminal_bench_app, "get_global_config_dict", lambda: {})
        monkeypatch.setattr(terminal_bench_app, "resolve_provider_config", lambda *_: MagicMock())
        monkeypatch.setattr(terminal_bench_app, "resolve_provider_metadata", lambda *_: {})
        server = TerminalBench21ResourcesServer(
            config=TerminalBench21ResourcesServerConfig(
                sandbox_provider="test",
                sandbox_config={},
                evaluation_timeout=30,
                host="",
                port=0,
                entrypoint="",
                name="terminal_bench_2_1_resources_server",
            ),
            server_client=MagicMock(spec=ServerClient),
        )

        result = await server._create_sandbox(
            TerminalBench21SeedSessionRequest(
                task_name="terminal-bench/test-task",
                docker_image="terminal-bench/test-task:latest",
                task_folder=str(tmp_path),
            )
        )

        assert result is sandbox
        sandbox.start_with_setup.assert_awaited_once()
        spec, setup = sandbox.start_with_setup.call_args.args
        assert spec is not None
        await setup(sandbox)
        sandbox.exec.assert_awaited_once()
        assert sandbox.exec.call_args.args[0] == "apt-get update"

    async def test_create_sandbox_setup_failure_propagates(self, monkeypatch, tmp_path: Path) -> None:
        sandbox = AsyncMock()
        sandbox.start_with_setup = AsyncMock(side_effect=RuntimeError("setup command failed"))
        monkeypatch.setattr(terminal_bench_app, "AsyncSandbox", lambda _provider: sandbox)
        monkeypatch.setattr(terminal_bench_app, "get_global_config_dict", lambda: {})
        monkeypatch.setattr(terminal_bench_app, "resolve_provider_config", lambda *_: MagicMock())
        monkeypatch.setattr(terminal_bench_app, "resolve_provider_metadata", lambda *_: {})
        server = TerminalBench21ResourcesServer(
            config=TerminalBench21ResourcesServerConfig(
                sandbox_provider="test",
                sandbox_config={},
                evaluation_timeout=30,
                host="",
                port=0,
                entrypoint="",
                name="terminal_bench_2_1_resources_server",
            ),
            server_client=MagicMock(spec=ServerClient),
        )

        with pytest.raises(RuntimeError, match="setup command failed"):
            await server._create_sandbox(
                TerminalBench21SeedSessionRequest(
                    task_name="terminal-bench/test-task",
                    docker_image="terminal-bench/test-task:latest",
                    task_folder=str(tmp_path),
                )
            )
