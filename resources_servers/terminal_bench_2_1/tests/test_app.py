# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import resources_servers.terminal_bench_2_1.app as terminal_bench_app
from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_2_1.app import (
    _BULLSEYE_SECURITY_SNAPSHOT_SETUP,
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
    TerminalBench21SeedSessionRequest,
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
        sandbox.exec.return_value = MagicMock(return_code=0)
        await setup(sandbox)
        assert sandbox.exec.await_count == 2
        assert "/etc/apt/sources.list" in sandbox.exec.await_args_list[0].args[0]
        assert sandbox.exec.await_args_list[1].args[0] == "apt-get update"
        sandbox.exec.reset_mock()
        sandbox.exec.return_value = MagicMock(return_code=1)
        with pytest.raises(RuntimeError, match="Failed to prepare TerminalBench package sources"):
            await setup(sandbox)
        sandbox.exec.assert_awaited_once()

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


@pytest.mark.parametrize("distro", ["debian:bullseye", "debian:bookworm", "ubuntu:noble"])
def test_security_snapshot_preserves_other_repositories(tmp_path: Path, distro: str) -> None:
    distro_id, codename = distro.split(":")
    os_release = tmp_path / "os-release"
    os_release.write_text(f"ID={distro_id}\nVERSION_CODENAME={codename}\n")
    sources = tmp_path / "sources.list"
    live = "deb http://deb.debian.org/debian-security bullseye-security main"
    unrelated = "# " + live + "\ndeb http://deb.debian.org/debian bullseye main\n"
    sources.write_text(unrelated + live + "\n")
    command = ["bash", "-c", _BULLSEYE_SECURITY_SNAPSHOT_SETUP, "--", str(os_release), str(sources)]
    subprocess.run(command, check=True)
    expected = live
    if distro == "debian:bullseye":
        expected = (
            "deb [check-valid-until=no] "
            "https://snapshot.debian.org/archive/debian-security/20260831T235959Z/ bullseye-security main"
        )
    assert sources.read_text() == unrelated + expected + "\n"
    subprocess.run(command, check=True)
    assert sources.read_text() == unrelated + expected + "\n"
