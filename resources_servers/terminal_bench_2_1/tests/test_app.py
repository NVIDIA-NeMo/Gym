# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from shlex import quote
from shutil import copyfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import Request

import resources_servers.terminal_bench_2_1.app as terminal_bench_app
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_2_1.app import (
    _BULLSEYE_SECURITY_SNAPSHOT_SETUP,
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

    @pytest.mark.parametrize("replace_destination", [False, True], ids=["in-place", "replace-file"])
    @pytest.mark.parametrize(
        "reward_text,expected_reward,evaluation_completed",
        [("1\n", 1.0, True), ("0\n", 0.0, True), ("", 0.0, False), ("invalid", 0.0, False), (None, 0.0, False)],
        ids=["pass", "fail", "empty", "invalid", "missing"],
    )
    async def test_verify_reads_downloaded_reward(
        self,
        server: TerminalBench21ResourcesServer,
        tmp_path: Path,
        replace_destination: bool,
        reward_text: str | None,
        expected_reward: float,
        evaluation_completed: bool,
    ) -> None:
        def download_reward(remote_path: str, local_path: str) -> None:
            assert remote_path == "/logs/verifier/reward.txt"
            if reward_text is None:
                raise FileNotFoundError(remote_path)
            destination = Path(local_path)
            if replace_destination:
                # Docker cp replaces the file rather than writing to its existing inode.
                downloaded = tmp_path / "downloaded-reward.txt"
                downloaded.write_text(reward_text)
                downloaded.replace(destination)
            else:
                destination.write_text(reward_text)

        sandbox = MagicMock()
        sandbox.exec = AsyncMock(return_value=SimpleNamespace(stdout="verifier output", stderr="", return_code=0))
        sandbox.download = AsyncMock(side_effect=download_reward)
        sandbox.stop = AsyncMock()
        server._upload_folder = AsyncMock()
        server._session_id_to_sandbox["test-session"] = sandbox
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "test-session"}

        result = await server.verify(request, _verify_request(tmp_path))

        assert result.reward == expected_reward
        assert result.evaluation_completed is evaluation_completed
        assert result.test_output == "verifier output"
        sandbox.download.assert_awaited_once()
        sandbox.stop.assert_awaited_once()
        assert "test-session" not in server._session_id_to_sandbox

    @pytest.mark.parametrize("timeout", [1800, None])
    async def test_golden_patch_preserves_services_and_runs_verifier_separately(
        self, server: TerminalBench21ResourcesServer, tmp_path: Path, timeout: int | None
    ) -> None:
        server.config.is_verifying_golden_patch = True
        server.config.evaluation_timeout = timeout
        sandbox = MagicMock()
        sandbox.exec = AsyncMock(return_value=SimpleNamespace(stdout="/app\n", stderr=None, return_code=0))
        sandbox.download = AsyncMock(side_effect=lambda remote_path, local_path: Path(local_path).write_text("1"))
        sandbox.stop = AsyncMock()
        server._create_sandbox = AsyncMock(return_value=sandbox)
        server._upload_folder = AsyncMock()

        await server.verify(MagicMock(), _verify_request(tmp_path))

        assert sandbox.exec.await_args_list == [
            call("pwd"),
            call("bash /app/solve.sh", timeout_s=timeout, preserve_background_services=True),
            call("mkdir -p /logs/verifier", timeout_s=timeout),
            call("bash /tests/test.sh", timeout_s=timeout),
        ]

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

    @pytest.mark.parametrize("test_exit_code", [0, 1, 127])
    @pytest.mark.parametrize("logs_state", ["missing", "directory", "file"])
    async def test_verify_prepares_reward_directory(
        self, monkeypatch, tmp_path: Path, test_exit_code: int, logs_state: str
    ) -> None:
        verifier_dir = tmp_path / "logs" / "verifier"
        if logs_state == "directory":
            verifier_dir.mkdir(parents=True)
        elif logs_state == "file":
            verifier_dir.parent.mkdir()
            verifier_dir.write_text("not a directory")
        script = tmp_path / "test.sh"
        # Like the benchmark scripts, report the test command's outcome without
        # relying on a test runner to create the output directory as a side effect.
        script.write_text(
            f"bash -c 'exit {test_exit_code}'\n"
            f"if [ $? -eq 0 ]; then echo 1; else echo 0; fi > {quote(str(verifier_dir))}/reward.txt\n"
        )

        async def execute(command, **kwargs):
            command = command.replace("/logs/verifier", quote(str(verifier_dir)))
            command = command.replace("/tests/test.sh", quote(str(script)))
            result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=10)
            return MagicMock(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

        async def download(remote_path, local_path):
            assert remote_path == "/logs/verifier/reward.txt"
            copyfile(verifier_dir / "reward.txt", local_path)

        sandbox = AsyncMock()
        sandbox.exec.side_effect = execute
        sandbox.download.side_effect = download
        server = TerminalBench21ResourcesServer(
            config=TerminalBench21ResourcesServerConfig(
                sandbox_provider="test",
                sandbox_config={},
                evaluation_timeout=30,
                host="",
                port=0,
                entrypoint="",
                name="",
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        server._session_id_to_sandbox["test-session"] = sandbox
        monkeypatch.setattr(TerminalBench21ResourcesServer, "_upload_folder", AsyncMock())
        body = TerminalBench21VerifyRequest(
            task_name="terminal-bench/test-task",
            docker_image="test-task:latest",
            task_folder=str(tmp_path),
            responses_create_params={"input": []},
            response=NeMoGymResponse(
                id="response",
                created_at=0,
                model="test",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ),
        )

        result = await server.verify(Request({"type": "http", "session": {SESSION_ID_KEY: "test-session"}}), body)

        assert result.evaluation_completed == (logs_state != "file")
        assert result.reward == (1.0 if test_exit_code == 0 and logs_state != "file" else 0.0)
        if logs_state == "file":
            sandbox.download.assert_not_awaited()
            assert not any("test.sh" in call.args[0] for call in sandbox.exec.await_args_list)
        assert result.test_output == ""
        sandbox.stop.assert_awaited_once()
        assert "test-session" not in server._session_id_to_sandbox


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
