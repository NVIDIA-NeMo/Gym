# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import resources_servers.terminal_bench_2_1.app as terminal_bench_app
from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_2_1.app import (
    GOLDEN_PATCH_SOLVE_SH_PATCHES,
    TEST_SH_PATCHES,
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


PINNED_TASKS = terminal_bench_app.PARENT_DIR / "benchmarks/terminal_bench_2_1/data/inkling-small-tasks/tasks"
TASK = "terminal-bench/example"
PATCHES = {TASK: [("apt-get update", "apt-get update && echo patched")]}


def _server() -> TerminalBench21ResourcesServer:
    return TerminalBench21ResourcesServer(
        config=TerminalBench21ResourcesServerConfig(
            sandbox_provider="test",
            sandbox_config={},
            host="",
            port=0,
            entrypoint="",
            name="terminal_bench_2_1_resources_server",
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def _recording_sandbox(uploads: dict[str, str]) -> AsyncMock:
    sandbox = AsyncMock()
    sandbox.exec.return_value = MagicMock(return_code=0)

    async def upload(*, local_path: str, remote_path: str) -> None:
        uploads[remote_path] = Path(local_path).read_text()

    sandbox.upload.side_effect = upload
    return sandbox


class TestPatchedUpload:
    async def test_matching_patch_changes_only_the_uploaded_script(self, tmp_path: Path) -> None:
        (tmp_path / "test.sh").write_text("apt-get update\npytest\n")
        (tmp_path / "test_outputs.py").write_text("apt-get update  # not a script\n")
        uploads: dict[str, str] = {}

        await _server()._upload_folder(_recording_sandbox(uploads), tmp_path, "/tests", PATCHES, TASK)

        assert uploads["/tests/test.sh"] == "apt-get update && echo patched\npytest\n"
        assert uploads["/tests/test_outputs.py"] == "apt-get update  # not a script\n"

    async def test_patch_may_match_any_one_script_in_the_folder(self, tmp_path: Path) -> None:
        (tmp_path / "helper.sh").write_text("echo helper\n")
        (tmp_path / "test.sh").write_text("apt-get update\n")
        uploads: dict[str, str] = {}

        await _server()._upload_folder(_recording_sandbox(uploads), tmp_path, "/tests", PATCHES, TASK)

        assert uploads["/tests/helper.sh"] == "echo helper\n"
        assert uploads["/tests/test.sh"] == "apt-get update && echo patched\n"

    @pytest.mark.parametrize("scripts", [{"test.sh": "echo no installs here\n"}, {"notes.txt": "apt-get update\n"}])
    async def test_patch_matching_no_script_fails_naming_task_and_pattern(
        self, tmp_path: Path, scripts: dict[str, str]
    ) -> None:
        for name, content in scripts.items():
            (tmp_path / name).write_text(content)

        with pytest.raises(ValueError, match=r"terminal-bench/example: patches matched no \.sh file.*apt-get update"):
            await _server()._upload_folder(_recording_sandbox({}), tmp_path, "/tests", PATCHES, TASK)

    async def test_task_without_patches_uploads_unchanged(self, tmp_path: Path) -> None:
        (tmp_path / "test.sh").write_text("apt-get update\n")
        uploads: dict[str, str] = {}

        await _server()._upload_folder(
            _recording_sandbox(uploads), tmp_path, "/tests", PATCHES, "terminal-bench/other"
        )

        assert uploads == {"/tests/test.sh": "apt-get update\n"}


def _configured_patches() -> list[tuple[str, str, str]]:
    return [
        (subfolder, task, old)
        for subfolder, table in (("tests", TEST_SH_PATCHES), ("solution", GOLDEN_PATCH_SOLVE_SH_PATCHES))
        for task, patches in table.items()
        for old, _ in patches
    ]


@pytest.mark.skipif(not PINNED_TASKS.exists(), reason="run benchmarks.terminal_bench_2_1.prepare_inkling_small first")
@pytest.mark.parametrize(
    ("subfolder", "task", "old"), _configured_patches(), ids=lambda value: str(value).replace("\n", " ")[:40]
)
def test_configured_patches_match_the_pinned_task_scripts(subfolder: str, task: str, old: str) -> None:
    scripts = (PINNED_TASKS / task.split("/", 1)[1] / subfolder).rglob("*.sh")
    assert any(old in script.read_text() for script in scripts)
