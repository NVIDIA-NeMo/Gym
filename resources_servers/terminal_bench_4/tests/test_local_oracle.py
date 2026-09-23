# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.terminal_bench_4 import oracle, verifier
from resources_servers.terminal_bench_4.app import empty_response
from resources_servers.terminal_bench_4.oracle import OracleHarness
from resources_servers.terminal_bench_4.task import PackageLoader, content_hash
from resources_servers.terminal_bench_4.tests.test_task import package
from resources_servers.terminal_bench_4.transfers import stage_trusted_directory
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext


async def test_local_loader_uses_configured_path_and_revalidates(tmp_path):
    source = package(tmp_path / "source")
    config = source / "task.toml"
    config.write_text(config.read_text().replace("terminal-bench/test", "private/test"))
    ref = "sha256:" + content_hash(source)
    loader = PackageLoader(local_paths={"private/test": source})
    loader._download = AsyncMock(side_effect=AssertionError("Must never download local packages"))
    task = await loader.load("private/test", ref)
    assert task.stage_tests and task.path == source
    with pytest.raises(ValueError, match="configured local task"):
        await loader.load("private/unknown", ref)
    (source / "instruction.md").write_text("Changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        await loader.load("private/test", ref)


async def test_local_loader_requires_absolute_path():
    with pytest.raises(ValueError, match="absolute"):
        await PackageLoader(local_paths={"test": Path("relative")}).load("test", "sha256:" + "a" * 64)


@pytest.mark.parametrize("target", ["/solution", "/tests"])
async def test_trusted_staging_uses_root_and_keeps_workspace_untouched(tmp_path, target):
    source = tmp_path / "assets"
    source.mkdir(mode=0o700)
    (source / "solve.sh").write_text("exit 0")
    (source / "solve.sh").chmod(0o600)
    captured = []

    async def upload(path, destination):
        with tarfile.open(path) as tar:
            captured.extend(tar.getmembers())
        assert destination.startswith("/tmp/.nemo-gym-trusted-")

    sandbox = SimpleNamespace(
        upload=AsyncMock(side_effect=upload), exec=AsyncMock(return_value=SimpleNamespace(return_code=0))
    )
    await stage_trusted_directory(sandbox, source, target)
    for call in sandbox.exec.await_args_list:
        assert call.kwargs["user"] == "root"
        assert "/home/agent" not in call.args[0]
    assert all(member.uid == member.gid == 0 for member in captured)
    script = next(member for member in captured if member.name == "./solve.sh")
    assert script.mode == (0o644 if target == "/solution" else 0o600)
    assert (source / "solve.sh").stat().st_mode & 0o777 == 0o600


async def test_trusted_staging_rejects_escape_and_arbitrary_destination(tmp_path):
    source = tmp_path / "assets"
    source.mkdir()
    sandbox = SimpleNamespace(upload=AsyncMock(), exec=AsyncMock())
    with pytest.raises(ValueError, match="fixed destination"):
        await stage_trusted_directory(sandbox, source, "/home/agent")
    (source / "escape").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="escapes"):
        await stage_trusted_directory(sandbox, source, "/solution")
    sandbox.upload.assert_not_called()


@pytest.mark.parametrize("user", [None, "agent", 1000, "root"])
@pytest.mark.parametrize("exit_code", [0, 1])
async def test_oracle_preserves_agent_identity_and_exit_status(tmp_path, monkeypatch, user, exit_code):
    source = tmp_path / "solution"
    source.mkdir()
    (source / "solve.sh").write_text("exit 0")
    stage = AsyncMock()
    monkeypatch.setattr(oracle, "stage_trusted_directory", stage)
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="uid=test\n/task\n", stderr=""),
                SimpleNamespace(return_code=exit_code, stdout="solution", stderr=""),
            ]
        )
    )
    context = HarnessContext(
        session_id="safe-id",
        task_id="test",
        rollout_id="run",
        instruction="task",
        user=user,
        workdir="/task",
        setup_timeout_sec=30,
        mcp_servers=[],
        skills_dir=None,
    )
    harness = OracleHarness(
        sandbox=sandbox,
        context=context,
        solution_dir=source,
        directory=tmp_path / "oracle",
        response=empty_response(NeMoGymResponseCreateParamsNonStreaming(input=[]), "unused"),
    )
    await harness.setup()
    response, outcome, extra = await harness.execute(30)
    stage.assert_awaited_once_with(sandbox, source, "/solution")
    for call in sandbox.exec.await_args_list:
        assert call.kwargs["user"] == user and call.kwargs["cwd"] == "/task"
    assert outcome.reason == ("completed" if exit_code == 0 else "nonzero_exit")
    assert extra["oracle_exit_code"] == exit_code and response.output == []
    assert json.loads((tmp_path / "oracle/identity.json").read_text())["requested_user"] == user


@pytest.mark.parametrize("stage_tests", [False, True])
@pytest.mark.parametrize("user", [None, "root", "grader"])
async def test_verifier_stages_only_opted_in_packages_and_keeps_configured_user(
    tmp_path, monkeypatch, stage_tests, user
):
    stage = AsyncMock()
    monkeypatch.setattr(verifier, "stage_trusted_directory", stage)
    monkeypatch.setattr(verifier, "download_dir", AsyncMock())
    monkeypatch.setattr(verifier, "parse_reward", lambda _: {"reward": 1})
    environment = SimpleNamespace(
        task=SimpleNamespace(
            stage_tests=stage_tests,
            path=tmp_path,
            config=SimpleNamespace(verifier=SimpleNamespace(user=user, env={}, timeout_sec=30)),
        ),
        main=object(),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0)),
    )
    await verifier.run_verifier(environment, tmp_path, [])
    assert stage.await_count == int(stage_tests)
    assert environment.exec.await_args_list[0].kwargs["user"] == "root"
    assert environment.exec.await_args_list[1].kwargs["user"] == user
