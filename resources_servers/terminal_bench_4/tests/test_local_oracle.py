# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from resources_servers.terminal_bench_4 import oracle, verifier
from resources_servers.terminal_bench_4.task import PackageLoader, content_hash
from resources_servers.terminal_bench_4.tests.test_task import package
from resources_servers.terminal_bench_4.transfers import stage_trusted_directory


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


@pytest.mark.parametrize(
    "build_files",
    [(), ("Dockerfile",), ("docker-compose.yaml",), ("Dockerfile", "docker-compose.yaml")],
)
async def test_local_loader_infers_staging_from_verifier_not_agent_layout(tmp_path, build_files):
    source = package(tmp_path / "source")
    # An agent build spec must not suppress verifier test injection.
    (source / "environment/docker-compose.yaml").write_text("services: {}\n")
    tests = source / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("exit 0\n")
    for name in build_files:
        (tests / name).write_text("FROM verifier\n" if name == "Dockerfile" else "services: {}\n")
    loader = PackageLoader(local_paths={"terminal-bench/test": source})
    task = await loader.load("terminal-bench/test", "sha256:" + content_hash(source))
    assert task.stage_tests is (not build_files)


async def test_local_loader_revalidates_and_recomputes_staging_on_layout_change(tmp_path):
    source = package(tmp_path / "source")
    tests = source / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("exit 0\n")
    loader = PackageLoader(local_paths={"terminal-bench/test": source})
    ref = "sha256:" + content_hash(source)
    assert (await loader.load("terminal-bench/test", ref)).stage_tests
    dockerfile = tests / "Dockerfile"
    dockerfile.write_text("FROM verifier\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        await loader.load("terminal-bench/test", ref)
    ref = "sha256:" + content_hash(source)
    assert not (await loader.load("terminal-bench/test", ref)).stage_tests
    dockerfile.unlink()
    with pytest.raises(ValueError, match="hash mismatch"):
        await loader.load("terminal-bench/test", ref)
    assert (await loader.load("terminal-bench/test", "sha256:" + content_hash(source))).stage_tests


async def test_mixed_local_packages_choose_staging_independently(tmp_path):
    paths = {}
    for name in ("baked", "injected"):
        source = package(tmp_path / name)
        config = source / "task.toml"
        config.write_text(config.read_text().replace("terminal-bench/test", name))
        (source / "tests").mkdir()
        (source / "tests/test.sh").write_text("exit 0\n")
        paths[name] = source
    (paths["baked"] / "tests/Dockerfile").write_text("FROM verifier\n")
    loader = PackageLoader(local_paths=paths)
    for name, expected in (("baked", False), ("injected", True), ("baked", False)):
        task = await loader.load(name, "sha256:" + content_hash(paths[name]))
        assert task.stage_tests is expected


@pytest.mark.parametrize("baked", [False, True])
async def test_public_package_staging_behavior_is_unchanged(tmp_path, baked):
    source = package(tmp_path / "source")
    (source / "tests").mkdir()
    (source / "tests/test.sh").write_text("exit 0\n")
    if baked:
        (source / "tests/Dockerfile").write_text("FROM verifier\n")
    ref = "sha256:" + content_hash(source)
    loader = PackageLoader(tmp_path / "cache")
    cached = loader.root / "terminal-bench/test" / ref[7:]
    cached.parent.mkdir(parents=True)
    source.rename(cached)
    task = await loader.load("terminal-bench/test", ref)
    assert not task.stage_tests


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


async def test_oracle_staging_uses_the_fixed_trusted_destination(tmp_path, monkeypatch):
    source = tmp_path / "solution"
    source.mkdir()
    (source / "solve.sh").write_text("exit 0")
    stage = AsyncMock()
    monkeypatch.setattr(oracle, "stage_trusted_directory", stage)
    sandbox = SimpleNamespace(exec=AsyncMock())
    await oracle.stage_solution(sandbox, source)
    stage.assert_awaited_once_with(sandbox, source, "/solution")
    sandbox.exec.assert_not_awaited()  # Staging must not execute the solution.


async def test_oracle_staging_requires_trusted_solve_script(tmp_path, monkeypatch):
    stage = AsyncMock()
    monkeypatch.setattr(oracle, "stage_trusted_directory", stage)
    with pytest.raises(FileNotFoundError, match="solution/solve.sh"):
        await oracle.stage_solution(SimpleNamespace(), tmp_path)
    stage.assert_not_awaited()


@pytest.mark.parametrize("stage_tests", [False, True])
@pytest.mark.parametrize("user", [None, "root", "grader"])
async def test_verifier_stages_only_required_packages_and_keeps_configured_user(
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
    if stage_tests:
        stage.assert_awaited_once_with(environment.main, tmp_path / "tests", "/tests")
    else:
        stage.assert_not_awaited()
    assert environment.exec.await_args_list[0].kwargs["user"] == "root"
    assert environment.exec.await_args_list[1].kwargs["user"] == user
