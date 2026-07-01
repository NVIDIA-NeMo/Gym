# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import subprocess
from pathlib import Path

import pytest

from responses_api_agents.deep_swe import benchmark as benchmark_module
from responses_api_agents.deep_swe.benchmark import (
    ensure_checkout,
    materialize_task_snapshot,
    resolve_task,
    task_tree_digest,
    validate_checkout,
)


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    task = repo / "tasks" / "task-one"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('schema_version = "1.1"\n')
    (task / "instruction.md").write_text("Do the task")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_validate_checkout_and_resolve_task(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)

    assert validate_checkout(repo, commit, 1) == repo.resolve()
    assert resolve_task(repo, "task-one") == (repo / "tasks" / "task-one").resolve()
    assert len(task_tree_digest(repo)) == 64

    with pytest.raises(ValueError, match="task count mismatch"):
        validate_checkout(repo, commit, 2)
    with pytest.raises(ValueError, match="Invalid DeepSWE task_id"):
        resolve_task(repo, "../escape")
    with pytest.raises(ValueError, match="Unknown DeepSWE task_id"):
        resolve_task(repo, "missing")


def test_materialized_snapshot_uses_committed_read_only_tasks(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    snapshot_parent = tmp_path / "snapshots"
    snapshot_parent.mkdir()
    (repo / "tasks" / "task-one" / "instruction.md").write_text("mutated working tree")
    snapshot = materialize_task_snapshot(repo, snapshot_parent)
    assert (snapshot / "tasks" / "task-one" / "instruction.md").read_text() == "Do the task"
    assert not (snapshot / ".git").exists()
    assert (snapshot.stat().st_mode & 0o777) == 0o500
    assert ((snapshot / "tasks" / "task-one" / "instruction.md").stat().st_mode & 0o777) == 0o400


def test_materialized_snapshot_rejects_committed_symlinks_and_cleans_up(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    (repo / "tasks" / "task-one" / "linked-instruction").symlink_to("instruction.md")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add unsafe task symlink")
    snapshot_parent = tmp_path / "snapshots"
    snapshot_parent.mkdir()

    with pytest.raises(RuntimeError, match="contains a symlink"):
        materialize_task_snapshot(repo, snapshot_parent)

    assert list(snapshot_parent.iterdir()) == []


def test_materialized_snapshot_cleans_up_failed_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _repo(tmp_path)
    snapshot_parent = tmp_path / "snapshots"
    snapshot_parent.mkdir()

    def failed_archive(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        archive = Path(next(argument.split("=", 1)[1] for argument in command if argument.startswith("--output=")))
        archive.write_text("partial archive")
        partial = archive.parent / "partial"
        partial.mkdir()
        (partial / "file").write_text("partial")
        return subprocess.CompletedProcess(command, 1, "", "failed")

    monkeypatch.setattr(benchmark_module.subprocess, "run", failed_archive)
    with pytest.raises(RuntimeError, match="Could not materialize"):
        materialize_task_snapshot(repo, snapshot_parent)
    assert list(snapshot_parent.iterdir()) == []


def test_validate_checkout_errors(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ValueError, match="not a git repository"):
        validate_checkout(plain, "deadbeef", None)

    repo, commit = _repo(tmp_path / "nested")
    (repo / "extra.txt").write_text("next")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "next")
    with pytest.raises(ValueError, match="commit mismatch"):
        validate_checkout(repo, commit, 1)

    no_tasks_repo, _ = _repo(tmp_path / "no-tasks")
    _git(no_tasks_repo, "rm", "-r", "tasks")
    _git(no_tasks_repo, "commit", "-m", "remove tasks")
    no_tasks_commit = _git(no_tasks_repo, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="no tasks directory"):
        validate_checkout(no_tasks_repo, no_tasks_commit, None)


def test_validate_checkout_rejects_dirty_and_ignored_task_content(tmp_path: Path) -> None:
    dirty_repo, dirty_commit = _repo(tmp_path / "dirty")
    (dirty_repo / "tasks" / "task-one" / "instruction.md").write_text("changed")
    with pytest.raises(ValueError, match="modified or untracked"):
        validate_checkout(dirty_repo, dirty_commit, 1)

    ignored_repo, _ = _repo(tmp_path / "ignored")
    task = ignored_repo / "tasks" / "task-one"
    (task / ".gitignore").write_text("ignored.txt\n")
    _git(ignored_repo, "add", ".")
    _git(ignored_repo, "commit", "-m", "ignore generated task content")
    ignored_commit = _git(ignored_repo, "rev-parse", "HEAD")
    (task / "ignored.txt").write_text("unexpected build context")
    with pytest.raises(ValueError, match="modified or untracked"):
        validate_checkout(ignored_repo, ignored_commit, 1)


@pytest.mark.asyncio
async def test_ensure_checkout_accepts_pinned_path(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    checkout = await ensure_checkout(
        git_url="unused",
        commit=commit,
        cache_dir=tmp_path / "cache",
        benchmark_path=repo,
        expected_task_count=1,
    )
    assert checkout == repo.resolve()


@pytest.mark.asyncio
async def test_ensure_checkout_clones_and_reuses_cache(tmp_path: Path) -> None:
    source, commit = _repo(tmp_path / "source-root")
    cache = tmp_path / "cache"
    first, second = await asyncio.gather(
        ensure_checkout(
            git_url=str(source),
            commit=commit,
            cache_dir=cache,
            benchmark_path=None,
            expected_task_count=1,
        ),
        ensure_checkout(
            git_url=str(source),
            commit=commit,
            cache_dir=cache,
            benchmark_path=None,
            expected_task_count=1,
        ),
    )
    assert first == second == (cache / commit).resolve()
    assert task_tree_digest(first) == task_tree_digest(source)


@pytest.mark.asyncio
async def test_ensure_checkout_removes_failed_partial_clone(tmp_path: Path) -> None:
    source, _ = _repo(tmp_path / "source-root")
    cache = tmp_path / "cache"
    with pytest.raises(subprocess.CalledProcessError):
        await ensure_checkout(
            git_url=str(source),
            commit="deadbeef",
            cache_dir=cache,
            benchmark_path=None,
            expected_task_count=1,
        )
    assert not (cache / "deadbeef").exists()
