# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.terminal_bench_2_1 import prepare as preparation


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit_task(source: Path, instruction: str) -> str:
    task = source / "tasks" / "task-a"
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text('[task]\nname = "org/task-a"\n[environment]\ndocker_image = "image:1"\n')
    (task / "instruction.md").write_text(instruction)
    _git(source, "add", "-A")
    _git(source, "commit", "-q", "-m", instruction)
    return _git(source, "rev-parse", "HEAD")


@pytest.fixture
def task_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source = tmp_path / "upstream" / "terminal-bench-2-1"
    source.mkdir(parents=True)
    _git(source, "init", "-q")
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    monkeypatch.setattr(preparation, "TASK_REPOSITORY", str(source))
    monkeypatch.setattr(preparation, "BENCHMARK_DIR", benchmark_dir)
    monkeypatch.setattr(preparation, "OUTPUT_PATH", benchmark_dir / "data" / "benchmark.jsonl")
    monkeypatch.setattr(preparation, "NUM_TASKS", 1)
    return source, benchmark_dir


def _instruction(output: Path) -> str:
    (row,) = [json.loads(line) for line in output.read_text().splitlines()]
    return row["responses_create_params"]["input"][0]["content"]


def test_task_revision_selects_an_older_commit(task_repository: tuple[Path, Path]) -> None:
    source, benchmark_dir = task_repository
    old = _commit_task(source, "old instruction")
    _commit_task(source, "new instruction")

    output = preparation.prepare(task_revision=old)

    assert _instruction(output) == "old instruction"
    assert _git(benchmark_dir / "terminal-bench-2-1", "rev-parse", "HEAD") == old


def test_without_a_revision_the_default_branch_is_used(task_repository: tuple[Path, Path]) -> None:
    source, _ = task_repository
    _commit_task(source, "old instruction")
    _commit_task(source, "new instruction")

    assert _instruction(preparation.prepare()) == "new instruction"


def test_task_revision_moves_an_existing_checkout(task_repository: tuple[Path, Path]) -> None:
    source, benchmark_dir = task_repository
    old = _commit_task(source, "old instruction")
    _commit_task(source, "new instruction")
    preparation.prepare()

    output = preparation.prepare(task_revision=old)

    assert _instruction(output) == "old instruction"
    assert _git(benchmark_dir / "terminal-bench-2-1", "rev-parse", "HEAD") == old


def test_task_revision_fetches_a_commit_the_checkout_lacks(task_repository: tuple[Path, Path]) -> None:
    source, _ = task_repository
    _commit_task(source, "first instruction")
    preparation.prepare()
    later = _commit_task(source, "later instruction")

    assert _instruction(preparation.prepare(task_revision=later)) == "later instruction"


@pytest.mark.parametrize("revision", ["7131e437", "main", "G" * 40, "A" * 40])
def test_task_revision_must_be_a_full_commit_hash(task_repository: tuple[Path, Path], revision: str) -> None:
    source, benchmark_dir = task_repository
    _commit_task(source, "instruction")

    with pytest.raises(ValueError, match="full 40-character commit hash"):
        preparation.prepare(task_revision=revision)

    assert not (benchmark_dir / "terminal-bench-2-1").exists()


def test_task_revision_does_not_reset_a_modified_checkout(task_repository: tuple[Path, Path]) -> None:
    source, benchmark_dir = task_repository
    old = _commit_task(source, "old instruction")
    _commit_task(source, "new instruction")
    preparation.prepare()
    (benchmark_dir / "terminal-bench-2-1" / "tasks" / "task-a" / "instruction.md").write_text("edited")

    with pytest.raises(ValueError, match="is modified"):
        preparation.prepare(task_revision=old)
