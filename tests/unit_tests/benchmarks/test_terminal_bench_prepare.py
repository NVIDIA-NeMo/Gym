# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.terminal_bench_2_1 import prepare as preparation


def git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True, errors="replace").strip()


@pytest.fixture
def task_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "--quiet")
    git(source, "config", "user.name", "Test")
    git(source, "config", "user.email", "test@example.com")
    task = source / "tasks/example"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('[task]\nname="example"\n[environment]\ndocker_image="example:1"\n')
    (task / "instruction.md").write_text("First instruction")
    (task / "tests").mkdir()
    (task / "tests/test.sh").write_text("echo first")
    (source / ".gitignore").write_text("ignored.txt\n")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "first")
    first = git(source, "rev-parse", "HEAD")
    (task / "instruction.md").write_text("Second instruction")
    git(source, "commit", "--quiet", "-am", "second")
    second = git(source, "rev-parse", "HEAD")
    benchmark = tmp_path / "benchmarks/terminal_bench_2_1"
    monkeypatch.setattr(preparation, "BENCHMARK_DIR", benchmark)
    monkeypatch.setattr(preparation, "OUTPUT_PATH", benchmark / "data/benchmark.jsonl")
    monkeypatch.setattr(preparation, "NUM_TASKS", 1)
    return source, first, second


def test_requested_commit_controls_prepared_tasks_and_can_be_reused(task_source: tuple[Path, str, str]) -> None:
    source, first, second = task_source
    output = preparation.prepare(task_revision=first, task_repository=str(source))
    row = json.loads(output.read_text())
    assert row["responses_create_params"]["input"][0]["content"] == "First instruction"
    checkout = preparation.BENCHMARK_DIR / "terminal-bench-2-1"
    assert git(checkout, "rev-parse", "HEAD") == first
    assert git(checkout, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    first_output = output.read_bytes()
    assert preparation.prepare(task_revision=first, task_repository=str(source)).read_bytes() == first_output
    preparation.prepare(task_revision=second, task_repository=str(source))
    assert json.loads(output.read_text())["responses_create_params"]["input"][0]["content"] == "Second instruction"


def test_unpinned_preparation_preserves_existing_checkout(task_source: tuple[Path, str, str]) -> None:
    source, first, _ = task_source
    preparation.prepare(task_revision=first, task_repository=str(source))
    output = preparation.prepare(task_repository=str(source))
    assert json.loads(output.read_text())["responses_create_params"]["input"][0]["content"] == "First instruction"


@pytest.mark.parametrize("revision", ["main", "v1.0", "a" * 12, "z" * 40])
def test_invalid_revision_does_not_clone(task_source: tuple[Path, str, str], revision: str) -> None:
    source, _, _ = task_source
    with pytest.raises(ValueError, match="full 40-character"):
        preparation.prepare(task_revision=revision, task_repository=str(source))
    assert not preparation.BENCHMARK_DIR.exists()


@pytest.mark.parametrize("relative", ["instruction.md", "tests/untracked.txt", "tests/ignored.txt"])
def test_modified_task_checkout_is_preserved(task_source: tuple[Path, str, str], relative: str) -> None:
    source, first, second = task_source
    output = preparation.prepare(task_revision=first, task_repository=str(source))
    original_output = output.read_bytes()
    checkout = preparation.BENCHMARK_DIR / "terminal-bench-2-1"
    changed = checkout / "tasks/example" / relative
    changed.write_text("Local change")
    with pytest.raises(ValueError, match="local changes"):
        preparation.prepare(task_revision=second, task_repository=str(source))
    assert changed.read_text() == "Local change"
    assert git(checkout, "rev-parse", "HEAD") == first
    assert output.read_bytes() == original_output


def test_missing_commit_is_fetched_from_origin(task_source: tuple[Path, str, str]) -> None:
    source, first, _ = task_source
    preparation.prepare(task_revision=first, task_repository=str(source))
    (source / "tasks/example/instruction.md").write_text("Later instruction")
    git(source, "commit", "--quiet", "-am", "later")
    later = git(source, "rev-parse", "HEAD")
    output = preparation.prepare(task_revision=later, task_repository=str(source))
    assert json.loads(output.read_text())["responses_create_params"]["input"][0]["content"] == "Later instruction"


def test_different_repository_is_rejected(task_source: tuple[Path, str, str]) -> None:
    source, first, _ = task_source
    preparation.prepare(task_revision=first, task_repository=str(source))
    with pytest.raises(ValueError, match="different origin"):
        preparation.prepare(task_revision=first, task_repository=str(source.parent / "other"))


def test_non_commit_object_is_rejected(task_source: tuple[Path, str, str]) -> None:
    source, first, _ = task_source
    preparation.prepare(task_revision=first, task_repository=str(source))
    tree = git(source, "rev-parse", f"{first}^{{tree}}")
    with pytest.raises(ValueError, match="not another Git object"):
        preparation.prepare(task_revision=tree, task_repository=str(source))


def test_cli_accepts_repository_and_revision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "--quiet")
    git(source, "config", "user.name", "Test")
    git(source, "config", "user.email", "test@example.com")
    for index in range(preparation.NUM_TASKS):
        task = source / "tasks" / str(index)
        task.mkdir(parents=True)
        (task / "task.toml").write_text(f'[task]\nname="task-{index}"\n[environment]\ndocker_image="example:1"\n')
        (task / "instruction.md").write_text(f"Task {index}")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "tasks")
    revision = git(source, "rev-parse", "HEAD")
    # Run the actual script from an isolated benchmark directory, with no network or model dependency.
    script = tmp_path / "benchmarks/terminal_bench_2_1/prepare.py"
    script.parent.mkdir(parents=True)
    script.write_text(Path(preparation.__file__).read_text())
    subprocess.run(
        [sys.executable, str(script), "--task-repository", str(source), "--task-revision", revision], check=True
    )
    rows = [json.loads(line) for line in (script.parent / "data/benchmark.jsonl").read_text().splitlines()]
    assert len(rows) == preparation.NUM_TASKS
    assert {row["task_name"] for row in rows} == {f"task-{index}" for index in range(preparation.NUM_TASKS)}
    assert git(script.parent / "terminal-bench-2-1", "rev-parse", "HEAD") == revision
