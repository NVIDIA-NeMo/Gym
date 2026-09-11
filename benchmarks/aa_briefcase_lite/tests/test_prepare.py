# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.aa_briefcase_lite import prepare


def _write_tasks(dataset_dir: Path, tasks: list[dict]) -> None:
    (dataset_dir / "tasks.jsonl").write_text("\n".join(json.dumps(task) for task in tasks) + "\n", encoding="utf-8")


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / prepare.REVISION_MARKER).write_text(prepare.PINNED_REVISION, encoding="utf-8")
    source_dir = dataset_dir / "week" / "documents"
    source_dir.mkdir(parents=True)
    (source_dir / "source.bin").write_bytes(b"\x00\xff\x80materialized source")
    for relative in ["shared.txt", "scenario.md", "week.md", *(f"w1_t{i}.md" for i in range(1, 5))]:
        (dataset_dir / relative).write_text(f"Synthetic content for {relative}\n", encoding="utf-8")
    tasks = [
        {
            "task_id": f"w1_t{i}",
            "week": 1,
            "task_md_path": f"w1_t{i}.md",
            "deliverable_filenames": [f"deliverable_{i}.pdf"],
            "shared_files": ["shared.txt"],
            "week_files": ["week/documents/"],
            "scenario_overview_path": "scenario.md",
            "week_overview_path": "week.md",
            "checks": [{"rubric": "Synthetic verifier-only content", "judge": "fixture"}],
        }
        for i in range(1, 5)
    ]
    _write_tasks(dataset_dir, tasks)
    monkeypatch.setenv(prepare.DATASET_ENV, str(dataset_dir))
    monkeypatch.delenv(prepare.REVISION_ENV, raising=False)
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "output" / "tasks.jsonl")
    return dataset_dir, tasks


def test_prepare_four_rows_is_deterministic_and_excludes_judge_data(dataset):
    dataset_dir, tasks = dataset

    output = prepare.prepare()
    first = output.read_bytes()
    rows = [json.loads(line) for line in first.splitlines()]

    assert [row["task_id"] for row in rows] == ["w1_t1", "w1_t2", "w1_t3", "w1_t4"]
    for task, row in zip(tasks, rows, strict=True):
        assert row == {
            **{key: value for key, value in task.items() if key != "checks"},
            "responses_create_params": {"input": []},
            "dataset_dir": str(dataset_dir),
            "dataset_revision": prepare.PINNED_REVISION,
        }
    assert b"rubric" not in first
    assert b"judge" not in first
    assert b"checks" not in first
    assert prepare.prepare().read_bytes() == first


@pytest.mark.parametrize("change", ["duplicate_extra", "duplicate_replacement", "missing", "unknown"])
def test_prepare_requires_exactly_four_unique_task_ids(dataset, change):
    dataset_dir, tasks = dataset
    if change == "duplicate_extra":
        tasks.append(tasks[0])
    elif change == "duplicate_replacement":
        tasks[-1] = tasks[0]
    elif change == "missing":
        tasks.pop()
    else:
        tasks[-1]["task_id"] = "w1_t5"
    _write_tasks(dataset_dir, tasks)

    with pytest.raises(ValueError, match="four unique"):
        prepare.prepare()
    assert not prepare.OUTPUT_FPATH.exists()


def test_prepare_rejects_wrong_revision(dataset):
    dataset_dir, _ = dataset
    (dataset_dir / prepare.REVISION_MARKER).write_text("wrong-revision", encoding="utf-8")

    with pytest.raises(RuntimeError, match="revision mismatch"):
        prepare.prepare()


def test_prepare_allows_explicit_revision_override(dataset, monkeypatch):
    dataset_dir, _ = dataset
    (dataset_dir / prepare.REVISION_MARKER).write_text("fixture-revision\n", encoding="utf-8")
    monkeypatch.setenv(prepare.REVISION_ENV, "fixture-revision")

    rows = prepare.prepare().read_text(encoding="utf-8").splitlines()

    assert all(json.loads(row)["dataset_revision"] == "fixture-revision" for row in rows)


@pytest.mark.parametrize("relative", ["missing.txt", "../outside.txt"])
def test_prepare_rejects_missing_or_escaping_task_file(dataset, relative):
    dataset_dir, tasks = dataset
    (dataset_dir.parent / "outside.txt").write_text("outside", encoding="utf-8")
    tasks[-1]["task_md_path"] = relative
    _write_tasks(dataset_dir, tasks)

    with pytest.raises(FileNotFoundError, match="Invalid or missing dataset path"):
        prepare.prepare()


@pytest.mark.parametrize("target_kind", ["file", "directory", "missing"])
def test_prepare_rejects_nested_symlink_escape(dataset, target_kind):
    dataset_dir, _ = dataset
    target = dataset_dir.parent / "outside"
    if target_kind == "directory":
        target.mkdir()
        (target / "source.txt").write_text("outside", encoding="utf-8")
    elif target_kind == "file":
        target.write_text("outside", encoding="utf-8")
    (dataset_dir / "week" / "documents" / "link").symlink_to(target, target_is_directory=target_kind == "directory")

    with pytest.raises(FileNotFoundError, match="Invalid or missing dataset path"):
        prepare.prepare()


def test_prepare_accepts_nested_internal_symlink(dataset):
    dataset_dir, _ = dataset
    (dataset_dir / "week" / "documents" / "link").symlink_to(dataset_dir / "shared.txt")

    assert len(prepare.prepare().read_text(encoding="utf-8").splitlines()) == 4


@pytest.mark.parametrize("relative", ["shared.txt", "week/documents/source.bin"])
def test_prepare_rejects_unmaterialized_lfs_pointer(dataset, relative):
    dataset_dir, _ = dataset
    (dataset_dir / relative).write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 1234\n"
    )

    with pytest.raises(ValueError, match="Unmaterialized Git LFS pointer"):
        prepare.prepare()


def test_prepare_preserves_existing_output_when_last_row_is_invalid(dataset):
    dataset_dir, tasks = dataset
    output = prepare.prepare()
    previous = output.read_bytes()
    del tasks[-1]["deliverable_filenames"]
    _write_tasks(dataset_dir, tasks)

    with pytest.raises(KeyError, match="deliverable_filenames"):
        prepare.prepare()
    assert output.read_bytes() == previous


def test_prepare_rejects_inconsistent_source_pool(dataset):
    dataset_dir, tasks = dataset
    tasks[-1]["shared_files"] = ["scenario.md"]
    _write_tasks(dataset_dir, tasks)

    with pytest.raises(ValueError, match="identical shared/week source pool"):
        prepare.prepare()


def test_dataset_revision_uses_git_without_marker(dataset, monkeypatch):
    dataset_dir, _ = dataset
    (dataset_dir / prepare.REVISION_MARKER).unlink()
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=prepare.PINNED_REVISION + "\n")

    monkeypatch.setattr(prepare.subprocess, "run", run)

    assert prepare._dataset_revision(dataset_dir) == prepare.PINNED_REVISION
    assert calls == [
        (["git", "-C", str(dataset_dir), "rev-parse", "HEAD"], {"check": True, "capture_output": True, "text": True})
    ]


@pytest.mark.parametrize("error", [OSError("git unavailable"), subprocess.CalledProcessError(128, "git")])
def test_dataset_revision_requires_git_or_marker(dataset, monkeypatch, error):
    dataset_dir, _ = dataset
    (dataset_dir / prepare.REVISION_MARKER).unlink()

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(prepare.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="must be a Git checkout or carry"):
        prepare._dataset_revision(dataset_dir)


def test_prepare_requires_dataset_env(monkeypatch):
    monkeypatch.delenv(prepare.DATASET_ENV, raising=False)

    with pytest.raises(RuntimeError, match=f"Set {prepare.DATASET_ENV}"):
        prepare.prepare()


def test_prepare_requires_tasks_file(dataset):
    dataset_dir, _ = dataset
    (dataset_dir / "tasks.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="tasks file not found"):
        prepare.prepare()
