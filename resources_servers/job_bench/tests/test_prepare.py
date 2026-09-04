# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from resources_servers.job_bench import prepare as prepare_module
from resources_servers.job_bench.prepare import build_prompt, build_row, prepare, write_jsonl
from resources_servers.job_bench.task_store import discover_tasks


def _task(cache_dir: Path, task_id: str):
    return discover_tasks(cache_dir, "main")[task_id]


def test_prompt_points_the_agent_at_the_sandbox_workspace(cache_dir: Path, task_id: str) -> None:
    prompt = build_prompt(_task(cache_dir, task_id), include_search_files=False)

    assert "/workspace/task_folder" in prompt
    assert "/workspace/output" in prompt
    assert "TASK_INSTRUCTIONS.txt" in prompt
    # Upstream withholds the search corpus, so it must not be advertised.
    assert "files_required_to_search" not in prompt


def test_prompt_advertises_search_files_only_when_they_are_mounted(cache_dir: Path, task_id: str) -> None:
    prompt = build_prompt(_task(cache_dir, task_id), include_search_files=True)

    assert "/workspace/files_required_to_search" in prompt


def test_row_carries_the_task_id_but_never_the_rubrics(cache_dir: Path, task_id: str) -> None:
    row = build_row(_task(cache_dir, task_id), include_search_files=False)

    assert row["task_id"] == task_id
    assert row["responses_create_params"]["input"][0]["role"] == "user"
    metadata = row["verifier_metadata"]
    assert metadata["task_id"] == task_id
    assert metadata["profession"] == "biostatisticians"
    assert metadata["num_rubrics"] == 2
    assert metadata["max_score"] == 15
    assert len(metadata["rubrics_sha256"]) == 64
    # The answer key stays on the control plane.
    assert "rubric" not in json.dumps(row).lower().replace("num_rubrics", "").replace("rubrics_sha256", "")


def test_write_jsonl_emits_one_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.jsonl"

    write_jsonl([{"a": 1}, {"a": 2}], path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["a"] for line in lines] == [1, 2]


def test_prepare_writes_the_benchmark_and_example_files(
    monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path, task_id: str
) -> None:
    monkeypatch.setitem(prepare_module.EXPECTED_TASK_COUNTS, "main", 1)
    benchmark_dir = tmp_path / "benchmarks"
    example_path = tmp_path / "example.jsonl"
    monkeypatch.setattr(prepare_module, "PACKAGE_DIR", tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    returned_cache, jsonl_path = prepare(
        split="main", cache_dir=cache_dir, benchmark_dir=benchmark_dir, download=False
    )

    assert returned_cache == cache_dir
    assert jsonl_path == benchmark_dir / "job_bench_main_benchmark.jsonl"
    assert json.loads(jsonl_path.read_text().splitlines()[0])["task_id"] == task_id
    assert (tmp_path / "data" / "example.jsonl").is_file()
    assert not example_path.exists()


def test_prepare_can_skip_the_example_file(monkeypatch: MonkeyPatch, cache_dir: Path, tmp_path: Path) -> None:
    monkeypatch.setitem(prepare_module.EXPECTED_TASK_COUNTS, "main", 1)
    monkeypatch.setattr(prepare_module, "PACKAGE_DIR", tmp_path)

    prepare(split="main", cache_dir=cache_dir, benchmark_dir=tmp_path / "b", download=False, write_example=False)

    assert not (tmp_path / "data" / "example.jsonl").exists()


def test_prepare_rejects_an_unknown_split(cache_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown JobBench split"):
        prepare(split="hard", cache_dir=cache_dir, benchmark_dir=tmp_path, download=False)


def test_prepare_fails_loudly_on_a_short_cache(cache_dir: Path, tmp_path: Path) -> None:
    # The fixture holds one task; the real main split holds 65.
    with pytest.raises(RuntimeError, match="expected 65"):
        prepare(split="main", cache_dir=cache_dir, benchmark_dir=tmp_path, download=False)


def test_download_is_skipped_when_the_split_is_already_present(monkeypatch: MonkeyPatch, cache_dir: Path) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("snapshot_download should not run for a populated cache")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)

    prepare_module.download_dataset(cache_dir, splits=("main",), force=False)


def test_download_moves_each_upstream_split_into_its_cache_directory(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "tasks"
    staging = cache_dir.parent / "_hf_snapshot"
    captured = {}

    def fake_snapshot_download(repo_id, *, repo_type, allow_patterns, local_dir):
        captured.update(repo_id=repo_id, repo_type=repo_type, allow_patterns=allow_patterns)
        # Upstream names the main split "dataset"; setup.sh renames it to "main".
        (Path(local_dir) / "dataset" / "lawyers" / "task1").mkdir(parents=True)
        (Path(local_dir) / "dataset" / "lawyers" / "task1" / "RUBRICS.json").write_text("{}")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    prepare_module.download_dataset(cache_dir, splits=("main",), force=False)

    assert captured["repo_id"] == "JobBench/job-bench"
    assert captured["repo_type"] == "dataset"
    assert captured["allow_patterns"] == ["dataset/**"]
    assert (cache_dir / "main" / "lawyers" / "task1" / "RUBRICS.json").is_file()
    assert not staging.joinpath("dataset").exists()


def test_download_uses_the_easy_split_upstream_directory(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "tasks"
    captured = {}

    def fake_snapshot_download(repo_id, *, repo_type, allow_patterns, local_dir):
        captured["allow_patterns"] = allow_patterns
        (Path(local_dir) / "dataset_easy" / "lawyers").mkdir(parents=True)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    prepare_module.download_dataset(cache_dir, splits=("easy",), force=False)

    assert captured["allow_patterns"] == ["dataset_easy/**"]
    assert (cache_dir / "easy" / "lawyers").is_dir()


def test_download_force_replaces_an_existing_split(monkeypatch: MonkeyPatch, cache_dir: Path) -> None:
    stale_marker = cache_dir / "main" / "biostatisticians" / "task1" / "RUBRICS.json"
    assert stale_marker.is_file()

    def fake_snapshot_download(repo_id, *, repo_type, allow_patterns, local_dir):
        (Path(local_dir) / "dataset" / "lawyers").mkdir(parents=True)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    prepare_module.download_dataset(cache_dir, splits=("main",), force=True)

    assert not stale_marker.exists()
    assert (cache_dir / "main" / "lawyers").is_dir()


def test_download_fails_when_the_snapshot_lacks_the_split(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="Expected .* in the Hugging Face snapshot"):
        prepare_module.download_dataset(tmp_path / "tasks", splits=("main",), force=False)


def test_cli_forwards_its_arguments_to_prepare(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    captured = {}
    monkeypatch.setattr(prepare_module, "prepare", lambda **kwargs: captured.update(kwargs) or (tmp_path, tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare.py",
            "--split",
            "easy",
            "--cache-dir",
            str(tmp_path / "c"),
            "--benchmark-dir",
            str(tmp_path / "b"),
            "--include-search-files",
            "--no-download",
            "--force",
            "--no-example",
        ],
    )

    prepare_module.main()

    assert captured["split"] == "easy"
    assert captured["cache_dir"] == tmp_path / "c"
    assert captured["include_search_files"] is True
    assert captured["download"] is False
    assert captured["force"] is True
    assert captured["write_example"] is False
