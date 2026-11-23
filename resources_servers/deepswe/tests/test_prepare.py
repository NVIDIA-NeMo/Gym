# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import shutil
import tarfile
from functools import partial
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import pytest
from pytest import MonkeyPatch

from benchmarks.deepswe.prepare import is_prepared_data_current
from resources_servers.deepswe import prepare as prepare_module
from resources_servers.deepswe.prepare import _write_jsonl, ensure_ripgrep
from resources_servers.deepswe.task_store import EXPECTED_TASK_COUNT, DeepSWETaskStore


def test_write_jsonl_uses_pinned_task_image(task_assets: Path, tmp_path: Path) -> None:
    store = DeepSWETaskStore(task_assets, expected_task_count=1)
    output_path = tmp_path / "benchmark.jsonl"

    _write_jsonl(store, output_path)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["task_id"] == "example-task"
    assert row["verifier_metadata"] == {"task_id": "example-task"}
    assert row["image"] == "public.example/project/example-task:v1.1"


@pytest.mark.parametrize("allow_download", [False, True], ids=["no-download", "download-failure"])
def test_ripgrep_preparation_failure_preserves_prepared_outputs(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch, allow_download: bool
) -> None:
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)
    for index in range(EXPECTED_TASK_COUNT):
        task_dir = source / "tasks" / f"example-task-{index}"
        shutil.copytree(task_assets / "example-task", task_dir)
        config = task_dir / "task.toml"
        config.write_text(config.read_text(encoding="utf-8").replace("example-task", task_dir.name), encoding="utf-8")
    monkeypatch.setattr(prepare_module, "_git_output", Mock(return_value=prepare_module.DEEPSWE_SOURCE_REVISION))
    monkeypatch.setattr(prepare_module, "ensure_ripgrep", partial(ensure_ripgrep, tmp_path / "runtime" / "rg"))
    monkeypatch.setattr(prepare_module.urllib.request, "urlopen", Mock(side_effect=URLError("ripgrep unavailable")))

    tasks = tmp_path / "prepared-tasks"
    shutil.copytree(source / "tasks", tasks)
    (tasks / "example-task-0" / "instruction.md").write_text("Previously prepared instructions.\n", encoding="utf-8")
    marker = {
        "source_repository": prepare_module.DEEPSWE_REPOSITORY,
        "source_revision": prepare_module.DEEPSWE_SOURCE_REVISION,
        "image_source": "task.toml",
    }
    (tasks / prepare_module.CACHE_MARKER).write_text(json.dumps(marker) + "\n", encoding="utf-8")
    jsonl = tmp_path / "benchmark.jsonl"
    example = tmp_path / "example.jsonl"
    store = DeepSWETaskStore(tasks)
    _write_jsonl(store, jsonl)
    _write_jsonl(store, example, limit=5)

    def prepared_outputs() -> dict[str, bytes]:
        return {
            str(path.relative_to(tmp_path)): path.read_bytes()
            for path in [*tasks.rglob("*"), jsonl, example]
            if path.is_file()
        }

    before = prepared_outputs()
    error = URLError if allow_download else FileNotFoundError
    with pytest.raises(error, match="ripgrep"):
        prepare_module.prepare(
            source_dir=source,
            tasks_dir=tasks,
            jsonl_path=jsonl,
            example_jsonl_path=example,
            allow_download=allow_download,
        )

    assert prepared_outputs() == before


def _ripgrep_archive(executable: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(prepare_module.RIPGREP_MEMBER)
        info.size = len(executable)
        tar.addfile(info, io.BytesIO(executable))
    return buffer.getvalue()


def _serve(monkeypatch: MonkeyPatch, archive: bytes, *, executable: bytes) -> list[str]:
    requested = []

    def urlopen(url: str, timeout: float) -> io.BytesIO:
        requested.append(url)
        return io.BytesIO(archive)

    monkeypatch.setattr(prepare_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(prepare_module, "RIPGREP_ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(prepare_module, "RIPGREP_SHA256", hashlib.sha256(executable).hexdigest())
    return requested


def test_ripgrep_is_downloaded_checked_and_then_reused(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    requested = _serve(monkeypatch, _ripgrep_archive(b"rg"), executable=b"rg")
    path = ensure_ripgrep(tmp_path / "opencode-runtime" / "rg")
    assert path.read_bytes() == b"rg" and path.stat().st_mode & 0o111
    assert ensure_ripgrep(path, allow_download=False) == path
    assert requested == [prepare_module.RIPGREP_URL]


def test_a_ripgrep_archive_with_another_digest_is_refused(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _serve(monkeypatch, _ripgrep_archive(b"rg"), executable=b"rg")
    monkeypatch.setattr(prepare_module, "RIPGREP_ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="ripgrep archive"):
        ensure_ripgrep(tmp_path / "rg")
    assert not (tmp_path / "rg").exists()


def test_a_ripgrep_executable_with_another_digest_is_refused(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _serve(monkeypatch, _ripgrep_archive(b"other"), executable=b"rg")
    with pytest.raises(ValueError, match="expected"):
        ensure_ripgrep(tmp_path / "rg")
    assert not (tmp_path / "rg").exists()


def test_a_stale_ripgrep_without_download_is_reported(tmp_path: Path) -> None:
    (tmp_path / "rg").write_bytes(b"stale")
    with pytest.raises(FileNotFoundError, match="Pinned ripgrep"):
        ensure_ripgrep(tmp_path / "rg", allow_download=False)


@pytest.mark.parametrize("ripgrep", [None, b"stale", b"rg"], ids=["missing", "stale", "pinned"])
def test_a_cached_dataset_is_current_only_with_the_pinned_ripgrep(
    tmp_path: Path, monkeypatch: MonkeyPatch, ripgrep: bytes | None
) -> None:
    requested = _serve(monkeypatch, _ripgrep_archive(b"rg"), executable=b"rg")
    monkeypatch.setattr(prepare_module, "ensure_ripgrep", partial(ensure_ripgrep, tmp_path / "rg"))
    if ripgrep is not None:
        (tmp_path / "rg").write_bytes(ripgrep)
    jsonl = tmp_path / "deepswe.jsonl"
    assert not is_prepared_data_current(jsonl)
    jsonl.write_text("{}\n", encoding="utf-8")

    assert is_prepared_data_current(jsonl) is (ripgrep == b"rg")
    assert requested == []
