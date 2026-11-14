# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from resources_servers.deepswe import prepare as prepare_module
from resources_servers.deepswe.prepare import _write_jsonl, ensure_ripgrep
from resources_servers.deepswe.task_store import DeepSWETaskStore


def test_write_jsonl_uses_pinned_task_image(task_assets: Path, tmp_path: Path) -> None:
    store = DeepSWETaskStore(task_assets, expected_task_count=1)
    output_path = tmp_path / "benchmark.jsonl"

    _write_jsonl(store, output_path)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["task_id"] == "example-task"
    assert row["verifier_metadata"] == {"task_id": "example-task"}
    assert row["image"] == "public.example/project/example-task:v1.1"


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
