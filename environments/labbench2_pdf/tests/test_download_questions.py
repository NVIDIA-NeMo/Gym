# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import datasets
import httpx
import pytest
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from environments.labbench2_pdf import download_questions as download_questions_module
from environments.labbench2_pdf.data_utils import QUESTION_FIELDS
from environments.labbench2_pdf.download_questions import QuestionDownloadOptions, download_questions


def _row(item_id: str, benchmark: str) -> dict:
    return {
        "id": item_id,
        "tag": benchmark,
        "version": "2.0",
        "question": f"Question for {item_id}?",
        "ideal": "answer",
        "sources": ["https://doi.org/10.1000/example"],
        "unneeded_upstream_field": "not exported",
    }


def test_hf_download_uses_token_without_recording_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    loader = Mock(return_value=[_row("figqa2-id", "figqa2")])
    monkeypatch.setattr(datasets, "load_dataset", loader)
    manifest = download_questions(QuestionDownloadOptions(output_dir=tmp_path, benchmarks=("figqa2",)))

    loader.assert_called_once_with(
        "EdisonScientific/labbench2", "figqa2", split="train", revision="main", token="test-hf-token"
    )
    assert "test-hf-token" not in json.dumps(manifest)
    assert "test-hf-token" not in (tmp_path / "source_manifest.json").read_text()


@pytest.mark.parametrize("error_type", ["dataset_not_found", "gated", "unauthorized", "forbidden"])
def test_hf_access_failure_explains_auth_and_preserves_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: str
) -> None:
    original = json.dumps(_row("original", "figqa2")) + "\n"
    snapshot = tmp_path / "figqa2.jsonl"
    snapshot.write_text(original)
    if error_type == "dataset_not_found":
        error = DatasetNotFoundError("not accessible")
    elif error_type == "gated":
        error = GatedRepoError(
            "accept terms", response=httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co"))
        )
    else:
        status = 401 if error_type == "unauthorized" else 403
        error = HfHubHTTPError(
            "access denied", response=httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co"))
        )
    monkeypatch.setattr(datasets, "load_dataset", Mock(side_effect=error))

    with pytest.raises(RuntimeError) as caught:
        download_questions(QuestionDownloadOptions(output_dir=tmp_path, benchmarks=("figqa2",), overwrite=True))

    assert caught.value.__cause__ is error
    assert "https://huggingface.co/datasets/EdisonScientific/labbench2" in str(caught.value)
    assert "accept its access terms" in str(caught.value)
    assert "HF_TOKEN" in str(caught.value)
    assert "hf auth login" in str(caught.value)
    assert "questions_dir" in str(caught.value)
    assert snapshot.read_text() == original


def test_hf_server_failure_is_not_misreported_as_access_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error = HfHubHTTPError(
        "unavailable", response=httpx.Response(503, request=httpx.Request("GET", "https://huggingface.co"))
    )
    monkeypatch.setattr(datasets, "load_dataset", Mock(side_effect=error))
    with pytest.raises(HfHubHTTPError) as caught:
        download_questions_module._load_hf_rows("EdisonScientific/labbench2", "figqa2", "main")
    assert caught.value is error


def test_download_questions_writes_valid_minimal_pinned_snapshot(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "questions"

    def fake_load(repo_id: str, benchmark: str, revision: str | None) -> list[dict]:
        assert repo_id == "example/labbench2"
        assert revision == "pinned-revision"
        return [_row(f"{benchmark}-id", benchmark)]

    monkeypatch.setattr(download_questions_module, "_load_hf_rows", fake_load)
    options = QuestionDownloadOptions(
        output_dir=output_dir,
        benchmarks=("litqa3", "figqa2"),
        repo_id="example/labbench2",
        revision="pinned-revision",
    )

    manifest = download_questions(options)

    assert manifest["repo_id"] == "example/labbench2"
    assert manifest["revision"] == "pinned-revision"
    assert manifest["configs"] == {"litqa3": 1, "figqa2": 1}
    exported = json.loads((output_dir / "litqa3.jsonl").read_text(encoding="utf-8"))
    assert set(exported) == set(QUESTION_FIELDS)
    assert "unneeded_upstream_field" not in exported
    assert json.loads((output_dir / "source_manifest.json").read_text())["configs"] == manifest["configs"]


def test_matching_question_snapshot_is_reused_without_network(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "questions"
    options = QuestionDownloadOptions(output_dir=output_dir, benchmarks=("litqa3",))

    def load_main(repo_id: str, benchmark: str, revision: str | None) -> list[dict]:
        assert revision == "main"
        return [_row("litqa3-id", benchmark)]

    monkeypatch.setattr(download_questions_module, "_load_hf_rows", load_main)
    first_manifest = download_questions(options)
    assert first_manifest["revision"] == "main"

    def unexpected_load(repo_id: str, benchmark: str, revision: str | None) -> list[dict]:
        raise AssertionError("a valid cached snapshot should be reused")

    monkeypatch.setattr(download_questions_module, "_load_hf_rows", unexpected_load)

    assert download_questions(options) == first_manifest


def test_refresh_downloads_current_main_snapshot(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "questions"
    calls = []

    def load_main(repo_id: str, benchmark: str, revision: str | None) -> list[dict]:
        assert revision == "main"
        calls.append(revision)
        return [_row(f"snapshot-{len(calls)}", benchmark)]

    monkeypatch.setattr(download_questions_module, "_load_hf_rows", load_main)
    download_questions(QuestionDownloadOptions(output_dir=output_dir, benchmarks=("litqa3",)))
    manifest = download_questions(
        QuestionDownloadOptions(output_dir=output_dir, benchmarks=("litqa3",), overwrite=True)
    )

    assert calls == ["main", "main"]
    assert manifest["revision"] == "main"
    assert json.loads((output_dir / "litqa3.jsonl").read_text())["id"] == "snapshot-2"


def test_failed_refresh_preserves_existing_question_snapshot(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "questions"
    output_dir.mkdir()
    original = json.dumps(_row("existing-id", "litqa3")) + "\n"
    (output_dir / "litqa3.jsonl").write_text(original, encoding="utf-8")

    def fail_load(repo_id: str, benchmark: str, revision: str | None) -> list[dict]:
        raise OSError("offline")

    monkeypatch.setattr(download_questions_module, "_load_hf_rows", fail_load)
    options = QuestionDownloadOptions(output_dir=output_dir, benchmarks=("litqa3",), overwrite=True)

    try:
        download_questions(options)
    except OSError as exc:
        assert str(exc) == "offline"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected the simulated download to fail")

    assert (output_dir / "litqa3.jsonl").read_text(encoding="utf-8") == original
