# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The expected-sample-ID file formats and NLTK tokenizer provisioning."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from resources_servers.voicebench.app import _ensure_nltk, _expected_ids


def _write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "ids"
    path.write_text(text)
    return str(path)


def test_json_list_of_ids(tmp_path: Path) -> None:
    assert _expected_ids(_write(tmp_path, json.dumps(["a", "b"])), "mmsu") == {"a", "b"}


def test_jsonl_rows_read_verifier_metadata_or_the_row_itself(tmp_path: Path) -> None:
    rows = [
        {"verifier_metadata": {"subset": "mmsu", "sample_id": "a"}},
        {"sample_id": "b"},
    ]
    text = "\n" + "\n".join(json.dumps(row) for row in rows) + "\n"
    assert _expected_ids(_write(tmp_path, text), "mmsu") == {"a", "b"}


def test_plain_text_lines_are_ids(tmp_path: Path) -> None:
    assert _expected_ids(_write(tmp_path, "a\n\n b \n"), "mmsu") == {"a", "b"}


def test_a_row_from_another_subset_is_rejected(tmp_path: Path) -> None:
    text = json.dumps({"verifier_metadata": {"subset": "openbookqa", "sample_id": "a"}})
    with pytest.raises(ValueError, match="different subset"):
        _expected_ids(_write(tmp_path, text), "mmsu")


def test_duplicates_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicates"):
        _expected_ids(_write(tmp_path, "a\na\n"), "mmsu")


def test_non_string_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonempty string list"):
        _expected_ids(_write(tmp_path, json.dumps([1, 2])), "mmsu")


def test_nltk_default_directory_comes_from_nltk_data_and_missing_tokenizers_are_downloaded(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NLTK_DATA", str(tmp_path / "nltk"))
    downloaded = []

    def download(name: str, download_dir: str, quiet: bool) -> bool:
        downloaded.append((name, download_dir))
        return True

    with patch("nltk.data.find", side_effect=LookupError), patch("nltk.download", side_effect=download):
        _ensure_nltk(None)

    assert downloaded == [("punkt", str(tmp_path / "nltk")), ("punkt_tab", str(tmp_path / "nltk"))]


def test_nltk_download_failure_is_reported(tmp_path: Path) -> None:
    with patch("nltk.data.find", side_effect=LookupError), patch("nltk.download", return_value=False):
        with pytest.raises(RuntimeError, match="could not provision NLTK tokenizer punkt"):
            _ensure_nltk(str(tmp_path / "nltk"))
