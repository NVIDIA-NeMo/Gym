# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import pytest

from . import flores200_model


@pytest.fixture
def model_bytes(monkeypatch: pytest.MonkeyPatch) -> bytes:
    data = b"verified tokenizer fixture bytes"
    monkeypatch.setattr(flores200_model, "MODEL_SIZE", len(data))
    monkeypatch.setattr(flores200_model, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    return data


def test_download_verifies_and_publishes_complete_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_bytes: bytes
) -> None:
    download = Mock(return_value=BytesIO(model_bytes))
    monkeypatch.setattr(flores200_model, "urlopen", download)

    model = flores200_model.prepare_flores200_model(tmp_path)

    assert model == tmp_path / "models" / "flores200sacrebleuspm"
    assert model.read_bytes() == model_bytes
    assert list(model.parent.iterdir()) == [model]
    download.assert_called_once_with(flores200_model.MODEL_URL, timeout=30)


def test_verified_cache_is_reused_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_bytes: bytes
) -> None:
    model = tmp_path / "models" / flores200_model.MODEL_FILENAME
    model.parent.mkdir()
    model.write_bytes(model_bytes)
    download = Mock(side_effect=AssertionError("cache reuse must not download"))
    monkeypatch.setattr(flores200_model, "urlopen", download)

    assert flores200_model.prepare_flores200_model(tmp_path) == model
    assert model.read_bytes() == model_bytes
    download.assert_not_called()


def test_corrupt_cache_is_rejected_without_replacement_or_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_bytes: bytes
) -> None:
    model = tmp_path / "models" / flores200_model.MODEL_FILENAME
    model.parent.mkdir()
    corrupt = b"x" * len(model_bytes)
    model.write_bytes(corrupt)
    download = Mock(side_effect=AssertionError("corrupt cache must fail closed"))
    monkeypatch.setattr(flores200_model, "urlopen", download)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        flores200_model.prepare_flores200_model(tmp_path)

    assert model.read_bytes() == corrupt
    download.assert_not_called()


@pytest.mark.parametrize("corruption", ["truncated", "wrong_digest", "oversized"])
def test_invalid_download_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_bytes: bytes, corruption: str
) -> None:
    data = {
        "truncated": model_bytes[:-1],
        "wrong_digest": b"x" * len(model_bytes),
        "oversized": model_bytes + b"x",
    }[corruption]
    monkeypatch.setattr(flores200_model, "urlopen", Mock(return_value=BytesIO(data)))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        flores200_model.prepare_flores200_model(tmp_path)

    assert not (tmp_path / "models").exists()


def test_network_failure_is_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flores200_model, "urlopen", Mock(side_effect=URLError("download unavailable")))

    with pytest.raises(URLError, match="download unavailable"):
        flores200_model.prepare_flores200_model(tmp_path)

    assert not (tmp_path / "models").exists()


def test_failed_atomic_publish_removes_staged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_bytes: bytes
) -> None:
    monkeypatch.setattr(flores200_model, "urlopen", Mock(return_value=BytesIO(model_bytes)))
    monkeypatch.setattr(Path, "replace", Mock(side_effect=PermissionError("cannot publish")))

    with pytest.raises(PermissionError, match="cannot publish"):
        flores200_model.prepare_flores200_model(tmp_path)

    assert list((tmp_path / "models").iterdir()) == []
