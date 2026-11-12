# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import pytest

from . import flores200_model
from .conftest import flores200_tokenizer_cache


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


def test_supplied_cache_is_used_read_only_without_network(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch, model_bytes: bytes
) -> None:
    from sacrebleu.tokenizers import tokenizer_spm

    cache_root = tmp_path / "supplied-cache"
    model = cache_root / "models" / flores200_model.MODEL_FILENAME
    model.parent.mkdir(parents=True)
    model.write_bytes(model_bytes)
    model.chmod(0o400)
    model.parent.chmod(0o500)
    cache_root.chmod(0o500)
    before = [(path.relative_to(cache_root), path.stat()) for path in cache_root.rglob("*")]
    previous_cache = tokenizer_spm.SACREBLEU_DIR
    monkeypatch.setenv("SACREBLEU", str(cache_root))
    download = Mock(side_effect=AssertionError("supplied cache must not download"))
    monkeypatch.setattr(flores200_model, "urlopen", download)
    cache = flores200_tokenizer_cache.__wrapped__(tmp_path_factory)
    try:
        assert next(cache) == model
        assert tokenizer_spm.SACREBLEU_DIR == str(cache_root)
        assert model.read_bytes() == model_bytes
        after = [(path.relative_to(cache_root), path.stat()) for path in cache_root.rglob("*")]
        assert [(path, stat.st_size, stat.st_mtime_ns, stat.st_mode) for path, stat in after] == [
            (path, stat.st_size, stat.st_mtime_ns, stat.st_mode) for path, stat in before
        ]
        download.assert_not_called()
    finally:
        cache.close()
        cache_root.chmod(0o700)
        model.parent.chmod(0o700)
        model.chmod(0o600)
    assert tokenizer_spm.SACREBLEU_DIR == previous_cache


@pytest.mark.parametrize("corruption", ["missing", "truncated", "wrong_digest"])
def test_invalid_supplied_cache_fails_closed_without_download(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    model_bytes: bytes,
    corruption: str,
) -> None:
    cache_root = tmp_path / "supplied-cache"
    model = cache_root / "models" / flores200_model.MODEL_FILENAME
    if corruption != "missing":
        model.parent.mkdir(parents=True)
        model.write_bytes(model_bytes[:-1] if corruption == "truncated" else b"x" * len(model_bytes))
    before = model.read_bytes() if model.exists() else None
    monkeypatch.setenv("SACREBLEU", str(cache_root))
    download = Mock(side_effect=AssertionError("invalid supplied cache must fail closed"))
    monkeypatch.setattr(flores200_model, "urlopen", download)
    cache = flores200_tokenizer_cache.__wrapped__(tmp_path_factory)

    with pytest.raises(FileNotFoundError if corruption == "missing" else ValueError):
        next(cache)

    assert (model.read_bytes() if model.exists() else None) == before
    assert cache_root.exists() == (corruption != "missing")
    download.assert_not_called()


def test_unset_cache_uses_verified_local_download(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch, model_bytes: bytes
) -> None:
    from sacrebleu.tokenizers import tokenizer_spm

    monkeypatch.delenv("SACREBLEU", raising=False)
    download = Mock(return_value=BytesIO(model_bytes))
    monkeypatch.setattr(flores200_model, "urlopen", download)
    previous_cache = tokenizer_spm.SACREBLEU_DIR
    cache = flores200_tokenizer_cache.__wrapped__(tmp_path_factory)
    try:
        model = next(cache)
        assert model.read_bytes() == model_bytes
        assert model == Path(tokenizer_spm.SACREBLEU_DIR) / "models" / flores200_model.MODEL_FILENAME
        download.assert_called_once_with(flores200_model.MODEL_URL, timeout=30)
    finally:
        cache.close()
    assert tokenizer_spm.SACREBLEU_DIR == previous_cache


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
