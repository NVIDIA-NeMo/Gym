# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small provenance helpers shared by Indic benchmark adapters."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LANGUAGE_NAMES = {
    "en": "English",
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}
QUALITY_STAGES = {
    "english_source",
    "first_judge_pass",
    "passed_after_correction",
    "failed_after_correction_review_needed",
    "not_judged_review_needed",
    "failed_first_judge_review_needed",
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo_id: str
    revision: str
    split: str
    text_field: str
    license: str | None
    expected_english_rows: int


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_hf_file(repo_id: str, revision: str, filename: str) -> Path:
    """Use the caller's normal HF credentials without logging or copying secrets."""
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("A full immutable 40-character HF revision is required")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision))


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_jsonl_with_manifest(path: Path, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> Path:
    if not rows:
        raise ValueError("Refusing to write an empty benchmark dataset")
    _write_atomic(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    manifest = {**metadata, "prepared_rows": len(rows), "prepared_sha256": sha256(path)}
    _write_atomic(
        path.with_suffix(".manifest.json"), json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return path
