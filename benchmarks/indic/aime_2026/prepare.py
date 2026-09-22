# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned Indic AIME 2026 with exact canonical problem and answer alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "aime_2026_benchmark.jsonl"
PROMPT_PATH = BENCHMARK_DIR.parents[1] / "prompts/generic/math.yaml"
DEFAULT_LANGUAGES = ("as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur")
BENCHMARK_ID = "indic/aime_2026"
SOURCE_ID = "ai4bharat/indic-aime-2026"
SOURCE_REVISION = "6cbc9d963bdd9f77e18f28de396f2f9b09bb180a"
SOURCE_SPLIT = "train"
SOURCE_LICENSE = "Apache-2.0"
EXPECTED_ENGLISH_ROWS = 30
CANONICAL_REPO = "MathArena/aime_2026"
CANONICAL_REVISION = "d2de22f3c656b4f56cf8981212186377d1e23bc3"
CANONICAL_FILE = "data/train-00000-of-00001.parquet"
CANONICAL_COLUMNS = {"problem_idx", "answer", "problem"}


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
SOURCE_COLUMNS = CANONICAL_COLUMNS | {
    f"problem_{LANGUAGE_NAMES[language]}_translation" for language in DEFAULT_LANGUAGES
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_hf_file(repo_id: str, revision: str, filename: str) -> Path:
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


def _index_rows(records: Sequence[Mapping[str, Any]], *, language: str | None) -> dict[str, Mapping[str, Any]]:
    if not records:
        raise ValueError(f"Empty AIME 2026 configuration: {language or 'canonical'}")
    indexed = {}
    for row in records:
        if not isinstance(row, Mapping) or set(row) != CANONICAL_COLUMNS:
            raise ValueError(f"Unexpected AIME 2026 source columns for {language or 'canonical'}")
        if type(row["problem_idx"]) is not int or row["problem_idx"] < 1:
            raise ValueError("AIME 2026 requires positive integer problem_idx values")
        if type(row["answer"]) is not int or not 0 <= row["answer"] <= 999:
            raise ValueError("AIME 2026 requires integer answers between 0 and 999 inclusive")
        if not isinstance(row["problem"], str) or not row["problem"].strip():
            raise ValueError("Nonempty translated problem required; English fallback is forbidden")
        identity = str(row["problem_idx"])
        if identity in indexed:
            raise ValueError(f"Duplicate AIME 2026 problem identity: {language}/{identity}")
        indexed[identity] = row
    return indexed


def _selection(
    languages: Sequence[str] | None, config_name: str | None, question_ids: Sequence[str | int] | None
) -> tuple[list[str], list[str] | None]:
    if isinstance(languages, str):
        raise ValueError("languages must be a sequence, not a string")
    if config_name is not None:
        if languages is not None and list(languages) != [config_name]:
            raise ValueError("config_name conflicts with languages")
        languages = [config_name]
    selected = list(DEFAULT_LANGUAGES if languages is None else languages)
    if not selected or any(not isinstance(code, str) for code in selected) or len(set(selected)) != len(selected):
        raise ValueError("languages must contain unique configuration codes")
    unsupported = set(selected) - {"en", *DEFAULT_LANGUAGES}
    if unsupported:
        raise ValueError(f"Unsupported AIME 2026 language configurations: {sorted(unsupported)}")
    if question_ids is None:
        return selected, None
    if isinstance(question_ids, str) or not question_ids:
        raise ValueError("question_ids must contain unique positive decimal IDs")
    normalized = []
    for value in question_ids:
        if type(value) is int and value > 0:
            identity = str(value)
        elif (
            isinstance(value, str)
            and value.isascii()
            and value.isdecimal()
            and str(int(value)) == value
            and value != "0"
        ):
            identity = value
        else:
            raise ValueError("question_ids must contain positive decimal IDs, as strings or integers, without padding")
        if identity in normalized:
            raise ValueError("question_ids must contain unique positive decimal IDs")
        normalized.append(identity)
    return selected, normalized


def load_source(
    *,
    languages: Sequence[str] | None = None,
    config_name: str | None = None,
    question_ids: Sequence[str | int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate source IDs and answers, then select translations without English fallback."""
    import pyarrow.parquet as pq

    selected, question_ids = _selection(languages, config_name, question_ids)
    canonical_path = download_hf_file(CANONICAL_REPO, CANONICAL_REVISION, CANONICAL_FILE)
    canonical = _index_rows(pq.read_table(canonical_path).to_pylist(), language=None)
    if set(canonical) != {str(index) for index in range(1, EXPECTED_ENGLISH_ROWS + 1)}:
        raise ValueError(f"Expected canonical AIME 2026 problem IDs 1 through {EXPECTED_ENGLISH_ROWS}")
    wanted = set(canonical) if question_ids is None else set(question_ids)
    if wanted - canonical.keys():
        raise ValueError(f"Unknown question IDs: {sorted(wanted - canonical.keys(), key=int)}")
    source_path = download_hf_file(SOURCE_ID, SOURCE_REVISION, "train.parquet")
    source_rows = pq.read_table(source_path).to_pylist()
    if any(set(row) != SOURCE_COLUMNS for row in source_rows):
        raise ValueError("Unexpected AIME 2026 source columns")
    english = _index_rows([{key: row[key] for key in CANONICAL_COLUMNS} for row in source_rows], language="en")
    if english.keys() != canonical.keys():
        raise ValueError("English AIME 2026 configuration must contain every canonical problem")
    for identity, row in english.items():
        if row != canonical[identity]:
            raise ValueError(f"Canonical field mismatch: en/{identity}")
    records, coverage = [], {}
    for language in selected:
        problem_column = "problem" if language == "en" else f"problem_{LANGUAGE_NAMES[language]}_translation"
        indexed = _index_rows(
            [
                {"problem_idx": row["problem_idx"], "answer": row["answer"], "problem": row[problem_column]}
                for row in source_rows
            ],
            language=language,
        )
        identities = sorted(wanted, key=int)
        for identity in identities:
            records.append(
                {
                    **indexed[identity],
                    "question_id": identity,
                    "language": language,
                    "language_name": LANGUAGE_NAMES[language],
                }
            )
        coverage[language] = {"published_rows": len(indexed), "selected_rows": len(identities)}
    return records, {
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source_split": SOURCE_SPLIT,
        "source_license": SOURCE_LICENSE,
        "source_configs": selected,
        "source_files": {"train.parquet": {"sha256": sha256(source_path), "rows": len(source_rows)}},
        "canonical_source_id": CANONICAL_REPO,
        "canonical_revision": CANONICAL_REVISION,
        "canonical_file_sha256": sha256(canonical_path),
        "english_rows": len(canonical),
        "coverage": coverage,
        "question_ids": question_ids,
    }


def build_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Use the English AIME question/answer schema and retain translation provenance."""
    return [
        {
            "question": row["problem"],
            "expected_answer": str(row["answer"]),
            "uuid": f"{BENCHMARK_ID}/{SOURCE_REVISION}/{row['language']}/{row['question_id']}",
            "question_id": row["question_id"],
            "language": row["language"],
            "language_name": row["language_name"],
        }
        for row in records
    ]


def prepare(
    *,
    languages: Sequence[str] | None = None,
    config_name: str | None = None,
    question_ids: Sequence[str | int] | None = None,
    output_fpath: str | None = None,
) -> Path:
    """Write native Gym rows and a hash-backed manifest after all alignment checks."""
    records, metadata = load_source(languages=languages, config_name=config_name, question_ids=question_ids)
    rows = build_rows(records)
    metadata.update(
        {
            "benchmark_id": BENCHMARK_ID,
            "evaluation_protocol": "gym_aime26",
            "adapter_sha256": sha256(Path(__file__)),
            "prompt_sha256": sha256(PROMPT_PATH),
        }
    )
    return write_jsonl_with_manifest(Path(output_fpath) if output_fpath else OUTPUT_FPATH, rows, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--config-name")
    parser.add_argument("--question-ids", nargs="+")
    parser.add_argument("--output-fpath")
    print(prepare(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
