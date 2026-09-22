# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned Indic GPQA Diamond data with Gym's English GPQA pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from benchmarks.gpqa.prepare import build_row


DIRECTORY = Path(__file__).resolve().parent
SOURCE_ID = "anushakamathofficial/Indic_GPQA_Diamond"
SOURCE_REVISION = "1a56d82dd1aa89b6f270b2cccaa541be4a8b07d6"
CANONICAL_SOURCE_ID = "Idavidrein/gpqa"
CANONICAL_SOURCE_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
LANGUAGES = {
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
DEFAULT_LANGUAGES = tuple(language for language in LANGUAGES if language != "en")
EXPECTED_ROWS = 198
TEXT_FIELDS = ("Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
SOURCE_FIELDS = {*TEXT_FIELDS, "language", "language_code", "judge_pass_stage"}
QUALITY_STAGES = {
    "english_source",
    "first_judge_pass",
    "passed_after_correction",
    "failed_after_correction_review_needed",
    "not_judged_review_needed",
    "failed_first_judge_review_needed",
}


def _validate_records(records: Sequence[Mapping[str, Any]], *, language: str | None) -> None:
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"GPQA {language or 'canonical'}: expected {EXPECTED_ROWS} rows")
    for index, row in enumerate(records):
        if any(not isinstance(row.get(field), str) or not row[field].strip() for field in TEXT_FIELDS):
            raise ValueError(f"GPQA {language or 'canonical'}/{index}: all text fields must be nonempty")
        if language is None:
            continue
        if set(row) != SOURCE_FIELDS:
            raise ValueError(f"GPQA {language}/{index}: unexpected source schema")
        if row["language_code"] != language or row["language"] != LANGUAGES[language]:
            raise ValueError(f"GPQA {language}/{index}: incorrect language metadata")
        stage = row["judge_pass_stage"]
        if stage not in QUALITY_STAGES or (stage == "english_source") != (language == "en"):
            raise ValueError(f"GPQA {language}/{index}: incorrect translation status")


def build_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    language: str,
    canonical_ids: Sequence[str],
    canonical_questions: Sequence[str],
    question_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Reuse English GPQA formatting and aligned choice positions across languages."""
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported GPQA language: {language}")
    _validate_records(records, language=language)
    if len(canonical_questions) != len(records) or any(
        not isinstance(value, str) or not value.strip() for value in canonical_questions
    ):
        raise ValueError("Canonical questions must be nonempty and cover every row")
    if (
        len(canonical_ids) != len(records)
        or any(not isinstance(value, str) or not value for value in canonical_ids)
        or len(set(canonical_ids)) != len(canonical_ids)
    ):
        raise ValueError("Canonical IDs must be nonempty, unique, and cover every row")
    known_ids = {str(index) for index in range(len(records))}
    if question_ids is not None and (
        isinstance(question_ids, str)
        or not question_ids
        or any(not isinstance(value, str) for value in question_ids)
        or len(set(question_ids)) != len(question_ids)
        or set(question_ids) - known_ids
    ):
        raise ValueError("question_ids must contain unique existing string row indices")
    wanted = set(question_ids) if question_ids is not None else known_ids

    result = []
    for index, row in enumerate(records):
        if str(index) not in wanted:
            continue
        result.append(
            {
                **build_row(row, shuffle_question=canonical_questions[index]),
                "uuid": str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"indic-gpqa-diamond/{SOURCE_REVISION}/{language}/{canonical_ids[index]}",
                    )
                ),
                "metadata": {
                    "benchmark": "indic/gpqa_diamond",
                    "language": language,
                    "subset_for_metrics": language,
                    "task_id": str(index),
                    "canonical_record_id": canonical_ids[index],
                    "source_row_index": index,
                    "judge_pass_stage": row["judge_pass_stage"],
                    "duplicate_choice_text": len({row[field] for field in TEXT_FIELDS[1:]}) != 4,
                    "source_revision": SOURCE_REVISION,
                    "canonical_source_revision": CANONICAL_SOURCE_REVISION,
                },
            }
        )
    return result


def _read_parquet(*, repo_id: str, revision: str, filename: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)
    return pq.read_table(path).to_pylist()


def _read_canonical() -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id=CANONICAL_SOURCE_ID,
        filename="gpqa_diamond.csv",
        repo_type="dataset",
        revision=CANONICAL_SOURCE_REVISION,
    )
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prepare(
    output_fpath: str | Path | None = None,
    *,
    languages: Sequence[str] | None = None,
    question_ids: Sequence[str] | None = None,
) -> Path:
    """Prepare selected languages and verify exact English-to-canonical alignment."""
    if isinstance(languages, str):
        raise ValueError("languages must be a list of language codes")
    selected = list(DEFAULT_LANGUAGES if languages is None else languages)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(LANGUAGES):
        raise ValueError(f"Select unique language codes from {list(LANGUAGES)}")

    canonical = _read_canonical()
    _validate_records(canonical, language=None)
    canonical_ids = [row.get("Record ID") for row in canonical]
    if any(not isinstance(value, str) or not value for value in canonical_ids) or len(set(canonical_ids)) != len(
        canonical_ids
    ):
        raise ValueError("Canonical GPQA Record IDs must be nonempty and unique")

    english = _read_parquet(
        repo_id=SOURCE_ID,
        revision=SOURCE_REVISION,
        filename="data/en/train.parquet",
    )
    _validate_records(english, language="en")
    for index, (published, original) in enumerate(zip(english, canonical, strict=True)):
        if any(published[field] != original[field] for field in TEXT_FIELDS):
            raise ValueError(f"GPQA English row {index} differs from the pinned canonical source")

    rows = []
    for language in selected:
        records = (
            english
            if language == "en"
            else _read_parquet(
                repo_id=SOURCE_ID,
                revision=SOURCE_REVISION,
                filename=f"data/{language}/train.parquet",
            )
        )
        rows.extend(
            build_rows(
                records,
                language=language,
                canonical_ids=canonical_ids,
                canonical_questions=[row["Question"] for row in canonical],
                question_ids=question_ids,
            )
        )

    output_path = Path(output_fpath) if output_fpath else DIRECTORY / "data/gpqa_diamond_benchmark.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=list(LANGUAGES))
    parser.add_argument("--question-ids", nargs="+")
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    print(
        prepare(
            args.output_path,
            languages=args.languages,
            question_ids=args.question_ids,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
