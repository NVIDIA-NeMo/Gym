# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned Indic GPQA Diamond data with Gym's English GPQA pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


DIRECTORY = Path(__file__).resolve().parent
SOURCE_ID = "ai4bharat/indic-gpqa"
SOURCE_REVISION = "c3c32b0a0ec7aeebe884c4c55c46730b4274a612"
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


def _validate_records(records: Sequence[Mapping[str, Any]], *, language: str) -> None:
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"GPQA {language}: expected {EXPECTED_ROWS} rows")
    suffix = f"_{LANGUAGES[language]}_translation" if language != "en" else ""
    fields = [f"{field}{suffix}" for field in TEXT_FIELDS]
    for index, row in enumerate(records):
        if any(not isinstance(row.get(field), str) or not row[field].strip() for field in fields):
            raise ValueError(f"GPQA {language}/{index}: all text fields must be nonempty")


def build_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    language: str,
    question_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Match English GPQA formatting and align choice positions across languages."""
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported GPQA language: {language}")
    _validate_records(records, language=language)
    if language != "en":
        _validate_records(records, language="en")
    record_ids = [row.get("Record ID") for row in records]
    if any(not isinstance(value, str) or not value for value in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("GPQA Record IDs must be nonempty and unique")
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

    suffix = f"_{LANGUAGES[language]}_translation" if language != "en" else ""
    result = []
    for index, row in enumerate(records):
        if str(index) not in wanted:
            continue
        example = {field: row[f"{field}{suffix}"] for field in TEXT_FIELDS}
        choices = [example[field] for field in TEXT_FIELDS[1:]]
        # The embedded English question keeps choice positions aligned across languages.
        seed = int(hashlib.md5(row["Question"].encode()).hexdigest(), 16)
        random.Random(seed).shuffle(choices)
        options = [{letter: text} for letter, text in zip("ABCD", choices, strict=True)]
        options_text = "\n".join(f"{letter}: {text}" for letter, text in zip("ABCD", choices, strict=True))
        result.append(
            {
                "question": example["Question"],
                "options_text": options_text,
                "problem": f"{example['Question']}\n{options_text}",
                "options": options,
                "expected_answer": "ABCD"[choices.index(example["Correct Answer"])],
                "uuid": str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"indic-gpqa-diamond/{SOURCE_REVISION}/{language}/{record_ids[index]}",
                    )
                ),
                "metadata": {
                    "benchmark": "indic/gpqa_diamond",
                    "language": language,
                    "subset_for_metrics": language,
                    "task_id": str(index),
                    "record_id": record_ids[index],
                    "source_row_index": index,
                    "duplicate_choice_text": len({example[field] for field in TEXT_FIELDS[1:]}) != 4,
                    "source_id": SOURCE_ID,
                    "source_revision": SOURCE_REVISION,
                },
            }
        )
    return result


def _read_parquet(*, repo_id: str, revision: str, filename: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)
    return pq.read_table(path).to_pylist()


def prepare(
    output_fpath: str | Path | None = None,
    *,
    languages: Sequence[str] | None = None,
    question_ids: Sequence[str] | None = None,
) -> Path:
    """Prepare selected languages from the pinned dataset."""
    if isinstance(languages, str):
        raise ValueError("languages must be a list of language codes")
    selected = list(DEFAULT_LANGUAGES if languages is None else languages)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(LANGUAGES):
        raise ValueError(f"Select unique language codes from {list(LANGUAGES)}")

    records = _read_parquet(
        repo_id=SOURCE_ID,
        revision=SOURCE_REVISION,
        filename="train.parquet",
    )
    rows = []
    for language in selected:
        rows.extend(
            build_rows(
                records,
                language=language,
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
