# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Indic GPQA Diamond data with Gym's English GPQA pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled


DIRECTORY = Path(__file__).resolve().parent
SOURCE_ID = "ai4bharat/indic-gpqa"
# Dataset commit and content hash recorded by the original Hub download, not credentials.
SOURCE_REVISION = "c3c32b0a0ec7aeebe884c4c55c46730b4274a612"  # pragma: allowlist secret
SOURCE_SHA256 = "34cd810650109f4c08b842e69dcadcff17da3a23c37a0f75d0d6a667b1e2c0cf"  # pragma: allowlist secret
SOURCE_SPLIT = "train"
PROMPT_PATH = DIRECTORY.parents[1] / "prompts/eval/aai/mcq-4choices.yaml"
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
                **({"template_metadata": {"allow_localized_answer_labels": True}} if language != "en" else {}),
                "uuid": str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"indic-gpqa-diamond/{language}/{record_ids[index]}",
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
                    "source_split": SOURCE_SPLIT,
                },
            }
        )
    return result


def _source_file(source_path: str | Path | None) -> Path:
    if source_path is None:
        try:
            source_path = hf_hub_download(
                repo_id=SOURCE_ID, repo_type="dataset", revision=SOURCE_REVISION, filename="train.parquet"
            )
        except (HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled) as error:
            raise RuntimeError(
                f"Cannot access {SOURCE_ID}@{SOURCE_REVISION}. Authenticate with `hf auth login` using an account "
                "authorized for this dataset, or pass source_path=/path/to/train.parquet from an authorized "
                "download of the pinned revision. See benchmarks/indic/gpqa_diamond/README.md; "
                "authentication alone does not grant access."
            ) from error
    path = Path(source_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"GPQA source_path must be an existing train.parquet file: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"GPQA source SHA-256 mismatch: expected {SOURCE_SHA256}, got {actual_hash}")
    return path


def prepare(
    output_fpath: str | Path | None = None,
    *,
    languages: Sequence[str] | None = None,
    question_ids: Sequence[str] | None = None,
    source_path: str | Path | None = None,
) -> Path:
    """Prepare selected languages from the pinned Hub file or its verified local copy."""
    if isinstance(languages, str):
        raise ValueError("languages must be a list of language codes")
    selected = list(DEFAULT_LANGUAGES if languages is None else languages)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(LANGUAGES):
        raise ValueError(f"Select unique language codes from {list(LANGUAGES)}")

    source_file = _source_file(source_path)
    records = list(load_dataset("parquet", data_files=str(source_file), split=SOURCE_SPLIT))
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
    provenance = {
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source_split": SOURCE_SPLIT,
        "source_sha256": SOURCE_SHA256,
        "source_rows": len(records),
        "languages": selected,
        "question_ids": list(question_ids) if question_ids is not None else None,
        "rows_per_language": {
            language: sum(row["metadata"]["language"] == language for row in rows) for language in selected
        },
        "prepared_rows": len(rows),
        "prepared_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(),
        "localized_answer_labels": "opted in for non-English rows only",
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=list(LANGUAGES))
    parser.add_argument("--question-ids", nargs="+")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--source-path", type=Path)
    args = parser.parse_args()
    print(
        prepare(
            args.output_path,
            languages=args.languages,
            question_ids=args.question_ids,
            source_path=args.source_path,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
