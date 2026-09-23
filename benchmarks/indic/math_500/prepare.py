# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Indic MATH-500 for the shared English math evaluation pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import get_token, hf_hub_download

from nemo_gym.global_config import HF_TOKEN_KEY_NAME, maybe_get_global_config_dict


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "math_500_benchmark.jsonl"
SOURCE_ID = "ai4bharat/indic-math-500"
SOURCE_REVISION = "29557d8eaa22621b82f3af5557ab60babcf3feb5"
EXPECTED_ROWS = 500
LANGUAGE_NAMES = {
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
DEFAULT_LANGUAGES = tuple(LANGUAGE_NAMES)


def build_rows(
    records: Sequence[Mapping[str, Any]], *, languages: Sequence[str] = DEFAULT_LANGUAGES
) -> list[dict[str, Any]]:
    """Select translated problems while preserving the original answers and task IDs."""
    if (
        isinstance(languages, str)
        or not languages
        or any(code not in (*DEFAULT_LANGUAGES, "en") for code in languages)
    ):
        raise ValueError(f"languages must be a nonempty sequence from {(*DEFAULT_LANGUAGES, 'en')}")
    if len(set(languages)) != len(languages):
        raise ValueError("languages must be unique")
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} MATH-500 tasks, got {len(records)}")
    identities = set()
    for record in records:
        for field in ("unique_id", "answer", "subject"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing or invalid {field}")
        if type(record.get("level")) is not int or not 1 <= record["level"] <= 5:
            raise ValueError("level must be an integer between 1 and 5")
        identity = record["unique_id"]
        if identity in identities:
            raise ValueError(f"Duplicate unique_id: {identity}")
        identities.add(identity)
    rows = []
    for language in languages:
        column = "problem" if language == "en" else f"problem_{LANGUAGE_NAMES[language]}_translation"
        for record in records:
            problem = record.get(column)
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError(f"Missing problem text: {language}/{record['unique_id']}")
            rows.append(
                {
                    "question": problem,
                    "expected_answer": record["answer"],
                    "unique_id": record["unique_id"],
                    "subject": record["subject"],
                    "level": record["level"],
                    "language": language,
                    "uuid": f"{SOURCE_ID}/{SOURCE_REVISION}/{language}/{record['unique_id']}",
                }
            )
    return rows


def prepare(*, languages: Sequence[str] = DEFAULT_LANGUAGES, output_fpath: str | None = None) -> Path:
    """Download the pinned test split and write native Gym tasks."""
    config = maybe_get_global_config_dict()
    token = config.get(HF_TOKEN_KEY_NAME) if config is not None else None
    source = hf_hub_download(
        repo_id=SOURCE_ID,
        filename="test.parquet",
        repo_type="dataset",
        revision=SOURCE_REVISION,
        token=token or get_token(),
    )
    records = load_dataset("parquet", data_files={"test": source}, split="test").to_list()
    rows = build_rows(records, languages=languages)
    output = Path(output_fpath) if output_fpath else OUTPUT_FPATH
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} problems to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    parser.add_argument("--output-fpath")
    prepare(**vars(parser.parse_args()))
