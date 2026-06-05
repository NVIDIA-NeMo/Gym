# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize reconstituted BunsenBench Chemistry MCQ rows into runnable Gym MCQA JSONL."""

from __future__ import annotations

import hashlib
import html
import json
import random
from pathlib import Path
from typing import Iterable


PROMPT_VERSION = "bunsen_chem_mcq_xml_choice_v1"
OPTION_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
UPSTREAM_CONFIG_METADATA_FIELDS = {
    "bunsen_bench_revision",
    "bunsen_bench_config",
    "bunsen_bench_config_version",
}
MANIFEST_METADATA_FIELDS = {
    "bunsen_id",
    "source",
    "source_dataset",
    "source_config",
    "source_split",
    "source_revision",
    "source_record_id",
    "source_row_index",
    "source_record_sha256",
    "canonical_problem_sha256",
    "bct_field",
    "bct_subfield",
}
PUBLIC_METADATA_FIELDS = UPSTREAM_CONFIG_METADATA_FIELDS | MANIFEST_METADATA_FIELDS
PAYLOAD_FIELDS = {"question", "choices", "answer", "answer_index", "source_meta"}
EXPECTED_RECONSTITUTED_FIELDS = PUBLIC_METADATA_FIELDS | PAYLOAD_FIELDS
REQUIRED_RECONSTITUTED_FIELDS = (MANIFEST_METADATA_FIELDS | PAYLOAD_FIELDS) - {"source_meta"}


def materialize_dataset(rows: Iterable[dict], output_path: Path) -> Path:
    rows = [dict(row) for row in rows]
    validate_reconstituted_rows(rows)
    materialized = [materialize_row(row) for row in rows]
    write_jsonl(materialized, output_path)
    return output_path


def validate_reconstituted_rows(rows: Iterable[dict]) -> None:
    seen_ids: set[str] = set()
    seen_locators: set[tuple[str, str]] = set()
    for row in rows:
        missing = REQUIRED_RECONSTITUTED_FIELDS - set(row)
        if missing:
            raise ValueError(f"Reconstituted row {row.get('bunsen_id', '?')} is missing fields: {sorted(missing)}")
        unexpected = set(row) - EXPECTED_RECONSTITUTED_FIELDS
        if unexpected:
            raise ValueError(
                f"Reconstituted row {row.get('bunsen_id', '?')} has unexpected fields: {sorted(unexpected)}"
            )
        if row["bunsen_id"] in seen_ids:
            raise ValueError(f"Duplicate bunsen_id: {row['bunsen_id']}")
        seen_ids.add(row["bunsen_id"])
        locator = (row["source"], row["source_record_id"])
        if locator in seen_locators:
            raise ValueError(f"Duplicate source locator: {locator}")
        seen_locators.add(locator)
        choices = row["choices"]
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError(f"Reconstituted row {row['bunsen_id']} must have at least two choices")
        answer = row["answer"]
        if answer not in choices:
            raise ValueError(f"Reconstituted row {row['bunsen_id']} answer is not present in choices")
        if row["answer_index"] != choices.index(answer):
            raise ValueError(f"Reconstituted row {row['bunsen_id']} has inconsistent answer_index")


def materialize_row(row: dict) -> dict:
    shuffled = _deterministic_shuffle(row["choices"], f"{row['bunsen_id']}:{PROMPT_VERSION}")
    expected_idx = _expected_index(shuffled, row["answer"])
    expected_letter = OPTION_LETTERS[expected_idx]
    options = [{OPTION_LETTERS[i]: choice} for i, choice in enumerate(shuffled)]
    options_text = _choices_xml(shuffled)
    metadata = {key: row[key] for key in sorted(PUBLIC_METADATA_FIELDS) if key in row}
    metadata["prompt_version"] = PROMPT_VERSION
    return {
        "question": row["question"],
        "options_text": options_text,
        "options": options,
        "expected_answer": expected_letter,
        "uuid": row["bunsen_id"],
        "metadata": metadata,
    }


def write_jsonl(rows: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    print(f"Wrote {count} rows to {output_path}")


def _deterministic_shuffle(values: list[str], seed_text: str) -> list[str]:
    values = list(values)
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], byteorder="big")
    rng = random.Random(seed)
    rng.shuffle(values)
    return values


def _expected_index(choices: list[str], expected_answer: str) -> int:
    matches = [idx for idx, choice in enumerate(choices) if choice == expected_answer]
    if len(matches) != 1:
        raise ValueError("Expected answer must appear exactly once after materialization")
    return matches[0]


def _choices_xml(choices: list[str]) -> str:
    lines = ["<choices>"]
    lines.extend(f"<choice>{html.escape(choice, quote=False)}</choice>" for choice in choices)
    lines.append("</choices>")
    return "\n".join(lines)
