# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned CC BY 4.0 Political Even-handedness CSV as Gym JSONL."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
SOURCE_FPATH = DATA_DIR / "eval_set.csv"
OUTPUT_FPATH = DATA_DIR / "even_handedness_benchmark.jsonl"
UPSTREAM_REVISION = "c5ed67908b56edc0781f47821241ca44114bd4ff"
UPSTREAM_SOURCE_SHA256 = "b02e49e2390c4f03225f176fa7a132858a3fd0d33eced9bae86ecb9f11670cf3"
UPSTREAM_SOURCE_URL = (
    f"https://raw.githubusercontent.com/anthropics/political-neutrality-eval/{UPSTREAM_REVISION}/eval_set.csv"
)
REQUIRED_COLUMNS = (
    "split",
    "main_category",
    "topic_name",
    "partisan",
    "template_category",
    "template",
    "stance_a",
    "stance_b",
    "prompt_a",
    "prompt_b",
    "prompt_a_group",
    "prompt_b_group",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download_source() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_FPATH.exists():
        with urlopen(UPSTREAM_SOURCE_URL) as response:
            SOURCE_FPATH.write_bytes(response.read())
    actual = _sha256(SOURCE_FPATH)
    if actual != UPSTREAM_SOURCE_SHA256:
        raise ValueError(f"Source CSV SHA-256 is {actual}; expected {UPSTREAM_SOURCE_SHA256}")


def _convert_row(index: int, row: dict[str, str]) -> dict:
    if not row["prompt_a"].strip() or not row["prompt_b"].strip():
        raise ValueError(f"Row {index} has an empty paired prompt")
    return {
        "id": f"even_handedness_{index:04d}",
        **{column: row[column] for column in REQUIRED_COLUMNS},
        "source_revision": UPSTREAM_REVISION,
        "source_sha256": UPSTREAM_SOURCE_SHA256,
        "responses_create_params": {
            "input": [{"role": "user", "content": row["prompt_a"]}],
        },
        "prompt_config": None,
    }


def prepare() -> Path:
    """Download, validate, and convert the complete 1,350-pair public set."""
    _download_source()
    with SOURCE_FPATH.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(REQUIRED_COLUMNS).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Source CSV is missing columns: {sorted(missing)}")
        rows = [_convert_row(index, row) for index, row in enumerate(reader)]
    if len(rows) != 1350:
        raise ValueError(f"Expected 1,350 paired tasks, found {len(rows)}")
    OUTPUT_FPATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
