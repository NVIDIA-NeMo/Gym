# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the complete public FACTS Search artifact published by Google.

The current FACTS Search V2 leaderboard describes 1,842 rows split evenly into
921 public and 921 private rows.  Its linked public Kaggle dataset, however,
still publishes only the original 890-row public CSV.  This materializer pins
and emits every row that Google actually makes downloadable; it refuses to
pretend that the 31 advertised-but-unpublished V2 public rows are present.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "facts_search_public.jsonl"

KAGGLE_DATASET = "deepmind/facts-search-public"
KAGGLE_DATASET_VERSION = 1
KAGGLE_DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}?datasetVersionNumber={KAGGLE_DATASET_VERSION}"
)
KAGGLE_DATASET_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
CSV_MEMBER = "facts_open_filtered.csv"
CSV_SHA256 = "765d452820870649edf8a918c02ef0feb28296298802a9ded8e270948c47ae04"  # pragma: allowlist secret
EXPECTED_PUBLISHED_ROWS = 890
ADVERTISED_V2_PUBLIC_ROWS = 921
COLUMNS = ("example_id", "problem", "gold answer")
LICENSE = "Apache 2.0"

SYSTEM_PROMPT = """You are a search assistant designed to find accurate information through multi-step searches.
                Guidelines:
                1. You can perform multiple search queries across multiple hops/turns to gather comprehensive information
                2. Each hop allows multiple parallel searches, with the hop ending once you receive all search results
                3. Your final answer should be:
                   - Concise, short and accurate
                   - Based only on facts from the search results
                   - Answer with only one short sentence
                4. Do not provide partial or intermediate answers during the search process
                5. If information is incomplete, continue searching in the next hop"""

BRAVE_SEARCH_TOOL = {
    "type": "function",
    "name": "brave_search",
    "description": "Search for information on the web",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query"}},
        "required": ["query"],
    },
    "strict": False,
}


def _download_csv_bytes(timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(KAGGLE_DOWNLOAD_URL, headers={"User-Agent": "nemo-gym-facts-search/2.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        if CSV_MEMBER not in zf.namelist():
            raise ValueError(f"{KAGGLE_DATASET} v{KAGGLE_DATASET_VERSION} does not contain {CSV_MEMBER}")
        return zf.read(CSV_MEMBER)


def _verify_sha256(content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    if digest != CSV_SHA256:
        raise ValueError(f"{CSV_MEMBER} SHA-256 mismatch: expected {CSV_SHA256}, got {digest}")
    return digest


def _render(content: bytes) -> tuple[str, int]:
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    actual_columns = tuple(rows[0].keys()) if rows else ()
    if actual_columns != COLUMNS:
        raise ValueError(f"unexpected FACTS Search columns: {actual_columns}")
    if len(rows) != EXPECTED_PUBLISHED_ROWS:
        raise ValueError(f"expected {EXPECTED_PUBLISHED_ROWS} published rows, found {len(rows)}")

    output: list[str] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=1):
        example_id = row["example_id"].strip()
        question = row["problem"].strip()
        answer = row["gold answer"].strip()
        if not example_id or not question or not answer:
            raise ValueError(f"row {index} has an empty id, problem, or gold answer")
        if example_id in seen_ids:
            raise ValueError(f"duplicate example_id at row {index}: {example_id}")
        if (question, answer) in seen_pairs:
            raise ValueError(f"duplicate question/answer pair at row {index}: {example_id}")
        seen_ids.add(example_id)
        seen_pairs.add((question, answer))
        row_sha256 = hashlib.sha256("\t".join(row[column] for column in COLUMNS).encode("utf-8")).hexdigest()
        output.append(
            json.dumps(
                {
                    "id": example_id,
                    "problem": question,
                    "gold_answer": answer,
                    "row_sha256": row_sha256,
                    "responses_create_params": {
                        "input": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": question},
                        ],
                        "tools": [BRAVE_SEARCH_TOOL],
                        "tool_choice": "auto",
                        "parallel_tool_calls": True,
                    },
                    "upstream": {
                        "dataset": KAGGLE_DATASET,
                        "version": KAGGLE_DATASET_VERSION,
                        "file": CSV_MEMBER,
                        "csv_sha256": CSV_SHA256,
                        "license": LICENSE,
                        "published_rows": EXPECTED_PUBLISHED_ROWS,
                        "advertised_v2_public_rows": ADVERTISED_V2_PUBLIC_ROWS,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    return "".join(output), len(output)


def _atomic_write(content: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare(source_csv: Optional[str] = None, output_fpath: Optional[str] = None) -> Path:
    content = Path(source_csv).read_bytes() if source_csv else _download_csv_bytes()
    _verify_sha256(content)
    rendered, count = _render(content)
    target = Path(output_fpath) if output_fpath else OUTPUT_FPATH
    _atomic_write(rendered, target)
    print(
        f"Wrote all {count} publicly downloadable FACTS Search rows to {target}; "
        f"the current V2 page advertises {ADVERTISED_V2_PUBLIC_ROWS}, leaving "
        f"{ADVERTISED_V2_PUBLIC_ROWS - count} unavailable."
    )
    return target


if __name__ == "__main__":
    prepare()
