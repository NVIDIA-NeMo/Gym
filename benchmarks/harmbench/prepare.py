# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare HarmBench's pinned DirectRequest text-classifier test protocol."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "harmbench_direct_request_text.jsonl"

UPSTREAM_REPOSITORY = "centerforaisafety/HarmBench"
UPSTREAM_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"  # pragma: allowlist secret
SOURCE_PATH = "data/behavior_datasets/harmbench_behaviors_text_test.csv"
SOURCE_URL = f"https://raw.githubusercontent.com/{UPSTREAM_REPOSITORY}/{UPSTREAM_REVISION}/{SOURCE_PATH}"
EXPECTED_COUNTS = {"standard": 159, "contextual": 81}


def _download_rows() -> list[dict[str, str]]:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _render(rows: list[dict[str, str]]) -> str:
    output: list[str] = []
    counts = {key: 0 for key in EXPECTED_COUNTS}
    seen: set[str] = set()
    for row in rows:
        category = row["FunctionalCategory"]
        if category == "copyright":
            continue
        if category not in EXPECTED_COUNTS:
            raise ValueError(f"unsupported HarmBench functional category {category!r}")
        behavior_id = row["BehaviorID"]
        if behavior_id in seen:
            raise ValueError(f"duplicate HarmBench behavior id {behavior_id!r}")
        seen.add(behavior_id)
        counts[category] += 1

        context = row["ContextString"]
        behavior = row["Behavior"]
        direct_request = f"{context}\n\n---\n\n{behavior}" if context else behavior
        output.append(
            json.dumps(
                {
                    "responses_create_params": {
                        "input": [{"role": "user", "content": direct_request}],
                        "temperature": 0.0,
                        "top_p": 1.0,
                        # HarmBench uses max_new_tokens=512 for every reported experiment
                        # (docs/evaluation_pipeline.md, scripts/run_pipeline.py).
                        "max_output_tokens": 512,
                    },
                    "harmbench_id": behavior_id,
                    "behavior": behavior,
                    "context": context,
                    "functional_category": category,
                    "semantic_category": row["SemanticCategory"],
                    "tags": [tag for tag in row["Tags"].split(", ") if tag],
                    "method": "DirectRequest",
                    "upstream_revision": UPSTREAM_REVISION,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"HarmBench category counts {counts}; expected {EXPECTED_COUNTS}")
    return "".join(output)


def _atomic_write(content: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=DATA_DIR, prefix=f".{OUTPUT_FPATH.name}.")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, OUTPUT_FPATH)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare() -> Path:
    _atomic_write(_render(_download_rows()))
    print(f"Wrote {sum(EXPECTED_COUNTS.values())} HarmBench DirectRequest rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
