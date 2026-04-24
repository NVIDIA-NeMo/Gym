# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare miniF2F test split for the `math_formal_lean` resources server."""

import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
SOURCE_FPATH = DATA_DIR / "minif2f-kimina.jsonl"
OUTPUT_FPATH = DATA_DIR / "minif2f_benchmark.jsonl"
# Pinned commit on Goedel-LM/Goedel-Prover-V2 so schema drift is detected as a 404, not silently.
SOURCE_URL = (
    "https://raw.githubusercontent.com/Goedel-LM/Goedel-Prover-V2/"
    "2e9036e118464aa96a8bebaf9f5b9d091aa3585c/dataset/minif2f.jsonl"
)
EXPECTED_TEST_ROWS = 244


def _ensure_header_ends_with_by(text: str) -> str:
    marker = ":= by"
    idx = text.rfind(marker)
    if idx != -1:
        return text[: idx + len(marker)] + "\n"
    return text


def _clean_lean_snippet(text: str) -> str:
    cleaned = text.replace(" by sorry", " by").replace("by sorry", "by").replace("sorry", "")
    return _ensure_header_ends_with_by(cleaned)


def _split_header_and_theorem(text: str) -> tuple[str, str]:
    # Header precedes the first "/--" (doc) or "theorem ".
    header_end = -1
    doc_idx = text.find("/--")
    if doc_idx != -1:
        header_end = doc_idx
    thm_idx = text.find("theorem ")
    if header_end == -1 or (thm_idx != -1 and thm_idx < header_end):
        header_end = thm_idx
    header = text[:header_end] if header_end > 0 else ""
    theorem = text[thm_idx:] if thm_idx != -1 else text
    return header, theorem


def _process_entry(entry: dict) -> dict:
    raw_code = entry.get("formal_statement") or entry.get("lean4_code") or ""
    header, theorem = _split_header_and_theorem(raw_code)
    return {
        "name": entry.get("name", ""),
        "split": entry.get("split", entry.get("category", "")),
        "informal_prefix": entry.get("informal_prefix", ""),
        "formal_statement": _clean_lean_snippet(theorem),
        "goal": entry.get("goal", ""),
        "header": header,
    }


def prepare() -> Path:
    """Download and prepare miniF2F test split. Returns the output JSONL path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_FPATH.exists():
        print(f"Downloading miniF2F from {SOURCE_URL}...")
        urllib.request.urlretrieve(SOURCE_URL, SOURCE_FPATH)

    test_rows: list[dict] = []
    with open(SOURCE_FPATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            processed = _process_entry(json.loads(line))
            if processed["split"] == "test":
                test_rows.append(processed)

    assert len(test_rows) == EXPECTED_TEST_ROWS, (
        f"Expected {EXPECTED_TEST_ROWS} test theorems, got {len(test_rows)}; upstream may have drifted."
    )

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    SOURCE_FPATH.unlink()

    print(f"Wrote {len(test_rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
