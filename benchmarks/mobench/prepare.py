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
"""Prepare MOBench test split for the `math_formal_lean` resources server."""

import json
import re
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
SOURCE_FPATH = DATA_DIR / "MOBench.jsonl"
OUTPUT_FPATH = DATA_DIR / "mobench_benchmark.jsonl"
# Pinned commit on Goedel-LM/Goedel-Prover-V2 so schema drift is detected as a 404, not silently.
SOURCE_URL = (
    "https://raw.githubusercontent.com/Goedel-LM/Goedel-Prover-V2/"
    "2e9036e118464aa96a8bebaf9f5b9d091aa3585c/dataset/MOBench.jsonl"
)
EXPECTED_TEST_ROWS = 360
# Mirror the minif2f-kimina header so compiled proofs share the same Mathlib surface.
LEAN4_HEADER = (
    "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
)


def _strip_trailing_sorry(text: str) -> str:
    return re.sub(r"(?s)(:=\s*by)\s*sorry\s*$", r"\1", text.strip())


def _split_prelude_and_theorem(code: str) -> tuple[str, str]:
    m = re.search(r"(?m)^\s*theorem\s", code)
    if not m:
        return "", code
    return code[: m.start()].strip(), code[m.start() :].strip()


def _extract_theorem_by(theorem_block: str) -> str:
    cleaned = _strip_trailing_sorry(theorem_block)
    m = re.search(r"(?s)^(.*?:=\s*by)\b", cleaned)
    if m:
        base = m.group(1).rstrip()
    else:
        m2 = re.search(r":=\s*by", cleaned)
        base = cleaned[: m2.end()].rstrip() if m2 else cleaned.rstrip() + " := by"
    return base + "\n"


def _process_entry(entry: dict) -> dict:
    candidate = entry.get("full_formal_statement") or entry.get("formal_statement") or ""
    prelude, theorem_block = _split_prelude_and_theorem(candidate)
    theorem_by = _extract_theorem_by(theorem_block or candidate)
    formal_statement = f"{prelude}\n\n{theorem_by}" if prelude else theorem_by
    return {
        "name": entry.get("name", ""),
        "split": entry.get("split", "test"),
        "informal_prefix": entry.get("informal_prefix", ""),
        "formal_statement": formal_statement,
        "goal": "",
        "header": LEAN4_HEADER,
    }


def prepare() -> Path:
    """Download and prepare MOBench test split. Returns the output JSONL path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_FPATH.exists():
        print(f"Downloading MOBench from {SOURCE_URL}...")
        urllib.request.urlretrieve(SOURCE_URL, SOURCE_FPATH)

    rows: list[dict] = []
    with open(SOURCE_FPATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(_process_entry(json.loads(line)))

    assert len(rows) == EXPECTED_TEST_ROWS, (
        f"Expected {EXPECTED_TEST_ROWS} rows, got {len(rows)}; upstream may have drifted."
    )

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    SOURCE_FPATH.unlink()

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
