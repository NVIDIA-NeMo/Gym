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
"""Prepare IMO AnswerBench data for NeMo Gym.

Downloads `answerbench_v2.csv` from the same pinned commit of the
google-deepmind/superhuman repo that NeMo Skills' `imo-answerbench`
benchmark reads, then writes Gym JSONL.  Same source URL => byte-identical
problem text and answers, so the Skills<->Gym comparison is apples-to-apples,
except for imo-bench-algebra-036: the pinned CSV leaves its Problem quote
unterminated (google-deepmind/superhuman#13), and this script realigns the
row that a plain CSV read shifts by one column.

Emits `question`, `expected_answer`, plus `problem_id`, `category`,
`subcategory`, `source` for domain-stratified metrics.
"""

import csv
import io
import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "imo_answerbench_benchmark.jsonl"

# Pinned commit — must match Skills' nemo_skills/dataset/imo-answerbench/prepare.py.
_SOURCE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/superhuman/"
    "326a740a6877d5ec098035c534d2fbd931fe83ee/imobench/answerbench_v2.csv"
)


def _realign_unterminated_problem(row: dict) -> dict:
    """Undo the shift caused by a Problem field whose closing quote is missing.

    The parser then ends Problem at the Short Answer's opening quote, so Problem gains
    a trailing ``\\n,<answer>"`` and every later cell moves one column left, leaving
    Source empty. Rows of any other shape are returned unchanged.
    """
    problem, separator, answer = row["Problem"].rpartition("\n,")
    if row["Source"] is not None or not separator or not answer.endswith('"'):
        return row
    return {
        "Problem ID": row["Problem ID"],
        "Problem": problem + "\n",
        "Short Answer": answer.removesuffix('"'),
        "Category": row["Short Answer"],
        "Subcategory": row["Category"],
        "Source": row["Subcategory"],
    }


def prepare() -> Path:
    """Download the CSV, convert to Gym JSONL, return the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading IMO AnswerBench from {_SOURCE_URL} ...")
    with urllib.request.urlopen(_SOURCE_URL, timeout=30) as response:
        content = response.read().decode("utf-8")

    entries = []
    for row in csv.DictReader(io.StringIO(content)):
        row = _realign_unterminated_problem(row)
        # A missing or extra cell means the columns are misaligned; fail before writing anything.
        if None in row or not all(row.values()):
            raise ValueError(f"Malformed AnswerBench CSV row {row['Problem ID']!r}: {row}")
        entries.append(
            {
                "problem_id": row["Problem ID"],
                "question": row["Problem"],
                "expected_answer": row["Short Answer"],
                "category": row["Category"],
                "subcategory": row["Subcategory"],
                "source": row["Source"],
            }
        )

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as out:
        for entry in entries:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
