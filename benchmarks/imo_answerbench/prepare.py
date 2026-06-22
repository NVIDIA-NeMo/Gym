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
problem text and answers, so the Skills<->Gym comparison is apples-to-apples.

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


def prepare() -> Path:
    """Download the CSV, convert to Gym JSONL, return the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading IMO AnswerBench from {_SOURCE_URL} ...")
    with urllib.request.urlopen(_SOURCE_URL, timeout=30) as response:
        content = response.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))
    count = 0
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as out:
        for row in reader:
            entry = {
                "problem_id": row["Problem ID"],
                "question": row["Problem"],
                "expected_answer": row["Short Answer"],
                "category": row["Category"],
                "subcategory": row["Subcategory"],
                "source": row["Source"],
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
