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
"""Prepare IMO ProofBench data for NeMo Gym.

Downloads ``proofbench.csv`` from the same pinned commit of
``google-deepmind/superhuman`` that NeMo Skills' ``imo-proofbench``
benchmark reads. Same source URL ⇒ byte-identical problem text,
reference solutions, rubrics and short answers.

Emits each row with: ``problem_id``, ``problem``, ``reference_solution``,
``rubric``, ``category``, ``level``, ``expected_answer``, ``source``.
The first four are consumed by the imo_proofbench_judge resource server
during ``verify()`` (the judge prompt's placeholders are
``{problem}`` / ``{reference_solution}`` / ``{rubric}`` /
``{predicted_answer}``); ``category`` and ``level`` drive the
per-domain metric breakdowns.
"""

import csv
import io
import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "imo_proofbench_benchmark.jsonl"

# Pinned commit — must match Skills' nemo_skills/dataset/imo-proofbench/prepare.py.
_SOURCE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/superhuman/"
    "c1ee02e03d4cdb2ab21cd01ac927d895f5287fc8/imobench/proofbench.csv"
)


def prepare() -> Path:
    """Download the CSV, convert to Gym JSONL, return the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading IMO ProofBench from {_SOURCE_URL} ...")
    with urllib.request.urlopen(_SOURCE_URL, timeout=30) as response:
        content = response.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))
    count = 0
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as out:
        for row in reader:
            entry = {
                "problem_id": row["Problem ID"],
                "problem": row["Problem"],
                "reference_solution": row["Solution"],
                "rubric": row["Grading guidelines"],
                "category": row["Category"],
                "level": row["Level"],
                "expected_answer": row["Short Answer"],
                "source": row["Source"],
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
