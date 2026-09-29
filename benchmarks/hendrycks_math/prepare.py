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
"""Prepare the hendrycks MATH benchmark (test split, 5000 problems).

Mirrors nemo-skills' `save_data_from_qwen("math", "test")`:
fetches Qwen2.5-Math's GitHub-hosted copy of the MATH test split,
renames `question` -> `problem` / `answer` -> `expected_answer`, and
keeps all other fields. The Gym-side wrapper further renames
`problem` -> `question` for Gym's convention.
"""

import json
import os
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "hendrycks_math_benchmark.jsonl"

URL = "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/refs/heads/main/evaluation/data/math/test.jsonl"


def prepare() -> Path:
    """Download the MATH test split, apply Skills' renames, write Gym JSONL."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    original_file = DATA_DIR / "original_test.jsonl"
    urllib.request.urlretrieve(URL, original_file)

    count = 0
    try:
        with open(original_file, "rt", encoding="utf-8") as fin, open(OUTPUT_FPATH, "w") as fout:
            for line in fin:
                entry = json.loads(line)

                # Match nemo-skills' save_data_from_qwen renames:
                #   answer -> expected_answer, question -> problem
                if "answer" in entry:
                    entry["expected_answer"] = entry.pop("answer")
                if "problem" not in entry:
                    entry["problem"] = entry.pop("question")

                # Gym convention: problem -> question
                entry["question"] = entry.pop("problem")

                fout.write(json.dumps(entry) + "\n")
                count += 1
    finally:
        os.remove(original_file)

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
