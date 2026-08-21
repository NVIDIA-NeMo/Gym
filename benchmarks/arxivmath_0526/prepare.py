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
"""Prepare the ArXivMath 05/2026 benchmark data.

Downloads MathArena's ArXivMath 05/2026 release from HuggingFace and converts it
to the Gym benchmark JSONL format with `question` and `expected_answer` fields.

ArXivMath is a research-level final-answer benchmark sourced from recent arXiv
papers. MathArena publishes a new problem set each month and scores each release
on its own leaderboard, so every month is a separate Gym benchmark.
"""

import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "arxivmath_0526_benchmark.jsonl"

HF_REPO_ID = "MathArena/arxivmath-0526"
# Pin to a specific revision for reproducibility.
HF_REVISION = "69f701c7f266bfba235ef302ff9b19093f7455bb"


def prepare() -> Path:
    """Download and prepare ArXivMath 05/2026 data. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading ArXivMath 05/2026 data from {HF_REPO_ID} (revision {HF_REVISION[:7]})...")
    ds = load_dataset(HF_REPO_ID, split="train", revision=HF_REVISION)

    count = 0
    with open(OUTPUT_FPATH, "w") as f:
        for row in ds:
            out = {
                "question": row["problem"],
                "expected_answer": str(row["answer"]),
            }
            f.write(json.dumps(out) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
