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
"""Prepare the APEX 2025 benchmark data.

Downloads MathArena's APEX 2025 finals problems from HuggingFace and converts
them to the Gym benchmark JSONL format with `question` and `expected_answer`
fields. Mirrors the `apex_shortlist` benchmark, but targets the 12-problem
competition finals set (`MathArena/apex_2025`) instead of the larger shortlist.

The dataset revision is pinned so scores stay reproducible.
"""

import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "apex25_benchmark.jsonl"

HF_REPO_ID = "MathArena/apex_2025"
# Pin to a specific revision for reproducibility (see README).
HF_REVISION = "ac8a641db12cc87be39e61ea89f2e04c80c5f2e7"


def prepare() -> Path:
    """Download and prepare APEX 2025 data. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading APEX 2025 data from {HF_REPO_ID} (revision {HF_REVISION[:7]})...")
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
