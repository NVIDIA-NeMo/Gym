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
"""Prepare the BrokenArXiv 04/2026 benchmark data.

Downloads MathArena's BrokenArXiv 04/2026 release from HuggingFace and converts
it to the Gym benchmark JSONL format.

Each row is a statement from a recent arXiv paper, perturbed so that it is false
as written, plus the unperturbed `original_problem` the judge compares against.
There is no `expected_answer`: the model is scored on whether it notices the
statement is false, not on a final answer.

MathArena publishes a new BrokenArXiv problem set each month and scores each
release on its own leaderboard, so every month is a separate Gym benchmark.
"""

import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "brokenarxiv_0426_benchmark.jsonl"

HF_REPO_ID = "MathArena/brokenarxiv-0426"
# Pin to a specific revision for reproducibility.
HF_REVISION = "1c64bf1ce72ebc997b0a92896a1414a4d010add0"  # pragma: allowlist secret


def prepare() -> Path:
    """Download and prepare BrokenArXiv 04/2026 data. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading BrokenArXiv 04/2026 data from {HF_REPO_ID} (revision {HF_REVISION[:7]})...")
    ds = load_dataset(HF_REPO_ID, split="train", revision=HF_REVISION)

    count = 0
    with open(OUTPUT_FPATH, "w") as f:
        for row in ds:
            out = {
                # `question` feeds the prompt template; the verifier reads it
                # back as the false statement shown to the model.
                "question": row["problem"],
                # The judge needs the true statement to decide whether a
                # "repaired" statement contradicts the paper.
                "original_problem": row["original_problem"],
            }
            f.write(json.dumps(out) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
