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
"""Prepare LiveCodeBench evaluation data for NeMo Gym.

Downloads the code_gen validation dataset from HuggingFace and converts it to
benchmark JSONL format. The source data was prepared by the LCB runner
(``livecodebench_accuracy_test_prep.py``) using the official livecodebench
library, so the test cases are identical to what NeMo-Skills uses.

The source JSONL has pre-baked model outputs and expected rewards (used for
grading accuracy tests). We strip those and keep only the fields needed for
fresh benchmark evaluation: ``question_content``, ``verifier_metadata``
(with ``unit_tests`` and ``problem_id``).
"""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "livecodebench_v5_validation.jsonl"

# HuggingFace dataset containing the code_gen validation data
HF_REPO_ID = "nvidia/nemotron-RL-coding-competitive_coding"
HF_FILENAME = "validation.jsonl"

# Date range for filtering to match Skills' test_v5_2408_2502 split.
# The source data covers 2024-07-01 to 2025-02-01 (322 problems).
# Set to None to use all problems.
DATE_FROM = "2024-08-01"
DATE_TO = "2025-03-01"


def prepare() -> Path:
    """Download and prepare LCB benchmark data."""
    from huggingface_hub import hf_hub_download

    print(f"Downloading validation data from {HF_REPO_ID}...")
    source_path = hf_hub_download(HF_REPO_ID, HF_FILENAME, repo_type="dataset")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # The source has multiple rows per problem (one per model output from the
    # accuracy test). We deduplicate by problem_id and keep only the fields
    # needed for benchmark evaluation.
    seen_problems = set()
    rows = []
    with open(source_path) as f:
        for line in f:
            row = json.loads(line)
            problem_id = row["verifier_metadata"]["problem_id"]
            if problem_id in seen_problems:
                continue
            seen_problems.add(problem_id)

            # Extract question_content from the prompt
            prompt_input = row.get("responses_create_params", {}).get("input", [])
            question_content = ""
            for msg in prompt_input:
                if msg.get("role") == "user":
                    question_content = msg.get("content", "")
                    break

            out = {
                "question_content": question_content,
                "verifier_metadata": row["verifier_metadata"],
            }
            rows.append(out)

    # Enrich with difficulty and filter by date range using the HF dataset
    try:
        from datasets import load_dataset

        ds = load_dataset("livecodebench/code_generation_lite", "release_v5", split="test", revision="refs/pr/7")
        hf_map = {ex.get("question_id", ""): ex for ex in ds}

        enriched = []
        for row in rows:
            pid = row["verifier_metadata"]["problem_id"]
            hf_row = hf_map.get(pid)
            if hf_row:
                # Add difficulty for per-subset metrics
                row["verifier_metadata"]["difficulty"] = hf_row.get("difficulty", "unknown")
                # Date filter
                date = hf_row.get("contest_date", "")
                if DATE_FROM and date < DATE_FROM:
                    continue
                if DATE_TO and date >= DATE_TO:
                    continue
            enriched.append(row)
        rows = enriched
    except Exception as e:
        print(f"Warning: enrichment/filtering failed ({e}), using all {len(rows)} problems")

    with open(OUTPUT_FPATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
