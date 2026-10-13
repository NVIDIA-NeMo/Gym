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
"""Prepare FrontierScience Research data for NeMo Gym.

Downloads ``openai/frontierscience`` (``research/test.jsonl``) from
Hugging Face and writes a Gym benchmark JSONL. The upstream ``answer``
field is the per-task 10-point grading rubric, so it is stored as
``expected_answer`` for compatibility with the shared simple-agent
verification contract.

Output: ``benchmarks/frontierscience_research/data/frontierscience_research_benchmark.jsonl``

Source: https://huggingface.co/datasets/openai/frontierscience
"""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "frontierscience_research_benchmark.jsonl"

HF_REPO = "openai/frontierscience"
HF_SPLIT = "test"
HF_DATA_FILE = "research/test.jsonl"

SUBJECTS = ("chemistry", "biology", "physics")


def _format_entry(entry: dict, problem_index: int, subject: str) -> dict:
    rubric = (entry.get("answer", "") or "").strip()
    return {
        "id": f"research-{problem_index}",
        "question": entry.get("problem", ""),
        "expected_answer": rubric,
        "subject": subject,
        "task_group_id": entry.get("task_group_id", "") or "",
        "scoring_threshold": 7.0,
        "scoring_max_points": 10.0,
    }


def prepare() -> Path:
    """Download the dataset and write the combined Gym JSONL."""
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {HF_REPO} ({HF_DATA_FILE}, split={HF_SPLIT}) ...")
    research_data = load_dataset(
        HF_REPO,
        data_files={HF_SPLIT: HF_DATA_FILE},
        split=HF_SPLIT,
    )
    print(f"Loaded {len(research_data)} research problems")

    count = 0
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as out:
        for idx, entry in enumerate(research_data):
            subject = (entry.get("subject", "") or "").lower()
            if subject not in SUBJECTS:
                continue
            row = _format_entry(entry, idx, subject)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
