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

"""Prepare SimpleQA-Verified evaluation data for NeMo Gym.

Downloads `codelion/SimpleQA-Verified` (the verified subset of OpenAI's
SimpleQA) from HuggingFace and converts to Gym JSONL format compatible with
the simpleqa resource server.

Output is raw data — no prompts baked in. Prompts are applied at rollout
time via `prompt_config=benchmarks/simpleqa/prompts/default.yaml`.

Source: https://huggingface.co/datasets/codelion/SimpleQA-Verified
"""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "simpleqa_benchmark.jsonl"


def prepare() -> Path:
    """Download SimpleQA-Verified and convert to Gym JSONL format.

    Mirrors Skills' format_entry_verified: id from `original_index`, full
    upstream row preserved as `metadata`, plus the canonical `question` and
    `expected_answer` fields the resource server reads.
    """
    from datasets import load_dataset

    print("Downloading codelion/SimpleQA-Verified from HuggingFace...")
    ds = load_dataset("codelion/SimpleQA-Verified", split="train")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, entry in enumerate(ds):
        # Mirror Skills' prepare.format_entry_verified: id falls back to a
        # stable per-row tag, metadata carries the full upstream row.
        row_id = entry.get("original_index", f"simpleqa_{idx}")
        row = {
            "id": row_id,
            "metadata": dict(entry),
            "question": entry["problem"],
            "expected_answer": entry["answer"],
        }
        rows.append(json.dumps(row, ensure_ascii=False) + "\n")

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
