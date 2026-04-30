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
"""Prepare LongBench-v2 data for NeMo Gym.

Mirrors `nemo_skills/dataset/longbench-v2/prepare.py`: loads the same
HuggingFace dataset (`THUDM/LongBench-v2`, single "train" split that
holds all 503 evaluation questions), preserves every Skills field
(`index`, `context`, `question`, `choice_A..D`, `expected_answer`,
`domain`, `sub_domain`, `difficulty`, `length`, `context_tokens`),
and additionally emits the `options` list and `grading_mode` that the
existing `mcqa` resource server consumes for grading.

LongBench v2 covers 6 long-context domains (8k-2M words):
single-doc QA, multi-doc QA, long in-context learning, long-dialogue
history, code-repo understanding, long structured data.

Dataset: https://huggingface.co/datasets/THUDM/LongBench-v2
Paper:   https://arxiv.org/abs/2412.15204
"""

import json
from pathlib import Path

import tiktoken
from datasets import load_dataset
from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "longbench_v2_benchmark.jsonl"

# tiktoken encoding name used by Skills' prepare.py for `context_tokens`.
TOKENIZER_NAME = "cl100k_base"


def prepare() -> Path:
    """Download LongBench-v2, convert to Gym JSONL, return the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading THUDM/LongBench-v2 (split='train', {TOKENIZER_NAME} for context tokens) ...")
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)

    count = 0
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as out:
        for entry in tqdm(dataset, desc="Writing longbench_v2_benchmark.jsonl"):
            record = {
                # Fields preserved verbatim from Skills' prepare.py
                "index": entry["_id"],
                "context": entry["context"],
                "question": entry["question"],
                "choice_A": entry["choice_A"],
                "choice_B": entry["choice_B"],
                "choice_C": entry["choice_C"],
                "choice_D": entry["choice_D"],
                "expected_answer": entry["answer"],
                "domain": entry["domain"],
                "sub_domain": entry["sub_domain"],
                "difficulty": entry["difficulty"],
                "length": entry["length"],
                # disallowed_special=() — some LongBench-v2 contexts contain
                # raw `<|endoftext|>` strings that tiktoken would otherwise
                # refuse to encode. We only need the count, so encode them
                # as plain text.
                "context_tokens": len(encoder.encode(entry["context"], disallowed_special=())),
                # Gym-side additions consumed by the `mcqa` resource server.
                # mcqa's verify() reads `options`, `expected_answer`, `grading_mode`.
                "options": [
                    {"A": entry["choice_A"]},
                    {"B": entry["choice_B"]},
                    {"C": entry["choice_C"]},
                    {"D": entry["choice_D"]},
                ],
                "grading_mode": "strict_single_letter_boxed",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
