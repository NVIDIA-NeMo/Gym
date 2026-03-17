# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prepare GPQA Diamond evaluation data for NeMo Gym.

Downloads GPQA Diamond from HuggingFace and converts to Gym JSONL format
compatible with the mcqa resource server.

Output is raw data (no prompts baked in). Use prompt_config at rollout time
to specify the prompt, or ng_materialize_prompts to produce RL-ready data.
"""

import hashlib
import json
import random
import uuid
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "gpqa_diamond_validation.jsonl"
OPTION_LETTERS = ["A", "B", "C", "D"]


def prepare() -> Path:
    """Download GPQA Diamond data and convert to Gym JSONL format."""
    from datasets import load_dataset

    print("Downloading GPQA Diamond from HuggingFace...")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for example in ds:
        # Build options list from the dataset columns
        choices = [
            example["Correct Answer"],
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]

        # Shuffle options deterministically using the question as seed
        seed = int(hashlib.md5(example["Question"].encode()).hexdigest(), 16)
        rng = random.Random(seed)
        rng.shuffle(choices)

        # Find which letter is the correct answer after shuffle
        correct_idx = choices.index(example["Correct Answer"])
        correct_letter = OPTION_LETTERS[correct_idx]

        # Format options as MCQA expects
        options = [{letter: text} for letter, text in zip(OPTION_LETTERS, choices)]
        options_text = "\n".join(f"{letter}: {text}" for letter, text in zip(OPTION_LETTERS, choices))

        row = {
            "question": example["Question"],
            "options_text": options_text,
            "options": options,
            "expected_answer": correct_letter,
            "grading_mode": "strict_single_letter_boxed",
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, example["Question"])),
        }
        rows.append(json.dumps(row) + "\n")

    with open(OUTPUT_FPATH, "w") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
