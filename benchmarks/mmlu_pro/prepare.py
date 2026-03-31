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

"""Prepare MMLU-Pro evaluation data for NeMo Gym.

Downloads MMLU-Pro from HuggingFace and converts to Gym JSONL format
compatible with the mcqa resource server.

Output is raw data (no prompts baked in). Use prompt_config at rollout time
to specify the prompt, or ng_materialize_prompts to produce RL-ready data.
"""

import json
import uuid
from pathlib import Path

from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mmlu_pro_benchmark.jsonl"
# MMLU-Pro has up to 10 choices per question
OPTION_LETTERS = [chr(ord("A") + i) for i in range(10)]


def prepare() -> Path:
    """Download MMLU-Pro test data and convert to Gym JSONL format."""
    from datasets import load_dataset

    print("Downloading MMLU-Pro from HuggingFace...")
    hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", token=hf_token)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for example in ds:
        choices = example["options"]  # list of strings, length <= 10
        letters = OPTION_LETTERS[: len(choices)]

        options = [{letter: text} for letter, text in zip(letters, choices)]
        options_text = "\n".join(f"{letter}: {text}" for letter, text in zip(letters, choices))

        # MMLU-Pro has duplicate questions with different options across categories, so we use both in the UUID seed.
        seed_str = json.dumps({"question": example["question"], "options": choices}, sort_keys=True)
        row_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed_str))

        row = {
            "question": example["question"],
            "options_text": options_text,
            "options": options,
            "expected_answer": example["answer"],
            "category": example["category"],  # not used for grading although useful
            "uuid": row_uuid,
        }
        rows.append(json.dumps(row) + "\n")

    with open(OUTPUT_FPATH, "w") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
