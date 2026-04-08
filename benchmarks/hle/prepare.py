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

"""Prepare HLE evaluation data for NeMo Gym.

Downloads Humanity's Last Exam (HLE) from HuggingFace and converts to Gym JSONL
format compatible with the math_with_judge resource server.

Only text-only questions are included (image questions are filtered out).
"""

from pathlib import Path

from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "hle_benchmark.jsonl"


def prepare() -> Path:
    """Download HLE test data and convert to Gym JSONL format."""
    import json

    from datasets import Value, load_dataset

    print("Downloading HLE from HuggingFace...")
    hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    ds = load_dataset("cais/hle", split="test", token=hf_token)
    # Cast image column to string to avoid requiring Pillow.
    # Text-only questions have an empty string
    # Image questions have base64 data.
    ds = ds.cast_column("image", Value("string"))

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped_image = 0
    for entry in ds:
        if entry["image"]:
            skipped_image += 1
            continue

        row = {
            "question": entry["question"],
            "expected_answer": entry["answer"],
            "answer_type": entry["answer_type"],  # not used for grading; useful for analysis
            "uuid": entry["id"],
        }
        rows.append(json.dumps(row) + "\n")

    with open(OUTPUT_FPATH, "w") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH} (skipped {skipped_image} image questions)")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
