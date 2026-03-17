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

Downloads LiveCodeBench v5 from HuggingFace and converts to Gym JSONL format
compatible with the code_gen resource server.

Output is raw data (no prompts baked in). Use prompt_config at rollout time
to specify the prompt, or ng_materialize_prompts to produce RL-ready data.
"""

import base64
import json
import pickle
import zlib
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "livecodebench_v5_validation.jsonl"

# LiveCodeBench date range for v5 — matches Skills' test_v5_2408_2502 split
DATE_FROM = "2024-08-01"
DATE_TO = "2025-03-01"


def _decode_test_cases(raw) -> list:
    """Decode test cases from the refs/pr/7 HF revision.

    The public_test_cases field is plain JSON. The private_test_cases field
    is either plain JSON or base64+zlib+pickle encoded (matching the encoding
    used by the livecodebench library).
    """
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def prepare() -> Path:
    """Download LiveCodeBench data and convert to Gym JSONL format."""
    from datasets import load_dataset

    print("Downloading LiveCodeBench from HuggingFace...")
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        "release_v5",
        split="test",
        revision="refs/pr/7",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for example in ds:
        # Filter by date range for v5
        contest_date = example.get("contest_date", "")
        if contest_date and (contest_date < DATE_FROM or contest_date >= DATE_TO):
            continue

        # Public test cases (plain JSON list of {input, output} dicts)
        pub = _decode_test_cases(example.get("public_test_cases", ""))
        inputs = [tc["input"] for tc in pub]
        outputs = [tc["output"] for tc in pub]

        # Private test cases (base64+zlib+pickle encoded in refs/pr/7 revision)
        priv = _decode_test_cases(example.get("private_test_cases", ""))
        inputs.extend(tc["input"] for tc in priv)
        outputs.extend(tc["output"] for tc in priv)

        row = {
            "question_content": example["question_content"],
            "verifier_metadata": {
                "unit_tests": {
                    "inputs": inputs,
                    "outputs": outputs,
                    "fn_name": None,
                },
                "difficulty": example.get("difficulty", "unknown"),
            },
            "problem_id": example.get("question_id", ""),
        }
        rows.append(json.dumps(row) + "\n")

    with open(OUTPUT_FPATH, "w") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
