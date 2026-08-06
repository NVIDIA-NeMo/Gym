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

"""Prepare SWE Bench Pro benchmark data for NeMo Gym."""

import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent.parent
DATA_DIR = BENCHMARK_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FPATH = DATA_DIR / "swebench_pro_benchmark.jsonl"


def prepare():
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")

    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for row in ds:
            row = row | {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": row["problem_statement"],
                        }
                    ],
                },
                "subset": "pro",
                "split": "test",
                "environment_setup_commit": "",
                "difficulty": "",
                "version": "",
                "hints_text": "",
                "created_at": "",
            }

            # Normalize to SWE Bench Verified format
            row["FAIL_TO_PASS"] = json.dumps(eval(row.pop("fail_to_pass")))
            row["PASS_TO_PASS"] = json.dumps(eval(row.pop("pass_to_pass")))

            fout.write(json.dumps(row) + "\n")

    print(f"Wrote {len(ds)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
