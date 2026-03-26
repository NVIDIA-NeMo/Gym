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
"""Prepare XSTest benchmark data."""

import json
from pathlib import Path

from pandas import read_csv


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "xstest_benchmark.jsonl"


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = read_csv("https://github.com/paul-rottger/xstest/blob/main/xstest_prompts.csv")

    with OUTPUT_FPATH.open("w") as f:
        for _, row in df.iterrows():
            sample = {
                "responses_create_params": {"input": [{"role": "user", "content": row["prompt"]}]},
                "verifier_metadata": {
                    "label": row["label"],
                    "type": row["type"],
                },
            }
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(df)} samples to {OUTPUT_FPATH}")

    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
