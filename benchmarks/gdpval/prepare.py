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
"""Prepare the GDPVal benchmark JSONL.

Downloads the ``openai/gdpval`` HuggingFace dataset and converts it into the
NeMo-Gym benchmark JSONL format: each row has ``responses_create_params`` (an
empty input — the Stirrup agent builds the actual prompt from the top-level
``prompt`` / ``sector`` / ``occupation`` fields) plus task metadata at the
top level so the GDPVal resources server can pick them up via /verify.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "gdpval_benchmark.jsonl"

HF_DATASET = "openai/gdpval"
HF_SPLIT = "train"


def prepare() -> Path:
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Pass HF_TOKEN explicitly — ``load_dataset`` doesn't always pick it up
    # from the env, and GDPVal's bucket aggressively rate-limits anonymous IPs.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT, token=hf_token)

    with OUTPUT_FPATH.open("w") as f:
        for row in ds:
            record = {
                # Empty input: the Stirrup agent constructs the user prompt
                # from the top-level ``prompt`` field at runtime.
                "responses_create_params": {"input": []},
                "task_id": row["task_id"],
                "sector": row.get("sector", ""),
                "occupation": row.get("occupation", ""),
                "prompt": row["prompt"],
                "reference_files": row.get("reference_files", []),
                "reference_file_urls": row.get("reference_file_urls", []),
                "rubric_json": row.get("rubric_json", {}),
                "rubric_pretty": row.get("rubric_pretty", ""),
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(ds)} tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
