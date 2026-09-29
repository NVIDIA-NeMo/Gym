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
"""Prepare ProofNet test split for the `math_formal_lean` resources server."""

import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
SOURCE_FPATH = DATA_DIR / "proofnet.jsonl"
OUTPUT_FPATH = DATA_DIR / "proofnet_benchmark.jsonl"
# Pinned commit on deepseek-ai/DeepSeek-Prover-V1.5 so schema drift is detected as a 404, not silently.
SOURCE_URL = (
    "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/"
    "2c4ba9119eef74d0d611f494261b2c5bae98c69a/datasets/proofnet.jsonl"
)
EXPECTED_TEST_ROWS = 186


def prepare() -> Path:
    """Download ProofNet, keep the test split. Returns the output JSONL path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_FPATH.exists():
        print(f"Downloading ProofNet from {SOURCE_URL}...")
        urllib.request.urlretrieve(SOURCE_URL, SOURCE_FPATH)

    test_rows: list[dict] = []
    with open(SOURCE_FPATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("split") == "test":
                test_rows.append(entry)

    assert len(test_rows) == EXPECTED_TEST_ROWS, (
        f"Expected {EXPECTED_TEST_ROWS} test rows, got {len(test_rows)}; upstream may have drifted."
    )

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(json.dumps(row) + "\n")

    SOURCE_FPATH.unlink()

    print(f"Wrote {len(test_rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
