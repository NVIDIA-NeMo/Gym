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
"""Prepare the comp-math-24-25 benchmark.

Source data is a static JSONL snapshot (committed as `data/test.txt`) mirroring
nemo-skills' `nemo_skills/dataset/comp-math-24-25/test.txt`. The Skills
pipeline simply copies that file to `test.jsonl`; this script does the same
copy, then renames `problem` -> `question` for Gym's convention while
preserving every other Skills field verbatim.
"""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
SOURCE_FPATH = DATA_DIR / "test.txt"
OUTPUT_FPATH = DATA_DIR / "comp_math_24_25_benchmark.jsonl"


def prepare() -> Path:
    """Convert the shipped static JSONL snapshot into Gym's benchmark format."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(SOURCE_FPATH, "rt", encoding="utf-8") as fin, open(OUTPUT_FPATH, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            entry["question"] = entry.pop("problem")
            fout.write(json.dumps(entry) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
