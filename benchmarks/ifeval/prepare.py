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

"""Prepare IFEval evaluation data for NeMo Gym.

Downloads the original IFEval test data from the google-research GitHub repo
and converts it to Gym JSONL format compatible with the
`instruction_following` resources server.

Mirrors `nemo_skills/dataset/ifeval/prepare.py` exactly, with two
benchmark-output renames so the resulting JSONL matches the existing
`instruction_following` server's `verify()` schema:

* `prompt` (server input) is the original `prompt` field — same value Skills
  also stores under `question`. Both are kept on the row so the data is a
  superset of Skills' output.
* `grading_mode="binary"` is added to align with Skills'
  `prompt_strict_accuracy` (1 if all instructions pass, 0 otherwise).
"""

import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
RAW_FPATH = DATA_DIR / "input_data.jsonl"
OUTPUT_FPATH = DATA_DIR / "ifeval_benchmark.jsonl"
URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/instruction_following_eval/data/input_data.jsonl"
)


def prepare() -> Path:
    """Download IFEval input data and convert to Gym JSONL format."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading IFEval input data from {URL} ...")
    urllib.request.urlretrieve(URL, RAW_FPATH)

    rows = []
    with open(RAW_FPATH, "rt", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            entry = json.loads(line)
            # Skills prepare.py also writes `question = prompt`; keep it so
            # the Gym JSONL is a strict superset of Skills'. The existing
            # `instruction_following` server reads `prompt`.
            entry["question"] = entry["prompt"]
            entry["id"] = idx
            # `binary` aligns the per-rollout reward with Skills' headline
            # `prompt_strict_accuracy` (1 if all instructions pass, else 0).
            entry["grading_mode"] = "binary"
            rows.append(entry)

    with open(OUTPUT_FPATH, "wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
