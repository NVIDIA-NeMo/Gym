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
"""Prepare BrowseComp rows for session harnesses, which bring their own system prompt and tools.

prepare.py's rows also carry the system prompt and tool list of the BrowseComp research agent. A session
harness such as opencode_agent refuses a request that sets ``tools`` and adds a system message to its own
instructions, so these rows hold only the user turn. Everything else follows prepare.py: the same samples in
the same order (400 with seed 42, or all 1266 with BROWSECOMP_RUN_FULL=1), the same user turn byte for byte,
and the same ``question`` and ``ground_truth`` for the judge.
"""

import json
import os
from pathlib import Path

import pandas

from benchmarks.browsecomp.prepare import BROWSECOMP_CSV_URL, DATA_DIR, QUERY_SUFFIX, _select_samples, decrypt


OUTPUT_FPATH = DATA_DIR / "browsecomp_opencode_benchmark.jsonl"
BROWSECOMP_NUM_ROWS = 1266


def map_browsecomp_sample_to_user_only_sample(row: dict) -> dict:
    """Decrypt one CSV row into a request that holds only prepare.py's user turn."""
    problem = decrypt(row["problem"], row["canary"])
    return {
        "responses_create_params": {"input": [{"role": "user", "content": problem + QUERY_SUFFIX}]},
        "ground_truth": decrypt(row["answer"], row["canary"]),
        "question": problem,
    }


def prepare() -> Path:
    """Download BrowseComp and write prepare.py's samples as user-only rows. Returns the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading BrowseComp dataset from {BROWSECOMP_CSV_URL} ...")
    df = pandas.read_csv(BROWSECOMP_CSV_URL)
    # Same check as prepare.py: the subset is drawn by row index, so another source size selects other questions.
    if len(df) != BROWSECOMP_NUM_ROWS:
        raise ValueError(f"Expected {BROWSECOMP_NUM_ROWS} BrowseComp rows, got {len(df)}")

    # Parsed exactly as prepare.py parses it, so both scripts select the same samples.
    run_full = os.environ.get("BROWSECOMP_RUN_FULL", "").lower() in ("1", "true", "yes")
    df = _select_samples(df, run_full)

    with open(OUTPUT_FPATH, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(map_browsecomp_sample_to_user_only_sample(row.to_dict())) + "\n")

    print(f"Wrote {len(df)} user-only problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
