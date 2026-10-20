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

"""BrowseComp rows shaped for the sandboxed opencode agent + the search resources server.

Same source data and same seed-42 400-row subset as benchmarks/browsecomp, but the row shape
differs in two ways that the sandboxed agent requires:

  * exactly ONE user message and no system message. The agent asserts a single user item and
    prepends its own system prompt to the saved row, so a second system message would be
    duplicated in the trajectory.
  * no `tools` block. The browsecomp benchmark declares search/browse/bash_command function
    tools for the browsecomp_agent harness; opencode brings its own tools instead.
"""

import json
import os
from pathlib import Path

import pandas

from benchmarks.browsecomp.prepare import (
    BROWSECOMP_CSV_URL,
    BROWSECOMP_SUBSET_N,
    BROWSECOMP_SUBSET_SEED,
    _select_samples,
    decrypt,
)


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "search_browsecomp_benchmark.jsonl"


def map_browsecomp_sample_to_search_row(row: dict) -> dict:
    problem = decrypt(row["problem"], row["canary"])
    answer = decrypt(row["answer"], row["canary"])

    return {
        "question": problem,
        "ground_truth": answer,
        "responses_create_params": {
            # The BARE question, with no appended format block. The legacy harness's user-prompt
            # template was the whole of "{problem_statement}" (19 bytes) -- it sent the question
            # and nothing else.
            # simple-evals' QUERY_SUFFIX was redundant here anyway: the frankenstein system prompt
            # already ends with the same Explanation/Exact Answer/Confidence block, so appending it
            # told the model the same thing twice, 220 chars per row, in a different voice.
            "input": [{"role": "user", "content": problem}],
        },
    }


def build_rows() -> list[dict]:
    """Download, decrypt and subset BrowseComp into search-benchmark rows."""
    print(f"Downloading BrowseComp dataset from {BROWSECOMP_CSV_URL} ...")
    df = pandas.read_csv(BROWSECOMP_CSV_URL)
    assert len(df) == 1266, f"Expected 1266 samples, got {len(df)}"

    run_full = os.environ.get("BROWSECOMP_RUN_FULL", "").lower() in ("1", "true", "yes")
    df = _select_samples(df, run_full)
    mode = "FULL 1266" if run_full else f"{BROWSECOMP_SUBSET_N}-sample subset (seed {BROWSECOMP_SUBSET_SEED})"
    print(f"search/browsecomp: building {mode} ({len(df)} rows)")

    return [map_browsecomp_sample_to_search_row(row.to_dict()) for _, row in df.iterrows()]


def write_rows(rows: list[dict], output_fpath: Path) -> Path:
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_fpath, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} problems to {output_fpath}")
    return output_fpath


def prepare() -> Path:
    """Prepare the full search/browsecomp benchmark. Returns the output file path."""
    return write_rows(build_rows(), OUTPUT_FPATH)


if __name__ == "__main__":
    prepare()
