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
"""Prepare AweAI-Team/Scale-SWE training data for NeMo Gym, from the Hub.

This is a training dataset, not an eval benchmark -- there is no fixed/frozen reference split,
and the output is meant to feed a training run, not to be scored against as a leaderboard.

Writes only the rows already known to be usable: `data/supported_instance_ids.txt` (17,696 of
the Hub's 20,181 instances) is the result of a full-set 3x golden-patch sweep -- a row not on
that list either never resolves, never produced a verdict, or resolves inconsistently across
passes (see `aggregate_golden_patch.py` and the README), so it is not worth carrying into the
training jsonl at all rather than shipping a row no agent run could usefully score.

Streams the Hub dataset and filters row-by-row rather than loading the full ~940 MB split just
to keep 88% of it -- but the *kept* rows are still buffered and shuffled (fixed seed, so re-runs
are reproducible) before writing, since the Hub's own row order is grouped by repo, not
randomised, and training on that order as-is would bias early steps toward whichever repos
happen to sort first.

    SCALE_SWE_LIMIT=200 python resources_servers/scale_swe/prepare_scale_swe.py
"""

import json
import os
import random
from pathlib import Path


SHUFFLE_SEED = 0

DATASET_NAME = "AweAI-Team/Scale-SWE"
SUPPORTED_IDS_FPATH = Path(__file__).parent / "data" / "supported_instance_ids.txt"
OUTPUT_FPATH = Path(__file__).parent / "data" / "scale_swe_training.jsonl"

AGENT_REF = {"type": "responses_api_agents", "name": "scale_swe_opencode_sandboxed_agent"}

# The Hub row's own fields this server's request model reads (see app.py's
# ScaleSWEInstanceRequest) -- extras like `github_url`/`parent_commit`/`pr_commit`/`user` are
# dropped rather than carried through unused. FAIL_TO_PASS/PASS_TO_PASS arrive as JSON-encoded
# strings, not lists, in this dataset -- kept exactly as-received, matching the request model's
# `str | list[str]` type, rather than decoded and possibly round-tripped differently.
ROW_FIELDS = (
    "instance_id",
    "repo",
    "language",
    "workdir",
    "image_url",
    "patch",
    "pre_commands",
    "problem_statement",
    "f2p_patch",
    "f2p_script",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


def _load_supported_ids(fpath: Path) -> set[str]:
    return {line.strip() for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()}


def prepare(supported_ids: set[str], limit: int = 0) -> Path:
    from datasets import load_dataset

    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    rows = []
    for example in dataset:
        if limit and len(rows) >= limit:
            break
        if example["instance_id"] not in supported_ids:
            continue

        row = {field: example[field] for field in ROW_FIELDS}
        row["responses_create_params"] = {"input": [{"role": "user", "content": row["problem_statement"]}]}
        row["agent_ref"] = AGENT_REF

        rows.append(row)

    random.Random(SHUFFLE_SEED).shuffle(rows)

    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} Scale-SWE problems to {OUTPUT_FPATH} (shuffled, seed={SHUFFLE_SEED})")
    missing = len(supported_ids) - len(rows)
    if missing:
        print(f"  {missing} supported id(s) from {SUPPORTED_IDS_FPATH.name} were not found in the Hub dataset")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare(
        _load_supported_ids(SUPPORTED_IDS_FPATH),
        int(os.environ.get("SCALE_SWE_LIMIT") or 0),
    )
