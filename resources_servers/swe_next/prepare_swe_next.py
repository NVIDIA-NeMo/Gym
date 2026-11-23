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
"""Prepare TIGER-Lab/SWE-Next training data for NeMo Gym, from the Hub.

This is a training dataset, not an eval benchmark -- there is no fixed/frozen reference split,
and the output is meant to feed a training run, not to be scored against as a leaderboard.

Writes only the rows already known to be usable: `data/supported_instance_ids.txt` is the result
of a full-set 3x golden-patch sweep -- a row not on that list either never resolves, never
produced a verdict, or resolves inconsistently across passes (see `aggregate_golden_patch.py` and
the README), so it is not worth carrying into the training jsonl at all rather than shipping a
row no agent run could usefully score.

    SWE_NEXT_LIMIT=200 python -m resources_servers.swe_next.prepare_swe_next
"""

import json
import os
import random
from pathlib import Path


SHUFFLE_SEED = 0

DATASET_NAME = "TIGER-Lab/SWE-Next"
DATASET_FILENAME = "SWE_Next_dataset.jsonl"
SUPPORTED_IDS_FPATH = Path(__file__).parent / "data" / "supported_instance_ids.txt"
OUTPUT_FPATH = Path(__file__).parent / "data" / "swe_next_training.jsonl"

AGENT_REF = {"type": "responses_api_agents", "name": "swe_next_opencode_sandboxed_agent"}


def _load_supported_ids(fpath: Path) -> set[str]:
    return {line.strip() for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()}


def _iter_raw_rows():
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=DATASET_NAME, filename=DATASET_FILENAME, repo_type="dataset"))
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_row(raw: dict) -> dict:
    row = {
        "instance_id": raw.get("instance_id", ""),
        "repo": raw.get("repo", ""),
        "language": "python",
        "workdir": "/testbed",
        "image_ref": raw.get("docker_image", ""),
        "base_commit": raw.get("base_commit", ""),
        "patch": raw.get("patch", "") or "",
        "test_patch": raw.get("test_patch", "") or "",
        "problem_statement": raw.get("problem_statement", "") or "",
        # Provenance only -- grading uses expected_output_json (see verification.py).
        "FAIL_TO_PASS": raw.get("FAIL_TO_PASS") or [],
        "PASS_TO_PASS": raw.get("PASS_TO_PASS") or [],
        "expected_output_json": raw.get("expected_output_json") or "{}",
    }
    row["responses_create_params"] = {"input": [{"role": "user", "content": row["problem_statement"]}]}
    row["agent_ref"] = AGENT_REF
    return row


def prepare(supported_ids: set[str], limit: int = 0) -> Path:
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for raw in _iter_raw_rows():
        if limit and len(rows) >= limit:
            break
        if raw.get("instance_id") not in supported_ids:
            continue
        rows.append(_build_row(raw))

    random.Random(SHUFFLE_SEED).shuffle(rows)

    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} SWE-Next problems to {OUTPUT_FPATH} (shuffled, seed={SHUFFLE_SEED})")
    missing = len(supported_ids) - len(rows)
    if missing:
        print(f"  {missing} supported id(s) from {SUPPORTED_IDS_FPATH.name} were not found in the Hub dataset")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare(
        _load_supported_ids(SUPPORTED_IDS_FPATH),
        int(os.environ.get("SWE_NEXT_LIMIT") or 0),
    )
