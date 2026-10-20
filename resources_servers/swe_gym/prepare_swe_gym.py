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
"""Prepare SWE-Gym/SWE-Gym training data for NeMo Gym, from the Hub.

This is a training dataset, not an eval benchmark: 2,438 SWE-bench-format Python tasks from 11 repos
with prebuilt images (MIT; https://huggingface.co/datasets/SWE-Gym/SWE-Gym).

Two stages, like the other SWE servers:

    # 1. every Hub row, for the 3x golden-patch sweep
    python resources_servers/swe_gym/prepare_swe_gym.py --raw
      -> data/swe_gym_training_raw.jsonl
    # 2. only the rows whose golden patch resolved in every pass
    python resources_servers/swe_gym/prepare_swe_gym.py --supported-ids results/swe_gym_supported_instance_ids.txt
      -> data/swe_gym_training.jsonl, data/supported_instance_ids.txt

Rows are shuffled with a fixed seed: the Hub order is grouped by repo, and training on it as-is would
bias early steps toward whichever repo sorts first. The prompt is the issue text alone; hints_text
(maintainer comments written after the issue, which often spell out the fix) is deliberately dropped.

Set SWE_GYM_LIMIT=N to write only the first N kept rows (smoke tests).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


SHUFFLE_SEED = 0
DATASET_NAME = "SWE-Gym/SWE-Gym"
IMAGE_NAMESPACE = "docker.io/xingyaoww"
DATA_DIR = Path(__file__).parent / "data"
SUPPORTED_IDS_FPATH = DATA_DIR / "supported_instance_ids.txt"
AGENT_REF = {"type": "responses_api_agents", "name": "swe_gym_opencode_sandboxed_agent"}

# The Hub row's own fields this server's request model reads (see app.py's SWEGymInstanceRequest).
# created_at / hints_text are dropped: the first is unused, the second leaks the fix.
ROW_FIELDS = ("instance_id", "repo", "version", "base_commit", "patch", "test_patch", "problem_statement")


def image_name(instance_id: str) -> str:
    """SWE-Gym publishes its images as ``xingyaoww/sweb.eval.x86_64.<id>`` with ``__`` spelt ``_s_``.

    Lower-cased like OpenHands does: Docker repository names must be lower case, so
    ``Project-MONAI__MONAI-3715`` lives at ``...project-monai_s_monai-3715``.
    """
    return f"{IMAGE_NAMESPACE}/sweb.eval.x86_64.{instance_id.replace('__', '_s_').lower()}:latest"


def as_list(value) -> list[str]:
    """FAIL_TO_PASS / PASS_TO_PASS arrive as lists (parquet) or as their JSON string (older dumps)."""
    if isinstance(value, str):
        return [str(x) for x in json.loads(value)] if value.strip() else []
    return [str(x) for x in (value or [])]


def build_row(example: dict) -> dict:
    row = {field: example[field] for field in ROW_FIELDS}
    row["version"] = str(row["version"])
    row["language"] = "python"
    row["image_name"] = image_name(example["instance_id"])
    row["dataset_name"] = DATASET_NAME
    row["FAIL_TO_PASS"] = as_list(example.get("FAIL_TO_PASS"))
    row["PASS_TO_PASS"] = as_list(example.get("PASS_TO_PASS"))
    row["responses_create_params"] = {"input": [{"role": "user", "content": row["problem_statement"]}]}
    row["agent_ref"] = AGENT_REF
    return row


def load_supported_ids(fpath: Path) -> set[str]:
    return {line.strip() for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()}


def prepare(supported_ids: set[str] | None, limit: int = 0) -> Path:
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for example in load_dataset(DATASET_NAME, split="train"):
        if supported_ids is not None and example["instance_id"] not in supported_ids:
            continue
        rows.append(build_row(example))
    random.Random(SHUFFLE_SEED).shuffle(rows)
    if limit:
        rows = rows[:limit]

    stem = "swe_gym_training_raw" if supported_ids is None else "swe_gym_training"
    output = DATA_DIR / f"{stem}.jsonl"
    with output.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")
    if supported_ids is not None:
        SUPPORTED_IDS_FPATH.write_text("".join(sorted(row["instance_id"] + "\n" for row in rows)), encoding="utf-8")
        missing = len(supported_ids) - len(rows)
        if missing:
            print(f"  {missing} supported id(s) were not found in the Hub dataset")
    print(f"Wrote {len(rows)} SWE-Gym problems to {output} (shuffled, seed={SHUFFLE_SEED})")
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--raw", action="store_true", help="write every Hub row (input to the golden-patch sweep)")
    mode.add_argument(
        "--supported-ids",
        type=Path,
        default=SUPPORTED_IDS_FPATH,
        help="supported_instance_ids.txt from aggregate_golden_patch.py (default: data/supported_instance_ids.txt)",
    )
    args = ap.parse_args()
    supported = None if args.raw else load_supported_ids(args.supported_ids)
    prepare(supported, int(os.environ.get("SWE_GYM_LIMIT") or 0))


if __name__ == "__main__":
    main()
