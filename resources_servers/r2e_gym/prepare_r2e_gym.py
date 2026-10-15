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
"""Prepare R2E-Gym/R2E-Gym-Subset training data for NeMo Gym, from the Hub.

This is a training dataset, not an eval benchmark: 4,578 synthetic-issue Python tasks over real commits
in ten repos (pandas, numpy, pillow, orange3, aiohttp, tornado, scrapy, pyramid, datalad, coveragepy),
each with a prebuilt image (https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset; the R2E-Gym project
is MIT-licensed, the dataset card lists no license of its own).

Two stages, like the other SWE servers:

    # 1. every Hub row, for the 3x golden-patch sweep
    python resources_servers/r2e_gym/prepare_r2e_gym.py --raw
      -> data/r2e_gym_training_raw.jsonl
    # 2. only the rows whose golden patch resolved in every pass
    python resources_servers/r2e_gym/prepare_r2e_gym.py --supported-ids results/r2e_gym_supported_instance_ids.txt
      -> data/r2e_gym_training.jsonl, data/supported_instance_ids.txt

What changes shape on the way out of the Hub row:
  * the golden patch is rebuilt from ``parsed_commit_content`` (``r2e_patch.golden_patch``, byte-identical
    to R2E-Gym's ``ParsedCommit.get_patch``) and the multi-megabyte parsed commit itself is dropped;
  * the prompt is the text inside ``[ISSUE] ... [/ISSUE]``, as R2E-Gym's own harness presents it;
  * ``execution_result_content``, ``modified_entity_summaries``, ``relevant_files`` and the Hub's
    ``prompt`` column (the issue-writing instruction, not a task prompt) are dropped as unused.

Rows are shuffled with a fixed seed: the Hub order is grouped by repo. Set R2E_GYM_LIMIT=N for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from resources_servers.r2e_gym.r2e_patch import DATASET_NAME, golden_patch, image_name, instance_id_for, issue_text


SHUFFLE_SEED = 0
DATA_DIR = Path(__file__).parent / "data"
SUPPORTED_IDS_FPATH = DATA_DIR / "supported_instance_ids.txt"
AGENT_REF = {"type": "responses_api_agents", "name": "r2e_gym_opencode_sandboxed_agent"}

# Hub fields carried through unchanged (see app.py's R2EGymInstanceRequest and task_data.py).
ROW_FIELDS = (
    "repo_name",
    "docker_image",
    "commit_hash",
    "problem_statement",
    "expected_output_json",
    "modified_files",
    "num_non_test_files",
    "num_non_test_func_methods",
    "num_non_test_lines",
)


def build_row(example: dict) -> dict:
    row = {field: example[field] for field in ROW_FIELDS}
    row["instance_id"] = instance_id_for(example["repo_name"], example["commit_hash"])
    row["language"] = "python"
    row["image_name"] = image_name(example["docker_image"])
    row["dataset_name"] = DATASET_NAME
    row["patch"] = golden_patch(json.loads(example["parsed_commit_content"]))
    row["responses_create_params"] = {"input": [{"role": "user", "content": issue_text(example["problem_statement"])}]}
    row["agent_ref"] = AGENT_REF
    return row


def load_supported_ids(fpath: Path) -> set[str]:
    return {line.strip() for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()}


def prepare(supported_ids: set[str] | None, limit: int = 0) -> Path:
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped_empty_patch = 0
    for example in load_dataset(DATASET_NAME, split="train"):
        instance_id = instance_id_for(example["repo_name"], example["commit_hash"])
        if supported_ids is not None and instance_id not in supported_ids:
            continue
        row = build_row(example)
        if not row["patch"].strip():
            skipped_empty_patch += 1  # nothing to validate or to learn from
            continue
        rows.append(row)
    random.Random(SHUFFLE_SEED).shuffle(rows)
    if limit:
        rows = rows[:limit]

    stem = "r2e_gym_training_raw" if supported_ids is None else "r2e_gym_training"
    output = DATA_DIR / f"{stem}.jsonl"
    with output.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")
    if supported_ids is not None:
        SUPPORTED_IDS_FPATH.write_text("".join(sorted(row["instance_id"] + "\n" for row in rows)), encoding="utf-8")
        missing = len(supported_ids) - len(rows)
        if missing:
            print(f"  {missing} supported id(s) were not found in the Hub dataset")
    if skipped_empty_patch:
        print(f"  skipped {skipped_empty_patch} row(s) whose commit has no Python-file diff")
    print(f"Wrote {len(rows)} R2E-Gym problems to {output} (shuffled, seed={SHUFFLE_SEED})")
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
    prepare(supported, int(os.environ.get("R2E_GYM_LIMIT") or 0))


if __name__ == "__main__":
    main()
