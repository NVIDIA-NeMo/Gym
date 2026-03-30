#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Converts CVDP local_export prompts to NeMo-Gym format.
#
# Input: prompts.jsonl produced by CVDP's local_export mode, with fields:
#   {id, prompt, system, user}
#
# Also requires the original CVDP dataset for verifier_metadata (harness_files,
# target_files), which the resource server (app.py) needs to run the Docker harness.
#
# Usage:
#   python convert_to_gym.py \
#       --prompts  prompts.jsonl \
#       --dataset  cvdp_dataset.jsonl \
#       --output   data/train.jsonl

import argparse
import json
from pathlib import Path


def _get_target_files(entry: dict) -> list[str]:
    """All output.context keys — matches dataset_processor.py line 1117."""
    output_context = (entry.get("output") or {}).get("context") or {}
    return list(output_context.keys())


def _get_harness_files(entry: dict) -> dict[str, str | None]:
    """Docker-compose + test scripts — passed as-is, matching CVDP."""
    return (entry.get("harness") or {}).get("files") or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CVDP export prompts to NeMo-Gym format")
    parser.add_argument("--prompts", required=True, help="prompts.jsonl from CVDP local_export mode")
    parser.add_argument("--dataset", required=True, help="Original CVDP dataset JSONL (for verifier_metadata)")
    parser.add_argument("--output", required=True, help="Output NeMo-Gym JSONL")
    args = parser.parse_args()

    # Index original dataset by id for verifier_metadata lookup
    dataset: dict[str, dict] = {}
    with open(args.dataset) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                dataset[entry["id"]] = entry

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with open(args.prompts) as fin, open(args.output, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row["id"]

            if task_id not in dataset:
                skipped += 1
                continue

            raw = dataset[task_id]
            target_files = _get_target_files(raw)
            if not target_files:
                skipped += 1
                continue

            gym_row = {
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": row["system"]},
                        {"role": "user", "content": row["user"]},
                    ]
                },
                "verifier_metadata": {
                    "task_id": task_id,
                    "categories": raw.get("categories", []),
                    "difficulty": raw.get("difficulty", ""),
                    "target_files": target_files,
                    "harness_files": _get_harness_files(raw),
                },
            }
            fout.write(json.dumps(gym_row) + "\n")
            written += 1

    print(f"Wrote {written} entries to {args.output}")
    if skipped:
        print(f"Skipped {skipped} entries (no dataset match or no target files)")


if __name__ == "__main__":
    main()
