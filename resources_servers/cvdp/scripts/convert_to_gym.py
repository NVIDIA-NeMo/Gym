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


# Code comprehension categories use subjective scoring, not docker-compose harness.
# They have no target_files or harness_files — instead they carry a reference answer.
CODE_COMPREHENSION_CATEGORIES = [6, 8, 9, 10]


def _get_category_num(entry: dict) -> int | None:
    """Extract the numeric category from the categories list (e.g. 'cid010' -> 10)."""
    categories = entry.get("categories", [])
    if categories and isinstance(categories[0], str) and categories[0].startswith("cid"):
        return int(categories[0][3:])
    return None


def _get_target_files(entry: dict) -> list[str]:
    """All output.context keys — matches dataset_processor.py line 1117."""
    output_context = (entry.get("output") or {}).get("context") or {}
    return list(output_context.keys())


def _get_harness_files(entry: dict) -> dict[str, str | None]:
    """Docker-compose + test scripts — passed as-is, matching CVDP."""
    return (entry.get("harness") or {}).get("files") or {}


def _get_context_files(entry: dict) -> dict[str, str]:
    """Companion RTL files from input.context that the model doesn't generate
    but are needed for compilation (e.g. floor_to_seven_segment.sv).

    Returns input.context files that are NOT in output.context (i.e. not
    target files the model is asked to produce)."""
    input_context = (entry.get("input") or {}).get("context") or {}
    target_keys = set(_get_target_files(entry))
    return {k: v for k, v in input_context.items() if k not in target_keys and v}


def _get_subjective_reference(entry: dict) -> str | None:
    """Reference answer for code-comprehension categories — from output.response."""
    return (entry.get("output") or {}).get("response")


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
            cat_num = _get_category_num(raw)
            is_comprehension = cat_num is not None and cat_num in CODE_COMPREHENSION_CATEGORIES

            target_files = _get_target_files(raw)
            # Code-generation categories require target_files; comprehension categories don't.
            if not target_files and not is_comprehension:
                skipped += 1
                continue

            context_files = _get_context_files(raw)
            verifier_metadata = {
                "task_id": task_id,
                "categories": raw.get("categories", []),
                "difficulty": raw.get("difficulty", ""),
                "target_files": target_files,
                "harness_files": _get_harness_files(raw),
            }
            if context_files:
                verifier_metadata["context_files"] = context_files

            # Code-comprehension categories carry the reference answer for BLEU/ROUGE scoring.
            if is_comprehension:
                ref = _get_subjective_reference(raw)
                if ref:
                    verifier_metadata["subjective_reference"] = ref
                else:
                    print(f"WARNING: no output.response for comprehension task {task_id}, skipping")
                    skipped += 1
                    continue

            gym_row = {
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": row["system"]},
                        {"role": "user", "content": row["user"]},
                    ]
                },
                "verifier_metadata": verifier_metadata,
            }
            fout.write(json.dumps(gym_row) + "\n")
            written += 1

    print(f"Wrote {written} entries to {args.output}")
    if skipped:
        print(f"Skipped {skipped} entries (no dataset match, no target files, or missing reference)")


if __name__ == "__main__":
    main()
