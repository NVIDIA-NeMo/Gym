#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Preprocess HotPotQA into JSONL for the abstention environment.

Downloads the HotPotQA dataset from HuggingFace if --download is passed,
then converts the raw data into the JSONL format expected by the
abstention resources server.  This version:

  - Includes the ground-truth ``answer`` field in every record (needed by
    the LLM judge).
  - Injects a system prompt instructing the model to use \\boxed{} format
    and \\boxed{[IDK]} when unsure.

Usage:
    # Download and preprocess in one step
    python dataset_preprocess.py \\
        --download \\
        --raw-data-dir /path/to/data/hotpotqa \\
        --output-dir ./data

    # Preprocess only (data already downloaded)
    python dataset_preprocess.py \\
        --raw-data-dir /path/to/data/hotpotqa \\
        --output-dir ./data
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SYSTEM_PROMPT = (
    "You will be given a question, answer with JUST the answer (no explanation).\n"
    "Put your final answer in \\boxed{your answer} format.\n"
    "If you do not know the answer, or you need more context or tools to answer the question,\n"
    "output \\boxed{[IDK]} instead of guessing.\n"
    "It is better to not provide an answer when unsure than to be wrong."
)


def download_hotpotqa(raw_data_dir: str, splits: list[str] | None = None) -> None:
    """Download HotPotQA fullwiki from HuggingFace and save as JSONL."""
    from datasets import load_dataset

    os.makedirs(raw_data_dir, exist_ok=True)
    splits = splits or ["train", "validation"]

    for split in splits:
        out_path = os.path.join(raw_data_dir, f"fullwiki_{split}.jsonl")
        if os.path.exists(out_path):
            print(f"  {out_path} already exists, skipping download.")
            continue
        print(f"  Downloading HotPotQA fullwiki {split}...")
        ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=split, cache_dir=raw_data_dir)
        ds.to_json(out_path)
        print(f"  Saved {len(ds)} records to {out_path}")


def build_record(row: dict, idx: int) -> dict:
    """Convert a single HotPotQA row into an abstention record."""
    question = row["question"]
    answer = row["answer"]
    record_id = row.get("id", idx)

    return {
        "id": record_id,
        "question": question,
        "answer": answer,
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "abstention_simple_agent",
        },
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "tools": [],
            "parallel_tool_calls": False,
        },
    }


def preprocess_split(
    raw_jsonl_path: Path,
    output_path: Path,
    max_samples: int | None = None,
) -> int:
    """Read a raw HotPotQA JSONL and write the preprocessed version."""
    count = 0
    with open(raw_jsonl_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if max_samples is not None and count >= max_samples:
                break
            row = json.loads(line)
            record = build_record(row, idx)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Preprocess HotPotQA for the abstention environment")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the HotPotQA dataset from HuggingFace first.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        required=True,
        help="Directory containing (or to download to) raw HotPotQA JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory for preprocessed JSONL files (default: ./data).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit the number of training samples (default: use all).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Limit the number of validation samples (default: use all).",
    )
    args = parser.parse_args()

    if args.download:
        print("Step 1: Downloading HotPotQA from HuggingFace...")
        download_hotpotqa(str(args.raw_data_dir))
    else:
        print("Skipping download (use --download to fetch from HuggingFace).")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("fullwiki_train.jsonl", "hotpotqa_train.jsonl", args.max_train_samples),
        ("fullwiki_validation.jsonl", "hotpotqa_val.jsonl", args.max_val_samples),
    ]

    for raw_name, out_name, max_samples in splits:
        raw_path = args.raw_data_dir / raw_name
        out_path = args.output_dir / out_name
        if not raw_path.exists():
            print(f"Warning: {raw_path} not found, skipping.")
            continue
        print(f"Preprocessing {raw_path} -> {out_path}...")
        count = preprocess_split(raw_path, out_path, max_samples)
        print(f"  Wrote {count} records to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
