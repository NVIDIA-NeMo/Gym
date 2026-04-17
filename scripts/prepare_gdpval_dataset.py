#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Download and convert the GDPVal dataset from HuggingFace to JSONL format
compatible with nemo-gym's rollout collection.

Usage:
    python scripts/prepare_gdpval_dataset.py
    python scripts/prepare_gdpval_dataset.py --output responses_api_agents/stirrup_agent/data/gdpval.jsonl
    python scripts/prepare_gdpval_dataset.py --dataset openai/gdpval --split train
"""

import argparse
import json
from pathlib import Path


def prepare_gdpval_jsonl(hf_dataset_name: str, output_path: str, split: str = "train") -> str:
    """Convert the HuggingFace GDPVal dataset to a JSONL file for nemo-gym."""
    from datasets import load_dataset

    ds = load_dataset(hf_dataset_name, split=split)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for row in ds:
            # metadata values must all be strings (OpenAI Metadata type constraint)
            record = {
                "responses_create_params": {
                    "input": "",
                    "model": "placeholder",
                    "metadata": {
                        "task_id": row["task_id"],
                        "sector": row.get("sector", ""),
                        "occupation": row.get("occupation", ""),
                        "prompt": row["prompt"],
                        "reference_files": json.dumps(row.get("reference_files", [])),
                        "reference_file_urls": json.dumps(row.get("reference_file_urls", [])),
                        "rubric_json": json.dumps(row.get("rubric_json", {})),
                        "rubric_pretty": row.get("rubric_pretty", ""),
                    },
                },
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(ds)} tasks to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare GDPVal dataset for nemo-gym rollouts")
    parser.add_argument(
        "--dataset",
        default="openai/gdpval",
        help="HuggingFace dataset name (default: openai/gdpval)",
    )
    parser.add_argument(
        "--output",
        default="responses_api_agents/stirrup_agent/data/gdpval.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    args = parser.parse_args()

    prepare_gdpval_jsonl(
        hf_dataset_name=args.dataset,
        output_path=args.output,
        split=args.split,
    )


if __name__ == "__main__":
    main()
