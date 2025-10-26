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

# Creates JSONL files where each line contains a task index
# python scripts/generate_task_indices.py --num_samples 100 --output data/gsm8k_sample.jsonl
# python scripts/generate_task_indices.py --start 0 --end 1000 --output data/gsm8k_train.jsonl

import argparse
import json
from pathlib import Path

def generate_task_indices(output_path: str, num_samples: int = None, start: int = 0, end: int = None) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if end is not None:
        indices = range(start, end)
    elif num_samples is not None:
        indices = range(start, start + num_samples)
    else:
        raise ValueError("Must specify either --num_samples or --end")

    with output_file.open("w") as f:
        for idx in indices:
            line = json.dumps({"task_idx": idx, "responses_create_params": {"input": []}})
            f.write(line + "\n")

    print(f"Generated {len(indices)} task indices to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--num_samples", "-n", type=int, help="Number of samples to generate (or use --end)"
    )
    parser.add_argument("--start", "-s", type=int, default=0, help="Starting task index (default: 0)")
    parser.add_argument(
        "--end", "-e", type=int, help="Ending task index, exclusive (or use --num_samples)"
    )

    args = parser.parse_args()

    if args.num_samples is None and args.end is None:
        parser.error("Must specify either --num_samples or --end")
    if args.num_samples is not None and args.end is not None:
        parser.error("Cannot specify both --num_samples and --end")

    generate_task_indices(args.output, args.num_samples, args.start, args.end)

if __name__ == "__main__":
    main()
