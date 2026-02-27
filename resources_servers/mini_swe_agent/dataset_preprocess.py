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

"""
Preprocess raw SWE-Bench / SWE-Gym datasets into the nemo-gym training format.

The mini_swe_agent expects each JSONL row to contain:
  - All original SWE-Bench fields (instance_id, patch, test_patch, problem_statement, repo, etc.)
  - responses_create_params: {"input": []}  (empty; the agent builds the prompt at runtime)
  - subset: dataset subset identifier ("gym" for SWE-Gym, "verified" for SWE-bench Verified)
  - split: dataset split identifier ("train" or "test")

Usage:
  # Preprocess the training set (SWE-Gym/SWE-Gym)
  python resources_servers/mini_swe_agent/dataset_preprocess.py \
      --input resources_servers/mini_swe_agent/data/train.jsonl

  # Preprocess the validation set (princeton-nlp/SWE-bench_Verified)
  python resources_servers/mini_swe_agent/dataset_preprocess.py \
      --input resources_servers/mini_swe_agent/data/validation.json
"""

import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm


def preprocess(input_path: str, output_path: str, subset: str, split: str) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r") as fin:
        lines = fin.readlines()

    num_written = 0
    with open(output_path, "w") as fout:
        for line in tqdm(lines, desc=f"Processing {input_path.name}"):
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)

            # already preprocessed
            if "responses_create_params" in sample:
                fout.write(json.dumps(sample) + "\n")
                num_written += 1
                continue

            # add fields expected by MiniSWEAgentRunRequest
            sample["responses_create_params"] = {"input": []}
            sample["subset"] = subset
            sample["split"] = split

            fout.write(json.dumps(sample) + "\n")
            num_written += 1

    print(f"Wrote {num_written} samples to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess raw SWE-Bench data for mini_swe_agent")
    parser.add_argument("--input", type=str, required=True, help="Path to raw input JSONL file")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to output JSONL file (defaults to --input for in-place)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="gym",
        help="Dataset subset identifier (e.g. 'gym' for SWE-Gym, 'verified' for SWE-bench Verified)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split identifier (e.g. 'train' or 'test')",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input
    preprocess(args.input, args.output, args.subset, args.split)
