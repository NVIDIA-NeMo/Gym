# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import argparse
import json
import sys
from pathlib import Path

import verifiers as vf

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_verifiers_dataset

def main():
    parser = argparse.ArgumentParser(description="Create dataset from verifiers environment")
    parser.add_argument("--env-id", required=True, help="Verifiers environment ID (e.g., reverse-text)")
    parser.add_argument("--env-args", default="{}", help="JSON string of environment arguments")
    parser.add_argument("--size", type=int, default=-1, help="Number of examples (-1 for all)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    env_args = json.loads(args.env_args)

    print(f"Loading verifiers environment: {args.env_id}")
    env = vf.load_environment(args.env_id, **env_args)

    print(f"Getting dataset (size={args.size}, seed={args.seed})")
    dataset_rows = load_verifiers_dataset(env, n=args.size, seed=args.seed)

    print(f"Dataset has {len(dataset_rows)} examples")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, row in enumerate(dataset_rows):
            output_row = {
                "task_idx": i,
                "vf_env_id": args.env_id,
                "responses_create_params": {
                    "input": row["prompt"],
                },
                "question": row["prompt"][-1]["content"] if row["prompt"] else "",
                "answer": row.get("answer", ""),
                "task": row["task"],
                "example_id": row["example_id"],
                "info": row.get("info", {}),
            }
            f.write(json.dumps(output_row) + "\n")

    print(f"Wrote {len(dataset_rows)} examples to {output_path}")


if __name__ == "__main__":
    main()
