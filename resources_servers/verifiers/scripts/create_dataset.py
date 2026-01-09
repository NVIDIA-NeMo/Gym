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
python scripts/create_dataset.py \
    --env-id reverse-text \
    --size 1000 \
    --output data/reverse_text_train.jsonl

python scripts/create_dataset.py \
    --env-id math-python \
    --env-args '{"difficulty": "easy"}' \
    --size 1000 \
    --seed 42 \
    --output data/math_train.jsonl
"""
import argparse
import json
from pathlib import Path

import verifiers as vf


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
    try:
        dataset = env.get_dataset(n=args.size, seed=args.seed)
    except ValueError:
        # Some environments (aime2025, ifeval, etc.) load dataset via different attributes
        # TODO: is there more standard way in verifiers.. check prime rl
        dataset = None
        for attr in ['dataset', 'train_dataset', 'eval_dataset']:
            ds = getattr(env, attr, None)
            if ds is not None:
                dataset = ds
                print(f"Found dataset in env.{attr}")
                break

        if dataset is None:
            raise ValueError(f"Environment {args.env_id} does not have a dataset")

        if args.seed is not None:
            dataset = dataset.shuffle(seed=args.seed)
        if args.size > 0:
            dataset = dataset.select(range(min(args.size, len(dataset))))

    print(f"Dataset has {len(dataset)} examples")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i in range(len(dataset)):
            row = {
                "task_idx": i,
                "vf_env_id": args.env_id,
                "responses_create_params": {
                    "input": dataset["prompt"][i],
                },
                "question": dataset["prompt"][i][-1]["content"] if dataset["prompt"][i] else "",
                "answer": dataset["answer"][i] if "answer" in dataset.column_names else "",
                "task": dataset["task"][i],
                "example_id": dataset["example_id"][i],
            }
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    main()
