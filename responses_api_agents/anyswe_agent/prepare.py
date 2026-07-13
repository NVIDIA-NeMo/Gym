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
"""Prepare SWE-bench Verified rows for AnySWE."""

import argparse
import json
import sys
from pathlib import Path


HF_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

_THIS_DIR = Path(__file__).parent


def _to_gym_row(inst: dict, split: str, sampling: dict) -> dict:
    # Keep rows runnable without collect-time overrides.
    return {
        "responses_create_params": {
            "input": [],
            **sampling,
            "metadata": {
                "instance_id": inst["instance_id"],
                "dataset_name": HF_DATASET,
                "split": split,
                "problem_statement": inst["problem_statement"],
                "instance_dict": json.dumps(inst),
            },
        },
    }


def build_dataset(output: Path, split: str, limit: int | None, instance_id: str | None, sampling: dict) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("`datasets` is required for dataset prep: pip install datasets")

    print(f"Loading {HF_DATASET} [{split}]...", flush=True)
    rows = load_dataset(HF_DATASET, split=split)

    if instance_id:
        rows = [r for r in rows if r["instance_id"] == instance_id]
        if not rows:
            sys.exit(f"instance_id {instance_id!r} not found in {HF_DATASET}")
    elif limit:
        rows = rows.select(range(min(limit, len(rows))))

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as f:
        for inst in rows:
            inst = dict(inst)
            f.write(json.dumps(_to_gym_row(inst, split, sampling)) + "\n")
            count += 1
    print(f"Wrote {count} rows -> {output}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=_THIS_DIR / "data" / "swebench_verified.jsonl")
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--limit", type=int, default=None, help="Only the first N instances (default: all)")
    p.add_argument("--instance-id", default=None, help="Only this instance")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Default model baked into each row")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--max-output-tokens", type=int, default=12288)
    args = p.parse_args()

    sampling = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
    }

    build_dataset(args.output, args.split, args.limit, args.instance_id, sampling)


if __name__ == "__main__":
    main()
