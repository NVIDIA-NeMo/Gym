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
"""Download reasoning_gym training data from HuggingFace.

Usage:
    python environments/reasoning_gym/prepare.py
    python environments/reasoning_gym/prepare.py --split train
"""

import argparse
import json
from pathlib import Path


def prepare(split: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    output_path = Path(__file__).parent / "data" / f"{split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("nvidia/Nemotron-RL-ReasoningGym-v1", split=split)

    with output_path.open("w") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(dataset)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train"])
    args = parser.parse_args()
    prepare(args.split)


if __name__ == "__main__":
    main()
