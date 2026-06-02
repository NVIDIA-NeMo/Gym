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
"""Download competitive coding training and validation data from HuggingFace.

Usage:
    python environments/code_gen/prepare.py
    python environments/code_gen/prepare.py --split train
    python environments/code_gen/prepare.py --split validation
"""

import argparse
from pathlib import Path


REPO_ID = "nvidia/nemotron-RL-coding-competitive_coding"

SPLITS = {
    "train": "opencodereasoning_filtered_25k_train.jsonl",
    "validation": "livecodebench_v5_2024-07-01_2025-02-01_validation.jsonl",
}

ARTIFACTS = {
    "train": "opencodereasoning_filtered_25k_train.jsonl",
    "validation": "validation.jsonl",
}


def prepare(split: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    output_path = Path(__file__).parent / "data" / SPLITS[split]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(repo_id=REPO_ID, filename=ARTIFACTS[split], repo_type="dataset")
    Path(downloaded).rename(output_path)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=list(SPLITS))
    args = parser.parse_args()
    prepare(args.split)


if __name__ == "__main__":
    main()
