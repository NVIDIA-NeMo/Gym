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
"""Download instruction_following training data from HuggingFace.

Usage:
    python environments/instruction_following/prepare.py
"""

import argparse
import shutil
from pathlib import Path


REPO_ID = "nvidia/Nemotron-RL-instruction_following"
ARTIFACTS = {
    "train": "instruction_following.jsonl",
}


def prepare(split: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    output_path = Path(__file__).parent / "data" / f"{split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy the published JSONL as is. datasets.load_dataset would parse it with Arrow's JSON reader, which
    # fails on the null entries in `kwargs` (apache/arrow#51305).
    downloaded = hf_hub_download(repo_id=REPO_ID, filename=ARTIFACTS[split], repo_type="dataset")
    shutil.copyfile(downloaded, output_path)

    with output_path.open() as f:
        num_rows = sum(1 for _ in f)
    print(f"Wrote {num_rows} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=list(ARTIFACTS))
    args = parser.parse_args()
    prepare(args.split)


if __name__ == "__main__":
    main()
