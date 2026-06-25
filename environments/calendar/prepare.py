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
"""Download calendar scheduling training/validation data from HuggingFace.

To regenerate from scratch instead of downloading, see scripts/ for the 3-step
synthesis pipeline (create_synth_conversations.py, generate_rollouts.py,
dataset_preprocess.py).

Usage:
    python environments/calendar/prepare.py
    python environments/calendar/prepare.py --split train
    python environments/calendar/prepare.py --split validation
"""

import argparse
from pathlib import Path


REPO_ID = "nvidia/Nemotron-RL-agent-calendar_scheduling"
ARTIFACTS = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
}


def prepare(split: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    output_path = Path(__file__).parent / "data" / ARTIFACTS[split]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=REPO_ID, filename=ARTIFACTS[split], repo_type="dataset")
    Path(downloaded).rename(output_path)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=list(ARTIFACTS))
    args = parser.parse_args()
    prepare(args.split)


if __name__ == "__main__":
    main()
