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
"""Generate a blackjack dataset by duplicating the example.jsonl template.

Each game is dealt on the fly during the resources server's `reset()`, so
every row is identical. This script just sizes the output JSONL.

Usage:
    python environments/blackjack/prepare.py --size 1000
    python environments/blackjack/prepare.py --size 1000 --split train
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000, help="Number of rows to generate.")
    parser.add_argument("--split", default="train", help="Output split name (train, validation, etc.)")
    args = parser.parse_args()

    here = Path(__file__).parent
    template_path = here / "data" / "example.jsonl"
    with template_path.open() as f:
        template = json.loads(f.readline())

    output_path = here / "data" / f"{args.split}.jsonl"
    with output_path.open("w") as f:
        for _ in range(args.size):
            f.write(json.dumps(template) + "\n")

    print(f"Wrote {args.size} rows to {output_path}")


if __name__ == "__main__":
    main()
