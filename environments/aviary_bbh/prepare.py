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
"""Generate task_idx JSONL for the BBH Aviary environment.

The Aviary app loads the underlying dataset internally. The Gym JSONL just
lists ``task_idx`` values to index into it.

Usage:
    python environments/aviary_bbh/prepare.py --size 1000 --output environments/aviary_bbh/data/train.jsonl
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for i in range(args.start, args.start + args.size):
            f.write(json.dumps({"task_idx": i, "responses_create_params": {"input": []}}) + "\n")
    print(f"Wrote {args.size} rows to {output_path}")


if __name__ == "__main__":
    main()
