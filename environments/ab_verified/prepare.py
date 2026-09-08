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
"""Generate the AutomationBench index dataset.

ab_aa, ab_zapier and ab_verified all score the same 600-task set
(6 domains x 100 tasks) and differ only in scoring, so this script is
identical across the three and produces byte-identical output.

A row carries only a task index; the taskset supplies task content at
rollout time, so this is an index into the task set, not a copy of it.

Usage:
    python environments/ab_verified/prepare.py
    python environments/ab_verified/prepare.py --size 166 --split train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


AGENT_REF = {"type": "responses_api_agents", "name": "verifiers_agent"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=600, help="number of tasks to index")
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or Path(__file__).parent / "data" / f"automationbench-{args.size}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for i in range(args.size):
            row = {
                "task_idx": i,
                "responses_create_params": {"input": []},
                "agent_ref": AGENT_REF,
            }
            f.write(json.dumps(row) + "\n")

    print(f"wrote {args.size} rows -> {out}")


if __name__ == "__main__":
    main()
