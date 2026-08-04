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
from itertools import islice
from pathlib import Path

import verifiers.v1 as vf


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Verifiers V1 taskset to Gym JSONL")
    parser.add_argument("--taskset", required=True)
    parser.add_argument("--taskset-config", default="{}", help="Additional taskset config as JSON")
    parser.add_argument("--size", type=int, default=-1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = vf.taskset_config_type(args.taskset).model_validate(
        {**json.loads(args.taskset_config), "id": args.taskset}
    )
    stop = None if args.size < 0 else args.offset + args.size
    rows = []
    for task_idx, task in enumerate(
        islice(vf.load_taskset(config), args.offset, stop),
        start=args.offset,
    ):
        data = task.data.model_dump(mode="json")
        params = {"input": data.get("prompt") or ""}
        if data.get("system_prompt"):
            params["instructions"] = data["system_prompt"]
        rows.append(
            {
                "task_idx": task_idx,
                "responses_create_params": params,
                "agent_ref": {
                    "type": "responses_api_agents",
                    "name": "verifiers_agent",
                },
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(f"{json.dumps(row, separators=(',', ':'))}\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} tasks to {output}")


if __name__ == "__main__":
    main()
