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
"""Build the AutomationBench dataset for ab_zapier.

AutomationBench ships its tasks as Python inside the `automation-bench`
package rather than as data files, so this loads the environment and
materializes each task's prompt into the row. Rows follow the same shape as
`responses_api_agents/verifiers_agent/scripts/create_dataset.py`; writing an
empty `input` makes the first model request fail with "Messages cannot be
empty".

Usage:
    python environments/ab_zapier/prepare.py
    python environments/ab_zapier/prepare.py --size 10 --domains sales
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VF_ENV_ID = "automationbench_zapier_env"
TOOLSET = "api"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="*", default=None,
                        help="subset of domains (default: all public domains)")
    parser.add_argument("--size", type=int, default=-1,
                        help="number of tasks (-1 for the full taskset)")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    try:
        import verifiers as vf  # noqa: F401
        from automationbench_zapier_env import load_environment
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "automationbench_zapier_env is not installed. Install this environment first:\n"
            "    uv pip install -e environments/ab_zapier"
        ) from exc

    env = load_environment(domains=args.domains, max_turns=args.max_turns,
                           toolset=TOOLSET)
    dataset = env.dataset
    n = len(dataset) if args.size < 0 else min(args.size, len(dataset))

    out = args.out or Path(__file__).parent / "data" / f"automationbench-{n}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for i in range(n):
            row = dataset[i]
            prompt = row["prompt"]
            output_row = {
                "task_idx": i,
                "vf_env_id": VF_ENV_ID,
                "responses_create_params": {"input": prompt},
                "agent_ref": {
                    "type": "responses_api_agents",
                    "name": "verifiers_agent",
                },
                "question": prompt[-1]["content"] if prompt else "",
                "answer": row.get("answer", ""),
                "example_id": row["example_id"],
                "info": row.get("info", {}),
            }
            f.write(json.dumps(output_row) + "\n")

    print(f"wrote {n} rows -> {out}")


if __name__ == "__main__":
    main()
