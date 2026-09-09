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
package rather than as data files, so this pulls the real taskset via
`get_combined_dataset` and sizes the output from it instead of assuming a
count. Each row carries the task index the agent resolves against that same
taskset at rollout time.

ab_aa and ab_zapier index the identical taskset and differ only in scoring,
so both produce byte-identical output for the same --domains.

Usage:
    python environments/ab_zapier/prepare.py
    python environments/ab_zapier/prepare.py --domains sales hr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="*", default=None,
                        help="subset of domains (default: all public domains)")
    parser.add_argument("--size", type=int, default=None,
                        help="cap the number of tasks (default: the full taskset)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    try:
        from automationbench.domains import DEFAULT_DOMAINS, get_combined_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "automation-bench is not installed. Install this environment first:\n"
            "    uv pip install -e environments/ab_zapier"
        ) from exc

    domains = list(args.domains) if args.domains else list(DEFAULT_DOMAINS)
    dataset = get_combined_dataset(domains)

    n = len(dataset) if args.size is None else min(args.size, len(dataset))
    out = args.out or Path(__file__).parent / "data" / f"automationbench-{n}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    agent_ref = {"type": "responses_api_agents", "name": "verifiers_agent"}
    with out.open("w") as f:
        for i in range(n):
            row = {
                "task_idx": i,
                "responses_create_params": {"input": []},
                "agent_ref": agent_ref,
            }
            f.write(json.dumps(row) + "\n")

    print(f"wrote {n} rows from {len(domains)} domains ({', '.join(domains)}) -> {out}")


if __name__ == "__main__":
    main()
