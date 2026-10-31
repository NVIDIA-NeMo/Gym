#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check that every row of a training file loads in this env, without calling a model.

Per row, the steps the env runs before the first policy call: the agent_ref name (unless --agent ''), the run-request
model, the harness the row selects (a row that names a harness must not fall back to generic), the harness's
conversion of the history into tool items, its tool list, and the replay of recorded file edits into the session
workspace. Then a dry draw of which rows obstacle injection would arm at --obstacle-row-rate (counts only; the draw
hashes row content, so a content change moves it). The first failing row stops the run with
its line number. Prints, and with --report writes, counts per file: rows, harness, rows whose tools have obstacle
texts, obstacle draw (keys are "sandbox:<kind>" for sandbox rows, "other:<kind>" otherwise).

    PYTHONPATH=<Gym root> python validate_rows.py train_mix.clean.jsonl --report validation.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from resources_servers.agentic_turn_simulation.app import AgenticTurnSimulationRunRequest, _input_items, _row_tools
from resources_servers.agentic_turn_simulation.harnesses import select_harness
from resources_servers.agentic_turn_simulation.obstacles import ObstacleInjectionConfig, arm


AGENT = "agentic_turn_simulation_agent"


def validate_file(path: Path, *, agent: str = AGENT, row_rate: float = 0.3) -> dict[str, object]:
    """Counts for one file; raises SystemExit naming the first row that does not load."""
    config = ObstacleInjectionConfig(enabled=True, row_rate=row_rate)
    harnesses, armed, obstacle_tools, rows = Counter(), Counter(), Counter(), 0
    with open(path, "rb", buffering=1 << 22) as f:
        for n, line in enumerate(f, 1):
            row = json.loads(line)
            metadata = row.get("verifier_metadata") or {}
            if agent and (row.get("agent_ref") or {}).get("name") != agent:
                raise SystemExit(f"{path}:{n}: agent_ref {row.get('agent_ref')!r}, wanted {agent!r}")
            body = AgenticTurnSimulationRunRequest.model_validate(row)
            harness = select_harness(body.verifier_metadata)
            if harness.name == "generic" and metadata.get("harness") not in (None, "generic"):
                raise SystemExit(f"{path}:{n}: row names harness {metadata['harness']!r} but selects generic")
            items = _input_items(body)
            prefix = harness.to_tool_items(items)
            tool_names = harness.tool_names(_row_tools(body))
            workspace = harness.new_workspace(body.verifier_metadata or {})
            if workspace is not None:
                workspace.replay(prefix)
            if any(name in tool_names for name in harness.obstacle_texts):
                obstacle_tools[harness.name] += 1
            obstacle = arm(
                config,
                harness=harness,
                tool_names=tool_names,
                input_items=items,
                metadata=body.verifier_metadata or {},
            )
            group = "sandbox" if harness.name == "sandbox" else "other"
            armed[f"{group}:{obstacle.kind if obstacle is not None else 'not_armed'}"] += 1
            harnesses[harness.name] += 1
            rows += 1
    return {
        "rows": rows,
        "harness": dict(harnesses.most_common()),
        "rows_with_obstacle_tools": dict(obstacle_tools.most_common()),
        f"obstacle_draw_at_row_rate_{row_rate}": dict(sorted(armed.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument(
        "--agent", default=AGENT, help=f"expected agent_ref name (default {AGENT}; '' skips the check)"
    )
    parser.add_argument("--obstacle-row-rate", type=float, default=0.3)
    parser.add_argument("--report", type=Path, help="write the counts as JSON")
    args = parser.parse_args()
    report = {}
    for path in args.files:
        report[path.name] = validate_file(path, agent=args.agent, row_rate=args.obstacle_row_rate)
        print(path.name, json.dumps(report[path.name]))
    if args.report:
        with os.fdopen(os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            f.write(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
