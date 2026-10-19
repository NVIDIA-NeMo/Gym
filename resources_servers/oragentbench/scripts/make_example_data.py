# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write ``data/example.jsonl``: five synthetic rows in the exact shape the preparer emits.

ORAgentBench's task packets are third-party content (MIT code, CC BY 4.0 data) and are
fetched at preparation time rather than redistributed here. The rows below are invented for
this file: five toy operations-research briefs with the same fields the real rows carry.
Their ``docker_image`` and ``task_folder`` point at a synthetic fixture task under
``tests/fixtures/`` so a smoke run through ``verify()`` exercises the real container path.
"""

import json
import sys
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR / "scripts"))
from prepare_oragentbench import AGENT_NAME  # noqa: E402


FIXTURE_TASK_FOLDER = "resources_servers/oragentbench/tests/fixtures/toy_assignment"
FIXTURE_IMAGE = "oragentbench-fixture:toy_assignment"

INSTRUCTION = """You are solving a small operations-research exercise inside a sandbox.

Work only inside `/app`. Read `/app/PROBLEM-STATEMENT.md` and every file under `/app/data/`.
Write the model to `/app/submissions/model.md`, implement the solver in
`/app/submissions/solve.py` using PySCIPOpt, write the assignment to
`/app/submissions/solution.csv` with columns `worker,job`, and log every solve in
`/app/submissions/solve_log.md`.

Scenario: {scenario}
"""

SCENARIOS = [
    ("synthetic/toy_assignment_bakery", "easy", "assign three bakers to three morning shifts at minimum total cost."),
    ("synthetic/toy_assignment_clinic", "easy", "assign four nurses to four wards, one each, minimising travel time."),
    (
        "synthetic/toy_assignment_depot",
        "medium",
        "assign five trucks to five delivery zones with zone-specific costs.",
    ),
    ("synthetic/toy_assignment_lab", "medium", "assign six analysts to six instruments; each analyst runs one."),
    ("synthetic/toy_assignment_port", "hard", "assign eight cranes to eight berths under a per-berth cost matrix."),
]


def rows() -> list:
    out = []
    for name, band, scenario in SCENARIOS:
        out.append(
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": INSTRUCTION.format(scenario=scenario)}]
                },
                "task_name": name,
                "docker_image": FIXTURE_IMAGE,
                "task_folder": FIXTURE_TASK_FOLDER,
                "difficulty": band,
                "num_steps": 1,
                "agent_ref": {"type": "responses_api_agents", "name": AGENT_NAME},
            }
        )
    return out


def main() -> int:
    output = SERVER_DIR / "data" / "example.jsonl"
    with open(output, "w") as f:
        for row in rows():
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(SCENARIOS)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
