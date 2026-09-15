# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare caller-visible results from legacy and taskset routing smoke runs."""

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {str(row["instance_id"]): row for row in rows}


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response") or {}
    metadata = response.get("metadata") or {}
    observations = row.get("ng_agent_observations") or {}
    return {
        "evaluation_completed": row.get("evaluation_completed"),
        "harness_execution": metadata.get("harness_execution"),
        "observation_gaps": observations.get("gaps") or [],
        "patch_applied": row.get("patch_applied"),
        "resolved": row.get("resolved"),
        "response_status": response.get("status"),
        "reward": row.get("reward"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy", type=Path)
    parser.add_argument("taskset", type=Path)
    args = parser.parse_args()

    legacy = _load(args.legacy)
    taskset = _load(args.taskset)
    if legacy.keys() != taskset.keys():
        raise SystemExit(
            f"routing outputs contain different tasks: legacy={sorted(legacy)}, taskset={sorted(taskset)}"
        )

    mismatches = {
        task_id: {"legacy": _summary(legacy[task_id]), "taskset": _summary(taskset[task_id])}
        for task_id in legacy
        if _summary(legacy[task_id]) != _summary(taskset[task_id])
    }
    if mismatches:
        raise SystemExit(json.dumps(mismatches, indent=2, sort_keys=True))
    print(f"Routing parity passed for {len(legacy)} task(s)")


if __name__ == "__main__":
    main()
