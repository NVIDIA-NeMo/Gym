# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validator for the synthetic toy assignment task: same JSON contract as upstream's evaluators."""

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--env-dir", required=True)
    args = parser.parse_args()
    costs = {}
    with open(Path(args.env_dir) / "data" / "costs.csv") as f:
        for row in csv.DictReader(f):
            costs[(row["worker"], row["job"])] = float(row["cost"])
    workers = sorted({w for w, _ in costs})
    jobs = sorted({j for _, j in costs})
    errors = []
    assignment = {}
    with open(args.solution) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["worker", "job"]:
            errors.append(f"Expected columns worker,job, got {reader.fieldnames}")
        else:
            for row in reader:
                assignment[row["worker"]] = row["job"]
    if not errors:
        if sorted(assignment) != workers:
            errors.append(f"Workers {sorted(assignment)} do not match {workers}")
        if sorted(assignment.values()) != jobs:
            errors.append(f"Jobs {sorted(assignment.values())} do not cover {jobs} exactly once")
    objective = sum(costs.get(pair, 0.0) for pair in assignment.items()) if not errors else None
    print(json.dumps({"feasible": not errors, "errors": errors, "error_count": len(errors), "objective": objective}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
