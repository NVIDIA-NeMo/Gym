# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge disjoint completed Gym rollouts, checking original on-disk reward receipts."""

import argparse
import json
from pathlib import Path

from responses_api_agents.wildclawbench_agent.metrics import benchmark_metrics


def merge(paths: list[Path], output: Path) -> dict:
    rows = []
    for path in paths:
        for line in path.read_text().split("\n"):
            if line.strip():
                row = json.loads(line)
                receipt = json.loads((Path(row["artifact_dir"]) / "result.json").read_text())
                for key in (
                    "task_id",
                    "task_sha256",
                    "source_revision",
                    "workspace_sha256",
                    "native_reward",
                    "reward",
                    "mask_sample",
                    "evaluation_protocol",
                ):
                    if row[key] != receipt[key]:
                        raise ValueError(f"Native/Gym receipt mismatch for {key}: {path}")
                rows.append(row)
    metrics = benchmark_metrics(rows)
    if "wildclawbench/overall_score" not in metrics:
        raise ValueError(f"Incomplete or masked suite: {metrics}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.rollouts, args.output), indent=2))
