# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted material: Apache-2.0, Copyright 2023 VLMEvalKit Authors.
# Source: https://github.com/open-compass/VLMEvalKit/blob/ebac2d9bda8fa28dc426a3b2ccc8f53c886dc0c5/vlmeval/dataset/utils/hrbench.py
# Full license: resources_servers/opencode_multimodal_benchmarks/third-party-notices.md
"""VLMEvalKit benchmark aggregation, without a VLMEvalKit dependency."""

from collections import defaultdict
from typing import Any


MCQ_ABBREVIATIONS = {
    "coarse_perception": "CP",
    "finegrained_perception (instance-level)": "FP-S",
    "finegrained_perception (cross-instance)": "FP-C",
    "logic_reasoning": "LR",
    "relation_reasoning": "RR",
    "attribute_reasoning": "AR",
}


def accuracy(rows: list[dict[str, Any]]) -> float | None:
    """Unjudged rows invalidate their group's score, rather than count as wrong."""
    if not rows or any(row["hit"] is None for row in rows):
        return None
    return sum(row["hit"] for row in rows) / len(rows)


# HRBench cycle aggregation adapted from VLMEvalKit:
# https://github.com/open-compass/VLMEvalKit/blob/ebac2d9bda8fa28dc426a3b2ccc8f53c886dc0c5/vlmeval/dataset/utils/hrbench.py
def cycle_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match report_acc_hrbench: average cycle accuracies, including categories."""
    groups = defaultdict(list)
    for row in rows:
        cycle = row.get("cycle_category")
        if cycle is None or str(cycle) == "":
            raise ValueError("HRBench8K reporting requires cycle_category")
        groups[str(cycle)].append(row)
    report = []
    for cycle, group in sorted(groups.items()):
        for category in ["all", *sorted({row["category"] for row in group})]:
            selected = group if category == "all" else [row for row in group if row["category"] == category]
            report.append(
                {
                    "cycle": cycle,
                    "type": category,
                    "accuracy": accuracy(selected),
                }
            )
    for category in sorted({row["type"] for row in report}):
        values = [row["accuracy"] for row in report if row["type"] == category]
        average = sum(values) / len(groups) if all(value is not None for value in values) else None
        report.append({"cycle": "Average", "type": category, "accuracy": average})
    return report


def benchmark_report(benchmark: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return VLMEvalKit report columns and units as JSON-compatible records."""
    if not rows:
        return []
    # Category fields are optional task annotations, including on synthetic
    # examples. Their absence must not prevent reporting the overall score.
    rows = [{"category": "", "l2_category": ""} | row for row in rows]
    if benchmark == "AgentX":
        report = []
        for modality in ("Image", "Video"):
            group = [row for row in rows if row["modality"] == modality]
            report.append({"modality": modality, "count": len(group), "accuracy": accuracy(group)})
        return report
    if benchmark == "ZeroBench":
        return [{"split": "zerobench", "count": len(rows), "accuracy": accuracy(rows)}]
    if benchmark == "HRBench8K":
        return cycle_report(rows)
    if benchmark == "VTCBench":
        groups = {"Overall": rows}
        groups.update(
            {
                category: [row for row in rows if row["category"] == category]
                for category in sorted({row["category"] for row in rows if row["category"]})
            }
        )
        return [
            {
                name: (round(100 * value, 2) if (value := accuracy(group)) is not None else None)
                for name, group in groups.items()
            }
        ]
    if benchmark == "VStarBench":
        report = []
        for split in sorted({row.get("split", "none") for row in rows}):
            group = [row for row in rows if row.get("split", "none") == split]
            record = {"split": split, "Overall": accuracy(group)}
            for field in ("l2_category", "category"):
                for name in sorted({row[field] for row in rows if row[field]}):
                    record[MCQ_ABBREVIATIONS.get(name, name)] = accuracy([row for row in group if row[field] == name])
            report.append(record)
        return report
    raise ValueError(f"Unsupported benchmark report: {benchmark}")
