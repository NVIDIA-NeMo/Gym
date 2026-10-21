# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted material: Apache-2.0, Copyright 2023 VLMEvalKit Authors.
# Source: https://github.com/open-compass/VLMEvalKit/blob/ebac2d9bda8fa28dc426a3b2ccc8f53c886dc0c5/vlmeval/dataset/utils/hrbench.py
# Source: https://github.com/open-compass/VLMEvalKit/blob/ebac2d9bda8fa28dc426a3b2ccc8f53c886dc0c5/vlmeval/dataset/babyvision.py
# Full license: resources_servers/opencode_multimodal_benchmarks/third-party-notices.md
"""VLMEvalKit benchmark aggregation, without a VLMEvalKit dependency."""

from collections import defaultdict


# Original BabyVision category mapping and reporting convention:
# https://github.com/UniPat-AI/BabyVision/blob/7f92fd4b1dc1c68b7b936a9bc09c68b4a944a55a/babyvision_eval/compute_score.py
# Also present in VLMEvalKit:
# https://github.com/open-compass/VLMEvalKit/blob/ebac2d9bda8fa28dc426a3b2ccc8f53c886dc0c5/vlmeval/dataset/babyvision.py
BABYVISION_TYPES = {
    "Fine-grained Discrimination": "1",
    "Visual Tracking": "2",
    "Spatial Perception": "3",
    "Visual Pattern Recognition": "4",
}
MCQ_ABBREVIATIONS = {
    "coarse_perception": "CP",
    "finegrained_perception (instance-level)": "FP-S",
    "finegrained_perception (cross-instance)": "FP-C",
    "logic_reasoning": "LR",
    "relation_reasoning": "RR",
    "attribute_reasoning": "AR",
}


def accuracy(rows):
    """Unjudged rows invalidate their group's score, rather than count as wrong."""
    if not rows or any(row["hit"] is None for row in rows):
        return None
    return sum(row["hit"] for row in rows) / len(rows)


# HRBench cycle aggregation adapted from VLMEvalKit:
# https://github.com/open-compass/VLMEvalKit/blob/ebac2d9bda8fa28dc426a3b2ccc8f53c886dc0c5/vlmeval/dataset/utils/hrbench.py
def cycle_report(rows):
    """Match report_acc_hrbench: average cycle accuracies, including categories."""
    groups = defaultdict(list)
    for row in rows:
        cycle = row["cycle_category"]
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


def benchmark_report(benchmark, rows):
    """Return VLMEvalKit report columns and units as JSON-compatible records."""
    if not rows:
        return []
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
    if benchmark == "BabyVision":
        types = defaultdict(list)
        subtypes = defaultdict(list)
        for row in rows:
            category = row["category"]
            types[category].append(row)
            subtypes[(category, row["l2_category"])].append(row)
        groups = [("Overall", rows)]
        groups += [(name, types[name]) for name in sorted(types, key=lambda name: BABYVISION_TYPES.get(name, "0"))]
        groups += [
            (category + "/" + subtype, subtypes[(category, subtype)])
            for category, subtype in sorted(
                subtypes,
                key=lambda key: BABYVISION_TYPES.get(key[0], "0") + key[0] + "/" + key[1],
            )
        ]
        return [
            {
                "split": name,
                "tot": len(group),
                "hit": (sum(row["hit"] for row in group) if all(row["hit"] is not None for row in group) else None),
                "acc": (100 * value if (value := accuracy(group)) is not None else None),
            }
            for name, group in groups
        ]
    raise ValueError(f"Unsupported benchmark report: {benchmark}")
