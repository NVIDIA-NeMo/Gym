# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MMRB judge accuracy, shared by GenRM and LLM as a judge modes."""

from collections import defaultdict
from typing import Any

from resources_servers.genrm_eval.preferences import score


def compute_metrics(tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for task in tasks for row in task]
    if not rows:
        return {}
    valid = [r for r in rows if r.get("predicted_ranking") is not None]
    correct = sum(score(r["verifier_metadata"], r.get("predicted_ranking")) for r in rows)
    metrics = {
        "overall": {
            "accuracy": correct / len(rows),
            "accuracy_valid": correct / len(valid) if valid else 0.0,
            "invalid_judgment_rate": 1 - len(valid) / len(rows),
            "total_pairs": len({r["verifier_metadata"]["pair_id"] for r in rows}),
            "total_judgments": len(rows),
        }
    }
    slices = defaultdict(list)
    for row in rows:
        metadata = row["verifier_metadata"]
        for key in ("source", "order", "pair_type", "category", "subcategory", "task", "response_modality"):
            if key in metadata:
                slices[f"{key}/{metadata[key]}/accuracy"].append(score(metadata, row.get("predicted_ranking")))
    metrics.update({key: sum(values) / len(values) for key, values in slices.items()})
    return metrics
