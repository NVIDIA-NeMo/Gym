# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage-aware summaries: unreviewed generations are never model failures."""

from typing import Any


def summarize(rows: list[dict[str, Any]], expected_items: int | None = None) -> dict[str, Any]:
    declared = {row.get("verifier_metadata", {}).get("cohort_size") for row in rows}
    declared.discard(None)
    if expected_items is not None:
        declared.add(expected_items)
    conflict = len(declared) > 1
    cohort_size = next(iter(declared)) if len(declared) == 1 else None
    identities = {(row.get("benchmark"), row.get("orig_trajectory_id")) for row in rows}
    scored = [row for row in rows if row.get("scoring_status") == "scored" and not row.get("mask_sample")]
    measured_identities = {(row.get("benchmark"), row.get("orig_trajectory_id")) for row in scored}
    passed = sum(row.get("reward") == 1 for row in scored)
    failed = len(scored) - passed
    complete = (
        cohort_size is not None
        and len(identities) == cohort_size
        and len(measured_identities) == cohort_size
        and len(scored) == len(rows)
    )
    return {
        "scored": len(scored),
        "not_scored": len(rows) - len(scored),
        "passed": passed,
        "failed": failed,
        "observed_items": len(identities),
        "scored_items": len(measured_identities),
        "expected_items": cohort_size,
        "cohort_size_conflict": conflict,
        "missing_items": max(0, cohort_size - len(identities)) if cohort_size is not None else None,
        "scoring_coverage": len(measured_identities) / cohort_size if cohort_size is not None else None,
        "scored_subset_accuracy": passed / len(scored) if scored else None,
        "complete_accuracy": passed / len(scored) if complete and scored else None,
    }
