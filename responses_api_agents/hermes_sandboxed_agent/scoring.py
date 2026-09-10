# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score only completed agent attempts with conclusive verification; report coverage."""

from collections import Counter


def is_scored(row):
    return (
        row.get("hermes_finished") is True
        and row.get("evaluation_completed") is not False
        and row.get("score_valid") is not False
        and not row.get("_ng_failure_class")
        and isinstance(row.get("reward"), (int, float))
    )


def summarize(rows):
    scored = [row for row in rows if is_scored(row)]
    return {
        "attempted": len(rows),
        "scored": len(scored),
        "excluded": len(rows) - len(scored),
        "coverage": len(scored) / len(rows) if rows else 0.0,
        "accuracy": sum(row["reward"] for row in scored) / len(scored) if scored else None,
        "exclusions_by_kind": dict(
            Counter(row.get("failure_kind") or "incomplete_or_unknown" for row in rows if not is_scored(row))
        ),
    }
