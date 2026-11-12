# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Average repeated measurements without treating repeats as additional tasks."""

from math import comb
from typing import Any

from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME


def average_repeats(tasks: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], bool]:
    """Return one mean-reward row per task and whether all observed repeats cover it.

    Dataset size is checked by the caller. Missing judge results leave a repeat
    incomplete; duplicate task IDs or duplicate measurements are invalid.
    """
    rows = []
    seen = set()
    repeat_sets = []
    all_measured = True
    for repeats in tasks:
        if not repeats:
            continue
        metadata = repeats[0]["verifier_metadata"]
        identity = (metadata.get("benchmark"), metadata["index"])
        if identity in seen:
            raise ValueError("Duplicate benchmark sample ID")
        seen.add(identity)
        indices = {row.get(ROLLOUT_INDEX_KEY_NAME, 0) for row in repeats}
        if len(indices) != len(repeats):
            raise ValueError("Duplicate sample repeat")
        if any(row["verifier_metadata"] != metadata for row in repeats):
            raise ValueError("Inconsistent metadata across sample repeats")
        measured = [
            row for row in repeats if not row.get("mask_sample", False) and row.get("evaluation_completed", True)
        ]
        all_measured = all_measured and len(measured) == len(repeats)
        repeat_sets.append({row.get(ROLLOUT_INDEX_KEY_NAME, 0) for row in measured})
        if measured:
            rows.append(measured[0] | {"reward": sum(row["reward"] for row in measured) / len(measured)})
    balanced = all_measured and bool(repeat_sets) and all(indices == repeat_sets[0] for indices in repeat_sets)
    return rows, balanced


def pass_at_k(tasks: list[list[dict[str, Any]]], *, k: int) -> float | None:
    """Estimate success in k attempts; incomplete or masked attempts cannot certify it."""
    if k < 1:
        raise ValueError("k must be positive")
    estimates = []
    for repeats in tasks:
        valid = [row for row in repeats if not row.get("mask_sample", False) and row.get("evaluation_completed", True)]
        if len(valid) < k:
            return None
        if any(row["reward"] not in (0, 1) for row in valid):
            raise ValueError("pass@k requires binary rewards")
        n = len(valid)
        correct = sum(int(row["reward"]) for row in valid)
        estimates.append(1.0 if n - correct < k else 1.0 - comb(n - correct, k) / comb(n, k))
    return sum(estimates) / len(estimates) if estimates else None
