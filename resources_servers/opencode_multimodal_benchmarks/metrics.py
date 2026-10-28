# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Average repeated measurements without treating repeats as additional tasks."""

from typing import Any

from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME


def average_repeats(tasks: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], bool]:
    """Return one mean-reward row per task and whether all observed repeats cover it.

    Dataset size is checked by the caller. Missing judge results leave a repeat
    incomplete; duplicate task IDs or duplicate measurements are invalid.
    """
    rows = []
    seen = set()
    repeat_sets = []
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
        repeat_sets.append(indices)
        rows.append(repeats[0] | {"reward": sum(row["reward"] for row in repeats) / len(repeats)})
    balanced = bool(repeat_sets) and all(indices == repeat_sets[0] for indices in repeat_sets)
    return rows, balanced
