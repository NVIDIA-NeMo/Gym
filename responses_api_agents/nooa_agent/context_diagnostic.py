# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Describe rejected history changes without logging model or task content."""

import hashlib
import json
from collections.abc import Mapping, Sequence


def context_difference(
    previous: Sequence[Mapping[str, object]], current: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Return first differing item metadata and field hashes, never field values."""
    result: dict[str, object] = {"previous_items": len(previous), "current_items": len(current)}
    for index in range(max(len(previous), len(current))):
        before = previous[index] if index < len(previous) else {}
        after = current[index] if index < len(current) else {}
        if before == after:
            continue
        keys = sorted(
            key
            for key in set(before) | set(after)
            if before.get(key) != after.get(key) or (key in before) != (key in after)
        )
        result.update(
            first_item=index,
            fields={
                key: {
                    "before_present": key in before,
                    "after_present": key in after,
                    "before_type": type(before.get(key)).__name__,
                    "after_type": type(after.get(key)).__name__,
                    "before_sha256": hashlib.sha256(
                        json.dumps(before.get(key), sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "after_sha256": hashlib.sha256(
                        json.dumps(after.get(key), sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
                for key in keys
            },
        )
        break
    return result
