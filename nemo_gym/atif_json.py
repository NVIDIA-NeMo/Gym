# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict JSON helpers for Gym's ATIF conversion boundaries."""

from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any


def strict_json_loads(value: str | bytes) -> Any:
    """Parse standards-compliant JSON without narrowing arbitrary-size integers."""

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-standard JSON number {constant!r}")

    def finite_float(number: str) -> float:
        parsed = float(number)
        if not math.isfinite(parsed):
            raise ValueError(f"JSON number {number!r} is outside the finite float range")
        if parsed == 0.0:
            significand = number.lower().split("e", 1)[0]
            if any(character in "123456789" for character in significand):
                raise ValueError(f"JSON number {number!r} underflows the finite float range")
            return parsed
        if Decimal(number) != Decimal(str(parsed)):
            raise ValueError(f"JSON number {number!r} cannot be represented without precision loss")
        return parsed

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = item
        return result

    return json.loads(
        value,
        parse_constant=reject_constant,
        parse_float=finite_float,
        object_pairs_hook=reject_duplicate_keys,
    )


def json_values_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/integer equality aliasing."""

    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
            and left == right
        )
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(
                json_values_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
            )
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(json_values_equal(left[key], right[key]) for key in left)
        )
    return False
