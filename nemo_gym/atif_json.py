# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict JSON helpers for Gym's ATIF conversion boundaries."""

from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any


def strict_json_loads(payload: str | bytes) -> Any:
    """Decode RFC 8259 JSON without losing large integers or duplicate-key evidence."""

    def reject_non_json_constant(value: str) -> Any:
        raise ValueError(f"non-JSON numeric constant {value!r}")

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"JSON number {value!r} exceeds the finite float range")
        if parsed == 0.0:
            significand = value.lower().split("e", 1)[0]
            if any(character in "123456789" for character in significand):
                raise ValueError(f"JSON number {value!r} underflows the finite float range")
            return parsed
        if Decimal(value) != Decimal(str(parsed)):
            raise ValueError(f"JSON number {value!r} cannot be represented without precision loss")
        return parsed

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    return json.loads(
        payload,
        parse_constant=reject_non_json_constant,
        parse_float=parse_finite_float,
        object_pairs_hook=reject_duplicate_keys,
    )
