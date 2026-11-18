# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict JSON verdicts returned by the question-bank judge."""

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


NonEmptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]


class Judgment(BaseModel):
    """A complete structured judgment; evidence is ordinary text, not executable."""

    model_config = ConfigDict(extra="forbid", strict=True)

    verdict: Literal["correct", "incorrect", "unresolved"]
    reason_code: NonEmptyString
    evidence: NonEmptyString


def hash_text(text: str) -> str:
    """Hash exact UTF-8 text without normalizing task content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def parse_judgment(text: str) -> Judgment | None:
    """Accept exactly one verdict with unique keys and nonempty evidence."""
    try:
        return Judgment.model_validate(
            json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        )
    except (ValueError, ValidationError, RecursionError):
        return None
