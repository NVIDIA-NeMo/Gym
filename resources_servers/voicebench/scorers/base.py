# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared, JSON-serializable results for VoiceBench scoring modules."""

from dataclasses import dataclass

from pydantic import JsonValue


@dataclass(frozen=True)
class ScoringResult:
    """A row reward, sufficient aggregate statistics and inspectable diagnostics."""

    reward: float
    metrics: dict[str, float]
    details: dict[str, JsonValue]
    valid: bool = True
