# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measurements of explicitly named operation boundaries."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TimingBoundary = Literal["lifecycle", "executor", "transcript", "event_receipt"]


class OperationTiming(BaseModel):
    """One measurement; endpoint comparability and elapsed duration are independent evidence."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)

    boundary: TimingBoundary = Field(
        description=(
            "Semantic interval measured: lifecycle spans harness dispatch through result handling; "
            "executor spans instrumented execution; transcript spans native call/result records; "
            "event_receipt spans observer receipt. Storage or delivery format does not determine the boundary."
        )
    )
    timing_source: str = Field(min_length=1)
    clock_domain: str | None = Field(default=None, min_length=1)
    started_at: float | None = None
    completed_at: float | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    duration_source: Literal["monotonic", "reported", "wall_clock"] | None = None
    execution_id: str | None = None

    @model_validator(mode="after")
    def validate_measurement(self) -> "OperationTiming":
        if self.started_at is None and self.completed_at is None and self.duration_ms is None:
            raise ValueError("a timing requires at least one endpoint or duration")
        if self.started_at is not None and self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.duration_ms is None and self.duration_source is not None:
            raise ValueError("duration_source requires a duration")
        return self
