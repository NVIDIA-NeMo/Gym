# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-runtime clocks, kept separate from client request intervals."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionTiming(BaseModel):
    """Runtime interval in epoch seconds and duration in milliseconds.

    Only intervals with the same clock_domain are directly comparable. A remote
    Unix clock is not assumed synchronized with a client or another sandbox.
    Duration may use a monotonic clock even when endpoints use a wall clock.
    Unknown endpoints stay absent; duration never supplies an absolute start.
    These intervals describe tool execution, not GPU activity.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = Field(default=None, ge=0)
    clock_domain: str = Field(min_length=1)
    timing_source: str = Field(min_length=1)
    duration_source: Optional[str] = None
    execution_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_interval(self) -> "ExecutionTiming":
        if self.started_at is not None and self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        return self
