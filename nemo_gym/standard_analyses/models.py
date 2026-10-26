# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact projections of Gym evidence and backend-independent analysis results."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Count = Annotated[int, Field(ge=0, strict=True)]
Measurement = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
Identity = Annotated[str, Field(min_length=1)]
Cell = str | int | float | bool | None


class AnalysisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Call(AnalysisModel):
    id: Identity
    index: Count
    output_tokens: Count | None = None  # Includes reasoning tokens once.
    reasoning_tokens: Count | None = None
    input_tokens: Count | None = None
    duration_s: Measurement | None = None

    @model_validator(mode="after")
    def validate_tokens(self) -> Call:
        if self.output_tokens is not None and self.reasoning_tokens is not None:
            if self.reasoning_tokens > self.output_tokens:
                raise ValueError("reasoning_tokens cannot exceed output_tokens")
        return self


class Rollout(AnalysisModel):
    run_id: Identity
    task_id: Identity
    repeat: Count
    stage: int | str | None = None
    agent: Identity
    model: Identity
    benchmark: Identity
    config_id: Identity
    run_key: str | None = None
    success: Annotated[bool, Field(strict=True)] | None = None
    health: Literal["healthy", "unhealthy", "quarantined", "unknown"] = "unknown"
    capture_complete: Annotated[bool, Field(strict=True)] | None = None
    calls: tuple[Call, ...] = ()
    gaps: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_calls(self) -> Rollout:
        if len({c.id for c in self.calls}) != len(self.calls):
            raise ValueError("duplicate call identity")
        if len({c.index for c in self.calls}) != len(self.calls):
            raise ValueError("call indices must be unique within a rollout")
        return self

    @property
    def cohort(self) -> tuple[str, ...]:
        return self.run_id, self.benchmark, self.agent, self.model, self.config_id

    @property
    def key(self) -> tuple:
        return (*self.cohort, self.task_id, self.repeat, str(self.stage))

    @property
    def eligible(self) -> bool:
        return self.health not in ("unhealthy", "quarantined")

    @property
    def complete_time(self) -> bool:
        return (
            self.eligible
            and self.capture_complete is True
            and bool(self.calls)
            and all(c.duration_s is not None for c in self.calls)
        )

    @property
    def model_s(self) -> float | None:
        if not self.calls or any(c.duration_s is None for c in self.calls):
            return None
        return math.fsum(c.duration_s for c in self.calls)


class Options(AnalysisModel):
    thresholds: Annotated[tuple[Count, ...], Field(min_length=1)] = (1000, 10000)
    max_rows: Annotated[int, Field(gt=0, strict=True)] = 2000


class Table(AnalysisModel):
    columns: list[str]
    rows: list[list[Cell]]
    total_rows: Count
    truncated: bool

    @classmethod
    def build(cls, columns: list[str], rows: list[list[Cell]], limit: int) -> Table:
        if any(len(row) != len(columns) for row in rows):
            raise ValueError("Table row width does not match columns")
        return cls(columns=columns, rows=rows[:limit], total_rows=len(rows), truncated=len(rows) > limit)


class AnalysisResult(AnalysisModel):
    name: str
    description: str
    status: Literal["available", "partial", "unavailable"] = "unavailable"
    reason: str | None = None
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    coverage: dict[str, int] = Field(default_factory=dict)
    tables: dict[str, Table] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class Report(AnalysisModel):
    results: list[AnalysisResult]
    cohort: dict[str, str]
    options: Options
    schema_version: str = "1.0"
    analysis_version: str = "0.2.0"

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def scalars(self, *, namespace: str = "efficiency") -> dict[str, int | float]:
        result = {}
        for analysis in self.results:
            prefix = f"{namespace}/{analysis.name}"
            result[f"{prefix}/available"] = int(analysis.status != "unavailable")
            result[f"{prefix}/complete"] = int(analysis.status == "available")
            for key, value in {**analysis.metrics, **analysis.coverage}.items():
                if value is not None:
                    result[f"{prefix}/{key}"] = value
        return result
