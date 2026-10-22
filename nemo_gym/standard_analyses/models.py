# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral inputs and results; unknown measurements remain null."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def measurement(value: float | None, name: str, *, integer: bool = False) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (integer and int(value) != value)
    ):
        raise ValueError(f"{name} must be a finite nonnegative {'integer' if integer else 'number'}")


@dataclass(frozen=True)
class Call:
    id: str
    index: int
    output_tokens: int | None = None  # Includes reasoning tokens once.
    reasoning_tokens: int | None = None
    input_tokens: int | None = None
    duration_s: float | None = None
    started_at: float | None = None
    invocation_id: str | None = None
    status: str = "unknown"

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Call id is required")
        if self.index is None:
            raise ValueError("Call index is required")
        measurement(self.index, "call index", integer=True)
        for name in ("output_tokens", "reasoning_tokens", "input_tokens"):
            measurement(getattr(self, name), name, integer=True)
        measurement(self.duration_s, "duration_s")
        measurement(self.started_at, "started_at")
        if (
            self.output_tokens is not None
            and self.reasoning_tokens is not None
            and self.reasoning_tokens > self.output_tokens
        ):
            raise ValueError("reasoning_tokens cannot exceed output_tokens")


@dataclass(frozen=True)
class Tool:
    id: str
    index: int | None = None
    name: str = "unknown"
    duration_s: float | None = None
    started_at: float | None = None
    invocation_id: str | None = None
    status: str = "unknown"
    action: str | None = None  # read, write, explore, implement, verify
    target: str | None = None
    content_hash: str | None = None  # Observed state, never inferred from command text.

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Tool id is required")
        measurement(self.index, "tool index", integer=True)
        measurement(self.duration_s, "tool duration_s")
        measurement(self.started_at, "tool started_at")


@dataclass(frozen=True)
class Compaction:
    invocation_id: str
    after_call_id: str  # First model call AFTER compaction.

    def __post_init__(self):
        if not self.invocation_id or not self.after_call_id:
            raise ValueError("Compaction requires invocation and first post-compaction call identity")


@dataclass(frozen=True)
class Rollout:
    run_id: str
    task_id: str
    repeat: int
    agent: str
    model: str
    benchmark: str
    config_id: str  # Cohort identity: sampling/evaluation parameters, not a secret-bearing config.
    success: bool | None = None
    health: str = "unknown"
    capture_complete: bool | None = None
    wall_s: float | None = None
    calls: tuple[Call, ...] = ()
    tools: tuple[Tool, ...] = ()
    compactions: tuple[Compaction, ...] = ()
    environment_status: str = "unknown"
    gaps: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("run_id", "task_id", "agent", "model", "benchmark", "config_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} is required")
        measurement(self.repeat, "repeat", integer=True)
        measurement(self.wall_s, "wall_s")
        if self.success is not None and not isinstance(self.success, bool):
            raise ValueError("success must be true, false, or null; nonbinary rewards need a policy")
        if self.capture_complete is not None and not isinstance(self.capture_complete, bool):
            raise ValueError("capture_complete must be true, false, or null")
        if self.health not in ("healthy", "unhealthy", "quarantined", "unknown"):
            raise ValueError("invalid health")
        for name in ("calls", "tools"):
            ids = [item.id for item in getattr(self, name)]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {name} identity in rollout {self.task_id}/{self.repeat}")
        indices = [c.index for c in self.calls]
        if len(indices) != len(set(indices)):
            raise ValueError("call indices must be unique within a rollout")

    @property
    def cohort(self) -> tuple:
        return self.run_id, self.benchmark, self.agent, self.model, self.config_id

    @property
    def key(self) -> tuple:
        return (*self.cohort, self.task_id, self.repeat)

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


@dataclass(frozen=True)
class Options:
    thresholds: tuple[int, ...] = (1000, 10000)
    pass_k: tuple[int, ...] = (1, 2, 4, 8)
    max_rows: int = 2000

    def __post_init__(self):
        if type(self.max_rows) is not int or self.max_rows < 1:
            raise ValueError("max_rows must be positive")
        if not self.thresholds or any(type(t) is not int or t < 0 for t in self.thresholds):
            raise ValueError("thresholds must be nonnegative integers")
        if not self.pass_k or any(type(k) is not int or k < 1 for k in self.pass_k):
            raise ValueError("pass_k must contain positive integers")


@dataclass
class Table:
    columns: list[str]
    rows: list[list[Any]]
    total_rows: int
    truncated: bool

    @classmethod
    def build(cls, columns: list[str], rows: list[list[Any]], limit: int):
        if any(len(row) != len(columns) for row in rows):
            raise ValueError("Table row width does not match columns")
        return cls(columns, rows[:limit], len(rows), len(rows) > limit)


@dataclass
class AnalysisResult:
    name: str
    description: str
    status: str = "unavailable"
    reason: str | None = None
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    coverage: dict[str, int] = field(default_factory=dict)
    tables: dict[str, Table] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Report:
    results: list[AnalysisResult]
    cohort: dict[str, str]
    options: Options
    schema_version: str = "1.0"
    analysis_version: str = "0.1.0"

    def to_dict(self):
        return asdict(self)

    def scalars(self) -> dict[str, int | float]:
        result = {}
        for analysis in self.results:
            prefix = f"efficiency/{analysis.name}"
            result[f"{prefix}/available"] = int(analysis.status != "unavailable")
            result[f"{prefix}/complete"] = int(analysis.status == "available")
            for key, value in {**analysis.metrics, **analysis.coverage}.items():
                if value is not None:
                    result[f"{prefix}/{key}"] = value
        return result
