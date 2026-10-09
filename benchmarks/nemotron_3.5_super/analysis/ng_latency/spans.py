# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The span model the analysis reads: one record per timed event, with its clock domain and explicit parent.

Spans are built offline by :mod:`ng_latency.project` from the observability records in ``rollouts.jsonl``; no
Gym process writes them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


TIMELINE_SCHEMA_VERSION = "2.0"

SpanStatus = Literal["ok", "error", "timeout", "cancelled", "incomplete", "unknown"]
EndKind = Literal["measured", "derived", "open"]


class SpanIds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Optional[str] = None
    repeat: Optional[int] = None
    attempt: Optional[int] = None
    rollout_id: Optional[str] = None
    invocation_id: Optional[str] = None
    parent_invocation_id: Optional[str] = None
    model_call_id: Optional[str] = None
    client_call_id: Optional[str] = None
    tool_call_id: Optional[str] = None

    @model_serializer(mode="wrap")
    def _without_absent_ids(self, handler: Any) -> dict[str, Any]:
        # Absent identifiers are omitted, not written as null: most spans carry only a few of them.
        return {key: value for key, value in handler(self).items() if value is not None}

    def merged(self, other: "SpanIds | None") -> "SpanIds":
        if other is None:
            return self
        return SpanIds(**{**self.model_dump(exclude_none=True), **other.model_dump(exclude_none=True)})


class SpanClock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    host: Optional[str] = None
    domain: str


def clock_for(domain: Optional[str], role: str) -> SpanClock:
    """A clock from a record's ``clock_domain`` (``node:<host>`` or ``sandbox:<id>``)."""
    domain = domain or "unknown"
    host = domain.split(":", 1)[1] if domain.startswith("node:") else None
    return SpanClock(role=role, host=host, domain=domain)


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    span_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    status: SpanStatus = "ok"
    ids: SpanIds = Field(default_factory=SpanIds)
    clock: SpanClock
    start_wall_s: float
    duration_ns: Optional[int] = Field(default=None, ge=0)
    end_wall_s: Optional[float] = None
    end_kind: EndKind = "open"
    attrs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_end(self) -> "Span":
        if self.end_wall_s is not None and self.end_wall_s < self.start_wall_s:
            raise ValueError("end_wall_s must not precede start_wall_s")
        if self.end_kind == "open" and self.end_wall_s is not None:
            raise ValueError("open spans have no end_wall_s")
        if self.end_kind != "open" and self.end_wall_s is None:
            raise ValueError(f"{self.end_kind} spans need end_wall_s")
        return self

    @property
    def duration_s(self) -> Optional[float]:
        if self.duration_ns is not None:
            return self.duration_ns / 1e9
        if self.end_wall_s is not None:
            return self.end_wall_s - self.start_wall_s
        return None


def measured_span(
    name: str,
    *,
    clock: SpanClock,
    start_wall_s: float,
    end_wall_s: Optional[float],
    ids: SpanIds | None = None,
    parent: Span | None = None,
    status: SpanStatus = "ok",
    attrs: dict[str, Any] | None = None,
) -> Span:
    """A span whose timestamps were reported by another system (OpenCode, the model server)."""
    attributes = dict(attrs or {})
    end_kind: EndKind = "measured"
    if end_wall_s is None:
        end_kind = "open"
    elif end_wall_s < start_wall_s:
        attributes["end_wall_s_raw"] = end_wall_s
        attributes["end_before_start"] = True
        end_wall_s = None
        end_kind = "open"
    return Span(
        name=name,
        parent_span_id=parent.span_id if parent is not None else None,
        status=status,
        ids=ids or SpanIds(),
        clock=clock,
        start_wall_s=start_wall_s,
        end_wall_s=end_wall_s,
        end_kind=end_kind,
        attrs=attributes,
    )


def timed_span(
    name: str,
    *,
    clock: SpanClock,
    start_wall_s: float,
    duration_ms: Optional[float],
    end_wall_s: Optional[float] = None,
    ids: SpanIds | None = None,
    parent: Span | None = None,
    status: SpanStatus = "ok",
    attrs: dict[str, Any] | None = None,
) -> Span:
    """A span the recording process timed itself: the end is start plus the monotonic duration when there is
    one, else the recorded end."""
    if duration_ms is None:
        return measured_span(
            name,
            clock=clock,
            start_wall_s=start_wall_s,
            end_wall_s=end_wall_s,
            ids=ids,
            parent=parent,
            status=status,
            attrs=attrs,
        )
    duration_ns = int(round(max(0.0, float(duration_ms)) * 1_000_000))
    return Span(
        name=name,
        parent_span_id=parent.span_id if parent is not None else None,
        status=status,
        ids=ids or SpanIds(),
        clock=clock,
        start_wall_s=start_wall_s,
        duration_ns=duration_ns,
        end_wall_s=start_wall_s + duration_ns / 1e9,
        end_kind="derived",
        attrs=dict(attrs or {}),
    )


def serialize(spans: Iterable[Span]) -> list[dict[str, Any]]:
    return [span.model_dump(mode="json") for span in spans]
