# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Recruitee CLI state definitions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class RecruiteeActionRecord(BaseModel):
    """Represents a single recorded action performed via the Recruitee CLI."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: f"recruitee_{uuid4().hex}")
    action_key: str
    params: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_result_dict(self) -> Dict[str, Any]:
        return {"id": self.id, **self.params}


class RecruiteeState(BaseModel):
    """Holds the accumulated state of all actions performed against Recruitee."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    actions: Dict[str, List[RecruiteeActionRecord]] = Field(default_factory=dict)

    def record_action(self, action_key: str, params: Dict[str, Any]) -> RecruiteeActionRecord:
        record = RecruiteeActionRecord(action_key=action_key, params=params)
        self.actions.setdefault(action_key, []).append(record)
        return record

    def find_actions(self, action_key: str, filters: Dict[str, Any]) -> List[RecruiteeActionRecord]:
        records = self.actions.get(action_key, [])
        if not filters:
            return list(records)
        results: List[RecruiteeActionRecord] = []
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
