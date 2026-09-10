# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Drive CLI state definitions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class GoogleDriveActionRecord(BaseModel):
    """Represents a single recorded action performed via the Google Drive CLI."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: f"google_drive_{uuid4().hex}")
    action_key: str
    params: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_result_dict(self) -> Dict[str, Any]:
        return {"id": self.id, **self.params}


class GoogleDriveState(BaseModel):
    """Holds the accumulated state of all actions, folders, and files in Google Drive."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    actions: Dict[str, List[GoogleDriveActionRecord]] = Field(default_factory=dict)
    folders: List[Dict[str, Any]] = Field(default_factory=list)
    files: List[Dict[str, Any]] = Field(default_factory=list)

    def record_action(self, action_key: str, params: Dict[str, Any]) -> GoogleDriveActionRecord:
        record = GoogleDriveActionRecord(action_key=action_key, params=params)
        self.actions.setdefault(action_key, []).append(record)
        return record

    def find_actions(
        self, action_key: str, filters: Dict[str, Any]
    ) -> List[GoogleDriveActionRecord]:
        records = self.actions.get(action_key, [])
        if not filters:
            return list(records)
        results: List[GoogleDriveActionRecord] = []
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
