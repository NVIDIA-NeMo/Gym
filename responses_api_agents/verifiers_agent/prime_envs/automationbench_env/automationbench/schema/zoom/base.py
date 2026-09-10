# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoom schema base classes."""

import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from automationbench.schema.zoom.meeting import Meeting
    from automationbench.schema.zoom.participant import Participant
    from automationbench.schema.zoom.recording import Recording
    from automationbench.schema.zoom.registrant import Registrant
    from automationbench.schema.zoom.webinar import Webinar


def generate_zoom_meeting_id() -> int:
    """Generate a Zoom-style meeting ID (numeric)."""
    return int(uuid.uuid4().int % 10**10)


def generate_zoom_uuid() -> str:
    """Generate a Zoom-style UUID."""
    return str(uuid.uuid4())


class ZoomRecord(BaseModel):
    """Base class for all Zoom records."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: int = Field(default_factory=generate_zoom_meeting_id, description="Zoom record ID")

    def to_display_dict(self) -> dict:
        """Return dict for agent display. Override in subclasses."""
        return {"id": self.id}


class ZoomState(BaseModel):
    """Zoom application state containing all object collections."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def strip_scratchpad_fields(cls, data: object) -> object:
        """Remove scratchpad fields not part of the schema."""
        if isinstance(data, dict):
            data = dict(data)
            data.pop("actions", None)
        return data

    meetings: list["Meeting"] = Field(default_factory=list)
    webinars: list["Webinar"] = Field(default_factory=list)
    recordings: list["Recording"] = Field(default_factory=list)
    registrants: list["Registrant"] = Field(default_factory=list)
    participants: list["Participant"] = Field(default_factory=list)
    users: list[dict] = Field(default_factory=list)

    def get_by_id(self, collection: str, record_id: int | str) -> Optional["Meeting | Registrant"]:
        """Find a record by ID in a collection."""
        records = getattr(self, collection, [])
        # Handle string or int IDs
        if isinstance(record_id, str):
            try:
                record_id = int(record_id)
            except ValueError:
                pass
        for record in records:
            if record.id == record_id:
                return record
        return None

    def find_by_field(
        self,
        collection: str,
        field: str,
        value: str | int,
        case_insensitive: bool = True,
    ) -> list["Meeting | Registrant"]:
        """Find records by field value."""
        records = getattr(self, collection, [])
        results = []
        for record in records:
            record_value = getattr(record, field, None)
            if record_value is None:
                continue
            if case_insensitive and isinstance(record_value, str) and isinstance(value, str):
                if record_value.lower() == value.lower():
                    results.append(record)
            elif record_value == value:
                results.append(record)
        return results

    def update_record(
        self, collection: str, record_id: int | str, updates: dict
    ) -> Optional["Meeting | Registrant"]:
        """Update a record by ID."""
        records = getattr(self, collection, [])
        # Handle string or int IDs
        if isinstance(record_id, str):
            try:
                record_id = int(record_id)
            except ValueError:
                pass
        for i, record in enumerate(records):
            if record.id == record_id:
                updated_data = record.model_dump()
                updated_data.update(updates)
                model_class = type(record)
                records[i] = model_class(**updated_data)
                return records[i]
        return None
