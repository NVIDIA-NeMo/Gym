# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Calendar Calendar model."""

from typing import Literal, Optional

from pydantic import Field

from automationbench.schema.google_calendar.base import GoogleCalendarRecord


class Calendar(GoogleCalendarRecord):
    """Represents a single Google Calendar (the container, not an event within it).

    Distinct from CalendarEvent — this models the calendar itself.
    """

    summary: str = Field(..., description="Display name or title of the calendar")
    description: Optional[str] = Field(None, description="Optional text describing the calendar")

    # Access and visibility
    access_role: Literal["freeBusyReader", "reader", "writer", "owner"] = Field(
        default="owner", description="The current user's permission level for this calendar"
    )
    hidden: bool = Field(
        default=False, description="Indicates whether the calendar is hidden from view"
    )
    deleted: bool = Field(
        default=False, description="Indicates whether the calendar has been removed"
    )

    # Time zone
    time_zone: str = Field(default="UTC", description="Time zone associated with the calendar")

    # Primary calendar flag
    primary: bool = Field(default=False, description="True if this is the user's primary calendar")

    def to_display_dict(self) -> dict:
        """Build a dict representation for agent consumption."""
        return {
            "id": self.id,
            "summary": self.summary,
            "description": self.description,
            "accessRole": self.access_role,
            "hidden": self.hidden,
            "deleted": self.deleted,
            "timeZone": self.time_zone,
            "primary": self.primary,
        }
