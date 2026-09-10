# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Schema models for Google Calendar."""

from automationbench.schema.google_calendar.base import (
    GoogleCalendarRecord,
    GoogleCalendarState,
    generate_google_calendar_id,
)
from automationbench.schema.google_calendar.calendar import Calendar
from automationbench.schema.google_calendar.event import CalendarEvent

# Resolve forward references once all imports are complete
GoogleCalendarState.model_rebuild()

__all__ = [
    "GoogleCalendarRecord",
    "GoogleCalendarState",
    "generate_google_calendar_id",
    "Calendar",
    "CalendarEvent",
]
