# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Calendar tools for AutomationBench.

Tools map 1:1 with Zapier Google Calendar actions.
"""

from automationbench.tools.zapier.google_calendar.calendar import (
    google_calendar_create_calendar,
    google_calendar_find_calendars,
    google_calendar_get_calendar_info,
)
from automationbench.tools.zapier.google_calendar.event import (
    google_calendar_add_attendees,
    google_calendar_create_detailed_event,
    google_calendar_delete_event,
    google_calendar_find_event,
    google_calendar_get_event,
    google_calendar_get_event_by_id,
    google_calendar_quick_add_event,
    google_calendar_update_event,
)

__all__ = [
    # Calendar actions
    "google_calendar_create_calendar",
    "google_calendar_find_calendars",
    "google_calendar_get_calendar_info",
    # Event actions
    "google_calendar_quick_add_event",
    "google_calendar_create_detailed_event",
    "google_calendar_update_event",
    "google_calendar_delete_event",
    "google_calendar_find_event",
    "google_calendar_add_attendees",
    "google_calendar_get_event",
    "google_calendar_get_event_by_id",
]
