# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Pydantic model definitions representing the AutomationBench simulation world state."""

from automationbench.schema.google_calendar import (
    CalendarEvent,
    GoogleCalendarRecord,
    GoogleCalendarState,
)
from automationbench.schema.salesforce import (
    Account,
    Campaign,
    Case,
    Contact,
    Event,
    Lead,
    Note,
    Opportunity,
    SalesforceRecord,
    SalesforceState,
    Task,
)
from automationbench.schema.world import WorldMeta, WorldState

__all__ = [
    "Account",
    "CalendarEvent",
    "Campaign",
    "Case",
    "Contact",
    "Event",
    "GoogleCalendarRecord",
    "GoogleCalendarState",
    "Lead",
    "Note",
    "Opportunity",
    "SalesforceRecord",
    "SalesforceState",
    "Task",
    "WorldMeta",
    "WorldState",
]
