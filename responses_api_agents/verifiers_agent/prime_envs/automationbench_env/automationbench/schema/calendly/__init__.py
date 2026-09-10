# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Calendly schema model definitions used by AutomationBench."""

from automationbench.schema.calendly.base import (
    CalendlyRecord,
    CalendlyState,
    generate_calendly_uri,
    generate_calendly_uuid,
)
from automationbench.schema.calendly.event import ScheduledEvent
from automationbench.schema.calendly.event_type import EventType
from automationbench.schema.calendly.invitee import Invitee, QuestionAnswer
from automationbench.schema.calendly.user import User

# Resolve forward references declared in CalendlyState after all imports are complete
CalendlyState.model_rebuild()

__all__ = [
    # Base
    "CalendlyRecord",
    "CalendlyState",
    "generate_calendly_uuid",
    "generate_calendly_uri",
    # Models
    "User",
    "EventType",
    "ScheduledEvent",
    "Invitee",
    "QuestionAnswer",
]
