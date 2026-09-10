# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoom schema model definitions used by AutomationBench."""

from automationbench.schema.zoom.base import (
    ZoomRecord,
    ZoomState,
    generate_zoom_meeting_id,
    generate_zoom_uuid,
)
from automationbench.schema.zoom.meeting import Meeting
from automationbench.schema.zoom.participant import Participant
from automationbench.schema.zoom.recording import Recording
from automationbench.schema.zoom.registrant import Registrant
from automationbench.schema.zoom.webinar import Webinar

# Resolve forward references declared in ZoomState after all imports are complete
ZoomState.model_rebuild()

__all__ = [
    "ZoomRecord",
    "ZoomState",
    "generate_zoom_meeting_id",
    "generate_zoom_uuid",
    "Meeting",
    "Recording",
    "Registrant",
    "Participant",
    "Webinar",
]
