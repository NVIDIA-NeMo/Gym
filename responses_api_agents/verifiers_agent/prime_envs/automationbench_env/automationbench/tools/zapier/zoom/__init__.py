# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoom tools for AutomationBench.

Tools map 1:1 with Zapier Zoom actions.
"""

from automationbench.tools.zapier.zoom.meeting import (
    zoom_create_meeting,
    zoom_create_meeting_registrant,
    zoom_create_webinar_registrant,
    zoom_find_meeting,
    zoom_find_meeting_participants,
    zoom_get_meeting_summary,
    zoom_get_recording,
    zoom_list_meetings,
    zoom_list_recordings,
    zoom_update_meeting,
)

__all__ = [
    # Meeting actions
    "zoom_find_meeting",
    "zoom_create_meeting",
    "zoom_update_meeting",
    "zoom_get_meeting_summary",
    # Recording actions
    "zoom_list_meetings",
    "zoom_list_recordings",
    "zoom_get_recording",
    # Registrant actions
    "zoom_create_meeting_registrant",
    "zoom_create_webinar_registrant",
    "zoom_find_meeting_participants",
]
