# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Sheets spreadsheet tools: create, find."""

import json
from typing import Optional

from automationbench.schema.google_sheets import Spreadsheet, Worksheet, generate_google_sheets_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def google_sheets_create_spreadsheet(
    world: WorldState,
    title: str,
    drive: Optional[str] = None,
    spreadsheet_to_copy: Optional[str] = None,
    headers: Optional[list[str]] = None,
) -> str:
    """
    Create a new spreadsheet.

    Args:
        title: Title of the new spreadsheet (required).
        drive: Google Drive location (My Drive or Shared Drive).
        spreadsheet_to_copy: Spreadsheet ID to duplicate.
        headers: Column headers for the new spreadsheet.

    Returns:
        JSON string with created spreadsheet details.
    """
    spreadsheet = Spreadsheet(
        id=generate_google_sheets_id(),
        title=title,
        drive=drive,
        spreadsheet_to_copy=spreadsheet_to_copy,
        headers=headers or [],
    )

    world.google_sheets.spreadsheets.append(spreadsheet)

    # Create a default worksheet (Sheet1) if headers provided
    if headers and not spreadsheet_to_copy:
        worksheet = Worksheet(
            id=generate_google_sheets_id(),
            spreadsheet_id=spreadsheet.id,
            title="Sheet1",
            headers=headers,
        )
        world.google_sheets.worksheets.append(worksheet)

    return json.dumps({"success": True, "spreadsheet": spreadsheet.to_display_dict()})


register_metadata(
    google_sheets_create_spreadsheet,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "create_spreadsheet",
        "type": "write",
        "action_id": "core:3115165",
    },
)


def google_sheets_get_spreadsheet_by_id(
    world: WorldState,
    spreadsheet: Optional[str] = None,
    spreadsheet_id: Optional[str] = None,
    id: Optional[str] = None,
    includeGridData: bool = True,
) -> str:
    """
    Get a spreadsheet by ID.

    Args:
        spreadsheet: Spreadsheet ID (required).
        spreadsheet_id: Alias for spreadsheet.
        id: Alias for spreadsheet.
        includeGridData: Whether to include grid data in response.

    Returns:
        JSON string with spreadsheet details.
    """
    spreadsheet = spreadsheet or spreadsheet_id or id or ""
    spreadsheet_obj = world.google_sheets.get_spreadsheet_by_id(spreadsheet)
    if spreadsheet_obj:
        result = spreadsheet_obj.to_display_dict()
        worksheets = world.google_sheets.get_worksheets_for_spreadsheet(spreadsheet)
        result["worksheets"] = [
            {"id": ws.id, "title": ws.title, "headers": ws.headers} for ws in worksheets
        ]
        if not includeGridData:
            # Return only properties, not grid data
            result.pop("cells", None)
        return json.dumps({"success": True, "spreadsheet": result})
    return json.dumps({"error": f"Spreadsheet with id '{spreadsheet}' not found"})


register_metadata(
    google_sheets_get_spreadsheet_by_id,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "get_spreadsheet_by_id",
        "type": "search",
        "action_id": "core:3115147",
    },
)
