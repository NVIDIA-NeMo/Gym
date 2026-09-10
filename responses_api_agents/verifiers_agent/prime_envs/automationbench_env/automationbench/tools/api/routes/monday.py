# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Monday.com API route handlers for api_fetch.

Dispatches REST-style Monday.com API requests to the appropriate impl tool functions.
Modeled after the Monday.com API v2 (https://developer.monday.com/api-reference).
"""

from automationbench.tools.api.impl.monday import (
    monday_boards_list,
    monday_items_create,
    monday_items_find,
    monday_items_update_column,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatcher
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"monday/v2/boards$", "boards_list"),
    ("POST", r"monday/v2/items:find$", "items_find"),
    ("POST", r"monday/v2/items:create$", "items_create"),
    ("POST", r"monday/v2/items/([^/:]+)/columns:update$", "items_update_column"),
]

_HANDLERS = {
    "boards_list": lambda w, ids, p, b: monday_boards_list(w),
    "items_find": lambda w, ids, p, b: monday_items_find(w, **b),
    "items_create": lambda w, ids, p, b: monday_items_create(w, **b),
    "items_update_column": lambda w, ids, p, b: monday_items_update_column(w, item_id=ids[0], **b),
}


route_monday = make_router(_ROUTES, _HANDLERS)
