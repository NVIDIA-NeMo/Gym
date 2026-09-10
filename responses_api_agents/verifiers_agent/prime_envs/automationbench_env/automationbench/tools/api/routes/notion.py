# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Notion REST API route handlers for api_fetch.

Dispatches REST-style Notion API requests to the appropriate impl tool functions.
Modeled after the Notion API v1 (https://developers.notion.com/reference).
"""

from automationbench.tools.api.impl.notion import notion_pages_create, notion_pages_update
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("POST", r"notion/v1/pages$", "pages_create"),
    ("PATCH", r"notion/v1/pages/([^/]+)$", "pages_update"),
]

_HANDLERS = {
    "pages_create": lambda w, ids, p, b: notion_pages_create(w, **b),
    "pages_update": lambda w, ids, p, b: notion_pages_update(w, page_id=ids[0], **b),
}


route_notion = make_router(_ROUTES, _HANDLERS)
