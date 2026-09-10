# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Confluence Cloud REST API routing layer.

Dispatches incoming REST-style Confluence requests to their matching implementation functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.confluence import confluence_pages_create
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatch map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("POST", r"confluence/wiki/api/v2/pages$", "pages_create"),
]

_HANDLERS = {
    "pages_create": lambda w, ids, p, b: confluence_pages_create(w, **b),
}


route_confluence = make_router(_ROUTES, _HANDLERS)
