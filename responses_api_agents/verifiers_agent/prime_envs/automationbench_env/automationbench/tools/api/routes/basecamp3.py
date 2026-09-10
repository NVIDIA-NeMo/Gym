# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Basecamp3 REST API routing layer for api_fetch.

Dispatches incoming REST-style Basecamp3 requests to their matching implementation functions.
Modeled after the Basecamp3 API (https://github.com/basecamp/bc3-api).
"""

from automationbench.tools.api.impl.basecamp3 import basecamp3_todos_create
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatch map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    (
        "POST",
        r"basecamp3/([^/]+)/buckets/([^/]+)/todosets/([^/]+)/todolists/([^/]+)/todos$",
        "todos_create",
    ),
]

_HANDLERS = {
    "todos_create": lambda w, ids, p, b: basecamp3_todos_create(
        w, ids[0], ids[1], ids[2], ids[3], **b
    ),
}


route_basecamp3 = make_router(_ROUTES, _HANDLERS)
