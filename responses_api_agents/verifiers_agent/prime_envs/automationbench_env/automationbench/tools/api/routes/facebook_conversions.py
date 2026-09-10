# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Facebook Conversions API routing layer.

Dispatches incoming REST-style Facebook Conversions requests to their matching implementation functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.facebook_conversions import (
    facebook_conversions_events_create,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatch map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("POST", r"facebook/conversions/v25/([^/]+)/events$", "events_create"),
]

_HANDLERS = {
    "events_create": lambda w, ids, p, b: facebook_conversions_events_create(
        w, ids[0], **{**p, **b}
    ),
}


route_facebook_conversions = make_router(_ROUTES, _HANDLERS)
