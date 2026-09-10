# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Conversions API request dispatchers.

Forwards REST-style LinkedIn Conversions API (CAPI) calls to the corresponding API-native tool functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.linkedin_conversions import (
    linkedin_conversions_events_create,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("POST", r"linkedin/conversions/rest/conversionEvents$", "events_create"),
]

_HANDLERS = {
    "events_create": lambda w, ids, p, b: linkedin_conversions_events_create(w, **b),
}


route_linkedin_conversions = make_router(_ROUTES, _HANDLERS)
