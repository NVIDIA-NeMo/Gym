# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Routing layer for the Facebook Lead Ads API.

Dispatches incoming REST-style requests to the appropriate API-native tool functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.facebook_lead_ads import (
    facebook_lead_ads_ads_find,
    facebook_lead_ads_leadgen_forms_create,
    facebook_lead_ads_leads_list,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and handler map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    # Create a lead-gen form attached to a Page
    ("POST", r"facebook/lead_ads/v25/([^/]+)/leadgen_forms$", "leadgen_forms_create"),
    # Retrieve leads collected by a form or ad
    ("GET", r"facebook/lead_ads/v25/([^/]+)/leads$", "leads_list"),
    # Search for an ad by name inside an ad account (act_{adAccountId}/ads)
    ("GET", r"facebook/lead_ads/v25/act_([^/]+)/ads$", "ads_find"),
]

_HANDLERS = {
    "leadgen_forms_create": lambda w, ids, p, b: facebook_lead_ads_leadgen_forms_create(
        w, ids[0], **b
    ),
    "leads_list": lambda w, ids, p, b: facebook_lead_ads_leads_list(w, ids[0], **p),
    "ads_find": lambda w, ids, p, b: facebook_lead_ads_ads_find(w, ids[0], **p),
}


route_facebook_lead_ads = make_router(_ROUTES, _HANDLERS)
