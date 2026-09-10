# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Ads API request dispatchers.

Forwards REST-style LinkedIn Marketing API (REST) calls to the corresponding API-native tool functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.linkedin_ads import (
    linkedin_ads_analytics_create,
    linkedin_ads_conversions_create,
    linkedin_ads_dmp_segment_companies_add,
    linkedin_ads_dmp_segment_users_add,
    linkedin_ads_dmp_segment_users_remove,
    linkedin_ads_dmp_segments_create,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    # Build a new audience segment under a given ad account
    ("POST", r"linkedin/ads/rest/adAccounts/([^/]+)/dmpSegments$", "dmp_segments_create"),
    # Enroll a user into a segment
    ("POST", r"linkedin/ads/rest/dmpSegments/([^/]+)/users$", "dmp_segment_users_add"),
    # Enroll a company into a segment
    ("POST", r"linkedin/ads/rest/dmpSegments/([^/]+)/companies$", "dmp_segment_companies_add"),
    # Remove a user or email address from a segment
    ("DELETE", r"linkedin/ads/rest/dmpSegments/([^/]+)/users$", "dmp_segment_users_remove"),
    # Record a conversion event
    ("POST", r"linkedin/ads/rest/conversions$", "conversions_create"),
    # Request an analytics or performance report
    ("POST", r"linkedin/ads/rest/adAnalytics$", "analytics_create"),
]

_HANDLERS = {
    "dmp_segments_create": lambda w, ids, p, b: linkedin_ads_dmp_segments_create(w, ids[0], **b),
    "dmp_segment_users_add": lambda w, ids, p, b: linkedin_ads_dmp_segment_users_add(
        w, ids[0], **{**p, **b}
    ),
    "dmp_segment_companies_add": lambda w, ids, p, b: linkedin_ads_dmp_segment_companies_add(
        w, ids[0], **{**p, **b}
    ),
    "dmp_segment_users_remove": lambda w, ids, p, b: linkedin_ads_dmp_segment_users_remove(
        w, ids[0], **{**p, **b}
    ),
    "conversions_create": lambda w, ids, p, b: linkedin_ads_conversions_create(w, **b),
    "analytics_create": lambda w, ids, p, b: linkedin_ads_analytics_create(w, **b),
}


route_linkedin_ads = make_router(_ROUTES, _HANDLERS)
