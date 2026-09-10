# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Routing layer for the Facebook Pages Graph API.

Dispatches incoming REST-style Graph API requests to the appropriate API-native tool functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.facebook_pages import (
    facebook_pages_accounts_list,
    facebook_pages_feed_create,
    facebook_pages_feed_list,
    facebook_pages_photos_create,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and handler map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"facebook/v25/me/accounts$", "accounts_list"),
    ("POST", r"facebook/v25/([^/]+)/feed$", "feed_create"),
    ("GET", r"facebook/v25/([^/]+)/feed$", "feed_list"),
    ("POST", r"facebook/v25/([^/]+)/photos$", "photos_create"),
]

_HANDLERS = {
    "accounts_list": lambda w, ids, p, b: facebook_pages_accounts_list(w, **p),
    "feed_create": lambda w, ids, p, b: facebook_pages_feed_create(w, ids[0], **b),
    "feed_list": lambda w, ids, p, b: facebook_pages_feed_list(w, ids[0], **p),
    "photos_create": lambda w, ids, p, b: facebook_pages_photos_create(w, ids[0], **b),
}


route_facebook_pages = make_router(_ROUTES, _HANDLERS)
