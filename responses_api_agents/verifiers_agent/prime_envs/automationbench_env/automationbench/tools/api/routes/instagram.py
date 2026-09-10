# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Instagram Graph API request dispatchers for api_fetch.

Translates REST-style Instagram Content Publishing API calls into calls to the matching impl functions.
Modeled after the Instagram Graph API (https://developers.facebook.com/docs/instagram-api/guides/content-publishing).
"""

from automationbench.tools.api.impl.instagram import (
    instagram_create_container,
    instagram_get_by_id,
    instagram_media_publish,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTES = [
    ("POST", r"instagram/v25/([^/]+)/media$", "create_container"),
    ("POST", r"instagram/v25/([^/]+)/media_publish$", "media_publish"),
    ("GET", r"instagram/v25/([^/]+)$", "get_by_id"),
]

_HANDLERS = {
    "create_container": lambda w, ids, p, b: instagram_create_container(w, ids[0], **b),
    "media_publish": lambda w, ids, p, b: instagram_media_publish(w, ids[0], **b),
    "get_by_id": lambda w, ids, p, b: instagram_get_by_id(w, ids[0], fields=p.get("fields", "")),
}

route_instagram = make_router(_ROUTES, _HANDLERS)
