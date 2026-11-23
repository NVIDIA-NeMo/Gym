# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the agent's browser out of the mock hub's internals.

A hub app keeps its session state behind plain HTTP endpoints on its own origin:
``GET /state?sid=`` (current state), ``GET /go?sid=`` (initial and current state
plus their diff, which reward programs read), ``POST /post?sid=``, and its seed
data ships as JS bundles under ``/assets/`` (e.g. ``/assets/seed-issues_index-*.js``).
The app fetches all of these itself (fetch, XHR, module imports), so the browser
must be able to reach them. A *top-level* navigation to one of them, or
``view-source:`` of a hub page, is only ever the agent reading data the real site
never shows. A Qwen screen did this (E30); our checkpoints never did (0 of ~6,900
held-out episodes). ``file:`` URLs (the container's file system, which holds the
datasets with their reward programs) are blocked as well.

Only top-level navigations are checked; there is no request interception (a
Playwright route makes every request a paused CDP request and disables the HTTP
cache). The worker checks a URL before ``navigate``/``tabs_create`` load it, checks
the page after ``navigate`` back/forward, and checks every tab after a computer call
that could have opened such a URL (a typed URL, Ctrl+U, or Alt+Left/Right once a
blocked page is in the history). A blocked navigation is answered with a recoverable
``net::ERR_BLOCKED_BY_CLIENT`` tool message.

A tab on ``view-source:<url>`` reports only ``<url>`` to Playwright (``page.url`` has no
prefix), so for hub pages the worker also asks the document (``VIEW_SOURCE_DOCUMENT_JS``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote, urlparse

from resources_servers.cuagym.session_sid import HubScope


BLOCKED_ERROR = "net::ERR_BLOCKED_BY_CLIENT"

# Hub state endpoints (secureMockApiPlugin + each app's vite.config.js). Matched like connect's
# ``use(route)``: case-insensitive, the path itself or the path followed by "/" or ".".
STATE_ENDPOINTS = ("/go", "/state", "/post", "/state-inspector", "/_cua_session")
DATA_BUNDLE_SUFFIXES = (".js", ".mjs", ".cjs", ".map", ".json")
DEV_SERVER_PREFIXES = ("/@fs/", "/@id/", "/@vite/", "/src/", "/node_modules/")

_TYPED_URL_HINT = re.compile(
    r"view-source:|file:|/(?:go|state|post|state-inspector|_cua_session)\b"
    r"|/assets/|/@(?:fs|id|vite)/|/src/|/node_modules/",
    re.IGNORECASE,
)
_CTRL = {"ctrl", "control", "cmd", "command"}

VIEW_SOURCE_REASON = "view-source of a mock page"
# Chromium renders view-source:<url> as an HTMLViewSourceDocument: a body-level div.line-gutter-backdrop and a
# table of td.line-number / td.line-content cells. Playwright reports such a tab's URL as plain <url> (browser
# check 2026-10-18: a Ctrl+U tab read http://<hub>/explore?sid=...), so a URL check cannot see it.
VIEW_SOURCE_DOCUMENT_JS = (
    "() => !!(document.querySelector('body > div.line-gutter-backdrop')"
    " && document.querySelector('td.line-number + td.line-content'))"
)


def blocked_navigation(url: str, scope: HubScope) -> str | None:
    """Why a top-level navigation to ``url`` is blocked, or None if it is allowed."""
    text = (url or "").strip()
    lower = text.lower()
    if lower.startswith("view-source:"):
        inner = text[len("view-source:") :].strip()
        if inner.lower().startswith("file:"):
            return "view-source of a local file"
        return VIEW_SOURCE_REASON if scope.contains(inner) else None
    if lower.startswith("file:"):
        return "local file"
    if not scope.contains(text):
        return None
    try:
        path = unquote(urlparse(text).path or "/").lower()
    except ValueError:
        return None
    for endpoint in STATE_ENDPOINTS:
        if path == endpoint or path.startswith((endpoint + "/", endpoint + ".")):
            return f"mock state endpoint {endpoint}"
    if path.startswith("/assets/") and path.endswith(DATA_BUNDLE_SUFFIXES):
        return "mock data bundle"
    if path.startswith(DEV_SERVER_PREFIXES):
        return "mock dev-server path"
    return None


def computer_call_may_open_internals(arguments: Mapping[str, Any], *, blocked_page_in_history: bool) -> bool:
    """Cheap pre-filter on a computer call: could it have put a tab on a blocked URL?

    True for typed text that looks like such a URL, for Ctrl/Cmd+U (view source),
    and, once a blocked page is in a tab's history, for Alt+Left/Right.
    """
    for action in arguments.get("actions") or []:
        if not isinstance(action, Mapping):
            continue
        kind = action.get("action")
        if kind == "type" and _TYPED_URL_HINT.search(str(action.get("text") or "")):
            return True
        if kind == "key_press":
            keys = {str(key).strip().lower() for key in action.get("keys") or []}
            if "u" in keys and keys & _CTRL:
                return True
            history_key = ("alt" in keys and keys & {"left", "right"}) or keys & {"browserback", "browserforward"}
            if blocked_page_in_history and history_key:
                return True
    return False


def blocked_message(url: str) -> str:
    """The tool message for a blocked navigation (Playwright's wording for a client-blocked load)."""
    return f"Tool error: Error: Page.goto: {BLOCKED_ERROR} at {url}"
