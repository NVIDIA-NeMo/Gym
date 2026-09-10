# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Routing layer for the Google Drive API.

Dispatches incoming REST-style requests to the appropriate API-native tool functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.google_drive import (
    google_drive_files_copy,
    google_drive_files_create,
    google_drive_files_delete,
    google_drive_files_get,
    google_drive_files_list,
    google_drive_files_update,
)
from automationbench.utils.routing import make_router

_ROUTES: list[tuple[str, str, str]] = [
    # List or search files
    ("GET", r"drive/v[23]/files$", "files_list"),
    # Action paths are placed before the generic single-file pattern
    ("POST", r"drive/v[23]/files/([^/]+)/copy$", "files_copy"),
    # Standard file CRUD operations
    ("GET", r"drive/v[23]/files/([^/]+)$", "files_get"),
    ("PATCH", r"drive/v[23]/files/([^/]+)$", "files_update"),
    ("POST", r"drive/v[23]/files$", "files_create"),
    ("DELETE", r"drive/v[23]/files/([^/]+)$", "files_delete"),
]

_HANDLERS = {
    "files_list": lambda w, ids, p, b: google_drive_files_list(w, **{**p, **b}),
    "files_get": lambda w, ids, p, b: google_drive_files_get(w, ids[0], **p),
    "files_update": lambda w, ids, p, b: google_drive_files_update(w, ids[0], **{**p, **b}),
    "files_copy": lambda w, ids, p, b: google_drive_files_copy(w, ids[0], **{**p, **b}),
    "files_create": lambda w, ids, p, b: google_drive_files_create(w, **{**p, **b}),
    "files_delete": lambda w, ids, p, b: google_drive_files_delete(w, ids[0]),
}


route_google_drive = make_router(_ROUTES, _HANDLERS)
