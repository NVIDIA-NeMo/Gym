# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Routing layer for the Gmail API.

Dispatches incoming REST-style requests to the appropriate API-native tool functions.
Parameters are forwarded as-is without any transformation.
"""

from automationbench.tools.api.impl.gmail import (
    gmail_drafts_create,
    gmail_drafts_delete,
    gmail_drafts_get,
    gmail_drafts_list,
    gmail_drafts_send,
    gmail_drafts_update,
    gmail_labels_create,
    gmail_labels_delete,
    gmail_labels_get,
    gmail_labels_list,
    gmail_labels_update,
    gmail_messages_delete,
    gmail_messages_get,
    gmail_messages_list,
    gmail_messages_modify,
    gmail_messages_send,
    gmail_messages_trash,
    gmail_messages_untrash,
    gmail_threads_delete,
    gmail_threads_get,
    gmail_threads_list,
    gmail_threads_modify,
    gmail_threads_trash,
    gmail_threads_untrash,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Route table and handler map
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    ("POST", r"gmail/v1/users/[^/]+/messages/send$", "messages_send"),
    ("POST", r"gmail/v1/users/[^/]+/messages/([^/]+)/trash$", "messages_trash"),
    ("POST", r"gmail/v1/users/[^/]+/messages/([^/]+)/untrash$", "messages_untrash"),
    ("POST", r"gmail/v1/users/[^/]+/messages/([^/]+)/modify$", "messages_modify"),
    ("GET", r"gmail/v1/users/[^/]+/messages/([^/]+)$", "messages_get"),
    ("DELETE", r"gmail/v1/users/[^/]+/messages/([^/]+)$", "messages_delete"),
    ("GET", r"gmail/v1/users/[^/]+/messages$", "messages_list"),
    ("POST", r"gmail/v1/users/[^/]+/drafts/send$", "drafts_send"),
    ("GET", r"gmail/v1/users/[^/]+/drafts/([^/]+)$", "drafts_get"),
    ("PUT", r"gmail/v1/users/[^/]+/drafts/([^/]+)$", "drafts_update"),
    ("DELETE", r"gmail/v1/users/[^/]+/drafts/([^/]+)$", "drafts_delete"),
    ("POST", r"gmail/v1/users/[^/]+/drafts$", "drafts_create"),
    ("GET", r"gmail/v1/users/[^/]+/drafts$", "drafts_list"),
    ("GET", r"gmail/v1/users/[^/]+/labels/([^/]+)$", "labels_get"),
    ("PUT", r"gmail/v1/users/[^/]+/labels/([^/]+)$", "labels_update"),
    ("PATCH", r"gmail/v1/users/[^/]+/labels/([^/]+)$", "labels_update"),
    ("DELETE", r"gmail/v1/users/[^/]+/labels/([^/]+)$", "labels_delete"),
    ("POST", r"gmail/v1/users/[^/]+/labels$", "labels_create"),
    ("GET", r"gmail/v1/users/[^/]+/labels$", "labels_list"),
    ("POST", r"gmail/v1/users/[^/]+/threads/([^/]+)/trash$", "threads_trash"),
    ("POST", r"gmail/v1/users/[^/]+/threads/([^/]+)/untrash$", "threads_untrash"),
    ("POST", r"gmail/v1/users/[^/]+/threads/([^/]+)/modify$", "threads_modify"),
    ("GET", r"gmail/v1/users/[^/]+/threads/([^/]+)$", "threads_get"),
    ("DELETE", r"gmail/v1/users/[^/]+/threads/([^/]+)$", "threads_delete"),
    ("GET", r"gmail/v1/users/[^/]+/threads$", "threads_list"),
]

_HANDLERS = {
    "messages_list": lambda w, ids, p, b: gmail_messages_list(w, **p),
    "messages_get": lambda w, ids, p, b: gmail_messages_get(w, ids[0], **p),
    "messages_send": lambda w, ids, p, b: gmail_messages_send(w, **b),
    "messages_delete": lambda w, ids, p, b: gmail_messages_delete(w, ids[0]),
    "messages_trash": lambda w, ids, p, b: gmail_messages_trash(w, ids[0]),
    "messages_untrash": lambda w, ids, p, b: gmail_messages_untrash(w, ids[0]),
    "messages_modify": lambda w, ids, p, b: gmail_messages_modify(w, ids[0], **b),
    "drafts_list": lambda w, ids, p, b: gmail_drafts_list(w, **p),
    "drafts_get": lambda w, ids, p, b: gmail_drafts_get(w, ids[0]),
    "drafts_create": lambda w, ids, p, b: gmail_drafts_create(w, **b),
    "drafts_update": lambda w, ids, p, b: gmail_drafts_update(w, ids[0], **b),
    "drafts_delete": lambda w, ids, p, b: gmail_drafts_delete(w, ids[0]),
    "drafts_send": lambda w, ids, p, b: gmail_drafts_send(w, **b),
    "labels_list": lambda w, ids, p, b: gmail_labels_list(w),
    "labels_get": lambda w, ids, p, b: gmail_labels_get(w, ids[0]),
    "labels_create": lambda w, ids, p, b: gmail_labels_create(w, **b),
    "labels_update": lambda w, ids, p, b: gmail_labels_update(w, ids[0], **b),
    "labels_delete": lambda w, ids, p, b: gmail_labels_delete(w, ids[0]),
    "threads_list": lambda w, ids, p, b: gmail_threads_list(w, **p),
    "threads_get": lambda w, ids, p, b: gmail_threads_get(w, ids[0]),
    "threads_trash": lambda w, ids, p, b: gmail_threads_trash(w, ids[0]),
    "threads_untrash": lambda w, ids, p, b: gmail_threads_untrash(w, ids[0]),
    "threads_delete": lambda w, ids, p, b: gmail_threads_delete(w, ids[0]),
    "threads_modify": lambda w, ids, p, b: gmail_threads_modify(w, ids[0], **b),
}


route_gmail = make_router(_ROUTES, _HANDLERS)
