# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zendesk support tools."""

from automationbench.tools.zapier.zendesk.tickets import (
    zendesk_add_comment_to_ticket,
    zendesk_add_tags_to_ticket,
    zendesk_create_organization,
    zendesk_create_ticket,
    zendesk_create_user,
    zendesk_delete_user,
    zendesk_find_group,
    zendesk_find_or_create_organization,
    zendesk_find_or_create_user,
    zendesk_find_organization,
    zendesk_find_ticket,
    zendesk_find_user,
    zendesk_get_groups,
    zendesk_get_organizations,
    zendesk_get_ticket,
    zendesk_get_ticket_comments,
    zendesk_get_tickets,
    zendesk_get_users,
    zendesk_remove_tags_from_ticket,
    zendesk_update_organization,
    zendesk_update_ticket,
    zendesk_update_user,
)

__all__ = [
    # Ticket tools
    "zendesk_create_ticket",
    "zendesk_update_ticket",
    "zendesk_get_ticket",
    "zendesk_get_tickets",
    "zendesk_find_ticket",
    "zendesk_add_comment_to_ticket",
    "zendesk_get_ticket_comments",
    "zendesk_add_tags_to_ticket",
    "zendesk_remove_tags_from_ticket",
    # User tools
    "zendesk_create_user",
    "zendesk_update_user",
    "zendesk_delete_user",
    "zendesk_get_users",
    "zendesk_find_user",
    "zendesk_find_or_create_user",
    # Organization tools
    "zendesk_create_organization",
    "zendesk_update_organization",
    "zendesk_get_organizations",
    "zendesk_find_organization",
    "zendesk_find_or_create_organization",
    # Group tools
    "zendesk_get_groups",
    "zendesk_find_group",
]
