# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoho Desk support tools."""

from automationbench.tools.zapier.zoho_desk.tickets import (
    zoho_desk_add_comment,
    zoho_desk_create_account,
    zoho_desk_create_contact,
    zoho_desk_create_ticket,
    zoho_desk_find_account,
    zoho_desk_find_contact,
    zoho_desk_find_or_create_contact,
    zoho_desk_find_ticket,
    zoho_desk_get_contacts,
    zoho_desk_get_tickets,
    zoho_desk_send_reply,
    zoho_desk_update_ticket,
)

__all__ = [
    # Ticket tools
    "zoho_desk_create_ticket",
    "zoho_desk_update_ticket",
    "zoho_desk_get_tickets",
    "zoho_desk_find_ticket",
    "zoho_desk_add_comment",
    "zoho_desk_send_reply",
    # Contact tools
    "zoho_desk_create_contact",
    "zoho_desk_get_contacts",
    "zoho_desk_find_contact",
    "zoho_desk_find_or_create_contact",
    # Account tools
    "zoho_desk_create_account",
    "zoho_desk_find_account",
]
