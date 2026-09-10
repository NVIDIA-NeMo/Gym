# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Freshdesk support tools."""

from automationbench.tools.zapier.freshdesk.tickets import (
    freshdesk_add_note_to_ticket,
    freshdesk_create_company,
    freshdesk_create_contact,
    freshdesk_create_ticket,
    freshdesk_find_contact,
    freshdesk_find_ticket,
    freshdesk_get_contacts,
    freshdesk_get_tickets,
    freshdesk_update_ticket,
)

__all__ = [
    # Ticket tools
    "freshdesk_create_ticket",
    "freshdesk_update_ticket",
    "freshdesk_get_tickets",
    "freshdesk_find_ticket",
    "freshdesk_add_note_to_ticket",
    # Contact tools
    "freshdesk_create_contact",
    "freshdesk_get_contacts",
    "freshdesk_find_contact",
    # Company tools
    "freshdesk_create_company",
]
