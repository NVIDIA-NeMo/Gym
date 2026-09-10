# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Intercom support tools."""

from automationbench.tools.zapier.intercom.tickets import (
    intercom_add_note,
    intercom_add_tag_to_contact,
    intercom_add_tag_to_conversation,
    intercom_create_company,
    intercom_create_contact,
    intercom_create_conversation,
    intercom_create_ticket,
    intercom_find_company,
    intercom_find_contact,
    intercom_find_conversation,
    intercom_find_lead,
    intercom_find_or_create_company,
    intercom_find_or_create_lead,
    intercom_find_ticket,
    intercom_get_conversations,
    intercom_list_companies,
    intercom_list_contacts,
    intercom_remove_tag_from_contact,
    intercom_reply_to_conversation,
    intercom_send_message,
    intercom_tag_company,
    intercom_update_contact,
    intercom_update_lead,
    intercom_update_ticket,
)

__all__ = [
    # Ticket tools
    "intercom_create_ticket",
    "intercom_update_ticket",
    "intercom_find_ticket",
    # Conversation tools
    "intercom_create_conversation",
    "intercom_get_conversations",
    "intercom_find_conversation",
    "intercom_reply_to_conversation",
    "intercom_send_message",
    # Contact tools
    "intercom_create_contact",
    "intercom_find_contact",
    "intercom_list_contacts",
    "intercom_update_contact",
    "intercom_add_tag_to_contact",
    "intercom_add_tag_to_conversation",
    "intercom_remove_tag_from_contact",
    # Lead tools
    "intercom_find_lead",
    "intercom_find_or_create_lead",
    "intercom_update_lead",
    # Company tools
    "intercom_create_company",
    "intercom_find_company",
    "intercom_find_or_create_company",
    "intercom_list_companies",
    "intercom_tag_company",
    # Note tools
    "intercom_add_note",
]
