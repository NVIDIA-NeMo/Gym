# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""HelpScout support tools."""

from automationbench.tools.zapier.helpscout.conversations import (
    helpscout_add_note,
    helpscout_create_conversation,
    helpscout_create_customer,
    helpscout_find_conversation,
    helpscout_find_customer,
    helpscout_find_mailbox,
    helpscout_find_or_create_customer,
    helpscout_find_user,
    helpscout_get_conversation,
    helpscout_get_conversations,
    helpscout_get_customer,
    helpscout_get_customers,
    helpscout_get_mailboxes,
    helpscout_get_users,
    helpscout_send_reply,
    helpscout_update_conversation,
    helpscout_update_customer,
)

__all__ = [
    # Conversation tools
    "helpscout_create_conversation",
    "helpscout_update_conversation",
    "helpscout_get_conversation",
    "helpscout_get_conversations",
    "helpscout_find_conversation",
    "helpscout_send_reply",
    "helpscout_add_note",
    # Customer tools
    "helpscout_create_customer",
    "helpscout_get_customers",
    "helpscout_get_customer",
    "helpscout_find_customer",
    "helpscout_find_or_create_customer",
    "helpscout_update_customer",
    # Mailbox tools
    "helpscout_get_mailboxes",
    "helpscout_find_mailbox",
    # User tools
    "helpscout_get_users",
    "helpscout_find_user",
]
