# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Slack tools for AutomationBench.

Tools map 1:1 with Zapier Slack actions.
"""

from automationbench.tools.zapier.slack.channels import (
    slack_archive_conversation,
    slack_create_channel,
    slack_get_conversation,
    slack_get_conversation_members,
    slack_invite_to_channel,
    slack_list_channels,
    slack_set_channel_topic,
)
from automationbench.tools.zapier.slack.messaging import (
    slack_add_reaction,
    slack_delete_message,
    slack_edit_message,
    slack_send_channel_message,
    slack_send_direct_message,
)
from automationbench.tools.zapier.slack.search import (
    slack_find_message,
    slack_find_message_in_channel,
    slack_get_channel_messages,
    slack_get_message,
    slack_get_message_reactions,
    slack_get_thread_replies,
    slack_list_channel_messages,
)
from automationbench.tools.zapier.slack.users import (
    slack_find_user,
    slack_find_user_by_email,
    slack_find_user_by_id,
    slack_find_user_by_name,
    slack_find_user_by_username,
    slack_set_status,
)

__all__ = [
    # Channel actions
    "slack_create_channel",
    "slack_invite_to_channel",
    "slack_archive_conversation",
    "slack_set_channel_topic",
    "slack_get_conversation",
    "slack_get_conversation_members",
    "slack_list_channels",
    # Messaging actions
    "slack_send_channel_message",
    "slack_send_direct_message",
    "slack_edit_message",
    "slack_delete_message",
    "slack_add_reaction",
    # Search actions
    "slack_find_message",
    "slack_find_message_in_channel",
    "slack_list_channel_messages",
    "slack_get_channel_messages",
    "slack_get_message",
    "slack_get_message_reactions",
    "slack_get_thread_replies",
    # User actions
    "slack_find_user_by_email",
    "slack_find_user_by_id",
    "slack_find_user_by_name",
    "slack_find_user_by_username",
    "slack_find_user",
    "slack_set_status",
]
