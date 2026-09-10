# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Schema definitions for Slack data models."""

from automationbench.schema.slack.base import (
    SlackRecord,
    SlackState,
    generate_slack_channel_id,
    generate_slack_message_ts,
    generate_slack_user_id,
)
from automationbench.schema.slack.channel import Channel
from automationbench.schema.slack.message import Message, Reaction
from automationbench.schema.slack.user import User

# Rebuild forward references after all imports
SlackState.model_rebuild()

__all__ = [
    "SlackRecord",
    "SlackState",
    "generate_slack_channel_id",
    "generate_slack_message_ts",
    "generate_slack_user_id",
    "Channel",
    "Message",
    "Reaction",
    "User",
]
