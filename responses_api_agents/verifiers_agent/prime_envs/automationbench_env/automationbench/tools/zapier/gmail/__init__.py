# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Gmail tools for AutomationBench.

Tools map 1:1 with Zapier Gmail actions.
"""

from automationbench.tools.zapier.gmail.actions import (
    gmail_archive_email,
    gmail_mark_as_read,
    gmail_mark_as_unread,
    gmail_star_messages,
    gmail_trash_email,
)
from automationbench.tools.zapier.gmail.attachment import gmail_get_attachment
from automationbench.tools.zapier.gmail.draft import (
    gmail_create_draft,
    gmail_create_draft_reply,
    gmail_create_draft_v2,
)
from automationbench.tools.zapier.gmail.label import (
    gmail_add_label_to_email,
    gmail_create_label,
    gmail_remove_label_from_email,
    gmail_remove_thread_label,
)
from automationbench.tools.zapier.gmail.message import (
    gmail_find_email,
    gmail_get_email_by_id,
    gmail_list_emails,
    gmail_reply_to_email,
    gmail_send_email,
)
from automationbench.tools.zapier.gmail.thread import (
    gmail_get_thread,
    gmail_get_threads,
)

__all__ = [
    # Message actions
    "gmail_send_email",
    "gmail_reply_to_email",
    "gmail_find_email",
    "gmail_get_email_by_id",
    "gmail_list_emails",
    # Draft actions
    "gmail_create_draft",
    "gmail_create_draft_v2",
    "gmail_create_draft_reply",
    # Label actions
    "gmail_add_label_to_email",
    "gmail_remove_label_from_email",
    "gmail_create_label",
    "gmail_remove_thread_label",
    # Message state actions
    "gmail_mark_as_read",
    "gmail_mark_as_unread",
    "gmail_archive_email",
    "gmail_trash_email",
    "gmail_star_messages",
    # Thread actions
    "gmail_get_threads",
    "gmail_get_thread",
    # Attachment actions
    "gmail_get_attachment",
]
