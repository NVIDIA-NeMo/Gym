# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Model representing a Gmail email attachment."""

from typing import Optional

from pydantic import Field

from automationbench.schema.gmail.base import GmailRecord


class Attachment(GmailRecord):
    """A file attached to a Gmail message."""

    message_id: str = Field(..., description="ID of the message this attachment belongs to")
    filename: str = Field(..., description="Name of the attached file")
    mime_type: Optional[str] = Field(None, description="Content type of the attachment")
    size: int = Field(default=0, description="File size measured in bytes")
    data: Optional[str] = Field(None, description="Attachment content encoded as base64")

    def to_display_dict(self) -> dict:
        """Build the dictionary representation shown to agents."""
        return {
            "attachmentId": self.id,
            "messageId": self.message_id,
            "filename": self.filename,
            "mimeType": self.mime_type,
            "size": self.size,
        }
