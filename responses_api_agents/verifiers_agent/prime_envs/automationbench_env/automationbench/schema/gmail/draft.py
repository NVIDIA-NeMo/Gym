# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Model representing a Gmail draft message."""

from typing import TYPE_CHECKING

from pydantic import Field

from automationbench.schema.gmail.base import GmailRecord

if TYPE_CHECKING:
    pass


class Draft(GmailRecord):
    """A Gmail draft that wraps an unsent message.

    Drafts serve as a holding container for messages that have not been sent yet.
    """

    message_id: str = Field(..., description="ID of the underlying message stored in this draft")

    def to_display_dict(self) -> dict:
        """Build the dictionary representation shown to agents."""
        return {
            "id": self.id,
            "message": {
                "id": self.message_id,
            },
        }
