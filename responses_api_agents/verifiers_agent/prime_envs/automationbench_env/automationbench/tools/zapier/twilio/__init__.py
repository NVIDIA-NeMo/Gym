# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Twilio tools for AutomationBench.

Tools map 1:1 with Zapier Twilio actions.
"""

from automationbench.tools.zapier.twilio.calls import (
    twilio_make_call,
    twilio_make_call_legacy,
)
from automationbench.tools.zapier.twilio.messaging import (
    twilio_send_sms,
    twilio_send_sms_legacy,
    twilio_send_whatsapp,
)

__all__ = [
    # SMS actions
    "twilio_send_sms",
    "twilio_send_sms_legacy",
    # WhatsApp actions
    "twilio_send_whatsapp",
    # Call actions
    "twilio_make_call",
    "twilio_make_call_legacy",
]
