# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Conversions API tools for AutomationBench.

Tools map 1:1 with Zapier LinkedIn Conversions actions.
"""

from automationbench.tools.zapier.linkedin_conversions.events import (
    linkedin_conversions_send_conversion_event,
)

__all__ = [
    "linkedin_conversions_send_conversion_event",
]
