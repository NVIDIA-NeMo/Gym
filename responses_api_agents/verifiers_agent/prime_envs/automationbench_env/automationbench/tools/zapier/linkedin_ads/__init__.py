# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Ads (Lead Gen Forms) tools for AutomationBench.

Tools map 1:1 with Zapier LinkedIn Lead Gen Forms actions.
"""

from automationbench.tools.zapier.linkedin_ads.audiences import (
    linkedin_ads_add_company_to_audience,
    linkedin_ads_add_contact_to_audience,
    linkedin_ads_create_audience,
    linkedin_ads_remove_email_from_audience,
)
from automationbench.tools.zapier.linkedin_ads.conversions import (
    linkedin_ads_create_conversion_event,
    linkedin_ads_create_report,
)

__all__ = [
    # Audience actions
    "linkedin_ads_create_audience",
    "linkedin_ads_add_contact_to_audience",
    "linkedin_ads_add_company_to_audience",
    "linkedin_ads_remove_email_from_audience",
    # Conversion and report actions
    "linkedin_ads_create_conversion_event",
    "linkedin_ads_create_report",
]
