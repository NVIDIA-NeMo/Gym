# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Ads tools for AutomationBench.

Tools map 1:1 with Zapier Google Ads actions.
"""

from automationbench.tools.zapier.google_ads.campaigns import (
    google_ads_find_campaign_by_id,
    google_ads_find_campaign_by_name,
    google_ads_get_all_campaigns,
    google_ads_set_campaign_status,
)
from automationbench.tools.zapier.google_ads.conversions import (
    google_ads_create_report,
    google_ads_send_offline_conversion,
    google_ads_send_offline_conversion_v2,
)
from automationbench.tools.zapier.google_ads.customer_lists import (
    google_ads_add_email_to_customer_list,
    google_ads_add_email_to_customer_list_v2,
    google_ads_add_email_to_customer_list_v3,
    google_ads_add_to_customer_list,
    google_ads_add_to_customer_list_v2,
    google_ads_add_to_customer_list_v3,
    google_ads_create_customer_list,
    google_ads_find_customer_list,
    google_ads_remove_from_customer_list,
    google_ads_remove_from_customer_list_v2,
)

__all__ = [
    # Customer list actions
    "google_ads_create_customer_list",
    "google_ads_find_customer_list",
    "google_ads_add_email_to_customer_list",
    "google_ads_add_email_to_customer_list_v2",
    "google_ads_add_email_to_customer_list_v3",
    "google_ads_add_to_customer_list",
    "google_ads_add_to_customer_list_v2",
    "google_ads_add_to_customer_list_v3",
    "google_ads_remove_from_customer_list",
    "google_ads_remove_from_customer_list_v2",
    # Campaign actions
    "google_ads_find_campaign_by_name",
    "google_ads_find_campaign_by_id",
    "google_ads_get_all_campaigns",
    "google_ads_set_campaign_status",
    # Conversion and report actions
    "google_ads_send_offline_conversion",
    "google_ads_send_offline_conversion_v2",
    "google_ads_create_report",
]
