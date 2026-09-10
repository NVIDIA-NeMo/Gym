# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""HubSpot CRM tools for AutomationBench.

Tools map 1:1 with Zapier HubSpot actions.
"""

from automationbench.tools.zapier.hubspot.crm import (
    hubspot_add_contact_to_deal,
    hubspot_create_company,
    hubspot_create_contact,
    hubspot_create_deal,
    hubspot_create_engagement,
    hubspot_create_ticket,
    hubspot_find_contact,
    hubspot_get_all_companies,
    hubspot_get_all_contacts,
    hubspot_get_all_deals,
    hubspot_get_all_tickets,
    hubspot_update_contact,
    hubspot_update_deal,
    hubspot_update_ticket,
)

__all__ = [
    "hubspot_create_contact",
    "hubspot_update_contact",
    "hubspot_create_company",
    "hubspot_create_deal",
    "hubspot_update_deal",
    "hubspot_create_ticket",
    "hubspot_update_ticket",
    "hubspot_add_contact_to_deal",
    "hubspot_create_engagement",
    "hubspot_find_contact",
    "hubspot_get_all_companies",
    "hubspot_get_all_contacts",
    "hubspot_get_all_deals",
    "hubspot_get_all_tickets",
]
