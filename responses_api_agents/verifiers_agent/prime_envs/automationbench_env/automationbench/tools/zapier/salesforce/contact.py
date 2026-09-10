# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Contact tools."""

import json
from typing import Optional

from automationbench.schema.salesforce import CampaignMember, Contact, generate_salesforce_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def salesforce_contact_create(
    world: WorldState,
    last_name: str,
    first_name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    account_id: Optional[str] = None,
    title: Optional[str] = None,
    lead_source: Optional[str] = None,
    mailing_street: Optional[str] = None,
    mailing_city: Optional[str] = None,
    mailing_state: Optional[str] = None,
    mailing_postal_code: Optional[str] = None,
    mailing_country: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """
    Create a new Contact in Salesforce.

    Args:
        last_name: Contact's last name (required)
        first_name: Contact's first name
        email: Email address
        phone: Phone number
        account_id: Related Account ID
        title: Job title
        lead_source: Source where contact originated
        mailing_street: Street address
        mailing_city: City
        mailing_state: State/Province
        mailing_postal_code: Zip/Postal code
        mailing_country: Country
        description: Additional notes

    Returns:
        JSON string with created contact or error.
    """
    contact = Contact(
        id=generate_salesforce_id(),
        last_name=last_name,
        first_name=first_name,
        email=email,
        phone=phone,
        account_id=account_id,
        title=title,
        lead_source=lead_source,
        mailing_street=mailing_street,
        mailing_city=mailing_city,
        mailing_state=mailing_state,
        mailing_postal_code=mailing_postal_code,
        mailing_country=mailing_country,
        description=description,
    )
    world.salesforce.contacts.append(contact)
    return json.dumps({"success": True, "contact": contact.to_display_dict()})


register_metadata(
    salesforce_contact_create,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_contact",
        "type": "write",
        "action_id": "core:3079407",
    },
)


def salesforce_contact_update(
    world: WorldState,
    id: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    mobile_phone: Optional[str] = None,
    fax: Optional[str] = None,
    title: Optional[str] = None,
    department: Optional[str] = None,
    account_id: Optional[str] = None,
    lead_source: Optional[str] = None,
    status: Optional[str] = None,
    assistant_name: Optional[str] = None,
    assistant_email: Optional[str] = None,
    mailing_street: Optional[str] = None,
    mailing_city: Optional[str] = None,
    mailing_state: Optional[str] = None,
    mailing_postal_code: Optional[str] = None,
    mailing_country: Optional[str] = None,
    description: Optional[str] = None,
    nda_status: Optional[str] = None,
) -> str:
    """
    Update an existing Contact in Salesforce.

    Args:
        id: Salesforce Contact ID to update
        first_name: New first name
        last_name: New last name
        email: New email address
        phone: New phone number
        mobile_phone: New mobile phone number
        fax: New fax number
        title: New job title
        department: New department
        account_id: New Account ID
        lead_source: New lead source
        status: Contact status (e.g., Active, Inactive)
        assistant_name: New assistant's name
        assistant_email: New assistant's email
        mailing_street: New street address
        mailing_city: New city
        mailing_state: New state/province
        mailing_postal_code: New zip/postal code
        mailing_country: New country
        description: New description

    Returns:
        JSON string with updated contact or error.
    """
    updates = {}
    # Allow empty strings to clear fields
    if first_name is not None:
        updates["first_name"] = first_name
    if last_name is not None:
        updates["last_name"] = last_name
    if email is not None:
        updates["email"] = email
    if phone is not None:
        updates["phone"] = phone
    if mobile_phone is not None:
        updates["mobile_phone"] = mobile_phone
    if fax is not None:
        updates["fax"] = fax
    if title is not None:
        updates["title"] = title
    if department is not None:
        updates["department"] = department
    if account_id is not None:
        updates["account_id"] = account_id
    if lead_source is not None:
        updates["lead_source"] = lead_source
    if status is not None:
        updates["status"] = status
    if assistant_name is not None:
        updates["assistant_name"] = assistant_name
    if assistant_email is not None:
        updates["assistant_email"] = assistant_email
    if mailing_street is not None:
        updates["mailing_street"] = mailing_street
    if mailing_city is not None:
        updates["mailing_city"] = mailing_city
    if mailing_state is not None:
        updates["mailing_state"] = mailing_state
    if mailing_postal_code is not None:
        updates["mailing_postal_code"] = mailing_postal_code
    if mailing_country is not None:
        updates["mailing_country"] = mailing_country
    if description is not None:
        updates["description"] = description
    if nda_status is not None:
        updates["nda_status"] = nda_status

    if not updates:
        return json.dumps({"error": "No fields to update"})

    contact = world.salesforce.update_record("contacts", id, updates)
    if contact is None:
        return json.dumps({"error": f"Contact with id '{id}' not found"})

    return json.dumps({"success": True, "contact": contact.to_display_dict()})


register_metadata(
    salesforce_contact_update,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "update_contact",
        "type": "write",
        "action_id": "core:3079408",
    },
)


def salesforce_contact_add_to_campaign(
    world: WorldState,
    CampaignId: Optional[str] = None,
    ContactId: Optional[str] = None,
    Status: Optional[str] = None,
    campaign_id: Optional[str] = None,
    contact_id: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """
    Add a Contact to a Campaign.

    Args:
        CampaignId: The Campaign ID to add the contact to
        ContactId: The Contact ID to add
        Status: Campaign member status (optional)

    Returns:
        JSON string with created campaign member or error.
    """
    CampaignId = CampaignId or campaign_id or ""
    ContactId = ContactId or contact_id or ""
    Status = Status or status
    # Verify campaign exists
    campaign = world.salesforce.get_by_id("campaigns", CampaignId)
    if campaign is None:
        return json.dumps({"error": f"Campaign with id '{CampaignId}' not found"})

    # Verify contact exists
    contact = world.salesforce.get_by_id("contacts", ContactId)
    if contact is None:
        return json.dumps({"error": f"Contact with id '{ContactId}' not found"})

    # Check if already a member
    for member in world.salesforce.campaign_members:
        if member.CampaignId == CampaignId and member.ContactId == ContactId:
            return json.dumps({"error": "Contact is already a member of this campaign"})

    campaign_member = CampaignMember(
        id=generate_salesforce_id(),
        CampaignId=CampaignId,
        ContactId=ContactId,
        Status=Status or "Sent",
    )
    world.salesforce.campaign_members.append(campaign_member)
    return json.dumps({"success": True, "campaign_member": campaign_member.to_display_dict()})


register_metadata(
    salesforce_contact_add_to_campaign,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "add_contact_to_campaign",
        "type": "write",
        "action_id": "core:3079412",
    },
)
