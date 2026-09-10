# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""HelpCrunch data models for AutomationBench."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _generate_id() -> str:
    return str(uuid4())


class HelpCrunchCustomerEvent(BaseModel):
    """An event associated with a HelpCrunch customer record."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_generate_id)
    event_name: str = ""
    data: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "event_name": self.event_name,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
        }


class HelpCrunchCustomer(BaseModel):
    """A customer contact stored in HelpCrunch."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_generate_id)
    email: Optional[str] = None
    name: Optional[str] = None
    user_id: Optional[str] = None  # Identifier from an external system
    phone: Optional[str] = None
    company: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    custom_data: dict = Field(default_factory=dict)
    events: list[HelpCrunchCustomerEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "user_id": self.user_id,
            "phone": self.phone,
            "company": self.company,
            "tags": self.tags,
            "custom_data": self.custom_data,
            "events": [e.to_display_dict() for e in self.events],
            "created_at": self.created_at.isoformat(),
        }


class HelpCrunchState(BaseModel):
    """Top-level HelpCrunch workspace state."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    customers: list[HelpCrunchCustomer] = Field(default_factory=list)

    def get_customer_by_id(self, customer_id: str) -> Optional[HelpCrunchCustomer]:
        """Look up a customer using their unique identifier."""
        for customer in self.customers:
            if customer.id == customer_id:
                return customer
        return None

    def get_customer_by_email(self, email: str) -> Optional[HelpCrunchCustomer]:
        """Look up a customer using their email address."""
        for customer in self.customers:
            if customer.email == email:
                return customer
        return None

    def get_customer_by_user_id(self, user_id: str) -> Optional[HelpCrunchCustomer]:
        """Look up a customer using their external system user ID."""
        for customer in self.customers:
            if customer.user_id == user_id:
                return customer
        return None
