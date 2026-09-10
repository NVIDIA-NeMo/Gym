# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Email data model representing outbound email records."""

from typing import Optional

from pydantic import Field

from automationbench.schema.salesforce.base import SalesforceRecord


class Email(SalesforceRecord):
    """Represents a Salesforce outbound email record."""

    to_address: str = Field(..., description="Primary recipient email address")
    cc_address: Optional[str] = Field(None, description="Carbon copy recipient address")
    bcc_address: Optional[str] = Field(None, description="Blind carbon copy recipient address")
    subject: str = Field(..., description="Subject line of the email")
    text_body: str = Field(default="", description="Plain-text version of the email body")
    html_body: Optional[str] = Field(None, description="HTML-formatted version of the email body")
    sender_type: str = Field(
        default="CurrentUser", description="Indicates who is listed as the sender"
    )
    related_to_id: Optional[str] = Field(
        None, description="ID of a Salesforce record this email is linked to"
    )

    def to_display_dict(self) -> dict:
        """Build a display-ready dict for agent consumption."""
        return {
            "Id": self.id,
            "ToAddress": self.to_address,
            "CcAddress": self.cc_address,
            "BccAddress": self.bcc_address,
            "Subject": self.subject,
            "TextBody": self.text_body,
            "HtmlBody": self.html_body,
        }
