# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""LinkedIn Post model."""

from typing import Literal, Optional

from pydantic import Field

from automationbench.schema.linkedin.base import (
    LinkedInRecord,
    generate_linkedin_post_id,
)


class PostContent(LinkedInRecord):
    """Media or link content attached to a LinkedIn post."""

    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    image_url: Optional[str] = None

    # Alternative field names from API (for parsing API responses)
    article_url: Optional[str] = Field(default=None, alias="articleUrl")
    article_title: Optional[str] = Field(default=None, alias="articleTitle")
    article_description: Optional[str] = Field(default=None, alias="articleDescription")


class Post(LinkedInRecord):
    """A LinkedIn post, share, or feed update."""

    id: str = Field(default_factory=generate_linkedin_post_id)
    # Author can be a person URN or company ID
    author_id: Optional[str] = None
    author_type: Literal["person", "company"] = "person"

    # Post content
    text: Optional[str] = None
    content: Optional[PostContent | str] = None

    # Visibility - update to include PUBLIC from API
    visibility: Literal["anyone", "connections-only", "PUBLIC"] = "anyone"

    # Lifecycle state from API
    lifecycle_state: str = Field(
        default="PUBLISHED", description="Publishing state of the post, e.g. PUBLISHED or DRAFT"
    )

    # Engagement metrics
    likes_count: int = 0
    comments_count: int = 0
    shares_count: int = 0
    views_count: int = 0

    # Engagement detail (arrays of engager info)
    likes: list[dict] = Field(default_factory=list)
    comments: list[dict] = Field(default_factory=list)

    # Timestamps - support both ISO and epoch
    created_at: Optional[int | str] = None  # Epoch millis or ISO datetime
    updated_at: Optional[int | str] = None  # Epoch millis or ISO datetime
    published_at: Optional[int | str] = None  # Epoch millis or ISO datetime

    # URL
    url: Optional[str] = None

    # Group association
    group_id: Optional[str] = None

    # Distribution
    is_reshare_disabled_by_author: bool = False

    # State
    is_deleted: bool = False

    @property
    def is_company_post(self) -> bool:
        """Return True when this post was authored by a company page."""
        return self.author_type == "company"
