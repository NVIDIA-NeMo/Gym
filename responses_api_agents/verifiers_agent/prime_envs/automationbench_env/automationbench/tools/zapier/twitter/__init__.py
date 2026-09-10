# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Twitter/X tools for AutomationBench.

Tools map 1:1 with Zapier X/Twitter actions.
"""

from automationbench.tools.zapier.twitter.tweets import (
    twitter_find_tweet,
    twitter_like_tweet,
    twitter_post_tweet,
    twitter_retweet,
)
from automationbench.tools.zapier.twitter.users import (
    twitter_find_user,
    twitter_follow_user,
    twitter_send_direct_message,
)

__all__ = [
    # Tweet actions
    "twitter_post_tweet",
    "twitter_like_tweet",
    "twitter_retweet",
    "twitter_find_tweet",
    # User actions
    "twitter_find_user",
    "twitter_follow_user",
    "twitter_send_direct_message",
]
