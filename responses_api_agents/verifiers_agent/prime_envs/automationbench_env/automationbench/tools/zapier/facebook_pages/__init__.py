# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Facebook Pages tools for AutomationBench.

Tools map 1:1 with Zapier Facebook Pages actions.
"""

from automationbench.tools.zapier.facebook_pages.posts import (
    facebook_pages_change_profile_picture,
    facebook_pages_create_photo,
    facebook_pages_create_post,
    facebook_pages_create_video,
    facebook_pages_post_insights,
)

__all__ = [
    "facebook_pages_create_post",
    "facebook_pages_create_photo",
    "facebook_pages_create_video",
    "facebook_pages_change_profile_picture",
    "facebook_pages_post_insights",
]
