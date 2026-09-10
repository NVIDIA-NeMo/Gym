# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Mailchimp tools for AutomationBench.

Tools map 1:1 with Zapier Mailchimp actions.
"""

from automationbench.tools.zapier.mailchimp.subscribers import (
    mailchimp_add_subscriber,
    mailchimp_add_subscriber_note,
    mailchimp_add_tag_to_subscriber,
    mailchimp_archive_subscriber,
    mailchimp_create_audience,
    mailchimp_create_campaign,
    mailchimp_create_tag,
    mailchimp_find_subscriber,
    mailchimp_list_subscribers,
    mailchimp_remove_tag_from_subscriber,
    mailchimp_send_campaign,
)

__all__ = [
    "mailchimp_add_subscriber",
    "mailchimp_create_audience",
    "mailchimp_create_campaign",
    "mailchimp_send_campaign",
    "mailchimp_create_tag",
    "mailchimp_add_tag_to_subscriber",
    "mailchimp_remove_tag_from_subscriber",
    "mailchimp_archive_subscriber",
    "mailchimp_add_subscriber_note",
    "mailchimp_find_subscriber",
    "mailchimp_list_subscribers",
]
