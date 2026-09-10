# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Schema definitions for Gmail data models."""

from automationbench.schema.gmail.attachment import Attachment
from automationbench.schema.gmail.base import GmailRecord, GmailState, generate_gmail_id
from automationbench.schema.gmail.draft import Draft
from automationbench.schema.gmail.label import Label, get_default_labels
from automationbench.schema.gmail.message import Message

# Rebuild forward references after all imports
GmailState.model_rebuild()

__all__ = [
    "GmailRecord",
    "GmailState",
    "generate_gmail_id",
    "Attachment",
    "Message",
    "Label",
    "Draft",
    "get_default_labels",
]
