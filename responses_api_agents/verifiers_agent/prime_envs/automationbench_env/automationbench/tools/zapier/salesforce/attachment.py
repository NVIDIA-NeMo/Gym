# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Attachment tools."""

import json
from typing import Optional

from automationbench.schema.salesforce import Attachment, generate_salesforce_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def salesforce_attachment_create(
    world: WorldState,
    File: str,
    Parent: str,
    Description: Optional[str] = None,
    IsPrivate: Optional[bool] = None,
) -> str:
    """
    Create a new Attachment in Salesforce.

    Args:
        File: File content (required)
        Parent: Parent record ID (required)
        Description: Attachment description
        IsPrivate: Whether attachment is private

    Returns:
        JSON string with created attachment or error.
    """
    attachment = Attachment(
        id=generate_salesforce_id(),
        File=File,
        Parent=Parent,
        Description=Description,
        IsPrivate=IsPrivate if IsPrivate is not None else False,
    )
    world.salesforce.attachments.append(attachment)
    return json.dumps({"success": True, "attachment": attachment.to_display_dict()})


register_metadata(
    salesforce_attachment_create,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_attachment",
        "type": "write",
        "action_id": "core:3079413",
    },
)
