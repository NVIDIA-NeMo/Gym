# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Folder tools."""

import json
from typing import Optional

from automationbench.schema.salesforce import Folder, generate_salesforce_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def salesforce_folder_find_or_create(
    world: WorldState,
    search_field: str,
    search_value: str,
    Name: Optional[str] = None,
    Type: Optional[str] = None,
    AccessType: Optional[str] = None,
) -> str:
    """
    Find a Folder, or create one if not found.

    Args:
        search_field: Field to search by (required)
        search_value: Value to search for (required)
        Name: Folder name for creation (defaults to search_value if searching by Name)
        Type: Folder type for creation
        AccessType: Access type for creation

    Returns:
        JSON with folder and whether it was created or found.
    """
    folders = world.salesforce.find_by_field("folders", search_field, search_value)
    if folders:
        return json.dumps({"found": True, "folder": folders[0].to_display_dict()})

    # Create new folder
    folder_name = Name if Name else (search_value if search_field == "Name" else "New Folder")

    folder = Folder(
        id=generate_salesforce_id(),
        Name=folder_name,
        Type=Type,
        AccessType=AccessType or "Public",
    )
    # Set the search field value if it's a valid field
    if hasattr(folder, search_field):
        setattr(folder, search_field, search_value)

    world.salesforce.folders.append(folder)
    return json.dumps({"created": True, "folder": folder.to_display_dict()})


register_metadata(
    salesforce_folder_find_or_create,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "find_record",
        "type": "search_or_write",
        "action_id": "core:3079390",
    },
)
