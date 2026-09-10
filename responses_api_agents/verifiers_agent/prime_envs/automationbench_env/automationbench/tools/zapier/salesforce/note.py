# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Note tools."""

import json
from typing import Optional

from automationbench.schema.salesforce import Note, generate_salesforce_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def salesforce_note_create(
    world: WorldState,
    parent_id: str,
    custom_type: Optional[str] = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
    is_private: Optional[bool] = None,
) -> str:
    """
    Create a new Note in Salesforce (note_v2).

    Args:
        parent_id: Parent record ID (required)
        custom_type: Object type selector (account, contact, lead, opportunity)
        title: Note title
        body: Note body/content
        is_private: Whether note is private

    Returns:
        JSON string with created note or error.
    """
    note = Note(
        id=generate_salesforce_id(),
        title=title or "",
        parent_id=parent_id,
        custom_type=custom_type,
        body=body,
        is_private=is_private if is_private is not None else False,
    )
    world.salesforce.notes.append(note)
    return json.dumps({"success": True, "note": note.to_display_dict()})


register_metadata(
    salesforce_note_create,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_note",
        "type": "write",
        "action_id": "core:3079417",
    },
)
