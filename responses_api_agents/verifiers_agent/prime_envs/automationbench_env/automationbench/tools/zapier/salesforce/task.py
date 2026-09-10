# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Task tools."""

import json
from datetime import datetime
from typing import Optional

from automationbench.schema.salesforce import Task, generate_salesforce_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def salesforce_task_create(
    world: WorldState,
    subject: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    due_date: Optional[str] = None,
    description: Optional[str] = None,
    contact_id: Optional[str] = None,
    who_id: Optional[str] = None,
    related_to_id: Optional[str] = None,
    what_id: Optional[str] = None,
    assigned_to_id: Optional[str] = None,
    owner_id: Optional[str] = None,
    activity_date: Optional[str] = None,
) -> str:
    """
    Create a new Task in Salesforce.

    Args:
        subject: Task subject (required)
        status: Task status (default: Not Started)
        priority: Task priority (default: Normal)
        due_date: Due date (ISO format)
        description: Task description
        contact_id: Related Contact or Lead ID (alias: who_id)
        who_id: Related Contact or Lead ID (Salesforce-style)
        related_to_id: Related Account, Opportunity, etc. (alias: what_id)
        what_id: Related Account, Opportunity, etc. (Salesforce-style)
        assigned_to_id: User ID of task assignee (alias: owner_id)
        owner_id: User ID of task assignee (Salesforce-style)

    Returns:
        JSON string with created task or error.
    """
    due_date = due_date or activity_date
    # Handle both snake_case and Salesforce-style parameter names
    effective_contact_id = contact_id or who_id
    effective_related_to_id = related_to_id or what_id
    effective_assigned_to_id = assigned_to_id or owner_id

    task = Task(
        id=generate_salesforce_id(),
        subject=subject or "",
        status=status or "Not Started",
        priority=priority or "Normal",
        due_date=datetime.fromisoformat(due_date) if due_date else None,
        description=description,
        contact_id=effective_contact_id,
        related_to_id=effective_related_to_id,
        assigned_to_id=effective_assigned_to_id,
        owner_id=effective_assigned_to_id,  # Set both for API compatibility
    )
    world.salesforce.tasks.append(task)
    return json.dumps({"success": True, "task": task.to_display_dict()})


register_metadata(
    salesforce_task_create,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_record",
        "type": "write",
        "action_id": "core:3079403",
    },
)
