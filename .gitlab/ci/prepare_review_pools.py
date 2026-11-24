# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a trusted child pipeline with a lock scoped to one merge request."""

import json
import sys
from pathlib import Path


def review_mr(event: dict) -> int | None:
    """Return the MR IID for a supported webhook, ignoring unrelated events.

    Args:
        event: Decoded GitLab webhook payload.

    Returns:
        A positive MR IID, or None for events that need no reconciliation.
    """
    attributes = event.get("object_attributes", {})
    if (
        event.get("object_kind") != "merge_request"
        or event.get("project", {}).get("id") != 191584
        or attributes.get("target_project_id") != 191584
        or attributes.get("target_branch") != "nv-internal-main"
        or attributes.get("action") not in {"open", "reopen", "update"}
        or attributes.get("state") != "opened"
        or attributes.get("draft", attributes.get("work_in_progress", False))
    ):
        return None
    iid = attributes.get("iid")
    if type(iid) is not int or iid <= 0:
        raise ValueError("Webhook must contain a positive integer MR IID")
    return iid


def child_pipeline(iid: int | None) -> dict:
    """Build child configuration using only a validated numeric MR identifier.

    Args:
        iid: Validated MR IID, or None for a no-op pipeline.

    Returns:
        GitLab child configuration with a per-MR resource group.
    """
    pipeline = {
        "workflow": {
            "rules": [
                {
                    "if": '$CI_PIPELINE_SOURCE == "parent_pipeline" && '
                    '$CI_COMMIT_BRANCH == "nv-internal-main" && '
                    '$CI_COMMIT_REF_PROTECTED == "true"'
                }
            ]
        },
        "stages": ["test"],
    }
    if iid is None:
        pipeline["review-pools-skipped"] = {
            "image": "python:3.12-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b",
            "variables": {"GIT_STRATEGY": "empty"},
            "script": ["echo 'Webhook does not request review-pool reconciliation'"],
        }
        return pipeline
    pipeline["include"] = [{"local": ".gitlab/ci/review-pools-run.yml"}]
    pipeline["gym-review-pools"] = {
        "extends": ".gym-review-pools",
        "resource_group": f"gym-review-pools-191584-{iid}",
        "variables": {"GYM_REVIEW_MR_IID": str(iid)},
    }
    return pipeline


def main() -> None:
    """Read the webhook file and write a JSON child pipeline artifact."""
    payload, output = map(Path, sys.argv[1:])
    if payload.stat().st_size > 1024 * 1024:
        raise ValueError("Webhook payload exceeds one MiB")
    event = json.loads(payload.read_text())
    if not isinstance(event, dict):
        raise ValueError("Webhook payload must be an object")
    output.write_text(json.dumps(child_pipeline(review_mr(event)), indent=2) + "\n")


if __name__ == "__main__":
    main()
