# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish package approval status into the exact GitLab merge-request pipeline."""

import importlib.util
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("review_approvals", ROOT / "scripts/ci/review_approvals.py")
approvals = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(approvals)


def head_pipeline(api: str, token: str, prefix: str, head: str) -> int:
    """Wait for the current MR pipeline without attaching status to another head."""
    for attempt in range(25):
        merge = approvals.api_json(api, token, prefix)
        if merge["sha"] != head:
            raise ValueError("Merge request head changed before pipeline attachment")
        pipeline = merge.get("head_pipeline") or {}
        if pipeline.get("sha") == head:
            return int(pipeline["id"])
        if attempt < 24:
            time.sleep(5)
    raise ValueError("No pipeline exists for the current merge-request head")


def run() -> int:
    """Run target-branch policy and update its job in the required MR pipeline."""
    project = int(os.environ["CI_PROJECT_ID"])
    event = json.loads(Path(os.environ["TRIGGER_PAYLOAD"]).read_text())
    if event.get("object_kind") != "merge_request" or event["project"]["id"] != project:
        raise ValueError("Expected a merge-request webhook for this project")
    attributes = event["object_attributes"]
    if attributes["target_branch"] != "nv-internal-main" or os.environ["CI_COMMIT_BRANCH"] != "nv-internal-main":
        raise ValueError("Approval checks must run from the protected target branch")
    number = int(attributes["iid"])
    api = os.environ["CI_API_V4_URL"]
    token = os.environ["GITLAB_TOKEN"]
    prefix = f"projects/{project}/merge_requests/{number}"
    merge = approvals.api_json(api, token, prefix)
    head = merge["sha"]
    if merge["state"] != "opened":
        print("Merge request is closed; no approval verdict is needed")
        return 0
    pipeline_id = head_pipeline(api, token, prefix, head)
    status_path = f"projects/{project}/statuses/{head}"
    payload = {
        "pipeline_id": pipeline_id,
        "name": "Package approvals",
        "state": "pending",
        "target_url": os.environ["CI_JOB_URL"],
    }
    approvals.api_json(api, token, status_path, method="POST", payload=payload)
    result = 2
    try:
        with tempfile.TemporaryDirectory(prefix="gym-approvals-") as directory:
            output = Path(directory) / "result.json"
            completed = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/ci/check-review-approvals.sh"),
                    "gitlab",
                    "--project-id",
                    str(project),
                    "--number",
                    str(number),
                    "--output",
                    str(output),
                ],
                check=False,
            )
            if completed.returncode == 0:
                if json.loads(output.read_text())["head"] != head:
                    raise ValueError("Merge request head changed during evaluation")
                result = 0
            else:
                result = completed.returncode
    finally:
        payload["state"] = "success" if result == 0 else "failed"
        approvals.api_json(api, token, status_path, method="POST", payload=payload)
    return result


if __name__ == "__main__":
    raise SystemExit(run())
