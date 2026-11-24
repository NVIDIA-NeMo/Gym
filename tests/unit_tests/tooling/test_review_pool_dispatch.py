# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify webhook routing and serialization boundaries for review pools."""

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("prepare_review_pools", ROOT / ".gitlab/ci/prepare_review_pools.py")
DISPATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISPATCH)


def event(iid: object = 1355) -> dict:
    """Return an eligible MR update webhook for routing tests."""
    return {
        "object_kind": "merge_request",
        "project": {"id": 191584},
        "object_attributes": {
            "iid": iid,
            "target_project_id": 191584,
            "target_branch": "nv-internal-main",
            "action": "update",
            "state": "opened",
            "draft": False,
        },
    }


class ReviewPoolDispatchTests(unittest.TestCase):
    """Exercise the boundary between untrusted webhooks and child CI config."""

    def test_review_pool_jobs_block_on_failure_and_require_code_owners(self) -> None:
        """Every review-pool job blocks, and the child runs the pinned adapter in enforcing mode."""
        parent = yaml.safe_load((ROOT / ".gitlab/ci/review-pools.yml").read_text())
        child = yaml.safe_load((ROOT / ".gitlab/ci/review-pools-run.yml").read_text())
        self.assertFalse(parent["gym-review-pools-prepare"]["allow_failure"])
        self.assertFalse(parent["gym-review-pools"]["allow_failure"])
        self.assertFalse(child[".gym-review-pools"]["allow_failure"])
        script = "\n".join(child[".gym-review-pools"]["script"])
        self.assertIn("--require-codeowners", script)
        self.assertIn("--codeowners-fallback @nvidia-nemo/gym_maintainers", script)
        pin = re.search(r"fetch --depth=1 origin ([0-9a-f]{40})", script)
        self.assertIsNotNone(pin, "child job must fetch the adapter by full commit SHA")
        readme = (ROOT / ".gitlab/ci/README.md").read_text()
        self.assertIn(
            f"The GitLab adapter is pinned to\n`{pin.group(1)}`", readme, "README pin differs from the job pin"
        )

    def test_invalid_payload_cannot_create_child(self) -> None:
        """Reject malformed payloads without producing runnable child configuration."""
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "event.json"
            output = Path(directory) / "child.json"
            payload.write_text("{invalid")
            result = subprocess.run(
                [sys.executable, str(ROOT / ".gitlab/ci/prepare_review_pools.py"), str(payload), str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            payload.write_text(json.dumps(event()))
            subprocess.run(
                [sys.executable, str(ROOT / ".gitlab/ci/prepare_review_pools.py"), str(payload), str(output)],
                check=True,
            )
            self.assertEqual(
                json.loads(output.read_text())["gym-review-pools"]["variables"], {"GYM_REVIEW_MR_IID": "1355"}
            )

    def test_independent_mrs_do_not_share_a_lock(self) -> None:
        """Different MRs run concurrently; repeated events share one MR lock."""
        first = DISPATCH.child_pipeline(DISPATCH.review_mr(event(1355)))["gym-review-pools"]
        second = DISPATCH.child_pipeline(DISPATCH.review_mr(event(1364)))["gym-review-pools"]
        repeated = DISPATCH.child_pipeline(DISPATCH.review_mr(event(1355)))["gym-review-pools"]
        self.assertNotEqual(first["resource_group"], second["resource_group"])
        self.assertEqual(first["resource_group"], repeated["resource_group"])
        self.assertEqual(first["variables"], {"GYM_REVIEW_MR_IID": "1355"})

    def test_invalid_iids_cannot_inject_configuration(self) -> None:
        """Reject string, boolean, zero, negative, and missing MR identifiers."""
        for iid in ["1355", "1\nscript: malicious", True, 0, -1, None]:
            with self.subTest(iid=iid), self.assertRaises(ValueError):
                DISPATCH.review_mr(event(iid))

    def test_irrelevant_events_do_not_launch_a_credentialed_job(self) -> None:
        """Drafts, approvals, closed MRs, and other project/branch events are no-ops."""
        for key, value in [
            ("draft", True),
            ("action", "approved"),
            ("state", "merged"),
            ("target_project_id", 1),
            ("target_branch", "main"),
        ]:
            payload = event()
            payload["object_attributes"][key] = value
            with self.subTest(key=key):
                iid = DISPATCH.review_mr(payload)
                self.assertIsNone(iid)
                pipeline = DISPATCH.child_pipeline(iid)
                self.assertNotIn("include", pipeline)
                self.assertNotIn("environment", pipeline["review-pools-skipped"])

    def test_webhook_cannot_supply_job_code_or_variables(self) -> None:
        """Only the validated IID crosses into the trusted job template."""
        payload = event()
        expected = DISPATCH.child_pipeline(DISPATCH.review_mr(deepcopy(payload)))
        payload.update({"script": "malicious", "variables": {"GH_TOKEN": "injected"}})
        payload["object_attributes"]["source_branch"] = "malicious"
        self.assertEqual(DISPATCH.child_pipeline(DISPATCH.review_mr(payload)), expected)


if __name__ == "__main__":
    unittest.main()
