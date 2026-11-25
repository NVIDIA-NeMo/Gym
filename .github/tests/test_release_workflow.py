# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = ["pyyaml>=6.0,<7"]
# ///
"""Exercise release guards without credentials, uploads, or package builds."""

import json
import os
import re
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load(next((ROOT / ".github/workflows").glob("release.y*ml")).read_text())
JOBS = WORKFLOW["jobs"]
FREEZE_JOBS = yaml.safe_load((ROOT / ".github/workflows/release-freeze.yml").read_text())["jobs"]


def evaluate(expression: str, context: dict[str, object]) -> bool:
    """Evaluate the boolean and equality subset used by release job guards."""
    source = expression.strip().removeprefix("${{").removesuffix("}}")
    source = source.replace("!cancelled()", "True").replace("!failure()", "True")
    source = re.sub(r"\b(?:github|inputs|needs)\.[\w.-]+", lambda match: repr(context[match[0]]), source)
    source = source.replace("&&", " and ").replace("||", " or ")
    source = re.sub(r"!(?!=)", "not ", source)
    return bool(eval(" ".join(source.split()), {"__builtins__": {}}, {}))


class ReleaseGuards(unittest.TestCase):
    """Cover publication eligibility and unsuccessful or skipped dependencies."""

    def setUp(self) -> None:
        """Create a successful manual release context for the canonical repo."""
        self.context = {
            "github.repository": "NVIDIA-NeMo/Gym",
            "github.event_name": "workflow_dispatch",
            "inputs.dry-run": False,
            "needs.validate-inputs.result": "success",
            "needs.release-workflow-tests.result": "success",
            "needs.bump.result": "success",
            "needs.build-wheel.result": "success",
            "needs.publish-wheel.result": "success",
        }

    def test_publish_requires_manual_non_dry_run(self) -> None:
        """Pushes and dry runs cannot request the production upload job."""
        for event in ("push", "workflow_dispatch"):
            for dry_run in (False, True):
                with self.subTest(event=event, dry_run=dry_run):
                    context = dict(self.context, **{"github.event_name": event, "inputs.dry-run": dry_run})
                    self.assertEqual(
                        evaluate(JOBS["publish-wheel"]["if"], context), event == "workflow_dispatch" and not dry_run
                    )
        self.context["github.repository"] = "example/fork"
        self.assertFalse(evaluate(JOBS["publish-wheel"]["if"], self.context))

    def test_missing_build_blocks_upload(self) -> None:
        """Failed, cancelled and skipped builds cannot reach publication."""
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.build-wheel.result"] = state
                self.assertFalse(evaluate(JOBS["publish-wheel"]["if"], self.context))

    def test_regression_failures_block_version_preparation(self) -> None:
        """Require the regression job to succeed before version preparation."""
        self.assertIn("release-workflow-tests", JOBS["bump"]["needs"])
        self.assertTrue(evaluate(JOBS["bump"]["if"], self.context))
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.release-workflow-tests.result"] = state
                self.assertFalse(evaluate(JOBS["bump"]["if"], self.context))

    def test_real_finalize_requires_upload(self) -> None:
        """A missing upload cannot finalize a real release; dry runs may continue."""
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.publish-wheel.result"] = state
                self.context["inputs.dry-run"] = False
                self.assertFalse(evaluate(JOBS["finalize"]["if"], self.context))
                self.context["inputs.dry-run"] = True
                self.assertTrue(evaluate(JOBS["finalize"]["if"], self.context))

    def test_summary_rejects_missing_required_jobs(self) -> None:
        """Run the actual shell summary against failures and permitted skips."""
        script = JOBS["release-summary"]["steps"][0]["run"]
        for publish in (False, True):
            baseline = {name: {"result": "success"} for name in JOBS["release-summary"]["needs"]}
            baseline["publish-wheel"]["result"] = "success" if publish else "skipped"
            for job in (
                "release-workflow-tests",
                "validate-inputs",
                "bump",
                "build-wheel",
                "publish-wheel",
                "finalize",
            ):
                for state in ("success", "failure", "cancelled", "skipped"):
                    with self.subTest(publish=publish, job=job, state=state):
                        needs = {name: dict(value) for name, value in baseline.items()}
                        needs[job]["result"] = state
                        env = dict(
                            os.environ,
                            NEEDS=json.dumps(needs),
                            PUBLISH_EXPECTED=str(publish).lower(),
                            RELEASE_SKIPPED="false",
                        )
                        result = subprocess.run(
                            ["bash", "-euo", "pipefail", "-c", script], env=env, capture_output=True
                        )
                        expected = state == ("skipped" if job == "publish-wheel" and not publish else "success")
                        self.assertEqual(result.returncode == 0, expected, result.stderr.decode())

    def test_validate_ref_treats_input_as_data(self) -> None:
        """Only full immutable SHAs pass; branch names and shell payloads fail."""
        script = JOBS["validate-inputs"]["steps"][0]["run"]
        for ref in ("a" * 40, "main", "abc1234", "$(exit 0)", "a" * 40 + "\n"):
            with self.subTest(ref=ref):
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", script],
                    env=dict(os.environ, RELEASE_REF=ref),
                    capture_output=True,
                )
                self.assertEqual(result.returncode == 0, ref == "a" * 40)

    def test_upload_has_no_checkout_or_password(self) -> None:
        """Keep source execution and build credentials outside the OIDC job."""
        self.assertEqual(JOBS["build-wheel"]["permissions"], {"contents": "read"})
        self.assertNotIn("secrets", JOBS["build-wheel"])
        upload = JOBS["publish-wheel"]
        self.assertEqual(upload["permissions"], {"id-token": "write"})
        self.assertEqual(upload["environment"]["name"], "pypi")
        self.assertEqual(len(upload["steps"]), 2)
        for step in upload["steps"]:
            self.assertNotIn("run", step)
            self.assertNotIn("password", step.get("with", {}))
            self.assertNotIn("checkout", step["uses"])

    def test_dispatch_ref_guard_preserves_push_rehearsals(self) -> None:
        """Reject manual runs on non-main refs before the version bump."""
        guard = next(step for step in JOBS["validate-inputs"]["steps"] if step["name"] == "Require dispatch from main")
        for event in ("push", "workflow_dispatch"):
            for ref in ("refs/heads/main", "refs/heads/r0.5.0", "refs/heads/pull-request/123", "refs/tags/main"):
                with self.subTest(event=event, ref=ref):
                    rejected = evaluate(guard["if"], {"github.event_name": event, "github.ref": ref})
                    self.assertEqual(rejected, event == "workflow_dispatch" and ref != "refs/heads/main")
        self.assertNotEqual(subprocess.run(["bash", "-c", guard["run"]], capture_output=True).returncode, 0)


class ReleaseCommitGuards(unittest.TestCase):
    """Run the real source validation steps against local GitHub API fixtures."""

    target_sha = "2" * 40
    candidate = "3" * 40
    api = """
gh() {
  [[ $1 == api && $3 == --jq && $# == 4 ]] || return 1
  case "$2" in
    "repos/$GITHUB_REPOSITORY/branches/$TARGET_BRANCH") reply=$BRANCH_REPLY ;;
    "repos/$GITHUB_REPOSITORY/compare/$EXPECTED_COMPARE") reply=$COMPARE_REPLY ;;
    *) return 1 ;;
  esac
  [[ -n "$reply" ]] || return 1
  jq -r "$4" <<< "$reply"
}
"""

    def run_step(self, step, **inputs):
        """Execute a workflow step, returning its exit status and Actions outputs."""
        env = dict(
            os.environ,
            GITHUB_REPOSITORY="NVIDIA-NeMo/Gym",
            RELEASE_REF=self.candidate,
            TARGET_BRANCH="r0.7.0",
            BRANCH_REPLY=json.dumps({"protected": True, "commit": {"sha": self.target_sha}}),
            COMPARE_REPLY=json.dumps({"status": "ahead"}),
            EXPECTED_COMPARE=f"{self.candidate}...{self.target_sha}",
        )
        env.update(inputs)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.touch()
            env["GITHUB_OUTPUT"] = str(output)
            result = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", self.api + step["run"]], env=env, capture_output=True
            )
            return result.returncode, output.read_text()

    def test_release_target_ancestry(self):
        """Only ancestors of the protected target pass."""
        step = next(s for s in JOBS["validate-inputs"]["steps"] if s["name"].startswith("Verify release commit"))
        self.assertEqual(step["if"], "github.event_name == 'workflow_dispatch'")
        for status in ("ahead", "identical", "behind", "diverged"):
            with self.subTest(status=status):
                code, _ = self.run_step(step, COMPARE_REPLY=json.dumps({"status": status}))
                self.assertEqual(code == 0, status in {"ahead", "identical"})
        for bad_reply in ("", json.dumps({"protected": False, "commit": {"sha": self.target_sha}})):
            self.assertNotEqual(self.run_step(step, BRANCH_REPLY=bad_reply)[0], 0)
        self.assertNotEqual(self.run_step(step, COMPARE_REPLY="")[0], 0)

    def test_release_target_uses_the_maintainer_branch_rule(self):
        """Only main and branches covered by the release rule qualify as targets."""
        step = next(s for s in JOBS["validate-inputs"]["steps"] if s["name"].startswith("Verify release commit"))
        for branch in ("main", "r0.7.0", "v0.2.0"):
            with self.subTest(branch=branch):
                self.assertEqual(self.run_step(step, TARGET_BRANCH=branch)[0], 0)
        for branch in ("v1-train-client", "r0.7.0-preview", "r1x2x3", "topic/prepare-release"):
            with self.subTest(branch=branch):
                self.assertNotEqual(self.run_step(step, TARGET_BRANCH=branch)[0], 0)

    def test_freeze_uses_main_as_the_source_branch(self):
        """Gym opts into the shared workflow's cutoff policy for main."""
        inputs = FREEZE_JOBS["code-freeze"]["with"]
        self.assertEqual(inputs["freeze-source-branch"], "main")
        self.assertEqual(inputs["freeze-commit"], "${{ inputs.freeze-commit }}")


if __name__ == "__main__":
    unittest.main()
