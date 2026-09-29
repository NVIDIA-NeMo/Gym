# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline preview contract tests: python3 -m unittest discover -s .github/scripts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from fern_preview import CONTENT_ROOTS, FERN_VERSION, preview_url, stage_content, verify_metadata


class PreviewTests(unittest.TestCase):
    """Exercise preview provenance and content-only staging."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / "incoming"
        self.trusted = self.root / "trusted"
        self.destination = self.root / "preview"
        for root in (self.artifact, self.trusted):
            (root / "fern").mkdir(parents=True)
        for content_root in CONTENT_ROOTS:
            for root in (self.artifact, self.trusted):
                (root / content_root).mkdir(parents=True, exist_ok=True)
        (self.trusted / "fern/docs.yml").write_text("instances: []\n")
        (self.trusted / "fern/index.mdx").write_text("old content")
        self.repo = "NVIDIA-NeMo/Gym"
        self.run = {
            "status": "completed",
            "path": ".github/workflows/fern-docs-preview-build.yml",
            "id": 123,
            "run_attempt": 2,
            "event": "pull_request",
            "conclusion": "success",
            "repository": {"full_name": self.repo},
            "head_repository": {"full_name": self.repo},
            "head_sha": "a" * 40,
            "pull_requests": [{"number": 42}],
        }
        self.pr = {
            "number": 42,
            "state": "open",
            "head": {"repo": {"full_name": self.repo}, "sha": "a" * 40},
            "base": {"repo": {"full_name": self.repo}},
        }
        metadata = self.artifact / "preview-metadata"
        metadata.mkdir()
        for name, value in {"pr_number": "42", "head_sha": "a" * 40, "run_id": "123", "run_attempt": "2"}.items():
            (metadata / name).write_text(value + "\n")

    def test_normal_content_uses_trusted_configuration(self):
        (self.artifact / "fern/index.mdx").write_text("new content")
        (self.artifact / "fern/fern.config.json").write_text(
            json.dumps({"version": "file:./fixture-cli", "organization": "fixture"})
        )
        (self.artifact / "fern/package.json").write_text(
            json.dumps({"scripts": {"generate:library": "echo fixture", "sanitize:generated": "echo fixture"}})
        )
        (self.artifact / "fern/docs.yml").write_text("instances: [fixture]\n")
        (self.artifact / "fern/.npmrc").write_text("registry=https://example.invalid\n")
        (self.artifact / "fern/tool.js").write_text("throw new Error('fixture');")
        stage_content(self.artifact, self.trusted, self.destination)
        docs = self.destination / "fern"
        self.assertEqual((docs / "index.mdx").read_text(), "new content")
        self.assertEqual((docs / "docs.yml").read_text(), "instances: []\n")
        self.assertEqual(
            json.loads((docs / "fern.config.json").read_text()),
            {
                "version": FERN_VERSION,
                "organization": "nvidia",
            },
        )
        for name in ("package.json", ".npmrc", "tool.js"):
            self.assertFalse((docs / "fern" / name).exists())

    def test_publisher_uses_fixed_cli_and_isolated_artifact(self):
        workflow = (Path(__file__).parents[1] / "workflows/fern-docs-preview-comment.yml").read_text()
        self.assertIn(f"fern-api@{FERN_VERSION}", workflow)
        self.assertIn('"$RUNNER_TEMP/fern-cli/node_modules/.bin/fern"', workflow)
        self.assertIn("path: incoming", workflow)
        self.assertIn("ref: ${{ github.sha }}", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("github.event.workflow_run.head_repository.full_name == github.repository", workflow)
        self.assertIn("run-id: ${{ github.event.workflow_run.id }}\n", workflow)
        self.assertNotIn("npm run", workflow)
        self.assertNotIn("npx", workflow)
        self.assertNotIn("jq -r .version", workflow)
        self.assertNotIn("get-slug-for-file", workflow)

    def test_run_type_status_and_workflow_are_verified(self):
        for field, value in (
            ("event", "push"),
            ("status", "in_progress"),
            ("path", ".github/workflows/other.yml"),
            ("head_sha", "invalid"),
        ):
            run = copy.deepcopy(self.run)
            run[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_metadata(self.artifact, run, self.pr, self.repo)

    def test_deleted_pages_are_not_restored(self):
        stage_content(self.artifact, self.trusted, self.destination)
        self.assertFalse((self.destination / "fern/index.mdx").exists())

    def test_links_are_rejected(self):
        (self.artifact / "fern/page.mdx").symlink_to(self.trusted / "fern/index.mdx")
        with self.assertRaises(ValueError):
            stage_content(self.artifact, self.trusted, self.destination)

    def test_current_pr_metadata(self):
        self.assertEqual(verify_metadata(self.artifact, self.run, self.pr, self.repo), 42)

    def test_mismatched_metadata(self):
        for field in ("pr_number", "head_sha", "run_id", "run_attempt"):
            path = self.artifact / "preview-metadata" / field
            original = path.read_text()
            path.write_text("999\n")
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_metadata(self.artifact, self.run, self.pr, self.repo)
            path.write_text(original)

    def test_forks_stale_heads_closed_and_unassociated_prs(self):
        for mutation in ("fork", "stale", "closed", "unassociated"):
            run, pr = copy.deepcopy(self.run), copy.deepcopy(self.pr)
            if mutation == "fork":
                pr["head"]["repo"]["full_name"] = "example/Evaluator"
            elif mutation == "stale":
                pr["head"]["sha"] = "b" * 40
            elif mutation == "closed":
                pr["state"] = "closed"
            else:
                run["pull_requests"] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_metadata(self.artifact, run, pr, self.repo)

    def test_url_is_display_only_and_requires_fern_https(self):
        url = "https://example.docs.buildwithfern.com/nemo/evaluator"
        self.assertEqual(preview_url(f"Published docs to {url} (preview)"), url)
        for value in (
            "https://example.invalid",
            "https://a.docs.buildwithfern.com.evil.invalid",
            "https://user@a.docs.buildwithfern.com",
            "http://a.docs.buildwithfern.com",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                preview_url(f"Published docs to {value} (preview)")


if __name__ == "__main__":
    unittest.main()
