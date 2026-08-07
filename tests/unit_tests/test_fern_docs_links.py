# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (REPO_ROOT / path).read_text()


class TestFernDocsLinks(unittest.TestCase):
    def test_model_call_capture_leads_with_a_copy_paste_consumer_workflow(self):
        for version in ("latest", "v0.5.0"):
            with self.subTest(version=version):
                guide = read(f"fern/versions/{version}/pages/model-server/model-call-capture.mdx")

                workflow = guide.index("## Run an evaluation with capture")
                author_guidance = guide.index("## If you build a custom agent")
                self.assertLess(workflow, author_guidance)
                self.assertIn("gym env start \\", guide)
                self.assertIn("++observability_enabled=true", guide)
                self.assertIn("++model_call_capture_dir=", guide)
                self.assertIn("gym eval run --no-serve \\", guide)
                self.assertIn("results/mcqa_rollouts.jsonl", guide)

    def test_model_call_capture_shows_the_rollout_attachment_shape(self):
        for version in ("latest", "v0.5.0"):
            guide = read(f"fern/versions/{version}/pages/model-server/model-call-capture.mdx")

            with self.subTest(version=version):
                self.assertIn('"ng_model_call_capture": {', guide)
                self.assertIn('"gaps": [', guide)
                for metric in (
                    "num_calls",
                    "tokens_in",
                    "tokens_out",
                    "tokens_reasoning",
                    "tokens_total",
                    "latency_total_ms",
                ):
                    self.assertIn(f'"{metric}":', guide)
                self.assertIn("<rollout_id>.capture.jsonl", guide)

    def test_model_call_capture_scopes_agent_observations_to_claude_code(self):
        for version in ("latest", "v0.5.0"):
            with self.subTest(version=version):
                guide = read(f"fern/versions/{version}/pages/model-server/model-call-capture.mdx")

                self.assertIn("Currently, only `claude_code_agent`", guide)
                self.assertIn("### Advanced: correlation, compaction, and sandbox evidence", guide)

    def test_model_server_index_links_model_call_capture(self):
        for version in ("latest", "v0.5.0"):
            with self.subTest(version=version):
                index = read(f"fern/versions/{version}/pages/model-server/index.mdx")

                self.assertIn('<Card title="Model-call capture" href="/model-server/model-call-capture">', index)

    def test_development_setup_uses_supported_cold_start_docs_command(self):
        guide = read("fern/versions/latest/pages/contribute/development-setup.mdx")

        self.assertNotIn("`fern docs dev` from the repository root", guide)
        self.assertIn("`make docs`", guide)
        self.assertIn(
            "For local documentation previews, ensure Node.js 22+ is installed.",
            guide,
        )
        self.assertNotIn("(including npm)", guide)
        self.assertIn("`make docs-login`", guide)
        self.assertIn(
            "https://github.com/NVIDIA-NeMo/Gym/blob/main/fern/README.md",
            guide,
        )

    def test_fern_tooling_uses_current_supported_node_version(self):
        package = json.loads(read("fern/package.json"))
        self.assertEqual(">=22", package["engines"]["node"])
        readme = read("fern/README.md")
        self.assertIn(
            "Install Node.js 22+, then install the Fern CLI globally",
            readme,
        )
        self.assertNotIn("(including npm)", readme)

        workflows = (
            ".github/workflows/fern-docs-ci.yml",
            ".github/workflows/fern-docs-preview-comment.yml",
            ".github/workflows/publish-fern-docs.yml",
        )
        for workflow in workflows:
            with self.subTest(workflow=workflow):
                self.assertIn("node-version: '22'", read(workflow))

    def test_prerequisites_disclose_default_quickstart_credentials(self):
        prerequisites = read("fern/versions/latest/pages/get-started/prerequisites.mdx")

        self.assertIn("## Quickstart Requirements", prerequisites)
        self.assertIn("OpenAI API key", prerequisites)
        self.assertIn("sufficient usage quota", prerequisites)
        self.assertIn("[Configure Models](/model-server)", prerequisites)
        self.assertIn("hosted provider or a local model", prerequisites)

    def test_main_evaluation_tutorial_links_include_the_tutorials_section(self):
        pages = REPO_ROOT / "fern/versions/latest/pages"
        broken_links = []
        canonical_links = 0

        for page in pages.rglob("*.mdx"):
            for line_number, line in enumerate(page.read_text().splitlines(), start=1):
                if re.search(r'(?:\]\(|href=")/evaluation-tutorials(?:[/#")])', line):
                    broken_links.append(f"{page.relative_to(REPO_ROOT)}:{line_number}")
                canonical_links += line.count("/tutorials/evaluation-tutorials")

        self.assertEqual([], broken_links)
        self.assertEqual(8, canonical_links)

    def test_v040_evaluation_tutorial_links_stay_in_the_frozen_version(self):
        pages = REPO_ROOT / "fern/versions/v0.4.0/pages"
        versionless_links = []
        versioned_links = 0

        for page in pages.rglob("*.mdx"):
            for line_number, line in enumerate(page.read_text().splitlines(), start=1):
                if re.search(r'(?:\]\(|href=")/evaluation-tutorials(?:[/#")])', line):
                    versionless_links.append(f"{page.relative_to(REPO_ROOT)}:{line_number}")
                versioned_links += line.count("/v0.4.0/tutorials/evaluation-tutorials")

        self.assertEqual([], versionless_links)
        self.assertEqual(5, versioned_links)

    def test_reward_profile_cards_use_the_generated_heading_anchor(self):
        for version in ("latest", "v0.4.0"):
            with self.subTest(version=version):
                pages = REPO_ROOT / f"fern/versions/{version}/pages/evaluation"
                broken_links = []
                canonical_links = 0
                expected_target = (
                    "/reference/cli-commands#gym-eval-profile"
                    if version == "latest"
                    else "/v0.4.0/reference/cli-commands#gym-eval-profile"
                )

                for page in pages.rglob("*.mdx"):
                    for line_number, line in enumerate(page.read_text().splitlines(), start=1):
                        if "#eval-profile" in line:
                            broken_links.append(f"{page.relative_to(REPO_ROOT)}:{line_number}")
                        canonical_links += line.count(expected_target)

                self.assertEqual([], broken_links)
                self.assertEqual(2, canonical_links)

    def test_internal_pages_are_linked_by_path_not_by_absolute_url(self):
        """An absolute docs.nvidia.com link pins the reader to the default version.

        Relative paths keep the reader inside the version they are already reading.
        Prose that names the domain without linking to it is fine.

        Scoped to `latest`. The frozen versions carry the same pattern but are left as
        they shipped.
        """
        pages = REPO_ROOT / "fern/versions/latest/pages"
        absolute_links = []

        for page in pages.rglob("*.mdx"):
            for line_number, line in enumerate(page.read_text().splitlines(), start=1):
                if re.search(r'(?:\]\(|href=")https?://docs\.nvidia\.com/nemo/gym', line):
                    absolute_links.append(f"{page.relative_to(REPO_ROOT)}:{line_number}")

        self.assertEqual([], absolute_links)

    def test_private_cli_compat_api_link_redirects_to_the_public_cli_page(self):
        redirects = read("fern/docs.yml")
        expected_redirect = """  - source: "/nemo/gym/nemo-gym/nemo_gym/cli/_compat"
    destination: "/nemo/gym/nemo-gym/nemo_gym/cli\""""

        self.assertIn(expected_redirect, redirects)


if __name__ == "__main__":
    unittest.main()
