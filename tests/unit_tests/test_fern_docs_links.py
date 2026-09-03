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

    def test_model_call_capture_labels_the_raw_record_as_synthetic(self):
        for version in ("latest", "v0.5.0"):
            with self.subTest(version=version):
                guide = read(f"fern/versions/{version}/pages/model-server/model-call-capture.mdx")

                self.assertIn("The following synthetic record shows the exact persisted field shape.", guide)

    def test_model_call_capture_routes_agent_observations_to_the_rollout_evidence_guide(self):
        guide = read("fern/versions/latest/pages/model-server/model-call-capture.mdx")

        self.assertNotIn("Currently, only `claude_code_agent`", guide)
        self.assertIn("/main/observability/rollout-evidence", guide)

    def test_observability_index_routes_both_observability_families(self):
        overview = read("fern/versions/latest/pages/observability/index.mdx")

        self.assertIn("Rollout evidence", overview)
        self.assertIn("OpenTelemetry", overview)
        self.assertIn("`observability_enabled`", overview)
        self.assertIn("`telemetry.enabled`", overview)
        self.assertIn('href="/main/observability/rollout-evidence"', overview)
        self.assertIn('href="/main/observability/opentelemetry"', overview)

    def test_rollout_evidence_guide_explains_the_complete_pipeline(self):
        path = REPO_ROOT / "fern/versions/latest/pages/observability/rollout-evidence.mdx"
        self.assertTrue(path.is_file(), "the task-oriented rollout evidence guide must exist")
        guide = path.read_text()

        for term in (
            "`ng_model_call_capture`",
            "`ng_agent_observations`",
            "`ng_trajectory`",
            "`ng_perf`",
            "`CaptureStore`",
            "`model_call_id`",
            "`model_ref`",
            "`response_id`",
            "`model_call_reference_unmatched`",
            "`model_call_reference_ambiguous`",
            "`model_call_reference_conflict`",
        ):
            with self.subTest(term=term):
                self.assertIn(term, guide)
        self.assertIn("flowchart LR", guide)
        self.assertIn("canonical normalized", guide)

    def test_rollout_evidence_guide_has_a_complete_joined_example(self):
        path = REPO_ROOT / "fern/versions/latest/pages/observability/rollout-evidence.mdx"
        self.assertTrue(path.is_file(), "the task-oriented rollout evidence guide must exist")
        guide = path.read_text()
        match = re.search(
            r"\{/\* joined-rollout-example:start \*/\}\s*```json\s*(.*?)\s*```\s*"
            r"\{/\* joined-rollout-example:end \*/\}",
            guide,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match, "the joined rollout example must be marked and valid JSON")
        example = json.loads(match.group(1))

        self.assertIn("ng_model_call_capture", example)
        self.assertIn("ng_agent_observations", example)
        trajectory = example["ng_trajectory"]
        self.assertEqual("1.0", trajectory["schema_version"])
        self.assertTrue(trajectory["invocations"])
        self.assertTrue(trajectory["model_calls"])
        self.assertTrue(trajectory["tool_calls"])
        self.assertTrue(trajectory["gaps"])
        self.assertTrue(trajectory["invocations"][0]["model_calls"])
        self.assertEqual(1.0, example["ng_perf"]["token_observability_coverage"])

    def test_rollout_evidence_joined_example_matches_runtime_schemas(self):
        page = REPO_ROOT / "fern/versions/latest/pages/observability/rollout-evidence.mdx"
        text = page.read_text(encoding="utf-8")
        match = re.search(
            r"\{/\* joined-rollout-example:start \*/\}\s*```json\s*(.*?)\s*```\s*"
            r"\{/\* joined-rollout-example:end \*/\}",
            text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        example = json.loads(match.group(1))

        observations = example["ng_agent_observations"]
        invocation = next(record for record in observations["records"] if record["kind"] == "agent_invocation")
        tool = next(record for record in observations["records"] if record["kind"] == "tool_call")
        self.assertEqual({"source", "records", "gaps"}, set(observations))
        self.assertIn("invocation_id", invocation)
        self.assertNotIn("output", tool, "agent-side tool timing does not own model-visible output")

        trajectory = example["ng_trajectory"]
        turn = trajectory["turns"][0]
        self.assertTrue({"task_id", "rollout_id", "timestamp"}.issubset(turn))

    def test_opentelemetry_has_an_explicit_landing_page(self):
        path = REPO_ROOT / "fern/versions/latest/pages/observability/opentelemetry.mdx"
        self.assertTrue(path.is_file(), "OpenTelemetry guidance must remain available from an explicit landing page")
        guide = path.read_text()

        self.assertIn("nemo-lens", guide)
        self.assertIn("NEMO_GYM_OTEL_EXPORTER", guide)
        self.assertIn("/main/observability/configuration", guide)

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

    def test_main_training_tutorial_links_include_the_tutorials_section(self):
        """`training-tutorials` sits under the same `section: Tutorials` as `evaluation-tutorials`.

        Both therefore publish under `/tutorials/`. `/training-tutorials/...` still resolves
        via a 308 from Fern, but the redirect is not something to rely on, and having the two
        sibling folders linked differently is what makes the prefix look optional.

        No exact link count here: there are enough of these that pinning a total would just
        mean editing this test every time a tutorial gains a cross-reference.
        """
        pages = REPO_ROOT / "fern/versions/latest/pages"
        redirected_links = []
        canonical_links = 0

        for page in pages.rglob("*.mdx"):
            for line_number, line in enumerate(page.read_text().splitlines(), start=1):
                if re.search(r'(?:\]\(|href=")/training-tutorials(?:[/#")])', line):
                    redirected_links.append(f"{page.relative_to(REPO_ROOT)}:{line_number}")
                canonical_links += line.count("/tutorials/training-tutorials")

        self.assertEqual([], redirected_links)
        self.assertGreater(canonical_links, 0)

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
