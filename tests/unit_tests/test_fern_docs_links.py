# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (REPO_ROOT / path).read_text()


class TestFernDocsLinks(unittest.TestCase):
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
        self.assertEqual(5, canonical_links)

    def test_v040_evaluation_tutorial_links_stay_in_the_frozen_version(self):
        pages = REPO_ROOT / "fern/versions/v0.4.0/pages"
        versionless_links = []
        versioned_links = 0

        for page in pages.rglob("*.mdx"):
            for line_number, line in enumerate(page.read_text().splitlines(), start=1):
                if re.search(r'(?:\]\(|href=")/evaluation-tutorials(?:[/#")])', line):
                    versionless_links.append(f"{page.relative_to(REPO_ROOT)}:{line_number}")
                versioned_links += line.count(
                    "/v0.4.0/tutorials/evaluation-tutorials"
                )

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
                            broken_links.append(
                                f"{page.relative_to(REPO_ROOT)}:{line_number}"
                            )
                        canonical_links += line.count(expected_target)

                self.assertEqual([], broken_links)
                self.assertEqual(2, canonical_links)

    def test_private_cli_compat_api_link_redirects_to_the_public_cli_page(self):
        redirects = read("fern/docs.yml")
        expected_redirect = """  - source: "/nemo/gym/nemo-gym/nemo_gym/cli/_compat"
    destination: "/nemo/gym/nemo-gym/nemo_gym/cli\""""

        self.assertIn(expected_redirect, redirects)


if __name__ == "__main__":
    unittest.main()
