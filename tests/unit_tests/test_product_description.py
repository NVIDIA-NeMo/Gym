# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Every place that says what NeMo Gym is uses the one description in ``nemo_gym/package_info.py``."""

import tomllib
from pathlib import Path

import pytest

from nemo_gym.cli.main import build_parser
from nemo_gym.package_info import __description__, __keywords__


REPO_ROOT = Path(__file__).resolve().parents[2]
OVERVIEW = "fern/versions/latest/pages/about/index.mdx"


def test_pypi_summary_is_the_package_description() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    assert project["description"] == __description__


def test_package_keywords_are_the_pypi_keywords() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    assert ", ".join(project["keywords"]) == __keywords__


@pytest.mark.parametrize(
    ("path", "line"),
    [
        ("AGENTS.md", __description__),
        (OVERVIEW, f'description: "{__description__}"'),
        ("docker/Dockerfile", f'LABEL org.opencontainers.image.description="{__description__}"'),
    ],
)
def test_file_states_the_package_description(path: str, line: str) -> None:
    assert line in (REPO_ROOT / path).read_text().splitlines()


def test_readme_and_docs_overview_open_with_the_same_paragraph() -> None:
    opening = [line for line in (REPO_ROOT / "README.md").read_text().splitlines() if line.startswith(__description__)]
    assert len(opening) == 1
    assert opening[0] in (REPO_ROOT / OVERVIEW).read_text().splitlines()


def _bullets(path: str, heading: str) -> list[str]:
    """The list items under the ``##`` heading whose title ends with ``heading``, without a trailing period."""
    bullets: list[str] = []
    in_section = False
    for line in (REPO_ROOT / path).read_text().splitlines():
        if line.startswith("## "):
            in_section = line.endswith(heading)
        elif in_section and line.startswith("- "):
            bullets.append(line.removesuffix("."))
    return bullets


@pytest.mark.parametrize("heading", ["When to Use NeMo Gym", "Used in Production", "What NeMo Gym Provides"])
def test_readme_and_docs_overview_list_the_same_items(heading: str) -> None:
    readme = _bullets("README.md", heading)
    assert readme, f"README.md has no list under {heading!r}"
    assert readme == _bullets(OVERVIEW, heading)


def test_root_help_states_the_package_description() -> None:
    assert build_parser().description == __description__
