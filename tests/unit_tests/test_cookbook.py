# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cookbook's recipe contract: every recipe is indexed, shows its expected results, has resolving links,
is run by a test, and its tasks load with a solution."""

import re
from pathlib import Path

import pytest

from nemo_gym.data.tasks.harbor.dockerfile import DockerfileNeedsBuild
from nemo_gym.data.tasks.harbor.errors import HarborTaskError
from nemo_gym.data.tasks.harbor.parameterized import is_template_folder, render_template_folder
from nemo_gym.data.tasks.harbor.task import discover_tasks, is_task_folder, iter_task_folders


COOKBOOK = Path(__file__).resolve().parents[2] / "cookbook"
INDEX = COOKBOOK / "README.md"

# A recipe is a directory two levels down (cookbook/<group>/<recipe>/) with its own README.
RECIPES = sorted(readme.parent for readme in COOKBOOK.glob("*/*/README.md"))
MARKDOWN = sorted(COOKBOOK.rglob("*.md"))
SOLUTION_FILES = ("solve.sh", "answer.md")
EXPECTED_RESULTS_HEADING = "## Check the result"

# Inline Markdown links, skipping images, external URLs, and pure in-page anchors.
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")


def _relative_links(markdown: Path) -> list[str]:
    targets = LINK.findall(markdown.read_text())
    return [t.split("#", 1)[0] for t in targets if not re.match(r"[a-z]+://|#|mailto:", t)]


def _holds_tasks(recipe: Path) -> bool:
    """Whether the recipe is a task, a parameterized template, or a dataset of task folders."""
    return is_task_folder(recipe) or is_template_folder(recipe) or bool(iter_task_folders(recipe))


def _section(markdown: str, heading: str) -> str | None:
    """The body of a level-2 section, up to the next level-2 heading, or None when the heading is absent."""
    lines = markdown.splitlines()
    if heading not in lines:
        return None
    body = lines[lines.index(heading) + 1 :]
    end = next((i for i, line in enumerate(body) if line.startswith("## ")), len(body))
    return "\n".join(body[:end])


def test_cookbook_has_recipes():
    assert RECIPES, f"no recipes found under {COOKBOOK}"


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: str(p.relative_to(COOKBOOK)))
def test_recipe_is_indexed(recipe: Path):
    linked = {(INDEX.parent / target).resolve() for target in _relative_links(INDEX)}
    assert (recipe / "README.md").resolve() in linked, f"{recipe.relative_to(COOKBOOK)} is missing from {INDEX.name}"


@pytest.mark.parametrize("markdown", MARKDOWN, ids=lambda p: str(p.relative_to(COOKBOOK)))
def test_relative_links_resolve(markdown: Path):
    broken = [t for t in _relative_links(markdown) if t and not (markdown.parent / t).exists()]
    assert not broken, f"{markdown.relative_to(COOKBOOK)} links to missing paths: {broken}"


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: str(p.relative_to(COOKBOOK)))
def test_recipe_tasks_load_with_a_solution(recipe: Path, tmp_path: Path):
    if not _holds_tasks(recipe):
        pytest.skip("recipe has no task folders")
    root = render_template_folder(recipe, tmp_path / recipe.name) if is_template_folder(recipe) else recipe
    try:
        tasks = discover_tasks(root)
    except HarborTaskError as error:
        # A Dockerfile that copies files only loads after its image is built, which needs a container runtime.
        # Check the files a solution needs here; the sandboxed smoke run loads and plays these tasks in full.
        if not isinstance(error.__cause__, DockerfileNeedsBuild):
            raise
        folders = [root] if is_task_folder(root) else iter_task_folders(root)
        missing = [f.name for f in folders if not any((f / "solution" / n).is_file() for n in SOLUTION_FILES)]
    else:
        assert tasks, f"{recipe.relative_to(COOKBOOK)} has no tasks"
        missing = [task.task_id for task in tasks if not task.has_solution]
    assert not missing, f"{recipe.relative_to(COOKBOOK)} tasks without a reference solution: {missing}"


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: str(p.relative_to(COOKBOOK)))
def test_recipe_shows_expected_results(recipe: Path):
    section = _section((recipe / "README.md").read_text(), EXPECTED_RESULTS_HEADING)
    assert section is not None, f"{recipe.relative_to(COOKBOOK)} README has no '{EXPECTED_RESULTS_HEADING}' section"
    shown = [line for line in section.splitlines() if line.strip() and not line.startswith("TODO")]
    assert shown, f"{recipe.relative_to(COOKBOOK)} '{EXPECTED_RESULTS_HEADING}' section shows no result yet"


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: str(p.relative_to(COOKBOOK)))
def test_recipe_is_run_by_a_test(recipe: Path):
    # Task-folder recipes are played by the sandboxed smoke run; notebooks run top to bottom; anything else
    # needs its own test_recipe.py.
    runnable = _holds_tasks(recipe) or any(recipe.glob("*.ipynb")) or (recipe / "test_recipe.py").is_file()
    assert runnable, f"{recipe.relative_to(COOKBOOK)} is not run by any test: add a test_recipe.py"
