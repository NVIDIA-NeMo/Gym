# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/generate_onboarding_index.py"


def make_repo(root):
    (root / "fern/versions/latest/pages/get-started").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.13.14"\n')
    recipes = []
    for name in ("native", "harbor", "verifiers"):
        page = f"get-started/{name}-onboarding.mdx"
        (root / "fern/versions/latest/pages" / page).write_text(f"# {name}\n")
        recipe = {
            "title": name.title(),
            "page": page,
            "prerequisites": "Model endpoint; local setup",
            "review_route": "Component maintainer (assignment pending)",
            "structural": "Offline tests included",
            "runtime": "Real-model run pending",
            "parity": "Not established",
            "trace": "Unavailable",
        }
        if name != "native":
            dependency = f"responses_api_agents/{name}_agent/requirements.txt"
            target = root / dependency
            target.parent.mkdir(parents=True)
            target.write_text(f"{name} @ git+https://example.test/{name}.git@{'a' * 40}\n")
            recipe.update(dependency_file=dependency, dependency_name=name)
        recipes.append(recipe)
    (root / "fern/onboarding-recipes.json").write_text(json.dumps(recipes))
    return root


def load_generator():
    assert SCRIPT.is_file(), "The recipe index generator has not been implemented"
    return runpy.run_path(str(SCRIPT))


def test_index_uses_real_source_pins_and_separate_validation_dimensions(tmp_path):
    repo = make_repo(tmp_path)
    result = load_generator()["render_index"](repo)

    assert "/main/get-started/native-onboarding" in result
    assert "/main/get-started/harbor-onboarding" in result
    assert "/main/get-started/verifiers-onboarding" in result
    assert ">=3.13.14" in result
    assert "a" * 40 in result
    assert "Structural" in result and "Runtime" in result and "Parity" in result and "Trace" in result
    assert "Real-model run pending" in result
    assert "assignment pending" in result
    assert "not a hosted registry" in result

    dependency = repo / "responses_api_agents/verifiers_agent/requirements.txt"
    dependency.write_text("verifiers @ git+https://example.test/verifiers.git@v0.1.14\n")
    assert "v0.1.14" in load_generator()["render_index"](repo)


@pytest.mark.parametrize("page", ["../outside.mdx", "/absolute.mdx", "get-started/missing.mdx"])
def test_rejects_missing_or_outside_recipe_pages(tmp_path, page):
    repo = make_repo(tmp_path)
    metadata = repo / "fern/onboarding-recipes.json"
    recipes = json.loads(metadata.read_text())
    recipes[0]["page"] = page
    metadata.write_text(json.dumps(recipes))

    with pytest.raises(ValueError, match="page"):
        load_generator()["render_index"](repo)


def test_rejects_duplicate_recipe_routes(tmp_path):
    repo = make_repo(tmp_path)
    metadata = repo / "fern/onboarding-recipes.json"
    recipes = json.loads(metadata.read_text())
    recipes.append(recipes[0])
    metadata.write_text(json.dumps(recipes))

    with pytest.raises(ValueError, match="Duplicate"):
        load_generator()["render_index"](repo)


@pytest.mark.parametrize("requirement", ["harbor>=0.1", "harbor @ git+https://example.test/harbor@main"])
def test_rejects_unpinned_adapter_dependencies(tmp_path, requirement):
    repo = make_repo(tmp_path)
    (repo / "responses_api_agents/harbor_agent/requirements.txt").write_text(requirement)

    with pytest.raises(ValueError, match="pin"):
        load_generator()["render_index"](repo)


def test_table_cells_escape_pipes_and_newlines(tmp_path):
    repo = make_repo(tmp_path)
    metadata = repo / "fern/onboarding-recipes.json"
    recipes = json.loads(metadata.read_text())
    recipes[0]["prerequisites"] = "CPU | GPU\naccess"
    metadata.write_text(json.dumps(recipes))

    assert "CPU &#124; GPU access" in load_generator()["render_index"](repo)


def test_check_detects_drift_without_rewriting_output(tmp_path):
    repo = make_repo(tmp_path)
    namespace = load_generator()
    assert "main" in namespace, "Generator needs a write/check command"
    main = namespace["main"]
    output = repo / "fern/versions/latest/pages/get-started/benchmark-onboarding.mdx"
    assert main(["--root", str(repo), "--check"]) == 1
    assert not output.exists()
    assert main(["--root", str(repo)]) == 0
    original = output.read_bytes()
    assert main(["--root", str(repo), "--check"]) == 0
    output.write_text("outdated\n")
    assert main(["--root", str(repo), "--check"]) == 1
    assert output.read_text() == "outdated\n"
    assert main(["--root", str(repo)]) == 0
    assert output.read_bytes() == original


def test_repository_index_is_generated_from_current_recipes_and_pins():
    metadata = ROOT / "fern/onboarding-recipes.json"
    assert metadata.is_file(), "Recipe metadata must be checked in"
    output = ROOT / "fern/versions/latest/pages/get-started/benchmark-onboarding.mdx"
    assert output.is_file(), "Generated recipe index must be checked in"
    assert output.read_text() == load_generator()["render_index"](ROOT)


def test_cli_reports_invalid_inputs_without_writing_output(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "fern/onboarding-recipes.json").write_text("not JSON")
    main = load_generator()["main"]

    assert main(["--root", str(repo)]) == 2
    assert not (repo / "fern/versions/latest/pages/get-started/benchmark-onboarding.mdx").exists()
