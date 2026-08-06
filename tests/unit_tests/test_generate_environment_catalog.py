# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import scripts.generate_environment_catalog as generator
import yaml
from scripts.generate_environment_catalog import (
    catalog_payload,
    render_catalog_html,
    render_catalog_json,
    write_catalog_artifacts,
)

from nemo_gym.environment_catalog import (
    CatalogCoverage,
    CatalogIssue,
    EnvironmentCatalog,
    EnvironmentCatalogEntry,
)
from nemo_gym.environment_versioning import LOCK_RELATIVE_PATH


def _catalog(repo_root: Path) -> EnvironmentCatalog:
    return EnvironmentCatalog(
        entries=(
            EnvironmentCatalogEntry(
                name="alpha",
                version="1.2.3",
                kind="environment",
                status="experimental",
                lifecycle="active",
                source="manifest",
                config_path=repo_root / "environments" / "alpha" / "config.yaml",
                manifest_path=repo_root / "environments" / "alpha" / "manifest.yaml",
                integration_profile="stock-loop",
                domain="math",
                description="Evaluate <unsafe> arithmetic & reasoning",
                modality="text",
                licensing="Apache-2.0",
                authors=("@team",),
                determinism="seeded",
                required_capabilities=frozenset({"text-model", "verification"}),
            ),
            EnvironmentCatalogEntry(
                name="legacy",
                kind="benchmark",
                status="no-manifest",
                source="legacy",
                config_path=repo_root / "benchmarks" / "legacy" / "config.yaml",
                manifest_path=None,
                description="Legacy benchmark",
            ),
        ),
        coverage=CatalogCoverage(total=2, with_manifest=1, without_manifest=1, invalid_manifests=1),
        issues=(
            CatalogIssue(
                path=repo_root / "environments" / "bad" / "manifest.yaml",
                message=f"Invalid manifest {repo_root / 'environments' / 'bad' / 'manifest.yaml'}: bad <field>",
            ),
        ),
    )


def test_catalog_payload_is_cli_shaped_deterministic_and_clone_relative(tmp_path: Path) -> None:
    payload = catalog_payload(_catalog(tmp_path), tmp_path)
    first = render_catalog_json(payload)
    second = render_catalog_json(catalog_payload(_catalog(tmp_path), tmp_path))

    assert first == second
    decoded = json.loads(first)
    assert set(decoded) == {"entries", "coverage", "issues", "filter_limitations"}
    assert decoded["entries"][0]["config_path"] == "environments/alpha/config.yaml"
    assert decoded["entries"][0]["manifest_path"] == "environments/alpha/manifest.yaml"
    assert decoded["entries"][0]["required_capabilities"] == ["text-model", "verification"]
    assert decoded["issues"][0]["path"] == "environments/bad/manifest.yaml"
    assert decoded["issues"][0]["message"].startswith("Invalid manifest environments/bad/manifest.yaml")
    assert str(tmp_path) not in first


def test_static_html_contains_all_records_and_accessible_structure(tmp_path: Path) -> None:
    payload = catalog_payload(_catalog(tmp_path), tmp_path)
    rendered = render_catalog_html(payload)

    assert '<html lang="en">' in rendered
    assert 'href="#catalog">Skip to catalog' in rendered
    assert "<caption>Published and migration-era runnable Gym units</caption>" in rendered
    assert 'scope="col"' in rendered
    assert 'scope="row"' in rendered
    assert "alpha" in rendered and "legacy" in rendered
    assert "text-model, verification" in rendered
    assert "environments/alpha/manifest.yaml" in rendered
    assert "Evaluate &lt;unsafe&gt; arithmetic &amp; reasoning" in rendered
    assert "bad &lt;field&gt;" in rendered
    assert "invalid-manifest" in rendered
    assert "<unsafe>" not in rendered
    assert str(tmp_path) not in rendered


def test_static_html_aggregates_deferred_migration_diagnostics(tmp_path: Path) -> None:
    deferred = tuple(
        CatalogIssue(
            path=tmp_path / "benchmarks" / f"draft-{index}" / "manifest.yaml",
            message=f"draft detail {index}",
            code="migration-draft",
        )
        for index in range(12)
    ) + (
        CatalogIssue(
            path=tmp_path / "resources_servers" / "ambiguous",
            message="ambiguous detail",
            code="ambiguous-legacy-resource",
        ),
        CatalogIssue(
            path=tmp_path / LOCK_RELATIVE_PATH,
            message="lock is corrupt",
            code="invalid-version-lock",
        ),
    )
    catalog = EnvironmentCatalog(
        entries=(),
        coverage=CatalogCoverage(total=0, with_manifest=0, without_manifest=0, invalid_manifests=0),
        issues=deferred,
    )

    payload = catalog_payload(catalog, tmp_path)
    rendered = render_catalog_html(payload)

    assert len(payload["issues"]) == 14
    assert "12 generated migration drafts" in rendered
    assert "1 ambiguous legacy component" in rendered
    assert 'href="catalog.json"' in rendered
    assert "invalid-version-lock" in rendered
    assert "lock is corrupt" in rendered
    assert "environment-composition-locks.json" in rendered
    assert "draft detail" not in rendered
    assert "ambiguous detail" not in rendered
    assert rendered.count("<li>") == 2


def test_write_and_check_catalog_artifacts_share_one_payload(tmp_path: Path) -> None:
    json_path = tmp_path / "site" / "catalog.json"
    html_path = tmp_path / "site" / "index.html"
    catalog = _catalog(tmp_path)

    assert not write_catalog_artifacts(
        catalog,
        repo_root=tmp_path,
        json_output=json_path,
        html_output=html_path,
    )
    assert write_catalog_artifacts(
        catalog,
        repo_root=tmp_path,
        json_output=json_path,
        html_output=html_path,
        check=True,
    )
    html_path.write_text("stale")
    assert not write_catalog_artifacts(
        catalog,
        repo_root=tmp_path,
        json_output=json_path,
        html_output=html_path,
        check=True,
    )
    assert html_path.read_text() == "stale"


def test_fail_on_issues_blocks_invalid_version_lock(tmp_path: Path, monkeypatch) -> None:
    catalog = EnvironmentCatalog(
        entries=(),
        coverage=CatalogCoverage(total=0, with_manifest=0, without_manifest=0, invalid_manifests=0),
        issues=(
            CatalogIssue(
                path=tmp_path / LOCK_RELATIVE_PATH,
                message="lock is corrupt",
                code="invalid-version-lock",
            ),
        ),
    )
    monkeypatch.setattr(generator, "discover_environment_catalog", lambda: catalog)

    assert (
        generator.main(
            [
                "--repo-root",
                str(tmp_path),
                "--json-output",
                "catalog.json",
                "--html-output",
                "index.html",
                "--fail-on-issues",
            ]
        )
        == 1
    )


def test_publication_workflow_is_main_only_ephemeral_and_least_privileged() -> None:
    workflow_path = Path(__file__).parents[2] / ".github/workflows/publish-environment-catalog.yml"

    workflow = yaml.load(workflow_path.read_text(), Loader=yaml.BaseLoader)
    triggers = workflow["on"]
    assert triggers["push"]["branches"] == ["main"]
    assert "paths" not in triggers["push"]
    assert "workflow_dispatch" in triggers
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {
        "group": "environment-catalog-pages",
        "cancel-in-progress": "false",
    }

    build = workflow["jobs"]["build"]
    assert build["if"] == "github.ref == 'refs/heads/main'"
    steps = {step["name"]: step for step in build["steps"]}
    assert steps["Checkout repository"]["uses"] == ("actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803")
    generation_command = steps["Generate and validate catalog"]["run"]
    assert generation_command.count("$RUNNER_TEMP/environment-catalog") == 4
    assert "--fail-on-issues" in generation_command
    assert "--check" not in generation_command
    assert steps["Upload GitHub Pages artifact"] == {
        "name": "Upload GitHub Pages artifact",
        "uses": "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9",
        "with": {"path": "${{ runner.temp }}/environment-catalog"},
    }

    deploy = workflow["jobs"]["deploy"]
    assert deploy["needs"] == "build"
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert deploy["environment"] == {
        "name": "github-pages",
        "url": "${{ steps.deployment.outputs.page_url }}",
    }
    assert deploy["steps"] == [
        {
            "name": "Deploy to GitHub Pages",
            "id": "deployment",
            "uses": "actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128",
        }
    ]
