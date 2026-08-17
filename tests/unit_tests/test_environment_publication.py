# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.environment.manifest import dump_manifest, load_manifest
from nemo_gym.environment.onboarding import verify_environment
from nemo_gym.environment.publication import (
    EnvironmentPublicationError,
    EnvironmentPublicationReport,
    finalize_publication,
)
from nemo_gym.environment.scaffold import scaffold_environment
from nemo_gym.environment.validation import validate_environment
from nemo_gym.registry import EnvironmentCatalogEntry, discover_environment_catalog, resolve_catalog_entry


def _entry(tmp_path: Path, **manifest_updates: object) -> EnvironmentCatalogEntry:
    directory = tmp_path / "environments" / "sample"
    directory.mkdir(parents=True)
    manifest = {
        "name": "sample",
        "version": "0.1.0",
        "kind": "environment",
        "integration_profile": "custom-gym-verifier",
        "domain": "other",
        "description": "A complete local fixture environment.",
        "modality": "text",
        "licensing": "Apache-2.0",
        "authors": ["contributor"],
        "reward": {"range": [0, 1], "higher_is_better": True},
        "resources_server": "sample",
        "agent_server": "simple_agent",
        "model_server": "policy_model",
        "datasets": [
            {
                "name": "example",
                "type": "example",
                "jsonl_fpath": "environments/sample/data/example.jsonl",
            }
        ],
    }
    manifest.update(manifest_updates)
    manifest_path = directory / "manifest.yaml"
    manifest_path.write_text(dump_manifest(manifest), encoding="utf-8")
    config_path = directory / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    return EnvironmentCatalogEntry(
        name="sample",
        kind="environment",
        status="experimental",
        path=directory,
        config_path=config_path,
        manifest_path=manifest_path,
    )


def _reports() -> tuple[SimpleNamespace, SimpleNamespace]:
    validation = SimpleNamespace(
        name="sample",
        version="0.1.0",
        kind="environment",
    )
    verifier = SimpleNamespace(name="sample", kind="environment", cases=(object(), object(), object()))
    return validation, verifier


async def test_scaffolded_verifier_journey_reaches_published_catalog_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(tmp_path))
    asset = scaffold_environment(
        root=tmp_path,
        kind="environment",
        name="sample",
    ).asset_dir
    manifest_path = asset / "manifest.yaml"
    manifest = load_manifest(manifest_path).model_copy(
        update={
            "description": "A complete local fixture environment.",
            "licensing": "Apache-2.0",
            "authors": ["contributor"],
        }
    )
    manifest_path.write_text(dump_manifest(manifest), encoding="utf-8")

    entries = discover_environment_catalog()
    entry = resolve_catalog_entry("sample", "environment", entries=entries)
    validation = validate_environment(entry.manifest_path, entry.config_path)
    verifier = await verify_environment(entry)
    report = finalize_publication(entry, validation, verifier, catalog_entries=entries)

    assert (report.name, report.kind, report.status) == ("sample", "environment", "experimental")
    assert report.verifier_cases == 3


@pytest.mark.parametrize(
    "updates",
    [
        {"description": "TODO: describe this environment"},
        {"authors": ["TODO"]},
        {"licensing": "unknown"},
    ],
)
def test_publication_rejects_scaffold_placeholders(tmp_path: Path, updates: dict[str, object]) -> None:
    entry = _entry(tmp_path, **updates)
    validation, verifier = _reports()

    with pytest.raises(EnvironmentPublicationError, match="scaffold placeholders"):
        finalize_publication(entry, validation, verifier, catalog_entries=(entry,))


def test_publication_report_serializes() -> None:
    report = EnvironmentPublicationReport(
        name="sample",
        version="1.0.0",
        kind="environment",
        status="experimental",
        manifest_path="environments/sample/manifest.yaml",
        verifier_cases=3,
    )

    assert report.to_dict()["verifier_cases"] == 3


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_manifest", "no manifest"),
        ("wrong_validation", "Validation report"),
        ("missing_catalog_entry", "Catalog did not resolve"),
        ("wrong_status", "must enter as experimental"),
    ],
)
def test_publication_rejects_inconsistent_inputs(tmp_path: Path, case: str, message: str) -> None:
    entry = _entry(tmp_path)
    validation, verifier = _reports()
    entries: tuple[EnvironmentCatalogEntry, ...] = (entry,)
    if case == "missing_manifest":
        entry = replace(entry, manifest_path=None)
    elif case == "wrong_validation":
        validation.name = "other"
    elif case == "missing_catalog_entry":
        entries = ()
    else:
        entries = (replace(entry, status="validated"),)

    with pytest.raises(EnvironmentPublicationError, match=message):
        finalize_publication(entry, validation, verifier, catalog_entries=entries)


def test_publication_rejects_benchmark_protocol_placeholders(tmp_path: Path) -> None:
    entry = _entry(
        tmp_path,
        kind="benchmark",
        canonical_split="TODO",
        standard_prompt_config="TODO",
        datasets=[
            {
                "name": "test",
                "type": "benchmark",
                "jsonl_fpath": "environments/sample/data/example.jsonl",
                "prepare_script": "environments/sample/prepare.py",
            }
        ],
    )
    validation, verifier = _reports()

    with pytest.raises(EnvironmentPublicationError, match="canonical_split, standard_prompt_config"):
        finalize_publication(entry, validation, verifier, catalog_entries=(entry,))
