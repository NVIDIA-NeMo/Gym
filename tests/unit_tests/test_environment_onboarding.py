# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import pytest

from nemo_gym.environment_onboarding import (
    PUBLISH_PLACEHOLDER,
    EnvironmentOnboardingError,
    find_publish_placeholders,
    publish_environment,
)
from nemo_gym.environment_onboarding import (
    test_environment as exercise_environment,
)
from nemo_gym.environment_scaffold import scaffold_environment
from nemo_gym.environment_validation import EnvironmentValidationError, validate_environment
from nemo_gym.registry import EnvironmentCatalogEntry


def _entry(result, kind: str) -> EnvironmentCatalogEntry:
    return EnvironmentCatalogEntry(
        name=result.asset_dir.name,
        kind=kind,
        path=result.asset_dir,
        config_path=result.asset_dir / "config.yaml",
        manifest_path=result.asset_dir / "manifest.yaml",
        status="experimental",
    )


def _replace_placeholders(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".jsonl", ".md", ".py", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8")
        if PUBLISH_PLACEHOLDER in text:
            path.write_text(text.replace(PUBLISH_PLACEHOLDER, "completed"), encoding="utf-8")


def test_named_test_exercises_generated_scorer_without_services(tmp_path: Path) -> None:
    entry = _entry(scaffold_environment(root=tmp_path, kind="environment", name="sample"), "environment")
    server = tmp_path / "resources_servers" / "sample"
    server.joinpath("helper.py").write_text("READY = True\n", encoding="utf-8")
    app = server / "app.py"
    app.write_text("from .helper import READY\n" + app.read_text(encoding="utf-8"), encoding="utf-8")

    report = exercise_environment(entry)

    assert report.resources_server == "sample"
    assert len(report.cases) == 3


def test_named_test_reports_fixture_failures_as_onboarding_errors(tmp_path: Path) -> None:
    entry = _entry(scaffold_environment(root=tmp_path, kind="environment", name="sample"), "environment")
    cases = tmp_path / "resources_servers" / "sample" / "tests" / "verifier_cases.jsonl"
    cases.write_text(
        cases.read_text(encoding="utf-8").replace('"expected_reward": 1.0', '"expected_reward": 0.5'), encoding="utf-8"
    )

    with pytest.raises(EnvironmentOnboardingError, match="Verifier fixture.*reward mismatch"):
        exercise_environment(entry)


def test_named_test_requires_a_manifest_and_exported_fixture(tmp_path: Path) -> None:
    entry = _entry(scaffold_environment(root=tmp_path, kind="environment", name="sample"), "environment")
    legacy = replace(entry, manifest_path=None)

    with pytest.raises(EnvironmentOnboardingError, match="has no manifest"):
        exercise_environment(legacy)

    _replace_placeholders(tmp_path)
    with pytest.raises(EnvironmentValidationError, match="no manifest"):
        publish_environment(legacy)

    app = tmp_path / "resources_servers" / "sample" / "app.py"
    app.write_text(
        app.read_text(encoding="utf-8").replace(
            "VERIFIER_FIXTURE = VerifierFixture(", "UNUSED_FIXTURE = VerifierFixture("
        ),
        encoding="utf-8",
    )
    with pytest.raises(EnvironmentOnboardingError, match="does not export VERIFIER_FIXTURE"):
        exercise_environment(entry, validate_first=False)


def test_benchmark_reuses_string_match_without_copying_the_server(tmp_path: Path) -> None:
    entry = _entry(
        scaffold_environment(
            root=tmp_path,
            kind="benchmark",
            name="string_match_simple",
            reuse_verifier="string_match",
        ),
        "benchmark",
    )

    report = validate_environment(entry)

    assert "string_match" in {component.implementation for component in report.components}
    assert not (tmp_path / "resources_servers" / "string_match_simple").exists()


def test_publish_rejects_asset_and_generated_component_placeholders(tmp_path: Path) -> None:
    entry = _entry(
        scaffold_environment(root=tmp_path, kind="environment", name="sample", profile="measured-loop"),
        "environment",
    )

    with pytest.raises(EnvironmentOnboardingError, match="scaffold placeholders") as caught:
        publish_environment(entry)

    message = str(caught.value)
    assert "environments/sample/manifest.yaml" in message
    assert "responses_api_agents/sample_agent/app.py" in message
    assert "resources_servers/sample/app.py" in message


def test_placeholder_scan_ignores_generated_directories(tmp_path: Path) -> None:
    authored = tmp_path / "component"
    authored.mkdir()
    authored.joinpath("app.py").write_text("ready = True\n", encoding="utf-8")
    ignored = authored / ".venv" / "lib"
    ignored.mkdir(parents=True)
    ignored.joinpath("dependency.py").write_text(PUBLISH_PLACEHOLDER, encoding="utf-8")

    assert find_publish_placeholders(authored) == ()


def test_placeholder_scan_fails_closed_on_unreadable_authored_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authored = tmp_path / "component"
    authored.mkdir()
    blocked = authored / "app.py"
    blocked.write_text("ready = True\n", encoding="utf-8")
    original_open = Path.open

    def open_file(path: Path, *args, **kwargs):
        if path == blocked:
            raise OSError("permission denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)

    with pytest.raises(EnvironmentOnboardingError, match="Could not inspect authored file"):
        find_publish_placeholders(authored)


def test_publish_is_an_idempotent_local_readiness_gate(tmp_path: Path) -> None:
    entry = _entry(scaffold_environment(root=tmp_path, kind="environment", name="sample"), "environment")
    _replace_placeholders(tmp_path)

    first = publish_environment(entry)
    second = publish_environment(entry)

    assert first == second
    assert first.status == "experimental"


def test_publish_rejects_unexpected_status(tmp_path: Path) -> None:
    entry = _entry(scaffold_environment(root=tmp_path, kind="environment", name="sample"), "environment")
    _replace_placeholders(tmp_path)

    with pytest.raises(EnvironmentOnboardingError, match="unexpected local status"):
        publish_environment(replace(entry, status="no-manifest"))
