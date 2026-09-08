# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.environment.onboarding import (
    EnvironmentOnboardingError,
    VerifierRunSpec,
    _import_resources_server,
    _runtime_component_paths,
    verify_environment,
)
from nemo_gym.environment.scaffold import scaffold_environment
from nemo_gym.environment.validation import EnvironmentValidationError
from nemo_gym.registry import EnvironmentCatalogEntry


def _scaffold_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "sample",
) -> EnvironmentCatalogEntry:
    monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(tmp_path))
    asset = scaffold_environment(root=tmp_path, kind="environment", name=name).asset_dir
    return EnvironmentCatalogEntry(
        name=name,
        path=asset,
        config_path=asset / "config.yaml",
        manifest_path=asset / "manifest.yaml",
        kind="environment",
        status="experimental",
    )


async def test_verifies_manifest_resources_server_in_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _scaffold_entry(tmp_path, monkeypatch)

    report = await verify_environment(entry)

    assert report.name == "sample"
    assert report.kind == "environment"
    assert report.resources_server == "sample"
    assert report.fixture_path == str(tmp_path / "resources_servers/sample/tests/verifier_cases.jsonl")
    assert [case.kind for case in report.cases] == ["full_reward", "zero_reward", "malformed"]
    assert report.to_dict()["cases"][0]["observed_rewards"] == (1.0,)


async def test_expected_reward_updates_are_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _scaffold_entry(tmp_path, monkeypatch)
    cases_path = tmp_path / "resources_servers/sample/tests/verifier_cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    cases[0]["expected_reward"] = 0.25
    cases_path.write_text("".join(f"{json.dumps(case)}\n" for case in cases), encoding="utf-8")

    with pytest.raises(EnvironmentOnboardingError, match="reward mismatch"):
        await verify_environment(entry)

    await verify_environment(entry, update_expected=True)

    updated = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    assert updated[0]["expected_reward"] == 1.0


async def test_uses_the_entrypoint_declared_by_resolved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _scaffold_entry(tmp_path, monkeypatch)
    app_path = tmp_path / "resources_servers/sample/app.py"
    app_path.rename(app_path.with_name("verifier.py"))
    config_path = tmp_path / "resources_servers/sample/configs/sample.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("app.py", "verifier.py"),
        encoding="utf-8",
    )

    report = await verify_environment(entry)

    assert report.fixture_path.endswith("resources_servers/sample/tests/verifier_cases.jsonl")


async def test_resources_server_entrypoint_supports_relative_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _scaffold_entry(tmp_path, monkeypatch)
    server_dir = tmp_path / "resources_servers/sample"
    app_path = server_dir / "app.py"
    app_path.with_name("verifier_impl.py").write_text(app_path.read_text(encoding="utf-8"), encoding="utf-8")
    app_path.write_text("from .verifier_impl import VERIFIER_FIXTURE\n", encoding="utf-8")

    report = await verify_environment(entry)

    assert len(report.cases) == 3


def test_bundled_component_is_remapped_to_the_child_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_root = tmp_path / "parent-site-packages"
    child_root = tmp_path / "child-site-packages"
    relative_server = Path("resources_servers/sample")
    parent_server = parent_root / relative_server
    child_server = child_root / relative_server
    parent_server.mkdir(parents=True)
    child_server.mkdir(parents=True)
    (parent_root / "parent_only_dependency.py").write_text("ORIGIN = 'parent'\n", encoding="utf-8")
    (parent_server / "app.py").write_text("ORIGIN = 'parent app'\n", encoding="utf-8")
    (child_server / "helper.py").write_text("ORIGIN = 'child'\n", encoding="utf-8")
    (child_server / "app.py").write_text(
        "from .helper import ORIGIN\n"
        "try:\n"
        "    import parent_only_dependency\n"
        "except ModuleNotFoundError:\n"
        "    PARENT_DEPENDENCY_VISIBLE = False\n"
        "else:\n"
        "    PARENT_DEPENDENCY_VISIBLE = True\n",
        encoding="utf-8",
    )
    spec = VerifierRunSpec(
        name="sample",
        kind="environment",
        resources_server="sample",
        manifest_path="manifest.yaml",
        app_path=str(parent_server / "app.py"),
        component_root=str(parent_root),
        server_dir=str(parent_server),
        bundled_component=True,
        reward_range=(0.0, 1.0),
        higher_is_better=True,
        determinism="unknown",
    )
    monkeypatch.setattr("nemo_gym.environment.onboarding.PARENT_DIR", child_root)

    paths = _runtime_component_paths(spec)
    module = _import_resources_server(*paths)

    assert paths == (child_server / "app.py", child_root.resolve(), child_server)
    assert module.ORIGIN == "child"
    assert module.PARENT_DEPENDENCY_VISIBLE is False


async def test_legacy_entry_has_an_actionable_error(tmp_path: Path) -> None:
    entry = EnvironmentCatalogEntry(
        name="legacy",
        path=tmp_path / "environments/legacy",
        config_path=tmp_path / "environments/legacy/config.yaml",
    )

    with pytest.raises(EnvironmentOnboardingError, match=r"legacy.*has no manifest\.yaml.*Add one"):
        await verify_environment(entry)


async def test_static_validation_runs_before_resources_server_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _scaffold_entry(tmp_path, monkeypatch)
    marker = tmp_path / "imported"
    app_path = tmp_path / "resources_servers/sample/app.py"
    app_path.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tmp_path / "environments/sample/data/example.jsonl").unlink()

    with pytest.raises(EnvironmentValidationError, match="Dataset file was not found"):
        await verify_environment(entry)

    assert not marker.exists()


@pytest.mark.parametrize(
    ("app_source", "message"),
    [
        (None, "was not found"),
        ("VERIFIER_FIXTURE = object()\n", "must export VERIFIER_FIXTURE as a VerifierFixture"),
        ("raise RuntimeError('import failed')\n", "Could not import resources-server entrypoint"),
    ],
)
async def test_rejects_an_unusable_resources_server_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    app_source: str | None,
    message: str,
) -> None:
    entry = _scaffold_entry(tmp_path, monkeypatch)
    app_path = tmp_path / "resources_servers/sample/app.py"
    if app_source is None:
        app_path.unlink()
    else:
        app_path.write_text(app_source, encoding="utf-8")

    with pytest.raises(EnvironmentOnboardingError, match=message):
        await verify_environment(entry)
