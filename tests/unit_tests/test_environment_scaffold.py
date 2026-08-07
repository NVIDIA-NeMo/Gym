# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from nemo_gym.environment_manifest import EnvironmentKind, IntegrationProfile, load_manifest
from nemo_gym.environment_scaffold import (
    SCAFFOLD_PLACEHOLDER,
    ScaffoldConflictError,
    ScaffoldError,
    scaffold_environment,
)


@pytest.mark.parametrize("kind", list(EnvironmentKind))
@pytest.mark.parametrize("profile", list(IntegrationProfile))
def test_scaffolds_every_kind_and_profile(tmp_path: Path, kind: EnvironmentKind, profile: IntegrationProfile) -> None:
    result = scaffold_environment(root=tmp_path, kind=kind, name="sample", profile=profile)
    parent = "benchmarks" if kind == EnvironmentKind.BENCHMARK else "environments"
    asset = tmp_path / parent / "sample"
    manifest = load_manifest(asset / "manifest.yaml")

    assert result.asset_dir == asset
    assert result.created
    assert not result.existing
    assert manifest.kind == kind
    assert manifest.integration_profile == profile
    assert manifest.resources_server == "sample"
    assert manifest.determinism.value == "unknown"
    assert SCAFFOLD_PLACEHOLDER in (asset / "manifest.yaml").read_text(encoding="utf-8")
    assert SCAFFOLD_PLACEHOLDER in (asset / "data" / "example.jsonl").read_text(encoding="utf-8")

    for path in (*result.created, *result.existing):
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    if kind == EnvironmentKind.BENCHMARK:
        assert manifest.canonical_split == SCAFFOLD_PLACEHOLDER
    else:
        assert not (asset / "prepare.py").exists()

    if profile in {IntegrationProfile.MEASURED_LOOP, IntegrationProfile.EXTERNAL_LOOP}:
        agent_dir = tmp_path / "responses_api_agents" / "sample_agent"
        agent_app = agent_dir / "app.py"
        assert SCAFFOLD_PLACEHOLDER in agent_app.read_text(encoding="utf-8")
        assert (agent_dir / "README.md").is_file()
        assert (agent_dir / "tests" / "test_app.py").is_file()
    else:
        assert not (tmp_path / "responses_api_agents" / "sample_agent").exists()

    assert (tmp_path / "resources_servers" / "sample" / "README.md").is_file()

    if profile == IntegrationProfile.CUSTOM_DRIVER:
        driver = asset / "rollout_driver.py"
        assert manifest.rollout_driver == f"{parent}.sample.rollout_driver:run_rollout_collection"
        assert SCAFFOLD_PLACEHOLDER in driver.read_text(encoding="utf-8")
    else:
        assert manifest.rollout_driver is None


def test_generated_benchmark_prepare_writes_clean_domain_rows(tmp_path: Path) -> None:
    result = scaffold_environment(root=tmp_path, kind="benchmark", name="science")
    namespace = runpy.run_path(str(result.asset_dir / "prepare.py"))
    output = tmp_path / "prepared.jsonl"

    namespace["prepare"](result.asset_dir / "data" / "source.jsonl", output)

    assert output.read_text(encoding="utf-8") == '{"question": "What is 6 x 7?", "expected_answer": "42"}\n'

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid source row 1"):
        namespace["prepare"](invalid, output)


def test_identical_rerun_is_a_noop(tmp_path: Path) -> None:
    first = scaffold_environment(root=tmp_path, kind="environment", name="repeatable")
    second = scaffold_environment(root=tmp_path, kind="environment", name="repeatable")

    assert not second.created
    assert set(second.existing) == set(first.created)


def test_conflict_preflight_creates_nothing(tmp_path: Path) -> None:
    manifest = tmp_path / "environments" / "occupied" / "manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("user content\n", encoding="utf-8")

    with pytest.raises(ScaffoldConflictError) as caught:
        scaffold_environment(root=tmp_path, kind="environment", name="occupied")

    assert caught.value.paths == (manifest,)
    assert list(manifest.parent.iterdir()) == [manifest]
    assert not (tmp_path / "resources_servers" / "occupied").exists()

    blocked = tmp_path / "environments" / "blocked"
    blocked.write_text("user content\n", encoding="utf-8")
    with pytest.raises(ScaffoldConflictError) as caught:
        scaffold_environment(root=tmp_path, kind="environment", name="blocked")

    assert caught.value.paths == (blocked,)
    assert blocked.read_text(encoding="utf-8") == "user content\n"


def test_rejects_roots_and_targets_that_traverse_symlinks(tmp_path: Path) -> None:
    root_file = tmp_path / "root-file"
    root_file.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(ScaffoldError, match="root must be a directory"):
        scaffold_environment(root=root_file, kind="environment", name="sample")

    outside = tmp_path / "outside"
    outside.mkdir()
    root_link = tmp_path / "root-link"
    root_link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ScaffoldError, match="root must not be a symlink"):
        scaffold_environment(root=root_link, kind="environment", name="sample")

    root = tmp_path / "root"
    root.mkdir()
    (root / "environments").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ScaffoldError, match="target traverses symlink"):
        scaffold_environment(root=root, kind="environment", name="sample")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "exercise", "name": "sample"}, "kind must be one of"),
        ({"kind": "environment", "name": "../sample"}, "name must contain only"),
        ({"kind": "environment", "name": "sample", "reuse_verifier": "../server"}, "resources-server name"),
        ({"kind": "environment", "name": "class"}, "may not be a keyword"),
        ({"kind": "environment", "name": "sample", "profile": "magic"}, "profile must be one of"),
        (
            {"kind": "environment", "name": "sample-name", "profile": "custom-driver"},
            "valid Python module",
        ),
        (
            {"kind": "environment", "name": "sample", "reuse_verifier": "missing"},
            "was not found",
        ),
        (
            {
                "kind": "environment",
                "name": "sample",
                "profile": "measured-loop",
                "reuse_verifier": "string_match",
            },
            "supports only the stock-loop",
        ),
    ],
)
def test_rejects_invalid_requests_without_writes(tmp_path: Path, kwargs: dict[str, str], message: str) -> None:
    with pytest.raises(ScaffoldError, match=message):
        scaffold_environment(root=tmp_path, **kwargs)

    assert not (tmp_path / "environments").exists()
    assert not (tmp_path / "benchmarks").exists()
