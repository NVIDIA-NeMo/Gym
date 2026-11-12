# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from scripts.component_locks import POLICY_KEYS, expected_policy, managed_projects, uses_gym


ROOT = Path(__file__).resolve().parents[3]
COMPONENT_ROOTS = ("resources_servers", "responses_api_agents", "responses_api_models", "environment_servers")


def _tracked_files() -> set[str]:
    return set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())


def _servers(tracked: set[str]) -> list[str]:
    return sorted(
        path.removesuffix("/app.py")
        for path in tracked
        if path.endswith("/app.py") and path.split("/")[0] in COMPONENT_ROOTS
    )


def test_servers_declare_locked_dependencies() -> None:
    tracked = _tracked_files()
    servers = _servers(tracked)
    assert servers
    missing = [
        f"{server}/{name}"
        for server in servers
        for name in ("pyproject.toml", "uv.lock")
        if f"{server}/{name}" not in tracked
    ]
    assert not missing, "Servers must declare dependencies in a committed pyproject.toml and uv.lock:\n" + "\n".join(
        missing
    )


def test_servers_do_not_use_requirements_txt() -> None:
    tracked = _tracked_files()
    legacy = [f"{server}/requirements.txt" for server in _servers(tracked) if f"{server}/requirements.txt" in tracked]
    assert not legacy, "Move these dependencies into pyproject.toml and commit uv.lock:\n" + "\n".join(legacy)


def test_component_uv_policy_matches_root_with_declared_exceptions() -> None:
    """Reject drift in copied constraints, exclusions, overrides, and sources."""
    baseline = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]
    for project in managed_projects():
        if project == ROOT or not uses_gym(project):
            continue
        actual = tomllib.loads((project / "pyproject.toml").read_text())["tool"]["uv"]
        assert actual["required-version"] == baseline["required-version"], str(project)
        for key in POLICY_KEYS:
            assert actual.get(key, []) == expected_policy(project)[key], f"{project}: {key}"
        for name, source in baseline.get("sources", {}).items():
            assert actual.get("sources", {}).get(name) == source, f"{project}: source {name}"


def test_locks_satisfy_shared_security_floors() -> None:
    """Check locked versions against the shared floors and explicit component exceptions."""
    failures = []
    for project in managed_projects():
        if project != ROOT and not uses_gym(project):
            continue
        requirements = [Requirement(value) for value in expected_policy(project)["constraint-dependencies"]]
        floors = {requirement.name: requirement.specifier for requirement in requirements}
        lock = tomllib.loads((project / "uv.lock").read_text())
        for package in lock["package"]:
            if package["name"] in floors and not floors[package["name"]].contains(
                package["version"], prereleases=True
            ):
                failures.append(f"{project}: {package['name']}=={package['version']}")
    assert not failures, "Locked versions below shared floors:\n" + "\n".join(failures)


@pytest.mark.parametrize("component", ["wmt_translation", "longmt_eval"])
def test_comet_locks_respect_encoder_dependency_limits(component: str) -> None:
    """Keep COMET 2.2.7 within the supported Transformers and Hub API ranges."""
    lock = tomllib.loads((ROOT / "resources_servers" / component / "uv.lock").read_text())
    versions = {package["name"]: package["version"] for package in lock["package"] if "version" in package}
    assert versions["unbabel-comet"] == "2.2.7"
    for value in ("transformers>=4.17,<5", "huggingface-hub>=0.19.3,<1"):
        requirement = Requirement(value)
        assert requirement.specifier.contains(versions[requirement.name]), f"{component}: {requirement}"


def test_codec_lock_preserves_the_model_torch_runtime() -> None:
    """Keep the scheduled codec installer aligned with the root model runtime."""
    versions = []
    for path in (ROOT / "uv.lock", ROOT / "docker/excluded_deps/uv.lock"):
        lock = tomllib.loads(path.read_text())
        versions.append({package["version"] for package in lock["package"] if package["name"] == "torch"})
    assert versions[0] and versions[0] == versions[1]


def test_osworld_roles_lock_cpu_torch_and_preserve_cloud_exclusions() -> None:
    """Audit the resolver output for both desktop roles without installing GPU libraries."""
    for component in ("resources_servers/osworld", "responses_api_agents/osworld_agent"):
        project = ROOT / component
        lock = tomllib.loads((project / "uv.lock").read_text())
        torch = [package for package in lock["package"] if package["name"] == "torch"]
        assert torch and all(
            package["source"]["registry"] == "https://download.pytorch.org/whl/cpu" for package in torch
        )
        assert "cryptography" in {package["name"] for package in lock["package"]}
    agent = tomllib.loads((ROOT / "responses_api_agents/osworld_agent/pyproject.toml").read_text())
    excluded = set(agent["tool"]["uv"]["exclude-dependencies"])
    assert {"azure-identity", "azure-mgmt-compute", "azure-mgmt-network", "volcengine-python-sdk"} <= excluded


def test_managed_git_requirements_pin_full_revisions() -> None:
    """Keep direct Git dependencies stable when refreshing an unrelated lock."""
    for project in managed_projects():
        metadata = tomllib.loads((project / "pyproject.toml").read_text())
        for value in metadata.get("project", {}).get("dependencies", []):
            requirement = Requirement(value)
            if requirement.url and requirement.url.startswith("git+"):
                revision = requirement.url.partition("#")[0].rpartition("@")[2]
                assert re.fullmatch("[0-9a-f]{40}", revision), f"{project}: {requirement}"
