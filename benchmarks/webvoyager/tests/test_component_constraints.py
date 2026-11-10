# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import tomllib
from pathlib import Path

from packaging.markers import Marker, default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[3]
CONSTRAINTS = REPO_ROOT / "benchmarks/webvoyager/component-constraints.txt"
IMAGE_LOCK = REPO_ROOT / "resources_servers/visual_browser/docker/requirements.lock"
PIN = re.compile(r"^([a-z0-9][a-z0-9.-]*)==([^\s=]+)$")


def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        match = PIN.match(line)
        assert match, f"{path.name}: not a normalized exact pin: {line!r}"
        name, version = match.groups()
        assert name not in pins, f"{path.name}: {name} pinned twice"
        pins[name] = version
    return pins


def test_component_constraints_are_exact_pins() -> None:
    pins = _pins(CONSTRAINTS)
    assert not any(name.startswith("nemo-gym") for name in pins)
    # The pins exist to stop drift of the packages that changed what the policy sees.
    assert pins["pillow"] == "12.3.0"
    assert pins["grpcio"] == "1.84.0"


def test_image_lock_agrees_with_component_constraints() -> None:
    constraints = _pins(CONSTRAINTS)
    lock = _pins(IMAGE_LOCK)
    assert set(lock) <= set(constraints), sorted(set(lock) - set(constraints))
    mismatched = {name: (version, constraints[name]) for name, version in lock.items() if constraints[name] != version}
    assert not mismatched, mismatched


def test_root_runtime_dependency_closure_is_constrained() -> None:
    """A new core dependency must not silently float in rebuilt browser roles."""
    constraints = _pins(CONSTRAINTS)
    image = _pins(IMAGE_LOCK)
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    for raw in project["project"]["dependencies"]:
        requirement = Requirement(raw)
        name = canonicalize_name(requirement.name)
        assert name in constraints, f"unconstrained core dependency: {name}"
        assert Version(constraints[name]) in requirement.specifier, raw

    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    packages = {package["name"]: package for package in lock["package"]}
    environment = {
        **default_environment(),
        "python_version": "3.13",
        "python_full_version": "3.13.14",
        "sys_platform": "linux",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_system": "Linux",
        "extra": "",
    }
    pending = [dependency["name"] for dependency in packages["nemo-gym"]["dependencies"]]
    runtime_packages = set()
    while pending:
        name = pending.pop()
        if name in runtime_packages:
            continue
        runtime_packages.add(name)
        pending.extend(
            dependency["name"]
            for dependency in packages[name].get("dependencies", [])
            if not dependency.get("marker") or Marker(dependency["marker"]).evaluate(environment)
        )

    assert not runtime_packages - constraints.keys(), sorted(runtime_packages - constraints.keys())
    assert not runtime_packages - image.keys(), sorted(runtime_packages - image.keys())


def test_browser_image_uv_satisfies_the_core_requirement() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    dockerfile = IMAGE_LOCK.with_name("visual-browser-runtime.Dockerfile").read_text()
    image = re.search(r"^ARG UV_IMAGE=ghcr.io/astral-sh/uv:([^\s@]+)$", dockerfile, re.MULTILINE)
    assert image is not None
    assert Version(image.group(1)) in SpecifierSet(project["tool"]["uv"]["required-version"])
