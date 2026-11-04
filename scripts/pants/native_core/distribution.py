# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the sampled project's real console scripts through Pants local distributions."""

import tomllib
from collections import Counter
from email.parser import Parser
from importlib.metadata import Distribution
from pathlib import Path, PurePosixPath


def distribution_targets(repo: Path) -> str:
    """Build the original project, preserving its metadata and build-directory layout.

    Test sources retain their ordinary Python dependency inference. The distribution
    build receives the same declared checkout files and uses its original PEP 517
    backend. A shared distribution works with both test resolves: dependency versions
    remain supplied by their separately validated uv-derived locks.
    """
    project = tomllib.loads((repo / "pyproject.toml").read_text())
    name = project["project"]["name"]
    egg_base = project.get("tool", {}).get("distutils", {}).get("egg_info", {}).get("egg_base", ".")
    if not isinstance(egg_base, str) or PurePosixPath(egg_base).is_absolute() or ".." in PurePosixPath(egg_base).parts:
        raise ValueError("The sampled project's egg_base must stay within its checkout")
    placeholder = "scripts/pants/native_core/build-directory"
    (repo / placeholder).write_text("Declare the sampled project's build metadata directory.\n")
    return (
        "files(name='native-project-metadata', sources=['pyproject.toml'])\n"
        f"files(name='native-build-directory', sources=[{placeholder!r}])\n"
        "relocated_files(name='native-project-build-dir', files_targets=[':native-build-directory'], "
        f"src='scripts/pants/native_core', dest={egg_base!r})\n"
        "python_distribution(name='native-project', "
        f"provides=python_artifact(name={name!r}), generate_setup=False, sdist=False, "
        "dependencies=[':assets', ':native-project-metadata', ':native-project-build-dir'])\n"
    )


def validate_distribution(expected_path: Path, installed: Distribution) -> dict[str, str]:
    """Reject project metadata drift before replacing it with editable metadata."""
    expected = Distribution.at(expected_path)
    for key in ("Name", "Version", "Requires-Python", "Requires-Dist", "Provides-Extra"):
        if Counter(expected.metadata.get_all(key, [])) != Counter(installed.metadata.get_all(key, [])):
            raise ValueError(f"Native project distribution differs from sampled editable metadata: {key}")
    expected_entries = Counter((entry.group, entry.name, entry.value) for entry in expected.entry_points)
    installed_entries = Counter((entry.group, entry.name, entry.value) for entry in installed.entry_points)
    if expected_entries != installed_entries:
        raise ValueError("Native project console/plugin entry points differ from sampled editable metadata")
    generators = []
    for package in (expected, installed):
        wheel = package.read_text("WHEEL")
        generator = Parser().parsestr(wheel or "").get("Generator")
        if not generator:
            raise ValueError("Project distribution does not identify its wheel generator")
        generators.append(generator)
    if generators[0] != generators[1]:
        raise ValueError(f"Native wheel backend differs from sampled editable build: {generators}")
    return {"name": expected.metadata["Name"], "version": expected.version, "generator": generators[0]}
