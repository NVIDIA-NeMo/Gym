# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require the selection fields in every environment and benchmark manifest.

The selection fields tell people whether an environment suits them: who maintains it, what it is for, which
capabilities and domains it covers, a benchmark's headline metric, and what a run needs. The schema keeps them
optional, so a manifest written before them still loads; this hook requires them in every manifest in this repository.
A setup or protocol that a main manifest lists takes the main manifest's answers and sets only those that differ, so
it is not checked on its own.

``unknown`` is accepted where nobody could establish a value: ``capabilities``, ``domains``, ``requirements.gpu``,
``requirements.network`` and ``primary_metric``. An empty ``capabilities`` or ``domains`` list means none applies. A
scaffold placeholder is not a value: ``TODO``, or ``TODO:`` followed by an instruction, in any case. This is the check
``gym env publish`` makes, so a real login such as ``todorov`` passes. This hook checks presence only;
``gym env validate`` checks the values against the schema.
"""

import sys
from pathlib import Path

import yaml


CATALOG_TREES = ("benchmarks", "environments")
FIELD_REFERENCE = "fern/versions/latest/pages/contribute/environments/new-environment.mdx"


def _is_placeholder(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().casefold()
    return text == "todo" or text.startswith("todo:")


def missing_selection_fields(manifest: dict) -> list[str]:
    """The selection fields a manifest leaves out or leaves as the scaffold's ``TODO``."""
    missing = []
    maintainers = manifest.get("maintainers")
    if not isinstance(maintainers, list) or not maintainers or any(map(_is_placeholder, maintainers)):
        missing.append("maintainers")
    if not manifest.get("intended_use"):
        missing.append("intended_use")
    for name in ("capabilities", "domains"):
        labels = manifest.get(name)
        if (
            labels is None
            or _is_placeholder(labels)
            or (isinstance(labels, list) and any(map(_is_placeholder, labels)))
        ):
            missing.append(name)
    requirements = manifest.get("requirements")
    requirements = requirements if isinstance(requirements, dict) else {}
    missing.extend(f"requirements.{key}" for key in ("gpu", "network") if requirements.get(key) is None)
    primary_metric = manifest.get("primary_metric")
    if manifest.get("kind") == "benchmark" and (primary_metric is None or _is_placeholder(primary_metric)):
        missing.append("primary_metric")
    return missing


def is_catalog_manifest(path: Path) -> bool:
    """Whether a file is a catalog manifest.

    As in catalog discovery, hidden folders and the task folders below a Harbor dataset's ``manifest.toml`` hold
    data, not catalog entries.
    """
    parts = path.parts
    if path.name != "manifest.yaml" or len(parts) < 3 or parts[0] not in CATALOG_TREES:
        return False
    if any(part.startswith(".") for part in parts[1:-1]):
        return False
    tree_root = Path(parts[0])
    for folder in path.parents[1:]:
        if folder == tree_root:
            break
        if (folder / "manifest.toml").is_file():
            return False
    return True


def check(paths: list[Path]) -> list[str]:
    """One problem per catalog manifest that lacks a selection field."""
    problems = []
    for path in paths:
        if not is_catalog_manifest(path):
            continue
        try:
            manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            continue  # `gym env validate` reports a manifest that cannot be read.
        if not isinstance(manifest, dict):
            continue
        missing = missing_selection_fields(manifest)
        if missing:
            problems.append(
                f"{path}: set {', '.join(missing)}. Write `unknown` for capabilities, domains, a requirement or a "
                "headline metric nobody knows, and replace every scaffold TODO. "
                f"See the field reference in {FIELD_REFERENCE}."
            )
    return problems


def main(argv: list[str]) -> int:
    problems = check([Path(arg) for arg in argv])
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
