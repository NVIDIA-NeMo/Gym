# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check capability-scoped external commits in an OSWorld runtime checkout.

The checker is deliberately read-only. It never fetches, checks out, merges,
or cherry-picks code. Runtime integration branches should contain the exact
external commit as an ancestor so authorship and provenance remain intact.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "runtime_dependencies.toml"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ManifestError(ValueError):
    """Raised when the dependency manifest does not satisfy its schema."""


@dataclass(frozen=True)
class RuntimeDependency:
    id: str
    repository: str
    branch: str
    commit: str
    upstream_pr: str
    upstream_state: str
    author: str
    capabilities: tuple[str, ...]
    integration_policy: str
    reason: str


def _required_string(raw: dict[str, Any], field: str, dependency_id: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"dependency {dependency_id!r} requires a non-empty {field!r}")
    return value.strip()


def load_manifest(path: Path) -> list[RuntimeDependency]:
    """Load and validate a runtime dependency manifest."""

    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(f"could not read {path}: {exc}") from exc

    if payload.get("schema_version") != 1:
        raise ManifestError(f"{path} must declare schema_version = 1")
    raw_dependencies = payload.get("dependency")
    if not isinstance(raw_dependencies, list):
        raise ManifestError(f"{path} must contain at least one [[dependency]] table")

    dependencies: list[RuntimeDependency] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_dependencies):
        if not isinstance(raw, dict):
            raise ManifestError(f"dependency entry {index} must be a table")
        dependency_id = _required_string(raw, "id", f"entry-{index}")
        if dependency_id in seen_ids:
            raise ManifestError(f"duplicate dependency id: {dependency_id}")
        seen_ids.add(dependency_id)

        commit = _required_string(raw, "commit", dependency_id)
        if not COMMIT_PATTERN.fullmatch(commit):
            raise ManifestError(f"dependency {dependency_id!r} commit must be a full lowercase SHA-1")
        capabilities = raw.get("capabilities")
        if (
            not isinstance(capabilities, list)
            or not capabilities
            or any(not isinstance(item, str) or not item.strip() for item in capabilities)
        ):
            raise ManifestError(f"dependency {dependency_id!r} requires non-empty string capabilities")
        integration_policy = _required_string(raw, "integration_policy", dependency_id)
        if integration_policy != "exact-ancestor":
            raise ManifestError(
                f"dependency {dependency_id!r} has unsupported integration_policy {integration_policy!r}"
            )

        dependencies.append(
            RuntimeDependency(
                id=dependency_id,
                repository=_required_string(raw, "repository", dependency_id),
                branch=_required_string(raw, "branch", dependency_id),
                commit=commit,
                upstream_pr=_required_string(raw, "upstream_pr", dependency_id),
                upstream_state=_required_string(raw, "upstream_state", dependency_id),
                author=_required_string(raw, "author", dependency_id),
                capabilities=tuple(item.strip() for item in capabilities),
                integration_policy=integration_policy,
                reason=_required_string(raw, "reason", dependency_id),
            )
        )
    return dependencies


def select_dependencies(
    dependencies: Sequence[RuntimeDependency],
    capabilities: Sequence[str],
) -> list[RuntimeDependency]:
    """Return dependencies required by any selected capability."""

    selected = set(capabilities)
    return [dependency for dependency in dependencies if selected.intersection(dependency.capabilities)]


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def dependency_status(
    repo_root: Path,
    dependency: RuntimeDependency,
    *,
    head: str = "HEAD",
) -> tuple[bool, str]:
    """Return whether an exact dependency commit is an ancestor of ``head``."""

    object_check = _git(repo_root, "cat-file", "-e", f"{dependency.commit}^{{commit}}")
    if object_check.returncode != 0:
        return False, "commit object is not present locally"
    ancestor_check = _git(repo_root, "merge-base", "--is-ancestor", dependency.commit, head)
    if ancestor_check.returncode == 0:
        return True, "exact commit is an ancestor"
    if ancestor_check.returncode == 1:
        return False, "commit exists locally but is not an ancestor"
    detail = ancestor_check.stderr.strip() or f"git exited {ancestor_check.returncode}"
    return False, f"could not evaluate ancestry: {detail}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"dependency manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Gym checkout to inspect (default: current directory)",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="revision whose ancestry should be checked (default: HEAD)",
    )
    parser.add_argument(
        "--capability",
        action="append",
        default=[],
        help="runtime capability to verify; may be repeated",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="verify every declared external dependency",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list declared dependencies without checking Git ancestry",
    )
    return parser


def _print_dependency(dependency: RuntimeDependency) -> None:
    print(f"{dependency.id}:")
    print(f"  capabilities: {', '.join(dependency.capabilities)}")
    print(f"  source:       {dependency.repository} {dependency.branch}")
    print(f"  commit:       {dependency.commit}")
    print(f"  upstream:     {dependency.upstream_pr} ({dependency.upstream_state})")
    print(f"  author:       {dependency.author}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dependencies = load_manifest(args.manifest.resolve())
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.list:
        for dependency in dependencies:
            _print_dependency(dependency)
        return 0

    if args.all and args.capability:
        print("ERROR: use either --all or --capability, not both", file=sys.stderr)
        return 2
    if args.all:
        selected = dependencies
    elif args.capability:
        selected = select_dependencies(dependencies, args.capability)
        missing_capabilities = sorted(
            set(args.capability)
            - {capability for dependency in dependencies for capability in dependency.capabilities}
        )
        if missing_capabilities:
            print(
                f"ERROR: undeclared capability: {', '.join(missing_capabilities)}",
                file=sys.stderr,
            )
            return 2
    else:
        print("ERROR: select --all, --list, or at least one --capability", file=sys.stderr)
        return 2

    failures = 0
    repo_root = args.repo_root.resolve()
    for dependency in selected:
        ok, detail = dependency_status(repo_root, dependency, head=args.head)
        print(f"{'OK' if ok else 'MISSING'} {dependency.id}: {detail}")
        if not ok:
            failures += 1
            print(f"  fetch: git fetch {dependency.repository} {dependency.branch}")
            print("  integrate the exact commit on a separate runtime branch; do not copy it into feature/osworld2")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
