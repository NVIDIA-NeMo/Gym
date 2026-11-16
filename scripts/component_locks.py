# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check or regenerate every managed dependency lock and its shared uv policy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPONENT_ROOTS = (
    "resources_servers",
    "responses_api_agents",
    "responses_api_models",
    "environment_servers",
    "benchmarks",
)
POLICY_KEYS = ("constraint-dependencies", "override-dependencies", "exclude-dependencies")


def managed_projects(root: Path = ROOT) -> list[Path]:
    """Find tracked managed projects, including newly staged components."""
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, text=True
    )
    projects = {root}
    for name in paths.splitlines():
        path = Path(name)
        if path.name == "uv.lock" and (root / path).is_file():
            projects.add((root / path).parent)
        elif path.name == "app.py" and len(path.parts) == 3 and path.parts[0] in COMPONENT_ROOTS:
            projects.add((root / path).parent)
    return sorted(projects)


def dependency_name(value: str) -> str:
    """Normalize a policy requirement's distribution name without importing Gym."""
    import re

    return re.split(r"[\s\[<>=!~;@]", value, maxsplit=1)[0].lower().replace("_", "-").replace(".", "-")


def expected_policy(project: Path, root: Path = ROOT) -> dict[str, list[str]]:
    """Compose the root policy with explicitly recorded component exceptions."""
    baseline = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]
    exceptions = json.loads((root / "scripts/component-lock-policy.json").read_text())
    local = exceptions.get(project.relative_to(root).as_posix(), {})
    result = {}
    for key in POLICY_KEYS:
        # Standalone evaluators inherit security floors, not Gym's runtime exclusions/overrides.
        inherited = baseline.get(key, []) if not local.get("standalone") or key == "constraint-dependencies" else []
        values = {dependency_name(value): value for value in inherited}
        values.update({dependency_name(value): value for value in local.get(key, [])})
        if key == "exclude-dependencies":
            for name in local.get("include-dependencies", []):
                values.pop(name, None)
        result[key] = list(values.values())
    return result


def uses_gym(project: Path) -> bool:
    """Identify projects that must inherit Gym's resolver policy."""
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    return any(dependency_name(value) == "nemo-gym" for value in metadata.get("project", {}).get("dependencies", []))


def uses_shared_policy(project: Path, root: Path = ROOT) -> bool:
    """Include explicitly registered standalone scorers in dependency policy checks."""
    exceptions = json.loads((root / "scripts/component-lock-policy.json").read_text())
    local = exceptions.get(project.relative_to(root).as_posix(), {})
    return uses_gym(project) or local.get("standalone") is True


def synchronize_policy(project: Path) -> None:
    """Update only shared policy fields while preserving component metadata."""
    import tomlkit

    path = project / "pyproject.toml"
    metadata = tomlkit.parse(path.read_text())
    uv = metadata.setdefault("tool", {}).setdefault("uv", {})
    for key, values in expected_policy(project).items():
        array = tomlkit.array().multiline(True)
        array.extend(values)
        uv[key] = array
    root_uv = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]
    uv["required-version"] = root_uv["required-version"]
    sources = uv.setdefault("sources", {})
    if uses_gym(project):
        for name, source in root_uv.get("sources", {}).items():
            sources[name] = source
    path.write_text(tomlkit.dumps(metadata))


def check_project(project: Path, *, update: bool) -> tuple[Path, int, str]:
    """Run uv for one project and retain every failure's diagnostics."""
    if not (project / "pyproject.toml").is_file():
        return project, 1, "Missing pyproject.toml"
    if project != ROOT and uses_shared_policy(project):
        if update:
            synchronize_policy(project)
        else:
            actual = tomllib.loads((project / "pyproject.toml").read_text()).get("tool", {}).get("uv", {})
            root_sources = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"].get("sources", {})
            source_drift = uses_gym(project) and any(
                actual.get("sources", {}).get(name) != value for name, value in root_sources.items()
            )
            required = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["required-version"]
            if (
                actual.get("required-version") != required
                or source_drift
                or any(actual.get(key, []) != values for key, values in expected_policy(project).items())
            ):
                return project, 1, "Shared uv policy differs; run scripts/component_locks.py --update"
    command = ["uv", "lock", "--project", str(project)]
    if not update:
        command.append("--check")
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return project, result.returncode, result.stdout


def main() -> None:
    """Check every lock by default, or explicitly regenerate locks and policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="Copy shared policy and regenerate all locks")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    projects = managed_projects()
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = executor.map(lambda project: check_project(project, update=args.update), projects)
        for project, code, output in results:
            relative = project.relative_to(ROOT).as_posix()
            print(f"{'FAIL' if code else 'PASS'} {relative}", flush=True)
            if code:
                failures.append(relative)
                print(output, file=sys.stderr, flush=True)
    print(f"Checked {len(projects)} projects; {len(failures)} failed.", flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
