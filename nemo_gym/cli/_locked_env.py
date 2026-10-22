# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefer committed component locks while retaining inherited parent requirements."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, unquote, urlsplit

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def _source_matches(requirement_url: str, direct_url: dict) -> bool:
    """Compare a parent distribution's recorded source with the locked install."""
    parsed_url = urlsplit(requirement_url)
    subdirectory = parse_qs(parsed_url.fragment).get("subdirectory", [""])[0]
    if subdirectory != direct_url.get("subdirectory", ""):
        return False
    if requirement_url.startswith("git+"):
        repository, _, revision = requirement_url[4:].partition("#")[0].rpartition("@")
        vcs = direct_url.get("vcs_info", {})
        return direct_url.get("url", "").removesuffix(".git") == repository.removesuffix(".git") and revision in (
            vcs.get("commit_id"),
            vcs.get("requested_revision"),
        )
    return unquote(parsed_url._replace(fragment="").geturl()) == unquote(direct_url.get("url", ""))


def check_parent_requirements(requirements: list[str], constraints: list[str]) -> None:
    """Reject parent/server skew instead of modifying a locked environment."""
    pending = [(value, True) for value in requirements] + [(value, False) for value in constraints]
    checked: set[tuple[str, bool]] = set()
    while pending:
        value, required = pending.pop()
        if (value, required) in checked:
            continue
        checked.add((value, required))
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            distribution = importlib.metadata.distribution(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            if not required:
                continue
            raise RuntimeError(f"Parent requires {value}, but it is absent from the component lock.") from None
        # This validates an installed version, not candidate selection by a resolver.
        if not requirement.specifier.contains(distribution.version, prereleases=True):
            raise RuntimeError(
                f"Parent requires {value}, but the component installed {requirement.name}=={distribution.version}."
            )
        if requirement.url:
            direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
            if not _source_matches(requirement.url, direct_url):
                raise RuntimeError(f"Parent source for {requirement.name} differs from the component lock: {value}")
        for extra in requirement.extras:
            provided = {canonicalize_name(v) for v in distribution.metadata.get_all("Provides-Extra", [])}
            if canonicalize_name(extra) not in provided:
                raise RuntimeError(f"Parent requires undeclared extra {requirement.name}[{extra}].")
            for dependency in distribution.requires or []:
                parsed = Requirement(dependency)
                if parsed.marker is None or parsed.marker.evaluate({"extra": extra}):
                    parsed.marker = None
                    pending.append((str(parsed), True))


def _wheel_project(project: Path, venv: Path) -> Path:
    """Restore source metadata shipped in the wheel for its relative Gym dependency.

    A wheel has no checkout-level pyproject. Reconstruct that same source layout
    beside the component venv, reusing the installed package, not a PyPI re-resolution.
    Keep it for the lifetime of the venv because Gym is installed editably from it.
    """
    import nemo_gym

    install_root = Path(nemo_gym.__file__).resolve().parent.parent
    snapshot = install_root / "nemo_gym/resources/locked_project"
    if not (snapshot / "pyproject.toml").is_file():
        raise RuntimeError("The installed Gym wheel has no locked-project metadata; reinstall a compatible wheel.")
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    relative_source = metadata["tool"]["uv"]["sources"]["nemo-gym"]["path"]
    source_root = (project / relative_source).resolve()
    relative = project.resolve().relative_to(source_root)
    root = venv.parent / ".nemo-gym-project" / venv.name
    # This overlay is private to the venv. Invalidate its links when either the
    # installed Gym or the selected external component moves; otherwise an
    # upgrade would keep importing files from the previous installation.
    identity = json.dumps([str(install_root), str(project.resolve())])
    marker = root / ".source.json"
    if root.exists() and (not marker.is_file() or marker.read_text() != identity):
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(identity)
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        target = root / name
        if not target.is_file() or target.read_bytes() != (snapshot / name).read_bytes():
            shutil.copyfile(snapshot / name, target)
    (root / "cache").mkdir(exist_ok=True)
    container = root
    for part in relative.parts:
        for child in (source_root / container.relative_to(root)).iterdir():
            target = container / child.name
            if (
                child.name != part
                and not target.exists()
                and (
                    container != root
                    or child.name
                    in (
                        "nemo_gym",
                        "benchmarks",
                        "environments",
                        "environment_servers",
                        "resources_servers",
                        "responses_api_agents",
                        "responses_api_models",
                    )
                )
            ):
                target.symlink_to(child, target_is_directory=child.is_dir())
        container = container / part
        container.mkdir(exist_ok=True)
    # An external component need not share the installed wheel's site-packages.
    # The reconstructed root must still build/import that exact Gym package.
    package_link = root / "nemo_gym"
    if package_link.is_symlink():
        package_link.unlink()
    package_link.symlink_to(install_root / "nemo_gym", target_is_directory=True)
    destination = container
    for child in project.iterdir():
        if child.name in (".venv", ".nemo-gym-project") or child.name.endswith(".setup.lock"):
            continue
        target = destination / child.name
        if child.name in ("pyproject.toml", "uv.lock"):
            if not target.is_file() or target.read_bytes() != child.read_bytes():
                shutil.copyfile(child, target)
        elif not target.exists():
            target.symlink_to(child, target_is_directory=child.is_dir())
    return destination


def _install_parent_compatible(
    project: Path,
    venv: Path,
    requirements: list[str],
    constraints: list[str],
    extras: tuple[str, ...],
    env: dict[str, str],
    verbose: bool,
) -> None:
    """Resolve the original project with inherited requirements, never mutate its lock.

    This is the pre-lock installer contract for parent-only provider plugins and
    compatible OpenAI versions. Ray constraints are also overrides, and apply only
    when the component requires Ray; they must not add Ray to other components.
    """
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    command = ["uv", "pip", "install", "--python", str(venv / "bin/python")]
    config = project / "uv.toml"
    if config.is_file():
        command.extend(["--config-file", str(config)])
    if metadata.get("tool", {}).get("uv", {}).get("package") is False:
        command.extend(["--requirements", str(project / "pyproject.toml")])
        for extra in extras:
            command.extend(["--extra", extra])
    else:
        selected = f".[{','.join(extras)}]" if extras else "."
        command.extend(["--editable", selected])
    command.extend(requirements)
    if verbose:
        command.append("--verbose")
    with TemporaryDirectory(prefix="gym-parent-constraints-") as directory:
        if constraints:
            path = Path(directory) / "constraints.txt"
            path.write_text("\n".join(constraints) + "\n")
            command.extend(["--constraint", str(path), "--overrides", str(path)])
        subprocess.run(command, cwd=project, env=env, check=True)


def sync_environment(
    project: Path,
    venv: Path,
    python: str,
    requirements: list[str],
    constraints: list[str],
    *,
    extras: tuple[str, ...] = (),
    verbose: bool = False,
) -> None:
    """Sync the lock; resolve compatibility only for inherited parent mismatches."""
    project = project.absolute()
    for filename in ("pyproject.toml", "uv.lock"):
        if not (project / filename).is_file():
            raise RuntimeError(f"Missing {filename} in {project}; run `uv lock` and commit the result before setup.")
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    gym_source = metadata.get("tool", {}).get("uv", {}).get("sources", {}).get("nemo-gym", {})
    relative_source = gym_source.get("path")
    if relative_source and not (project / relative_source / "pyproject.toml").is_file():
        project = _wheel_project(project, venv)
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    # An enclosing training workspace or an inherited active venv must not redirect sync.
    command = ["uv", "sync", "--locked", "--project", str(project), "--python", python, "--no-dev"]
    selected_extras = list(extras)
    if any(canonicalize_name(Requirement(value).name) == "nemo-lens" for value in requirements):
        selected_extras.append("telemetry")
    selected_extras = tuple(dict.fromkeys(selected_extras))
    for extra in selected_extras:
        command.extend(["--extra", extra])
    if verbose:
        command.append("--verbose")
    subprocess.run(command, env=env, check=True)
    check = [str(venv / "bin/python"), "-I", str(Path(__file__).resolve()), "--check"]
    for value in requirements:
        check.extend(["--requirement", value])
    for value in constraints:
        check.extend(["--constraint", value])
    try:
        subprocess.run(check, check=True)
    except subprocess.CalledProcessError as error:
        # Only an explicit parent-contract mismatch permits compatibility resolution.
        # Failed sync, stale locks, inaccessible sources and checker crashes stay fatal.
        if error.returncode != 2:
            raise
        print("Resolving component dependencies with inherited parent requirements.", file=sys.stderr)
        _install_parent_compatible(project, venv, requirements, constraints, selected_extras, env, verbose)
        subprocess.run(check, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--requirement", action="append", default=[])
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if args.check:
            check_parent_requirements(args.requirement, args.constraint)
        else:
            if args.project is None or args.venv is None or args.python is None:
                parser.error("sync requires --project, --venv and --python")
            sync_environment(
                args.project,
                args.venv,
                args.python,
                args.requirement,
                args.constraint,
                extras=tuple(args.extra),
                verbose=args.verbose,
            )
    except subprocess.CalledProcessError as error:
        print(
            f"Locked environment command failed: {error}\n"
            "See the command output above for dependency access, cache, build, or lock errors; "
            "setup will not fall back to an unlocked install.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    except RuntimeError as error:
        print(
            f"Locked environment setup failed: {error}\n"
            "Parent mismatches use compatibility resolution during component setup; "
            "other failures require fixing the component metadata or environment.",
            file=sys.stderr,
        )
        raise SystemExit(2 if args.check else 1) from error


if __name__ == "__main__":
    main()
