# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a disposable checkout for the full native Pants Python prototype."""

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

from distribution import distribution_targets
from editable import snapshot_editable
from runtime import exclude_instrumentation


EXTRAS = {
    "core": ["dev", "ray", "telemetry", "mlflow", "wandb"],
    "sandbox": ["dev", "ray", "sandbox", "openshell", "mlflow", "wandb"],
}
BATCH_SIZE = 128
UV_VERSION = "0.13.4"
HEADER = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"


def environment_names() -> list[str]:
    """Reuse the controller's explicit noncredential environment allowlist."""
    controller = Path(__file__).resolve().parents[1] / "full_ci.py"
    for node in ast.parse(controller.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ENVIRONMENT" for target in node.targets
        ):
            names = ast.literal_eval(node.value)
            return [name for name in names if name not in {"PYTHONPATH", "PYTHONHOME"}]
    raise ValueError("Cannot locate the full CI environment allowlist")


def snapshot_git(repo: Path) -> None:
    """Declare the sampled commit and its complete tree, without history or credentials."""
    snapshot = repo / "scripts/pants/native_core/git-state"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    if subprocess.check_output(["git", "submodule", "status", "--recursive"], cwd=repo):
        raise ValueError("Native Git snapshot needs explicit submodule support")
    subprocess.run(["git", "init", "--bare", str(snapshot)], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    objects = subprocess.check_output(["git", "rev-list", "--objects", "--no-object-names", "HEAD^{tree}"], cwd=repo)
    subprocess.run(
        ["git", "pack-objects", str(snapshot / "objects/pack/pack")],
        cwd=repo,
        input=sha.encode() + b"\n" + objects,
        check=True,
        capture_output=True,
    )
    (snapshot / "HEAD").write_text(sha + "\n")
    (snapshot / "shallow").write_text(sha + "\n")
    subprocess.run(["git", "--git-dir=" + str(snapshot), "read-tree", "HEAD"], check=True)
    (snapshot / "config").write_text("[core]\n\trepositoryformatversion = 0\n\tbare = false\n\tfilemode = true\n")
    shutil.rmtree(snapshot / "hooks")
    exclude_instrumentation(repo, git_dir=snapshot)


def snapshot_distribution(repo: Path) -> None:
    """Retain the real editable project's metadata built during baseline collection."""
    candidates = list((repo / ".venv").glob("lib/python*/site-packages/nemo_gym-*.dist-info"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one installed Gym distribution after collection, found {len(candidates)}")
    destination = repo / "scripts/pants/native_core/distribution"
    if destination.exists():
        shutil.rmtree(destination)
    snapshot_editable(candidates[0], source=repo, destination=destination)


def require_uv_version(uv: str, expected: str = UV_VERSION) -> None:
    actual = subprocess.check_output([uv, "--version"], text=True).split()[1]
    if actual != expected:
        raise ValueError(f"The baseline pins uv {expected}; pass --uv with that executable (got {actual})")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_pins(text: str) -> dict[str, dict[str, str | bool]]:
    """Reject nonexact requirements and mutable VCS references before benchmarking."""
    result = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        registry = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^;\s]+)(?:\s*;.*)?", line)
        vcs = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])? @ (git\+https://\S+@([a-f0-9]{40}))(?:\s*;.*)?", line)
        if registry:
            name = normalize(registry[1])
            pin = {"version": registry[2], "conditional": ";" in line}
        elif vcs:
            name = normalize(vcs[1])
            pin = {"url": vcs[2], "commit": vcs[3], "conditional": ";" in line}
        else:
            raise ValueError(f"Unsupported or nonexact requirement: {line}")
        if name in result and result[name] != pin:
            raise ValueError(f"Multiple conditional versions need explicit platform handling: {name}")
        result[name] = pin
    return result


def validate_lock(requirements: Path, lock: Path) -> dict[str, object]:
    """Validate PEX's full closure against uv, including content-addressed VCS artifacts."""
    pins = requirement_pins(requirements.read_text())
    payload = json.loads("\n".join(line for line in lock.read_text().splitlines() if not line.startswith(("#", "//"))))
    seen = set()
    for resolve in payload["locked_resolves"]:
        for item in resolve["locked_requirements"]:
            name = normalize(item["project_name"])
            if name not in pins:
                raise ValueError(f"PEX reintroduced a dependency absent from uv: {name}")
            pin = pins[name]
            if "version" in pin and item["version"] != pin["version"]:
                raise ValueError(f"Version mismatch for {name}: {item['version']} != {pin['version']}")
            if "url" in pin:
                urls = [artifact["url"] for artifact in item.get("artifacts", [])]
                if pin["url"] not in urls:
                    raise ValueError(f"PEX did not retain the exact VCS source for {name}: {urls}")
                if any(not artifact.get("hash") for artifact in item.get("artifacts", [])):
                    raise ValueError(f"Missing VCS artifact integrity hash for {name}")
            seen.add(name)
    missing = sorted(name for name, pin in pins.items() if not pin["conditional"] and name not in seen)
    if missing:
        raise ValueError(f"PEX omitted unconditional uv dependencies: {missing}")
    return {"distributions": len(seen), "uv_export_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest()}


def export_requirements(repo: Path, destination: Path, *, uv: str = "uv") -> None:
    from lock_assets import source_uv_version

    require_uv_version(uv, source_uv_version(repo))
    destination.mkdir(parents=True, exist_ok=True)
    for resolve, extras in EXTRAS.items():
        command = [uv, "export", "--locked", "--no-emit-project", "--no-hashes", "--no-annotate"]
        for extra in extras:
            command.extend(["--extra", extra])
        result = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=True)
        requirement_pins(result.stdout)
        (destination / f"{resolve}.requirements.txt").write_text(result.stdout)


def native_coverage_config(project: str, *, finder: str | None) -> str:
    """Keep sampled coverage policy, excluding only our generated import plumbing."""
    original = tomllib.loads(project)["tool"]["coverage"]
    data_file_setting = 'data_file = "results/.coverage"'
    if project.count(data_file_setting) != 1:
        raise ValueError("Coverage output configuration changed; inspect native coverage parity before continuing")
    config = project.replace(data_file_setting, 'data_file = ".coverage"')
    section = re.search(r"(?ms)^\[tool\.coverage\.run\]\s*\n(.*?)(?=^\[|\Z)", config)
    if section is None:
        raise ValueError("Missing sampled coverage run configuration")
    # These files are produced only by our runtime plugins. Real source and all
    # sampled omit/report/branch/context settings remain unchanged.
    additions = [".native-child-site/*", *([finder] if finder else [])]
    body = section[1]
    values = ", ".join(json.dumps(value) for value in additions)
    if "omit" in original["run"]:
        body, count = re.subn(r"(?m)^(omit\s*=\s*\[)", lambda match: match[1] + values + ", ", body, count=1)
        if count != 1:
            raise ValueError("Unsupported sampled coverage omit configuration")
    else:
        body = f"omit = [{values}]\n" + body
    config = config[: section.start(1)] + body + config[section.end(1) :]
    expected = dict(original, run=dict(original["run"], data_file=".coverage"))
    expected["run"]["omit"] = additions + original["run"].get("omit", [])
    if tomllib.loads(config)["tool"]["coverage"] != expected:
        raise ValueError("Native coverage preparation changed sampled coverage policy")
    return config


def write_build_configuration(repo: Path, *, incremental: bool = False) -> dict[str, object]:
    """Declare all test files and broad explicit assets without removing import edges."""
    native = repo / "scripts/pants/native_core"
    snapshot_git(repo)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
    links = {}
    for path in filter(None, tracked):
        if (repo / path).is_symlink():
            target = os.readlink(repo / path)
            if Path(target).is_absolute() or not (repo / path).resolve().is_relative_to(repo.resolve()):
                raise ValueError(f"Tracked symlink needs explicit external input support: {path}")
            links[path] = target
    (native / "symlinks.json").write_text(json.dumps(links, indent=2) + "\n")
    tracked = sorted(path for path in tracked if path and path not in links and (repo / path).is_file())
    tests = [
        path
        for path in tracked
        if path.startswith("tests/") and Path(path).name.startswith("test") and path.endswith(".py")
    ]
    sources = [path for path in tracked if path.endswith(".py") and path not in tests]
    assets = [
        path
        for path in tracked
        if not path.endswith(".py") and path != "pyproject.toml" and Path(path).name != "BUILD"
    ]
    # Separate resolves keep optional-dependency availability identical to the two CI jobs.
    lines = [HEADER]
    for resolve in EXTRAS:
        overrides = ""
        if incremental and resolve == "core":
            from incremental import MIGRATION_TEST

            candidate = json.loads((native / "incremental-policy.json").read_text()).get("candidate", MIGRATION_TEST)
            overrides = f", overrides={{{candidate!r}: {{'dependencies': [':incremental-assets'], 'batch_compatibility_tag': 'incremental-migration'}}}}"
        lines.append(f"python_sources(name={resolve + '-sources'!r}, sources={sources!r}, resolve={resolve!r})\n")
        lines.append(
            f"python_tests(name={resolve + '-tests'!r}, sources={tests!r}, resolve={resolve!r}, "
            f"dependencies=[':assets', ':native-project'], tags=['native-core-{resolve}'], "
            f"batch_compatibility_tag={'native-' + resolve!r}{overrides})\n"
        )
    # Several core tests inspect source files as fixtures (for example every agent
    # app's session declaration). Imports do not describe those reads. Start with
    # an explicit complete repository file dependency; narrow only with parity proof.
    runtime_assets = [str(path.relative_to(repo)) for path in (native / "git-state").rglob("*") if path.is_file()]
    runtime_assets.extend(
        str(path.relative_to(repo)) for path in (native / "distribution").rglob("*") if path.is_file()
    )
    runtime_assets.append("scripts/pants/native_core/runtime.py")
    runtime_assets.append("scripts/pants/native_core/editable.py")
    runtime_assets.append("scripts/pants/native_core/distribution.py")
    runtime_assets.append("scripts/pants/native_core/symlinks.json")
    lines.append(f"files(name='assets', sources={sorted(set(assets + sources + tests + runtime_assets))!r})\n")
    lines.append(distribution_targets(repo))
    if incremental:
        from incremental import build_target

        lines.append(build_target(repo))
    (repo / "BUILD.native").write_text("".join(lines))
    (native / "BUILD.native").write_text(
        HEADER
        + "\n".join(
            f"python_requirements(name={resolve!r}, source={resolve + '.requirements.txt'!r}, resolve={resolve!r}, "
            "module_mapping={'pyyaml': ['yaml'], 'hydra-core': ['hydra']})"
            for resolve in EXTRAS
        )
        + "\n"
    )
    uv_config = tomllib.loads((repo / "pyproject.toml").read_text())["tool"]["uv"]
    editable_manifest = native / "distribution/editable.json"
    finder = json.loads(editable_manifest.read_text())["finder"] if editable_manifest.exists() else None
    if any(path.startswith(".native-child-site/") for path in tracked):
        raise ValueError("Generated child site overlaps sampled source")
    if finder and (finder in tracked or Path(finder).name != finder):
        raise ValueError("Generated editable finder overlaps sampled source or escapes the checkout")
    (native / "coverage.toml").write_text(native_coverage_config((repo / "pyproject.toml").read_text(), finder=finder))
    excludes = uv_config.get("exclude-dependencies", [])
    overrides = uv_config.get("override-dependencies", [])
    config = f"""{HEADER}[GLOBAL]
pants_version = "2.30.2"
backend_packages = ["pants.backend.python", "pants.backend.project_info", "scripts.pants.native_core"]
pythonpath = ["%(buildroot)s"]
build_patterns = ["BUILD.native"]
pants_ignore = ["/.git", "/.venv", "/.pants-bin", "__pycache__"]
pants_ignore_use_gitignore = false
unmatched_build_file_globs = "error"
[source]
root_patterns = ["/"]
[python]
interpreter_constraints = ["CPython=={(repo / ".python-version").read_text().strip()}"]
enable_resolves = true
default_resolve = "core"
resolves = {{core = "scripts/pants/native_core/core.lock", sandbox = "scripts/pants/native_core/sandbox.lock"}}
resolves_to_excludes = {{ __default__ = {json.dumps(excludes)} }}
resolves_to_overrides = {{ __default__ = {json.dumps(overrides)} }}
run_against_entire_lockfile = true
[python-infer]
string_imports = true
[pytest]
install_from_resolve = "core"
requirements = ["pytest", "pytest-asyncio", "pytest-cov", "pytest-xdist", "requests-mock"]
junit_family = "xunit1"
[test]
extra_env_vars = {json.dumps(environment_names())}
batch_size = {BATCH_SIZE}
[coverage-py]
report = ["raw"]
config = "scripts/pants/native_core/coverage.toml"
install_from_resolve = "core"
[pex-cli]
# Pants' bundled PEX 2.66.0 has the VCS symlink hashing bug pex-tool/pex#3148.
version = "v2.101.6"
known_versions = [
  "v2.101.6|macos_x86_64|b46f0b05cbcb28ae11e1bd2d8ee80d4f75c3398292c347f2421f1d8e86985554|5317838",
  "v2.101.6|macos_arm64|b46f0b05cbcb28ae11e1bd2d8ee80d4f75c3398292c347f2421f1d8e86985554|5317838",
  "v2.101.6|linux_x86_64|b46f0b05cbcb28ae11e1bd2d8ee80d4f75c3398292c347f2421f1d8e86985554|5317838",
  "v2.101.6|linux_arm64|b46f0b05cbcb28ae11e1bd2d8ee80d4f75c3398292c347f2421f1d8e86985554|5317838",
]
"""
    (repo / "pants.native.toml").write_text(config)
    if not (repo / "pants.toml").exists():
        # The scie launcher discovers its version before forwarding CLI config.
        (repo / "pants.toml").write_text(HEADER + '[GLOBAL]\npants_version = "2.30.2"\n')
    manifest = {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "test_files": tests,
        "test_file_count": len(tests),
        "python_files": len(sources),
        "asset_files": len(assets),
        "extras": EXTRAS,
    }
    (native / "scope.json").write_text(json.dumps(manifest, indent=2) + "\n")
    exclude_instrumentation(repo)
    return manifest


def prepare(source: Path, destination: Path, *, uv: str = "uv") -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    subprocess.run(["git", "clone", "--shared", "--no-hardlinks", str(source), str(destination)], check=True)
    # A clone checks out only the committed source; prototype files are isolated additions.
    native = destination / "scripts/pants/native_core"
    native.mkdir(parents=True, exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, native / path.name)
    export_requirements(destination, native, uv=uv)
    return write_build_configuration(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("prepare")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--destination", type=Path, required=True)
    create.add_argument("--uv", default="uv")
    check = subparsers.add_parser("validate-lock")
    check.add_argument("requirements", type=Path)
    check.add_argument("lock", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source.resolve(), args.destination.resolve(), uv=args.uv)
    else:
        result = validate_lock(args.requirements, args.lock)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
