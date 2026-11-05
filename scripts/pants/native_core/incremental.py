# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit input boundary for the expensive repository migration tests."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


MIGRATION_TEST = "tests/unit_tests/tooling/test_add_legacy_agent_environment_servers.py"
LEGACY_MIGRATION_TEST = "tests/unit_tests/test_add_legacy_agent_environment_servers.py"
SCANNED_ROOTS = (
    "benchmarks/",
    "environments/",
    "environment_servers/",
    "resources_servers/",
    "responses_api_agents/",
    "responses_api_models/",
)
AUDITED_FILES = (
    MIGRATION_TEST,
    "scripts/add_legacy_agent_environment_servers.py",
    "tests/conftest.py",
    "tests/unit_tests/conftest.py",
)
ENVIRONMENT = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONPYCACHEPREFIX": ".incremental-bytecode",
    "HOME": "/nonexistent",
    "PATH": "/usr/bin:/bin",
}


def eligibility(source: Path, policy: Path) -> dict[str, object]:
    """A changed external-read boundary returns to forced execution for inspection."""
    expected = json.loads(policy.read_text())
    candidate = expected.get("candidate", MIGRATION_TEST)
    if candidate not in {MIGRATION_TEST, LEGACY_MIGRATION_TEST}:
        raise ValueError("Incremental policy names an unsupported test workload")
    required = {candidate, *AUDITED_FILES[1:]}
    if not required <= expected.get("audited_sha256", {}).keys():
        raise ValueError("Incremental audit omits required test/script/conftest inputs")
    failures = [
        name
        for name, digest in expected["audited_sha256"].items()
        if not (source / name).is_file() or hashlib.sha256((source / name).read_bytes()).hexdigest() != digest
    ]
    if expected.get("uv_lock_sha256") and (
        not (source / "uv.lock").is_file()
        or hashlib.sha256((source / "uv.lock").read_bytes()).hexdigest() != expected["uv_lock_sha256"]
    ):
        failures.append("audited dependency lock changed")
    conftests = {
        str(path.relative_to(source))
        for parent in (source / candidate).parents
        if parent.is_relative_to(source)
        for path in [parent / "conftest.py"]
        if path.exists()
    }
    if conftests != {name for name in AUDITED_FILES if name.endswith("conftest.py")}:
        failures.append("inherited conftest membership")
    try:
        declared_files(source)
    except ValueError as error:
        failures.append(str(error))
    return {"eligible": not failures, "forced_reasons": failures, "candidate": candidate}


def select_policy(source: Path, controller: Path) -> tuple[Path | None, dict[str, object]]:
    """Reuse an audit only while its exact observed code and inherited fixtures match."""
    policies = [controller / "incremental-policy.json", *sorted((controller / "incremental-policies").glob("*.json"))]
    best = {"eligible": False, "forced_reasons": ["No audited policy for sampled code"]}
    best_count = float("inf")
    for policy in policies:
        if not policy.is_file():
            continue
        decision = eligibility(source, policy)
        if decision["eligible"]:
            return policy, decision
        if len(decision["forced_reasons"]) < best_count:
            best = decision
            best_count = len(decision["forced_reasons"])
    return None, best


def declared_files(source: Path) -> list[str]:
    """Inventory YAML scans, resource AST inputs, imports and non-Python fixtures.

    Names as well as contents enter the generated target, so additions/deletions
    invalidate a scanner. Ordinary Python imports retain Pants inference.
    """
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0")
    explicit = {*AUDITED_FILES, LEGACY_MIGRATION_TEST, "pyproject.toml", "uv.lock", ".python-version"}
    selected = sorted(
        name
        for name in tracked
        if name
        and (
            name in explicit
            or name.startswith("resources_servers/")
            or name.endswith((".yaml", ".yml"))
            or (name.startswith(("tests/", "nemo_gym/")) and not name.endswith(".py"))
            or name.startswith("tests/helpers/")
        )
    )
    files = set()
    for name in selected:
        path = source / name
        if path.is_symlink():
            resolved = path.resolve()
            if not resolved.is_relative_to(source.resolve()) or not resolved.is_file():
                raise ValueError(f"Scanner link needs explicit external/directory support: {name}")
            files.add(resolved.relative_to(source.resolve()).as_posix())
        elif not path.is_file():
            raise ValueError(f"Missing scanner input: {name}")
        else:
            files.add(name)
    return sorted(files)


def build_target(source: Path) -> str:
    """Keep Git snapshots and the genuine project wheel out of this target closure."""
    native = source / "scripts/pants/native_core"
    files = declared_files(source)
    configuration = json.loads((native / "configuration-inputs.json").read_text())
    files += configuration["files"]
    links = {}
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0")
    for name in filter(None, tracked):
        path = source / name
        if path.is_symlink() and (
            name.startswith(SCANNED_ROOTS) or name.endswith((".yaml", ".yml")) or name in configuration["files"]
        ):
            links[name] = os.readlink(path)
            files = [item for item in files if item != name]
            resolved = path.resolve()
            if not resolved.is_relative_to(source) or not resolved.is_file():
                raise ValueError(f"Unsupported configuration link: {name}")
            files.append(resolved.relative_to(source).as_posix())
    (native / "incremental-links.json").write_text(json.dumps(links, sort_keys=True) + "\n")
    policy = json.loads((native / "incremental-policy.json").read_text())
    files += list(policy["audited_sha256"])
    files += [str(path.relative_to(source)) for path in (native / "distribution").rglob("*") if path.is_file()]
    files += [
        f"scripts/pants/native_core/{name}"
        for name in (
            "incremental_runtime.py",
            "runtime_fingerprint.py",
            "incremental.py",
            "editable.py",
            "runtime-inputs.json",
            "incremental-policy.json",
            "incremental-links.json",
            "configuration-inputs.json",
        )
    ]
    return f"files(name='incremental-assets', sources={sorted(set(files))!r})\n"
